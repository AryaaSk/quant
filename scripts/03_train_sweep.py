"""Transformer size-sweep driver for earnings-flagship-B.

For each (size_name, layers, model_dim, heads, dropout) in the SIZES table:
  1. Copy the base yaml to markets/earnings-flagship-B-<size_name>.yaml (mutates transformer
     section + slug)
  2. Locally: build state if needed (or reuse from base slug)
  3. Locally: copy state vectors from earnings-flagship-B/ -> earnings-flagship-B-<size>/
     (state vectors are size-independent; sharing them saves Voyage cost)
  4. Sync to H200 via bin/h200 sync, then run scripts/run_remote_pipeline.py for that slug
  5. Pull runs back via bin/h200 pull
  6. Run local backtest + report
  7. Append a summary row to runs/sweep_summary.csv

Idempotent: skips a size if its run directory already has metrics.json.

Usage:
  python scripts/03_train_sweep.py
  python scripts/03_train_sweep.py --only 25M,117M
  python scripts/03_train_sweep.py --sizes-from custom_sizes.yaml
  python scripts/03_train_sweep.py --device cpu       # for dry-runs without H200
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Local-only imports (this script doesn't need quant on path until per-slug runs)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Default size sweep. Each entry is (size_name, layers, model_dim, heads, dropout).
# Params estimate: per_layer ≈ 4*d² + 2*d*(ffn_mult*d) for ffn_mult=4 → 12*d² per layer + input_proj.
# Note: actual params are a bit higher than this estimate due to layer-norms + biases + output head;
# size_name reflects the conservative estimate for clarity.
SIZES = [
    # Reordered: predicted-best first, then LinkedIn headline, then completeness.
    # If H200 has issues mid-sweep, we have the two highest-priority results first.
    # name,   layers, model_dim, heads, dropout
    ("11M",      6,    384,       8,    0.10),  # ~11.3M — predicted sweet spot
    ("117M",    12,    896,      14,    0.15),  # ~117M  — LinkedIn headline number
    ("33M",      8,    576,      12,    0.12),  # ~32.8M — middle ground
    ("4M",       4,    256,       8,    0.10),  # ~3.6M  — underfit baseline
    ("203M",    16,   1024,      16,    0.20),  # ~203M  — upper bound
]

BASE_SLUG = "earnings-flagship-B"
BASE_YAML_PATH = REPO_ROOT / "markets" / f"{BASE_SLUG}.yaml"
SUMMARY_CSV = REPO_ROOT / "runs" / "sweep_summary.csv"


def estimate_params(layers: int, d: int, ffn_mult: int = 4, state_dim: int = 1664) -> int:
    per_layer = 4 * d * d + 2 * d * (ffn_mult * d)
    return layers * per_layer + state_dim * d


def make_variant_yaml(size_name: str, layers: int, d: int, heads: int, dropout: float) -> Path:
    """Produce markets/earnings-flagship-B-<size>.yaml mutated from the base."""
    base = yaml.safe_load(BASE_YAML_PATH.read_text())
    base["slug"] = f"{BASE_SLUG}-{size_name}"
    base["display_name"] = f"{base.get('display_name', BASE_SLUG)} — {size_name} sweep"
    base.setdefault("transformer", {})
    base["transformer"]["layers"] = layers
    base["transformer"]["model_dim"] = d
    base["transformer"]["heads"] = heads
    base["transformer"]["dropout"] = dropout
    base["transformer"]["ffn_mult"] = base["transformer"].get("ffn_mult", 4)
    # Each variant gets its own state dir to avoid clobbering
    for ds in base.get("data_sources", []):
        if ds.get("name") == "polymarket_contracts":
            continue
    # Point all text block voyage_cache_dirs to the SHARED base cache so we don't re-embed
    for block in base.get("state_vector", {}).get("blocks", []):
        params = block.get("params") or {}
        if params.get("mode") == "voyage" and params.get("voyage_cache_dir"):
            # Re-route to base slug's voyage cache (shared across all sizes)
            params["voyage_cache_dir"] = f"data/raw/{BASE_SLUG}/_voyage_cache"
    out_path = REPO_ROOT / "markets" / f"{BASE_SLUG}-{size_name}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return out_path


def reuse_state_and_structured(size_slug: str) -> None:
    """Hardlink (or copy) state + structured from base slug to size-specific slug.

    State vectors are size-independent: they're built from market/state-vector config that's
    the same for every size in the sweep (only transformer hyperparams differ). So we
    reuse the base slug's state directory by symlinking.
    """
    base_state = REPO_ROOT / "data" / "state" / BASE_SLUG
    base_struct = REPO_ROOT / "data" / "structured" / BASE_SLUG
    base_heldout = REPO_ROOT / "data" / "held_out" / BASE_SLUG
    target_state = REPO_ROOT / "data" / "state" / size_slug
    target_struct = REPO_ROOT / "data" / "structured" / size_slug
    target_heldout = REPO_ROOT / "data" / "held_out" / size_slug
    pairs = [(base_state, target_state), (base_struct, target_struct)]
    if base_heldout.exists():
        pairs.append((base_heldout, target_heldout))
    for src, dst in pairs:
        # If a symlink exists (from a prior buggy version), replace with a real copy.
        # rsync to H200 doesn't follow symlinks reliably, so we need real dirs on disk.
        if dst.is_symlink():
            dst.unlink()
        if dst.exists():
            # Already a real dir, skip
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def already_done(size_name: str) -> bool:
    runs_dir = REPO_ROOT / "runs" / f"{BASE_SLUG}-{size_name}"
    if not runs_dir.exists():
        return False
    for sub in sorted(runs_dir.iterdir(), reverse=True):
        if (sub / "metrics.json").exists():
            return True
    return False


def run_one(size_name: str, layers: int, d: int, heads: int, dropout: float, *, device: str) -> dict:
    size_slug = f"{BASE_SLUG}-{size_name}"
    n_params = estimate_params(layers, d)
    print(f"\n=== {size_name} ({n_params/1e6:.1f}M params, layers={layers}, dim={d}, heads={heads}, dropout={dropout}, device={device}) ===", flush=True)

    if already_done(size_name):
        print(f"  [{size_name}] already has metrics.json → skipping (use --force-rerun to override)")
        return {"size_name": size_name, "status": "skipped", "n_params_m": n_params / 1e6}

    yaml_path = make_variant_yaml(size_name, layers, d, heads, dropout)
    print(f"  yaml: {yaml_path}")

    reuse_state_and_structured(size_slug)
    print(f"  state vectors: reusing from {BASE_SLUG}")

    t0 = time.time()

    if device == "cpu":
        # Train locally on CPU; useful for dry-run validation
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run(
            [sys.executable, "scripts/03_train.py", size_slug, "--device", "cpu"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  TRAIN FAILED: {proc.stderr[-2000:]}")
            return {"size_name": size_name, "status": "train_failed", "n_params_m": n_params / 1e6}
        proc = subprocess.run(
            [sys.executable, "scripts/04_backtest.py", size_slug],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  BACKTEST FAILED: {proc.stderr[-2000:]}")
            return {"size_name": size_name, "status": "backtest_failed", "n_params_m": n_params / 1e6}
    else:
        # Train on H200, then pull back
        env = os.environ.copy()
        env["H200_GPU"] = "6"
        # Sync the new yaml + reused state to H200
        print(f"  syncing to H200...")
        proc = subprocess.run(["bin/h200", "sync"], cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            print(f"  SYNC FAILED")
            return {"size_name": size_name, "status": "sync_failed", "n_params_m": n_params / 1e6}
        print(f"  training on H200 GPU 6...")
        proc = subprocess.run(
            ["bin/h200", "run", "run_remote_pipeline.py", size_slug,
             "--device", "cuda", "--confirm-gpu-free"],
            cwd=REPO_ROOT, env=env,
        )
        if proc.returncode != 0:
            print(f"  REMOTE PIPELINE FAILED")
            return {"size_name": size_name, "status": "remote_failed", "n_params_m": n_params / 1e6}
        print(f"  pulling runs back...")
        proc = subprocess.run(
            ["bin/h200", "pull", f"runs/{size_slug}/"],
            cwd=REPO_ROOT, env=env,
        )
        if proc.returncode != 0:
            print(f"  PULL FAILED")
            return {"size_name": size_name, "status": "pull_failed", "n_params_m": n_params / 1e6}

    elapsed = time.time() - t0

    # Load metrics from latest run
    import json
    runs_dir = REPO_ROOT / "runs" / size_slug
    latest = sorted(runs_dir.iterdir(), reverse=True)[0]
    metrics_path = latest / "metrics.json"
    if not metrics_path.exists():
        print(f"  no metrics.json at {metrics_path}")
        return {"size_name": size_name, "status": "no_metrics", "n_params_m": n_params / 1e6}
    m = json.loads(metrics_path.read_text())
    summary = {
        "size_name": size_name,
        "status": "ok",
        "n_params_m": round(n_params / 1e6, 1),
        "layers": layers,
        "model_dim": d,
        "heads": heads,
        "dropout": dropout,
        "brier_improvement": round(m.get("brier_improvement", 0.0), 4),
        "net_return": round(m.get("net_return", 0.0), 4),
        "n_trades": m.get("n_trades", 0),
        "hit_rate": round(m.get("hit_rate", 0.0), 4),
        "sharpe": round(m.get("sharpe", 0.0), 3),
        "elapsed_s": round(elapsed, 1),
        "run_dir": str(latest.relative_to(REPO_ROOT)),
    }
    print(f"  result: brier_improvement={summary['brier_improvement']:+.4f} net_return={summary['net_return']:+.2%} trades={summary['n_trades']} elapsed={summary['elapsed_s']:.0f}s")
    return summary


def append_summary(row: dict) -> None:
    """Append a row to the sweep summary CSV (creates header on first write)."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = "size_name,status,n_params_m,layers,model_dim,heads,dropout,brier_improvement,net_return,n_trades,hit_rate,sharpe,elapsed_s,run_dir"
    if not SUMMARY_CSV.exists():
        SUMMARY_CSV.write_text(header + "\n")
    cols = header.split(",")
    line = ",".join(str(row.get(c, "")) for c in cols)
    with SUMMARY_CSV.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Comma-separated size_names to include (default all)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="cuda for H200, cpu for local dry-run")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-train even if metrics.json exists for this size")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None

    print(f"=== EARNINGS-FLAGSHIP-B TRANSFORMER SIZE SWEEP ===")
    print(f"  base slug: {BASE_SLUG}")
    print(f"  device:    {args.device}")
    print(f"  sizes:     {[s[0] for s in SIZES if not only or s[0] in only]}")
    print(f"  summary:   {SUMMARY_CSV}")

    for size_name, layers, d, heads, dropout in SIZES:
        if only and size_name not in only:
            continue
        if args.force_rerun:
            # Force-mode: rename existing run dir so already_done() returns False
            runs_dir = REPO_ROOT / "runs" / f"{BASE_SLUG}-{size_name}"
            if runs_dir.exists():
                rotated = runs_dir.with_suffix(f".rotated.{time.strftime('%Y%m%dT%H%M%S')}")
                runs_dir.rename(rotated)
                print(f"  rotated existing runs to {rotated}")
        row = run_one(size_name, layers, d, heads, dropout, device=args.device)
        append_summary(row)

    print(f"\n=== sweep complete. summary: {SUMMARY_CSV} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
