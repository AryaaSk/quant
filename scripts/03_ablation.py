"""Text-block ablation driver for earnings-flagship-B at the winning size.

For the winning transformer size, run 3 ablation variants:
  - earnings-flagship-B-no-sec   (text_sec block params: mode=handcrafted, force-zero-text)
  - earnings-flagship-B-no-trans (text_transcript block disabled)
  - earnings-flagship-B-no-news  (text_news block disabled)

Tests which text source contributes most. Real research artifact.

Usage:
  python scripts/03_ablation.py 117M --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_SLUG = "earnings-flagship-B"


def _disable_text_block(yaml_dict: dict, block_name: str) -> None:
    """Replace the given text block with a same-slot-width handcrafted-zero block.

    We don't drop the block from the schema (that would shift slot indices); we keep the
    slot range and force the builder to emit zeros via mode='handcrafted' with no
    sentiment lexicon and no entity_keywords.
    """
    for block in yaml_dict.get("state_vector", {}).get("blocks", []):
        if block.get("name") == block_name:
            params = block.setdefault("params", {})
            params["mode"] = "handcrafted"
            params["sentiment_lexicon"] = {}
            params["entity_keywords"] = []
            # Remove voyage-specific params so they don't confuse the builder
            for k in ["use_pca", "pca_min_samples", "pca_fit_max_samples", "voyage_cache_dir",
                      "encoder_name"]:
                params.pop(k, None)


def make_ablation_yaml(suffix: str, drop_block: str, base_yaml_path: Path) -> Path:
    """Produce markets/earnings-flagship-B-no-<suffix>.yaml with the given text block disabled."""
    base = yaml.safe_load(base_yaml_path.read_text())
    base["slug"] = f"{BASE_SLUG}-no-{suffix}"
    base["display_name"] = f"{base.get('display_name', BASE_SLUG)} — ablation: no {suffix}"
    _disable_text_block(base, drop_block)
    out_path = REPO_ROOT / "markets" / f"{BASE_SLUG}-no-{suffix}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return out_path


def already_done(slug: str) -> bool:
    runs_dir = REPO_ROOT / "runs" / slug
    if not runs_dir.exists():
        return False
    for sub in sorted(runs_dir.iterdir(), reverse=True):
        if (sub / "metrics.json").exists():
            return True
    return False


def run_ablation(ablation_slug: str, *, device: str) -> dict:
    print(f"\n=== ABLATION: {ablation_slug} (device={device}) ===", flush=True)
    if already_done(ablation_slug):
        print(f"  [{ablation_slug}] already has metrics.json → skipping")
        return {"slug": ablation_slug, "status": "skipped"}

    # Rebuild state (since text block params changed)
    structured_dst = REPO_ROOT / "data" / "structured" / ablation_slug
    state_dst = REPO_ROOT / "data" / "state" / ablation_slug
    if structured_dst.exists() and structured_dst.is_symlink():
        structured_dst.unlink()
    if state_dst.exists() and state_dst.is_symlink():
        state_dst.unlink()

    # Copy raw -> ablation_slug (so adapters don't re-fetch and rsync handles real dirs)
    raw_dst = REPO_ROOT / "data" / "raw" / ablation_slug
    raw_src = REPO_ROOT / "data" / "raw" / BASE_SLUG
    if raw_dst.is_symlink():
        raw_dst.unlink()
    if not raw_dst.exists():
        raw_dst.parent.mkdir(parents=True, exist_ok=True)
        # Use cp -al (hard-link) for speed if same filesystem; fallback to copytree
        try:
            subprocess.run(["cp", "-al", str(raw_src), str(raw_dst)], check=True,
                           capture_output=True, text=True)
        except subprocess.CalledProcessError:
            shutil.copytree(raw_src, raw_dst)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["H200_GPU"] = "6"

    t0 = time.time()

    if device == "cpu":
        # Local: 02_collect (rebuilds structured + state) → 03_train → 04_backtest
        for cmd in [
            [sys.executable, "scripts/02_collect.py", ablation_slug],
            [sys.executable, "scripts/03_train.py", ablation_slug, "--device", "cpu"],
            [sys.executable, "scripts/04_backtest.py", ablation_slug],
        ]:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            if proc.returncode != 0:
                print(f"  FAILED: {' '.join(cmd)}")
                return {"slug": ablation_slug, "status": "failed", "cmd": cmd[1]}
    else:
        # H200: rebuild state locally first (Voyage cache hits make this cheap),
        # then sync + run pipeline on H200
        proc = subprocess.run(
            [sys.executable, "scripts/02_collect.py", ablation_slug],
            cwd=REPO_ROOT, env=env,
        )
        if proc.returncode != 0:
            print(f"  LOCAL STATE BUILD FAILED")
            return {"slug": ablation_slug, "status": "state_failed"}
        proc = subprocess.run(["bin/h200", "sync"], cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            return {"slug": ablation_slug, "status": "sync_failed"}
        proc = subprocess.run(
            ["bin/h200", "run", "run_remote_pipeline.py", ablation_slug,
             "--device", "cuda", "--confirm-gpu-free"],
            cwd=REPO_ROOT, env=env,
        )
        if proc.returncode != 0:
            return {"slug": ablation_slug, "status": "remote_failed"}
        proc = subprocess.run(
            ["bin/h200", "pull", f"runs/{ablation_slug}/"],
            cwd=REPO_ROOT, env=env,
        )
        if proc.returncode != 0:
            return {"slug": ablation_slug, "status": "pull_failed"}

    elapsed = time.time() - t0
    runs_dir = REPO_ROOT / "runs" / ablation_slug
    latest = sorted(runs_dir.iterdir(), reverse=True)[0]
    m = json.loads((latest / "metrics.json").read_text())
    summary = {
        "slug": ablation_slug,
        "status": "ok",
        "brier_improvement": round(m.get("brier_improvement", 0.0), 4),
        "net_return": round(m.get("net_return", 0.0), 4),
        "n_trades": m.get("n_trades", 0),
        "sharpe": round(m.get("sharpe", 0.0), 3),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  result: brier_improvement={summary['brier_improvement']:+.4f} net_return={summary['net_return']:+.2%} elapsed={summary['elapsed_s']:.0f}s")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("size_name",
                        help="Winning transformer size (e.g. 117M); reads markets/earnings-flagship-B-<size>.yaml as base")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    base_yaml = REPO_ROOT / "markets" / f"{BASE_SLUG}-{args.size_name}.yaml"
    if not base_yaml.exists():
        print(f"ERROR: base yaml not found: {base_yaml}", file=sys.stderr)
        return 1

    summary_path = REPO_ROOT / "runs" / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = "slug,status,brier_improvement,net_return,n_trades,sharpe,elapsed_s"
    if not summary_path.exists():
        summary_path.write_text(header + "\n")

    print(f"=== EARNINGS-FLAGSHIP-B ABLATION at {args.size_name} ===")
    print(f"  base yaml: {base_yaml}")
    print(f"  device:    {args.device}")

    ablations = [
        ("sec", "text_sec"),
        ("trans", "text_transcript"),
        ("news", "text_news"),
    ]
    for suffix, block_name in ablations:
        make_ablation_yaml(suffix, block_name, base_yaml)
        row = run_ablation(f"{BASE_SLUG}-no-{suffix}", device=args.device)
        cols = header.split(",")
        line = ",".join(str(row.get(c, "")) for c in cols)
        with summary_path.open("a") as f:
            f.write(line + "\n")

    print(f"\n=== ablation complete. summary: {summary_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
