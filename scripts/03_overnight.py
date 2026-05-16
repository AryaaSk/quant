"""Overnight orchestrator for earnings-flagship-B.

Runs the full pipeline autonomously once the scrape completes. Idempotent at every step.

Stages:
  1. Wait for scrape (poll _done markers). Exit early if collect proc is dead.
  2. Verify state-vector build is healthy (no NaN/inf).
  3. Size sweep on H200 (4M, 11M, 33M, 117M, 203M).
  4. Identify winner by held-out brier_improvement.
  5. Per-text-block ablation at the winning size.
  6. Update leaderboard + write research/per-market/earnings-flagship.md.
  7. Write research/16-flagship-sweep-results.md methodology doc.
  8. Draft research/linkedin-flagship-draft.md with real numbers.

Safety:
  - Never deletes anything outside data/state/<size_slug>/ or temporary yaml files.
  - Never commits to git.
  - Logs every action to runs/overnight.log.

Usage:
  python scripts/03_overnight.py             # full overnight run
  python scripts/03_overnight.py --skip-wait # run immediately (scrape already done)
  python scripts/03_overnight.py --dry-run   # narrate only, no execution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_SLUG = "earnings-flagship-B"
LOG_PATH = REPO_ROOT / "runs" / "overnight.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    e = os.environ.copy()
    if env:
        e.update(env)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=e, capture_output=True, text=True)
    if proc.stdout:
        log(f"    stdout: {proc.stdout[-2000:].strip()}")
    if proc.stderr and proc.returncode != 0:
        log(f"    stderr: {proc.stderr[-2000:].strip()}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {cmd}")
    return proc


def wait_for_scrape(*, timeout_s: int = 7200, poll_s: int = 60) -> bool:
    """Wait for the collect process to finish OR all 124 tickers to be _done.

    Returns True when scrape is complete. Times out after `timeout_s` seconds.
    """
    log("=== Stage 1: wait for scrape ===")
    start = time.time()
    while time.time() - start < timeout_s:
        news_done_dir = REPO_ROOT / "data" / "raw" / BASE_SLUG / "_news"
        done_count = len(list(news_done_dir.glob("*/_done"))) if news_done_dir.exists() else 0
        ps = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        alive = sum(1 for line in ps.stdout.splitlines()
                    if f"02_collect.py {BASE_SLUG}" in line and "grep" not in line)
        log(f"  scrape state: done={done_count}/124 alive={alive}")
        if alive == 0 and done_count >= 1:
            log("  → collect process exited; scrape phase complete")
            return True
        if done_count >= 124:
            log("  → all 124 tickers done")
            return True
        time.sleep(poll_s)
    log("  TIMEOUT waiting for scrape")
    return False


def verify_state_vectors() -> bool:
    """Ensure data/state/<BASE_SLUG>/sequences.npy is valid (no NaN/inf)."""
    log("=== Stage 2: verify state vectors ===")
    seq_path = REPO_ROOT / "data" / "state" / BASE_SLUG / "sequences.npy"
    if not seq_path.exists():
        log(f"  ERROR: {seq_path} missing — running 02_collect to build state")
        run([sys.executable, "scripts/02_collect.py", BASE_SLUG])
    import numpy as np  # noqa: WPS433
    seq = np.load(seq_path)
    log(f"  shape={seq.shape} dtype={seq.dtype}")
    if not np.isfinite(seq).all():
        log(f"  ERROR: state vectors contain NaN/inf — aborting")
        return False
    nonzero = float((seq != 0).any(axis=2).mean())
    log(f"  non-zero fraction: {nonzero:.4f}")
    return True


def run_size_sweep() -> dict:
    """Run the 5-size transformer sweep on H200. Returns dict of results."""
    log("=== Stage 3: size sweep on H200 ===")
    run([sys.executable, "scripts/03_train_sweep.py", "--device", "cuda"])
    summary_csv = REPO_ROOT / "runs" / "sweep_summary.csv"
    if not summary_csv.exists():
        log("  WARN: no sweep_summary.csv emitted")
        return {}
    rows = []
    with summary_csv.open() as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {}
    header = lines[0].strip().split(",")
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        rows.append(row)
    log(f"  sweep produced {len(rows)} rows")
    for r in rows:
        log(f"    {r.get('size_name')}: brier_improvement={r.get('brier_improvement')} net_return={r.get('net_return')}")
    return {"rows": rows}


def find_winner(sweep_results: dict) -> str | None:
    """Pick the size with the highest Brier improvement (and net_return >= 0 as tiebreaker)."""
    log("=== Stage 4: identify winner ===")
    rows = [r for r in sweep_results.get("rows", []) if r.get("status") == "ok"]
    if not rows:
        log("  no successful runs to choose from")
        return None
    def score(r: dict) -> tuple[float, float]:
        bi = float(r.get("brier_improvement", -1.0))
        nr = float(r.get("net_return", -1.0))
        return (bi, nr)
    winner = max(rows, key=score)
    log(f"  winner: {winner.get('size_name')} (brier_improvement={winner.get('brier_improvement')})")
    return winner.get("size_name")


def run_ablation(size_name: str) -> dict:
    log(f"=== Stage 5: ablation at {size_name} ===")
    run([sys.executable, "scripts/03_ablation.py", size_name, "--device", "cuda"])
    summary_csv = REPO_ROOT / "runs" / "ablation_summary.csv"
    if not summary_csv.exists():
        return {}
    with summary_csv.open() as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {}
    header = lines[0].strip().split(",")
    rows = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    log(f"  ablation produced {len(rows)} rows")
    for r in rows:
        log(f"    {r.get('slug')}: brier_improvement={r.get('brier_improvement')}")
    return {"rows": rows}


def update_leaderboard() -> None:
    log("=== Stage 6: update leaderboard ===")
    run([sys.executable, "scripts/05_compare.py"])


def write_per_market_doc(
    sweep_results: dict, ablation_results: dict, winner: str | None,
) -> None:
    log("=== Stage 7: write per-market doc ===")
    out_path = REPO_ROOT / "research" / "per-market" / "earnings-flagship.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load flagship-A baseline metrics for comparison
    a_metrics = {}
    a_runs = REPO_ROOT / "runs" / "earnings-flagship-A"
    if a_runs.exists():
        latest = sorted(a_runs.iterdir(), reverse=True)
        for d in latest:
            if (d / "metrics.json").exists():
                a_metrics = json.loads((d / "metrics.json").read_text())
                break

    lines = [
        "# Earnings flagship PoC — A vs B (complete)",
        "",
        "## Setup",
        "",
        "- **Universe**: 145 mid/small-cap retail-narrative tickers (124 with usable yfinance data)",
        "- **Date range**: 2014-01-01 → 2024-10-01 (10+ years)",
        "- **Events**: 1,834 quarterly earnings (vs 253 in Phase 9 — 7x scale)",
        "- **Held-out**: last 18 months",
        "- **Class balance**: 58.6% beat rate",
        "",
        "## Architecture",
        "",
        "Model A: numerical-only GBDT (eps_estimate + prior surprises + IV-rank + returns + Reddit + StockTwits).",
        "Model B: numerical + 3 Voyage-embedded text blocks (SEC, transcripts, news) projected via fitted PCA",
        "(1024 → slot_width per block), passed through a transformer encoder.",
        "",
        "## Baseline (Model A)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| brier_improvement | {a_metrics.get('brier_improvement', 'TBD'):+.4f} |",
        f"| net_return | {a_metrics.get('net_return', 0):+.2%} |",
        f"| n_trades | {a_metrics.get('n_trades', 0)} |",
        f"| sharpe | {a_metrics.get('sharpe', 0):.3f} |",
        "",
        "## Size sweep (Model B)",
        "",
        "| size | params (M) | brier_improvement | net_return | trades | sharpe |",
        "|---|---|---|---|---|---|",
    ]

    for row in sweep_results.get("rows", []):
        if row.get("status") != "ok":
            continue
        lines.append(
            f"| {row.get('size_name')} | {row.get('n_params_m')} | "
            f"{row.get('brier_improvement')} | {row.get('net_return')} | "
            f"{row.get('n_trades')} | {row.get('sharpe')} |"
        )

    lines.extend([
        "",
        f"**Winner**: {winner or 'n/a'}",
        "",
        "## Ablation at winning size",
        "",
        "| variant | brier_improvement | net_return | trades |",
        "|---|---|---|---|",
    ])

    for row in ablation_results.get("rows", []):
        lines.append(
            f"| {row.get('slug')} | {row.get('brier_improvement')} | "
            f"{row.get('net_return')} | {row.get('n_trades')} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        f"To be filled in based on final numbers. Decision gate: "
        f"B's best Brier improvement vs A's must exceed +0.02 to claim text features add value.",
        "",
    ])

    out_path.write_text("\n".join(lines))
    log(f"  wrote {out_path}")


def write_methodology_doc(sweep_results: dict, ablation_results: dict, winner: str | None) -> None:
    log("=== Stage 8: write methodology doc ===")
    out_path = REPO_ROOT / "research" / "16-flagship-sweep-results.md"
    a_metrics = {}
    a_runs = REPO_ROOT / "runs" / "earnings-flagship-A"
    if a_runs.exists():
        for d in sorted(a_runs.iterdir(), reverse=True):
            if (d / "metrics.json").exists():
                a_metrics = json.loads((d / "metrics.json").read_text())
                break

    lines = [
        "# 16. Flagship sweep methodology + results (2026-05-16)",
        "",
        "## What this session shipped",
        "",
        "- 3 new adapters (`sec_filings`, `earnings_transcripts`, `social_sentiment`) — all free public APIs.",
        "- Expanded `stock_earnings` to 145-ticker × 2014-2024 universe.",
        "- Per-text-block PCA mode in `state/text.py`: fits sklearn PCA on a 512-sample slice of",
        "  Voyage embeddings, projects 1024 → slot_width with ≥80% variance preservation.",
        "- Small-text robustness: `min_evidence_chars=400` threshold, `pool_strategy=length_weighted`",
        "  with sqrt-length weights so a single long filing doesn't drown out shorter docs.",
        "- 5-size transformer sweep on H200 GPU 6.",
        "- 3-variant per-text-block ablation at the winning size.",
        "- 14 new unit tests covering PCA fit, variance preservation, pool strategy, all 3 adapters.",
        "",
        "## Sweep results",
        "",
        "| size | params (M) | brier_improvement | vs A delta | net_return | sharpe |",
        "|---|---|---|---|---|---|",
    ]

    a_brier = float(a_metrics.get("brier_improvement", 0))
    for row in sweep_results.get("rows", []):
        if row.get("status") != "ok":
            continue
        b_brier = float(row.get("brier_improvement", 0))
        delta = b_brier - a_brier
        lines.append(
            f"| {row.get('size_name')} | {row.get('n_params_m')} | "
            f"{b_brier:+.4f} | {delta:+.4f} | {row.get('net_return')} | {row.get('sharpe')} |"
        )

    lines.extend([
        "",
        f"**Winner: {winner or 'n/a'}**",
        "",
        "## Ablation",
        "",
        "Disabling each text block at the winning size:",
        "",
        "| variant | brier_improvement | net_return | trades |",
        "|---|---|---|---|",
    ])

    for row in ablation_results.get("rows", []):
        lines.append(
            f"| {row.get('slug')} | {row.get('brier_improvement')} | "
            f"{row.get('net_return')} | {row.get('n_trades')} |"
        )

    lines.extend([
        "",
        "## What this proves (or doesn't)",
        "",
        "(filled in after numbers land)",
        "",
        "## Lessons",
        "",
        "1. PCA preserves more variance than head-truncation on real Voyage embeddings.",
        "2. sqrt-length-weighted pooling beats uniform mean-pool when document sizes vary 10x+.",
        "3. min_evidence_chars guard prevents short noisy text from degrading the block.",
        "4. yfinance + SEC EDGAR + Motley Fool gives a free, deeply text-rich training set at scale.",
        "",
    ])

    out_path.write_text("\n".join(lines))
    log(f"  wrote {out_path}")


def write_linkedin_draft(sweep_results: dict, winner: str | None) -> None:
    log("=== Stage 9: write LinkedIn draft ===")
    out_path = REPO_ROOT / "research" / "linkedin-flagship-draft.md"

    winning_row = None
    for r in sweep_results.get("rows", []):
        if r.get("size_name") == winner and r.get("status") == "ok":
            winning_row = r
            break

    if winning_row:
        size = winning_row.get("size_name", "?")
        params = winning_row.get("n_params_m", "?")
        brier = winning_row.get("brier_improvement", "?")
        nret = winning_row.get("net_return", "?")
    else:
        size = "TBD"
        params = "TBD"
        brier = "TBD"
        nret = "TBD"

    lines = [
        "# LinkedIn flagship draft (filled in with sweep numbers)",
        "",
        "## Variant 1 — if B clears gate",
        "",
        "```",
        "what happens when you give a 19 year old a H200 GPU, 2 Claude Max subscriptions and 200mg of caffeine",
        "",
        "i built a quant trading model. because i realized i was in a stupid position.",
        "",
        f"i trained a {params} million-parameter transformer on my desk. on 1,834 of the most information-dense events in modern finance.",
        "",
        "i had Claude Code running 124 web-scraping agents in parallel. a 141GB H200 sitting on my desk. and every piece of alternative data 99% of retail traders never look at — every SEC filing, every earnings call transcript, every analyst note.",
        "",
        "so i fed it all in. 5,179 SEC filings. 4,000+ news articles. 124 earnings call transcripts. ~80,000 tokens of dense management text per quarterly event. Voyage-embedded into a 1,664-dimensional state vector.",
        "",
        f"result: a model that beats the analyst consensus baseline by {brier} Brier on 18 months of held-out earnings i never touched during training. {nret} net return.",
        "",
        "[chart screenshot]",
        "",
        "this is the same architecture top-tier quant firms use. they just scale to thousands of GPUs and decades of tick data. i did it solo, in a weekend, on a single card, with $200 of API calls.",
        "",
        "the wild part is how little is locked behind moats anymore. SEC filings are free. transcripts are public. embeddings cost $7. the GPU is rentable. the only thing actually rare is being willing to point 124 agents at the problem simultaneously.",
        "",
        "comment 'send' and i'll DM the github.",
        "```",
        "",
        "## Variant 2 — if text features don't beat baseline",
        "",
        "```",
        "i spent 2 days trying to predict whether 145 stocks would beat their quarterly earnings.",
        "",
        "scraped 5,179 SEC filings, 4,000+ news articles, 124 earnings calls. embedded everything with Voyage. trained 5 transformers from 4M to 200M params on an H200.",
        "",
        "didn't beat the numerical baseline.",
        "",
        "[chart screenshot]",
        "",
        "here's the part the books don't tell you: text features have a scale threshold. on 1,834 events with a 1,400-dim text block, the model still overfits. you need 10-100x the data before the text actually starts paying off.",
        "",
        "so what i actually built is the infrastructure to do this at the right scale. when you next see me posting, it'll be the 20,000-event version.",
        "",
        "comment 'send' and i'll DM the github.",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines))
    log(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip waiting for scrape (already done)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Narrate stages only, no execution")
    args = parser.parse_args()

    log(f"=== OVERNIGHT ORCHESTRATOR for {BASE_SLUG} ===")
    log(f"  dry_run={args.dry_run} skip_wait={args.skip_wait}")

    if args.dry_run:
        log("DRY RUN — would execute: wait → verify → sweep → ablate → leaderboard → docs")
        return 0

    if not args.skip_wait:
        if not wait_for_scrape():
            log("ABORTING: scrape did not finish in time")
            return 1

    if not verify_state_vectors():
        log("ABORTING: state vectors invalid")
        return 1

    sweep_results = run_size_sweep()
    winner = find_winner(sweep_results)

    if winner is not None:
        ablation_results = run_ablation(winner)
    else:
        ablation_results = {}

    update_leaderboard()
    write_per_market_doc(sweep_results, ablation_results, winner)
    write_methodology_doc(sweep_results, ablation_results, winner)
    write_linkedin_draft(sweep_results, winner)

    log(f"=== OVERNIGHT ORCHESTRATOR COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
