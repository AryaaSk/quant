"""Run state build + train + backtest + report sequentially on the remote H200.

This script is invoked ON the H200 (via `bin/h200 run run_remote_pipeline.py <slug>`).
It assumes:
- structured/<slug>/features.parquet already exists (copied from Mac)
- raw/<slug>/_voyage_cache exists with cached embeddings (copied from Mac)
- held_out manifest is already locked

Stages, in order:
1. build_state  (uses CUDA via sentence-transformers OR voyage cache)
2. train        (transformer on GPU 6/whichever)
3. backtest     (CPU, fast)
4. report       (writes equity.png, reliability.png, notes.md)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import load_market  # noqa: E402
from quant.pipeline.backtest import backtest  # noqa: E402
from quant.pipeline.report import report  # noqa: E402
from quant.pipeline.state import build_state  # noqa: E402
from quant.pipeline.train import train  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/run_remote_pipeline.py <slug> [--device cuda]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    device = "auto"
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1]

    market = load_market(slug)

    t0 = time.time()
    print(f"[{slug}] === stage: build_state ===", flush=True)
    # If state already exists (e.g. rsync'd from local) reuse it. The state was already
    # built locally with Voyage embeddings; H200 doesn't have VOYAGE_API_KEY by default
    # so re-running build_state with force=True would fail.
    seq_path = market.state_dir() / "sequences.npy"
    lab_path = market.state_dir() / "labels.parquet"
    if seq_path.exists() and lab_path.exists():
        print(f"  reusing existing state: {seq_path}", flush=True)
        seq, lab = seq_path, lab_path
    else:
        seq, lab = build_state(market, force=True)
        print(f"  sequences -> {seq}", flush=True)
        print(f"  labels    -> {lab}", flush=True)
    print(f"  build_state elapsed: {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"[{slug}] === stage: train (device={device}) ===", flush=True)
    ckpt = train(market, device=device)
    print(f"  ckpt -> {ckpt}", flush=True)
    print(f"  train elapsed: {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    print(f"[{slug}] === stage: backtest ===", flush=True)
    run_dir = ckpt.parent
    metrics = backtest(market, run_dir)
    print(f"  net_return = {metrics.get('net_return', 0):.2%}", flush=True)
    print(f"  brier_improvement = {metrics.get('brier_improvement', 0):.4f}", flush=True)
    print(f"  trades = {metrics.get('n_trades', 0)}, hit_rate = {metrics.get('hit_rate', 0):.2%}", flush=True)
    print(f"  backtest elapsed: {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    print(f"[{slug}] === stage: report ===", flush=True)
    report(market, run_dir)
    print(f"  report dir: {run_dir}", flush=True)
    print(f"  report elapsed: {time.time() - t3:.1f}s", flush=True)

    print(f"[{slug}] DONE. Total elapsed: {time.time() - t0:.1f}s", flush=True)
    print(f"[{slug}] run_dir: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
