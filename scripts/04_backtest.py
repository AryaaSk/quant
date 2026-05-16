"""Phase 5: walk-forward backtest on the trained model.

Usage: python scripts/04_backtest.py <slug> [--run <run_dir>]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import load_market  # noqa: E402
from quant.pipeline.backtest import backtest  # noqa: E402
from quant.pipeline.report import report  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/04_backtest.py <slug> [--run <run_dir>]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    market = load_market(slug)

    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        run_dir = Path(sys.argv[idx + 1])
    else:
        runs = sorted([p for p in market.runs_dir().glob("*") if p.is_dir()])
        if not runs:
            print(f"no runs for {slug}; train first", file=sys.stderr)
            return 1
        run_dir = runs[-1]

    print(f"[{slug}] backtest on {run_dir}")
    metrics = backtest(market, run_dir)
    print(f"  net_return = {metrics['net_return']:.2%}, trades = {metrics['n_trades']}, "
          f"Brier improvement = {metrics['brier_improvement']:.4f}")
    report(market, run_dir)
    print(f"  report: {run_dir / 'backtest_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
