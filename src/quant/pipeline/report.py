"""Stage 7: write per-run reports (equity curve, reliability diagram, notes.md).

Pulls metrics.json + trades.parquet from the run directory and produces:
- equity.png
- reliability.png
- backtest_report.html (light wrapper)
- notes.md (one-paragraph human summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import MarketConfig
from quant.eval.plots import equity_curve, reliability_diagram


def report(market: MarketConfig, run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    trades = pd.read_parquet(run_dir / "trades.parquet") if (run_dir / "trades.parquet").exists() else pd.DataFrame()

    eq_png = equity_curve(np.asarray(trades["equity_after"].tolist() or [market.backtest.starting_bankroll]),
                          run_dir / "equity.png")
    rel_png = reliability_diagram([tuple(b) for b in metrics.get("calibration_bins", [])], run_dir / "reliability.png")

    html = run_dir / "backtest_report.html"
    html.write_text(_html_template(market, metrics, eq_png.name, rel_png.name))

    notes = run_dir / "notes.md"
    notes.write_text(_notes_template(market, metrics))
    return run_dir


def _html_template(market: MarketConfig, metrics: dict, eq_name: str, rel_name: str) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td><pre>{json.dumps(v, default=str)[:120]}</pre></td></tr>"
        for k, v in metrics.items() if k != "calibration_bins"
    )
    return f"""<!doctype html>
<html><head><title>{market.slug} backtest</title>
<style>body{{font-family:system-ui;margin:2em;max-width:880px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px;text-align:left;font-family:ui-monospace}}img{{max-width:100%;margin:1em 0;border:1px solid #eee}}</style>
</head><body>
<h1>{market.display_name} ({market.slug})</h1>
<p>{market.notes}</p>
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>Equity curve</h2><img src="{eq_name}">
<h2>Reliability</h2><img src="{rel_name}">
</body></html>
"""


def _notes_template(market: MarketConfig, metrics: dict) -> str:
    return (
        f"# {market.slug} run notes\n\n"
        f"- net return: {metrics.get('net_return', 0):.2%}\n"
        f"- trades: {metrics.get('n_trades', 0)} / events: {metrics.get('n_events', 0)}\n"
        f"- hit rate: {metrics.get('hit_rate', 0):.2%}\n"
        f"- Brier model: {metrics.get('brier_model', 0):.4f}\n"
        f"- Brier market: {metrics.get('brier_market', 0):.4f}\n"
        f"- Brier improvement: {metrics.get('brier_improvement', 0):.4f}\n"
        f"- Sharpe: {metrics.get('sharpe', 0):.2f}\n"
        f"- max drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
    )
