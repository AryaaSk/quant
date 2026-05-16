"""Plotting helpers for backtest reports.

Each function writes a PNG and returns the path. Matplotlib is imported lazily so the
test suite does not require a display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def equity_curve(equity: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(equity, lw=1.5)
    ax.set_title("Equity curve")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def reliability_diagram(bins: list[tuple[float, float, int]], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [b[0] for b in bins]
    actuals = [b[1] for b in bins]
    counts = [b[2] for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.scatter(centers, actuals, s=[max(20, c * 5) for c in counts], alpha=0.7, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
