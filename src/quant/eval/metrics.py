"""Backtest metrics.

Brier: mean((p_model - y)^2). Lower = better.
Brier improvement vs market: brier(p_market) - brier(p_model). Positive = beats the market.
Sharpe: mean(returns) / std(returns) * sqrt(N). N is per-trade.
Max drawdown: largest peak-to-trough drop on the equity curve.
Calibration bins: 10 deciles, output (bin_center, bin_actual_freq, bin_count) for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestMetrics:
    brier_model: float
    brier_market: float
    brier_improvement: float
    accuracy_model: float
    accuracy_market: float
    n_events: int
    n_trades: int
    hit_rate: float
    gross_return: float
    net_return: float
    sharpe: float
    max_drawdown: float
    calibration_bins: list[tuple[float, float, int]]
    label_shuffle_brier_improvement: float | None = None

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, list) else [list(t) for t in v]) for k, v in self.__dict__.items()}


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


def sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return mean / std * np.sqrt(returns.size)


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    return float(np.min(drawdowns))


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[tuple[float, float, int]]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        count = int(mask.sum())
        if count > 0:
            bins.append(((lo + hi) / 2, float(np.mean(y[mask])), count))
        else:
            bins.append(((lo + hi) / 2, float("nan"), 0))
    return bins
