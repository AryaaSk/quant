"""Synthetic adapter: deterministic, signal-bearing data for the E2E smoke test.

Generates a daily time series with several numeric features, a periodic event schedule,
realized event outcomes that ARE predictable from the features (with noise), and a
companion market price that is close-to-but-not-equal-to the truth so the model has a
nontrivial calibration delta to discover.

Configurable via source_params:
  n_days, n_events, n_numeric_series, noise_level, seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import MarketConfig


class SyntheticAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        rng = np.random.default_rng(int(source_params.get("seed", 42)))
        n_days = int(source_params.get("n_days", 800))
        n_events = int(source_params.get("n_events", 200))
        n_numeric = int(source_params.get("n_numeric_series", 8))
        noise = float(source_params.get("noise_level", 0.25))

        start = pd.Timestamp("2023-01-01")
        days = pd.date_range(start, periods=n_days, freq="D")
        # Build numeric features as AR(1) processes.
        feats = np.zeros((n_days, n_numeric))
        feats[0] = rng.normal(size=n_numeric)
        rho = 0.9
        for t in range(1, n_days):
            feats[t] = rho * feats[t - 1] + (1 - rho) * rng.normal(size=n_numeric)

        # Schedule events evenly over the window.
        event_indices = np.linspace(40, n_days - 5, n_events, dtype=int)
        event_times = days[event_indices]
        # True signal: linear combo of first 3 features (lagged 1 day).
        true_signal = 0.7 * feats[event_indices - 1, 0] + 0.4 * feats[event_indices - 1, 1] - 0.3 * feats[event_indices - 1, 2]
        # Binary outcome with noise.
        prob_true = 1.0 / (1.0 + np.exp(-true_signal))
        y_realized = (rng.random(size=n_events) < prob_true).astype(np.float64)
        # Market price = noisy estimate of the true probability.
        p_market = np.clip(prob_true + rng.normal(scale=noise, size=n_events), 0.05, 0.95)

        rows: list[dict] = []
        scraped_at = pd.Timestamp.utcnow()
        # Daily numeric rows (each feature -> num__feat_i column).
        for t_idx, t in enumerate(days):
            row = {
                "timestamp": t,
                "source_published_at": t,
                "scraped_at": scraped_at,
                "source_url": "synthetic://row",
                "source_type": "synthetic.numeric",
            }
            for j in range(n_numeric):
                row[f"num__feat_{j}"] = float(feats[t_idx, j])
            rows.append(row)

        # Event-day rows: target_event_time, y_realized, p_market.
        for e_idx, t in enumerate(event_times):
            rows.append({
                "timestamp": t,
                "source_published_at": t,
                "scraped_at": scraped_at,
                "source_url": f"synthetic://event/{e_idx}",
                "source_type": "synthetic.event",
                "target_event_time": t,
                "y_realized": float(y_realized[e_idx]),
                "p_market": float(p_market[e_idx]),
                "text": "synthetic news: feat_0 trending; macro flows watched closely",
            })

        df = pd.DataFrame(rows)
        return df
