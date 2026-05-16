"""Market state block builder: current market prices for the event we are trying to predict.

For Kalshi/Polymarket: contract prices for the upcoming target contract at this timestep.
For Betfair: decimal odds for the relevant outcomes.
For Hyperliquid: current funding rate, basis, open interest.

The market state block typically embeds:
- The market's implied probability for each outcome bucket
- Recent price velocity (delta over last 1, 5, 30 days)
- Volume / liquidity proxy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class MarketStateBlockBuilder:
    series_names: Sequence[str]
    slot_width: int
    velocity_lags: Sequence[int] = (1, 5, 30)
    rolling_window: int = 60

    def __post_init__(self) -> None:
        needed = len(self.series_names) * (1 + len(self.velocity_lags))
        if needed > self.slot_width:
            raise ValueError(
                f"market state block needs {needed} slots for {len(self.series_names)} series "
                f"x (level + {len(self.velocity_lags)} velocities) but only {self.slot_width} available"
            )

    def fit(self, structured_df: pd.DataFrame) -> None:
        missing = [s for s in self.series_names if s not in structured_df.columns]
        if missing:
            raise KeyError(f"market state series not present: {missing}")

    def build(self, t: pd.Timestamp, structured_df: pd.DataFrame) -> np.ndarray:
        past = structured_df.loc[structured_df.index < t, list(self.series_names)]
        if past.empty:
            return np.zeros(self.slot_width, dtype=np.float32)

        out = np.zeros(self.slot_width, dtype=np.float32)
        idx = 0
        latest = past.iloc[-1]
        window = past.tail(self.rolling_window)
        for name in self.series_names:
            val = latest[name]
            if pd.isna(val):
                out[idx] = 0.0
            else:
                out[idx] = float(np.clip(val, -8.0, 8.0))
            idx += 1
            for lag in self.velocity_lags:
                if len(window) > lag:
                    prior = window.iloc[-(lag + 1) if lag + 1 <= len(window) else 0][name]
                    delta = (val - prior) if not pd.isna(val) and not pd.isna(prior) else 0.0
                    out[idx] = float(np.clip(delta, -8.0, 8.0))
                else:
                    out[idx] = 0.0
                idx += 1
        return out
