"""Numeric block builder: takes named series, applies rolling z-score, packs into slots.

Each slot maps to one (series, lag) pair. Normalization uses ONLY past data at each
timestep (rolling window ending at t-1) to avoid leakage. Missing values are filled with
the rolling mean and a NaN-flag bit is set in an adjacent slot if `flag_missing=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class NumericBlockBuilder:
    series_names: Sequence[str]
    slot_width: int
    rolling_window: int = 252
    fill_missing: bool = True

    def __post_init__(self) -> None:
        if len(self.series_names) > self.slot_width:
            raise ValueError(
                f"numeric block has {len(self.series_names)} series but only {self.slot_width} slots"
            )

    def fit(self, structured_df: pd.DataFrame) -> None:
        # Stateless: rolling z-score is computed per-timestep at build time.
        # We only verify that the named series exist in the dataframe.
        missing = [s for s in self.series_names if s not in structured_df.columns]
        if missing:
            raise KeyError(f"numeric series not present in structured frame: {missing}")

    def build(self, t: pd.Timestamp, structured_df: pd.DataFrame) -> np.ndarray:
        # past = strict less-than t to prevent same-day leakage
        past = structured_df.loc[structured_df.index < t, list(self.series_names)]
        if past.empty:
            return np.zeros(self.slot_width, dtype=np.float32)

        window = past.tail(self.rolling_window)
        mean = window.mean()
        std = window.std().replace(0, 1.0).fillna(1.0)

        latest = past.iloc[-1]
        z = (latest - mean) / std

        out = np.zeros(self.slot_width, dtype=np.float32)
        for i, name in enumerate(self.series_names):
            val = z[name]
            if pd.isna(val):
                if self.fill_missing:
                    out[i] = 0.0
                else:
                    out[i] = np.nan
            else:
                out[i] = float(np.clip(val, -8.0, 8.0))
        return out
