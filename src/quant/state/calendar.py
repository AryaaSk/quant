"""Calendar block builder: cyclical encodings of time-related features.

Slots:
  [0] sin(day_of_week)        [1] cos(day_of_week)
  [2] sin(month)              [3] cos(month)
  [4] sin(day_of_year)        [5] cos(day_of_year)
  [6] days_until_target_event (clipped, normalized)
  [7] days_since_last_event   (clipped, normalized)
  [8..] additional event flags (e.g. FOMC week) from yaml params
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class CalendarBlockBuilder:
    slot_width: int
    target_event_column: str = "target_event_time"
    last_event_column: str | None = None
    extra_flag_columns: Sequence[str] = ()

    def fit(self, structured_df: pd.DataFrame) -> None:
        # Stateless apart from a sanity check.
        if self.slot_width < 8:
            raise ValueError(f"calendar block needs at least 8 slots, got {self.slot_width}")

    def build(self, t: pd.Timestamp, structured_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(self.slot_width, dtype=np.float32)
        dow = t.dayofweek
        month = t.month
        doy = t.dayofyear
        out[0] = float(np.sin(2 * np.pi * dow / 7))
        out[1] = float(np.cos(2 * np.pi * dow / 7))
        out[2] = float(np.sin(2 * np.pi * month / 12))
        out[3] = float(np.cos(2 * np.pi * month / 12))
        out[4] = float(np.sin(2 * np.pi * doy / 365.25))
        out[5] = float(np.cos(2 * np.pi * doy / 365.25))

        if self.target_event_column in structured_df.columns:
            future_events = structured_df.loc[structured_df.index >= t, self.target_event_column].dropna()
            if not future_events.empty:
                next_event = pd.to_datetime(future_events.iloc[0])
                days = (next_event - t).days
                out[6] = float(np.clip(days / 30.0, 0, 12))

        if self.last_event_column and self.last_event_column in structured_df.columns:
            past_events = structured_df.loc[structured_df.index < t, self.last_event_column].dropna()
            if not past_events.empty:
                last_event = pd.to_datetime(past_events.iloc[-1])
                days = (t - last_event).days
                out[7] = float(np.clip(days / 30.0, 0, 12))

        for i, col in enumerate(self.extra_flag_columns):
            slot = 8 + i
            if slot >= self.slot_width or col not in structured_df.columns:
                break
            window = structured_df.loc[structured_df.index == t, col]
            if not window.empty:
                out[slot] = float(window.iloc[0] or 0.0)
        return out
