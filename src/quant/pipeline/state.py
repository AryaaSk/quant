"""Stage 3: convert the structured frame into a sequence of state vectors per event.

For each event (row where `target_event_time` is set and a label can be derived), we build
a `sequence_length`-long sequence of state vectors ending at the timestep BEFORE the event
fires. The resulting tensor has shape (num_events, sequence_length, state_dim).

Outputs:
  - `data/state/<slug>/sequences.npy`  shape (N, L, D), float32
  - `data/state/<slug>/labels.parquet` per-event labels + timestamps
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import MarketConfig
from quant.pipeline.collect import write_held_out_manifest
from quant.state.composer import StateBuilder


def build_state(market: MarketConfig, *, force: bool = False) -> tuple[Path, Path]:
    state_dir = market.state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    seq_path = state_dir / "sequences.npy"
    label_path = state_dir / "labels.parquet"
    if seq_path.exists() and label_path.exists() and not force:
        return seq_path, label_path

    structured = pd.read_parquet(market.structured_dir() / "features.parquet")
    structured.index = pd.to_datetime(structured.index)
    structured = structured.sort_index()

    builder = StateBuilder(market.state_vector)
    builder.fit(structured)

    events = _extract_events(market, structured)
    L = market.sequence_length
    D = market.state_vector.dim

    sequences = np.zeros((len(events), L, D), dtype=np.float32)
    labels = []

    for i, ev in enumerate(events.itertuples()):
        t_event = pd.Timestamp(ev.target_event_time)
        # Sequence ends at the timestep BEFORE the event (no peeking).
        end_idx = structured.index.searchsorted(t_event, side="left")
        start_idx = max(0, end_idx - L)
        steps = list(range(start_idx, end_idx))
        if len(steps) < L:
            # Pad on the left with the earliest timestep so all events have equal-length sequences.
            pad = L - len(steps)
            steps = [steps[0] if steps else 0] * pad + steps
        for j, step in enumerate(steps):
            t = structured.index[step]
            sequences[i, j, :] = builder.build(t, structured)

        labels.append({
            "event_id": i,
            "event_index": ev.Index,
            "target_event_time": t_event,
            "y_realized": ev.y_realized,
            "p_market": getattr(ev, "p_market", np.nan),
            "decimal_odds": getattr(ev, "decimal_odds", np.nan),
            "source_published_at_max": structured.loc[structured.index[end_idx - 1], "source_published_at_max"] if end_idx > 0 else pd.NaT,
            "scraped_at_max": structured.loc[structured.index[end_idx - 1], "scraped_at_max"] if end_idx > 0 else pd.NaT,
        })

    labels_df = pd.DataFrame(labels)
    np.save(seq_path, sequences)
    labels_df.to_parquet(label_path, index=False)

    # Lock the held-out fence based on the data's actual range, not wall clock.
    if not labels_df.empty:
        last_event = pd.to_datetime(labels_df["target_event_time"]).max()
        write_held_out_manifest(market, last_event)
    return seq_path, label_path


def _extract_events(market: MarketConfig, structured: pd.DataFrame) -> pd.DataFrame:
    if "target_event_time" not in structured.columns:
        raise RuntimeError("structured frame missing 'target_event_time'")
    if "y_realized" not in structured.columns:
        raise RuntimeError("structured frame missing 'y_realized'")
    events = structured.dropna(subset=["target_event_time", "y_realized"]).copy()
    # One row per distinct target_event_time (use last observation for that event).
    events = events.reset_index().rename(columns={"t": "obs_t"}) if events.index.name == "t" else events.reset_index()
    # The structured index may be named differently after groupby; restore canonical "Index".
    if "obs_t" in events.columns:
        events = events.set_index("obs_t")
    events = events.groupby("target_event_time", as_index=False).last()
    events["target_event_time"] = pd.to_datetime(events["target_event_time"])
    events = events.sort_values("target_event_time").reset_index(drop=True)
    events.index = pd.Index(range(len(events)), name="Index")
    return events
