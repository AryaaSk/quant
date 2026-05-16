"""Stage 4: turn (sequences, labels) into torch Datasets with train / val / held-out splits.

Splits are temporal (no random shuffling across the boundary). The held-out window is the
last `held_out_months` of event time; the val window is the `val_fraction` slice
immediately before held-out; everything earlier is train.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from quant.config import MarketConfig


class StateSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: pd.DataFrame, market: MarketConfig):
        self.sequences = sequences
        self.labels = labels.reset_index(drop=True)
        self.market = market
        self.targets = _make_targets(market, labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.sequences[idx]).float()
        target = self.targets[idx]
        p_market = self.labels.iloc[idx].get("p_market", float("nan"))
        return {
            "x": x,
            "y": torch.tensor(target, dtype=torch.long if self.market.label.kind == "bucketed" else torch.float32),
            "p_market": torch.tensor(float(p_market) if pd.notna(p_market) else 0.5),
        }


def _make_targets(market: MarketConfig, labels: pd.DataFrame) -> np.ndarray:
    y = labels["y_realized"].to_numpy()
    if market.label.kind == "binary":
        return (y > 0.5).astype(np.float32)
    edges = np.asarray(market.label.buckets, dtype=np.float64)
    targets = np.digitize(y, edges[1:-1], right=False).astype(np.int64)
    return targets


def make_splits(market: MarketConfig) -> dict[str, StateSequenceDataset]:
    seq_path = market.state_dir() / "sequences.npy"
    label_path = market.state_dir() / "labels.parquet"
    sequences = np.load(seq_path)
    labels = pd.read_parquet(label_path)
    labels["target_event_time"] = pd.to_datetime(labels["target_event_time"])

    manifest = json.loads((market.held_out_dir() / "MANIFEST.json").read_text())
    held_out_start = pd.Timestamp(manifest["held_out_start"])
    if held_out_start.tzinfo is not None:
        held_out_start = held_out_start.tz_localize(None)
    # Match dtype with labels column (which is tz-naive after collection -> structure).
    labels = labels.copy()
    labels["target_event_time"] = pd.to_datetime(labels["target_event_time"]).dt.tz_localize(None) if pd.to_datetime(labels["target_event_time"]).dt.tz is not None else pd.to_datetime(labels["target_event_time"])

    in_held_out = labels["target_event_time"] >= held_out_start
    in_train_or_val = ~in_held_out

    val_count = max(1, int(in_train_or_val.sum() * market.train.val_fraction))
    train_val_indices = labels[in_train_or_val].sort_values("target_event_time").index.to_numpy()
    train_indices = train_val_indices[:-val_count]
    val_indices = train_val_indices[-val_count:]
    held_indices = labels[in_held_out].sort_values("target_event_time").index.to_numpy()

    def _slice(indices):
        return StateSequenceDataset(sequences[indices], labels.iloc[indices].reset_index(drop=True), market)

    return {
        "train": _slice(train_indices),
        "val": _slice(val_indices),
        "held_out": _slice(held_indices),
    }
