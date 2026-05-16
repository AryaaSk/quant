"""StateBuilder: composes block builders into a contiguous state vector per the market yaml.

A market's `state_vector` config is a list of blocks. Each block declares its builder type
(numeric, market_state, text, calendar), slot range, and params. The composer instantiates
the right builder per block, fits them on the structured frame, and provides a `build(t)`
method that returns the full state vector at timestep t.

Sequence assembly happens in `pipeline/state.py`: it calls `composer.build(t)` for each of
the `sequence_length` timesteps leading up to an event and stacks them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.config import StateBlockConfig, StateVectorConfig
from quant.state.calendar import CalendarBlockBuilder
from quant.state.market_state import MarketStateBlockBuilder
from quant.state.numeric import NumericBlockBuilder
from quant.state.text import TextBlockBuilder


@dataclass
class StateVectorSpec:
    dim: int
    block_names: list[str]
    block_ranges: list[tuple[int, int]]


class StateBuilder:
    def __init__(self, schema: StateVectorConfig):
        self.schema = schema
        self.builders: list[object] = []
        for block in schema.blocks:
            self.builders.append(self._instantiate(block))

    def _instantiate(self, block: StateBlockConfig):
        params = dict(block.params or {})
        width = block.slot_width
        if block.builder == "numeric":
            return NumericBlockBuilder(
                series_names=params.get("series", []),
                slot_width=width,
                rolling_window=params.get("rolling_window", 252),
                fill_missing=params.get("fill_missing", True),
            )
        if block.builder == "market_state":
            return MarketStateBlockBuilder(
                series_names=params.get("series", []),
                slot_width=width,
                velocity_lags=tuple(params.get("velocity_lags", [1, 5, 30])),
                rolling_window=params.get("rolling_window", 60),
            )
        if block.builder == "text":
            return TextBlockBuilder(
                slot_width=width,
                text_column=params.get("text_column", "text"),
                window_days=params.get("window_days", 7),
                mode=params.get("mode", "handcrafted"),
                encoder_name=params.get("encoder_name", "sentence-transformers/all-mpnet-base-v2"),
                entity_keywords=tuple(params.get("entity_keywords", [])),
                sentiment_lexicon=params.get("sentiment_lexicon"),
                voyage_cache_dir=params.get("voyage_cache_dir"),
                min_evidence_chars=params.get("min_evidence_chars", 400),
                pool_strategy=params.get("pool_strategy", "length_weighted"),
                use_pca=params.get("use_pca", False),
                pca_min_samples=params.get("pca_min_samples", 64),
                pca_fit_max_samples=params.get("pca_fit_max_samples", 512),
            )
        if block.builder == "calendar":
            return CalendarBlockBuilder(
                slot_width=width,
                target_event_column=params.get("target_event_column", "target_event_time"),
                last_event_column=params.get("last_event_column"),
                extra_flag_columns=tuple(params.get("extra_flag_columns", [])),
            )
        raise ValueError(f"unknown block builder: {block.builder}")

    def fit(self, structured_df: pd.DataFrame) -> None:
        for builder in self.builders:
            builder.fit(structured_df)

    def build(self, t: pd.Timestamp, structured_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(self.schema.dim, dtype=np.float32)
        for block_cfg, builder in zip(self.schema.blocks, self.builders, strict=True):
            piece = builder.build(t, structured_df)
            if piece.shape[0] != block_cfg.slot_width:
                raise RuntimeError(
                    f"block '{block_cfg.name}' returned shape {piece.shape}, expected ({block_cfg.slot_width},)"
                )
            out[block_cfg.slot_start : block_cfg.slot_end] = piece
        return out

    def spec(self) -> StateVectorSpec:
        return StateVectorSpec(
            dim=self.schema.dim,
            block_names=[b.name for b in self.schema.blocks],
            block_ranges=[(b.slot_start, b.slot_end) for b in self.schema.blocks],
        )
