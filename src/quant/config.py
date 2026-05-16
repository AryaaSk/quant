"""Market yaml loading and pydantic validation.

A market yaml defines: data sources, label, time grain, state vector schema, model type,
and backtest costs. This module is the only place where raw yaml meets typed objects;
everything else in the pipeline consumes the validated MarketConfig.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETS_DIR = REPO_ROOT / "markets"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


class DataSourceConfig(BaseModel):
    name: str
    kind: Literal["numeric", "market", "text", "synthetic"]
    params: dict = Field(default_factory=dict)


class StateBlockConfig(BaseModel):
    name: str
    slots: str  # "start:end" half-open
    builder: Literal["numeric", "market_state", "text", "calendar"]
    params: dict = Field(default_factory=dict)

    @property
    def slot_start(self) -> int:
        return int(self.slots.split(":")[0])

    @property
    def slot_end(self) -> int:
        return int(self.slots.split(":")[1])

    @property
    def slot_width(self) -> int:
        return self.slot_end - self.slot_start


class StateVectorConfig(BaseModel):
    dim: int
    blocks: list[StateBlockConfig]

    @model_validator(mode="after")
    def _validate_block_layout(self) -> "StateVectorConfig":
        cursor = 0
        for block in self.blocks:
            if block.slot_start != cursor:
                raise ValueError(
                    f"state vector block '{block.name}' starts at {block.slot_start}, "
                    f"expected {cursor} (blocks must be contiguous and ordered)"
                )
            if block.slot_end > self.dim:
                raise ValueError(
                    f"block '{block.name}' ends at {block.slot_end}, exceeds state dim {self.dim}"
                )
            cursor = block.slot_end
        if cursor != self.dim:
            raise ValueError(
                f"blocks cover [0, {cursor}) but state dim is {self.dim} (must be exactly covered)"
            )
        return self


class LabelConfig(BaseModel):
    kind: Literal["binary", "bucketed"]
    target_event: str
    buckets: list[float] | None = None  # required for bucketed; len(buckets) = num_classes + 1

    @model_validator(mode="after")
    def _validate(self) -> "LabelConfig":
        if self.kind == "bucketed":
            if self.buckets is None or len(self.buckets) < 3:
                raise ValueError("bucketed labels require at least 3 bucket edges (-> >=2 classes)")
        return self

    @property
    def num_classes(self) -> int:
        if self.kind == "binary":
            return 2
        return len(self.buckets) - 1


class TransformerConfig(BaseModel):
    layers: int = 6
    model_dim: int = 256
    heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1


class TrainConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 42
    val_fraction: float = 0.15


class BacktestConfig(BaseModel):
    commission_bps: int = 20          # bps of stake (Kalshi taker = 20)
    base_slippage_bps: int = 30       # additional execution cost
    safety_margin_bps: int = 150      # additional edge required beyond cost
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.02
    starting_bankroll: float = 1000.0


class MarketConfig(BaseModel):
    slug: str
    display_name: str
    platform: str
    notes: str = ""
    time_grain: Literal["daily", "weekly", "hourly"] = "daily"
    held_out_months: int = 6
    sequence_length: int = 32
    data_sources: list[DataSourceConfig]
    label: LabelConfig
    state_vector: StateVectorConfig
    model: Literal["transformer", "gbdt"] = "transformer"
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    def raw_dir(self) -> Path:
        return DATA_DIR / "raw" / self.slug

    def structured_dir(self) -> Path:
        return DATA_DIR / "structured" / self.slug

    def state_dir(self) -> Path:
        return DATA_DIR / "state" / self.slug

    def held_out_dir(self) -> Path:
        return DATA_DIR / "held_out" / self.slug

    def runs_dir(self) -> Path:
        return RUNS_DIR / self.slug


def load_market(slug_or_path: str | Path) -> MarketConfig:
    """Load a market yaml. Accepts a slug ('kalshi-cpi') or an explicit path."""
    p = Path(slug_or_path)
    if not p.exists():
        p = MARKETS_DIR / f"{slug_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"market config not found: {slug_or_path}")
    with p.open() as f:
        data = yaml.safe_load(f)
    return MarketConfig(**data)


def list_markets() -> list[str]:
    return sorted(p.stem for p in MARKETS_DIR.glob("*.yaml"))


def env(name: str, default: str | None = None) -> str | None:
    """Read an env var with .env fallback (lazy, no dotenv dependency)."""
    if name in os.environ:
        return os.environ[name]
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip()
                # Strip optional quotes
                if value.startswith(('"', "'")) and value.endswith(value[0]):
                    value = value[1:-1]
                return value
    return default
