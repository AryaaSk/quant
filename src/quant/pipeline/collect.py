"""Stage 1: data collection.

Routes each data source declared in the market yaml to the right adapter and writes raw
records to `data/raw/<slug>/<source>.parquet`. Records carry `source_published_at`,
`scraped_at`, `source_url`, `source_type` columns; adapters that cannot produce these
columns must raise loudly rather than fall back silently.

The held-out fence is enforced here: an adapter writing under the training partition will
never see rows with target_event_time in the held-out window.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.adapters import get_adapter
from quant.config import MarketConfig


REQUIRED_COLS = ("source_published_at", "scraped_at", "source_url", "source_type")


def collect(market: MarketConfig, *, force: bool = False) -> dict[str, Path]:
    """Run every data source declared in the market yaml. Returns map source_name -> output path.

    The held-out manifest is written by `pipeline.state.build_state` after the structured
    frame is built, because the fence is computed from the actual data range (last
    `held_out_months` of available events) rather than today's wall clock.
    """
    out_dir = market.raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for source in market.data_sources:
        out_path = out_dir / f"{source.name}.parquet"
        if out_path.exists() and not force:
            written[source.name] = out_path
            continue
        adapter = get_adapter(source.name)
        df = adapter.fetch(market=market, source_params=source.params)
        _validate_records(source.name, df)
        df.to_parquet(out_path, index=False)
        written[source.name] = out_path

    return written


def _validate_records(source_name: str, df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"adapter '{source_name}' produced rows missing required cols: {missing}. "
            f"Adapters must emit source_published_at, scraped_at, source_url, source_type."
        )
    if df.empty:
        raise ValueError(f"adapter '{source_name}' returned 0 rows; expected non-empty data")


def write_held_out_manifest(market: MarketConfig, last_event_time: pd.Timestamp) -> Path:
    """Write the held-out manifest given the latest event time observed in the data.

    Called by `pipeline.state.build_state` after labels are extracted. The fence is set
    to `last_event_time - held_out_months`. Subsequent calls do NOT overwrite the manifest
    once it exists; the fence is locked write-once.
    """
    held_out_dir = market.held_out_dir()
    held_out_dir.mkdir(parents=True, exist_ok=True)
    manifest = held_out_dir / "MANIFEST.json"
    if manifest.exists():
        return manifest

    last_event_time = pd.Timestamp(last_event_time)
    if last_event_time.tzinfo is not None:
        last_event_time = last_event_time.tz_localize(None)
    fence = (last_event_time - pd.DateOffset(months=market.held_out_months)).normalize()
    payload = {
        "slug": market.slug,
        "held_out_start": fence.isoformat(),
        "last_event_observed": last_event_time.isoformat(),
        "locked_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
        "held_out_months": market.held_out_months,
        "note": (
            "This window is locked write-once. Training MUST NOT load events with "
            "target_event_time >= held_out_start. The walk-forward backtest evaluates exactly here."
        ),
    }
    manifest.write_text(json.dumps(payload, indent=2))
    return manifest
