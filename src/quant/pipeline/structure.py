"""Stage 2: structure raw records into a wide per-timestep feature frame.

Reads `data/raw/<slug>/<source>.parquet`, pivots/joins them onto a daily time index, and
writes `data/structured/<slug>/features.parquet`. Each row corresponds to one timestep at
the market's `time_grain`; each column is one feature.

Key invariant: every row in the structured frame remembers `source_published_at_max` and
`scraped_at_max` aggregated over the rows that contributed to it. The temporal-leak test
in `tests/test_temporal_leak.py` asserts these timestamps are <= the row index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import MarketConfig


def structure(market: MarketConfig, *, force: bool = False) -> Path:
    out_dir = market.structured_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    if out_path.exists() and not force:
        return out_path

    raw_frames = _load_raw(market)
    structured = _build_structured_frame(market, raw_frames)
    structured.to_parquet(out_path)
    return out_path


def _load_raw(market: MarketConfig) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source in market.data_sources:
        p = market.raw_dir() / f"{source.name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"raw not found for source '{source.name}': {p}")
        out[source.name] = pd.read_parquet(p)
    return out


def _build_structured_frame(market: MarketConfig, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Index granularity follows the market's time_grain. Daily markets get one row per
    # date; hourly markets retain the hour.
    def _bucket(ts_series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(ts_series)
        if market.time_grain == "hourly":
            return ts.dt.floor("h")
        if market.time_grain == "weekly":
            return ts.dt.to_period("W").dt.start_time
        return ts.dt.normalize()

    all_ts: list[pd.Timestamp] = []
    for df in raw.values():
        if "timestamp" in df.columns:
            all_ts.extend(_bucket(df["timestamp"]).tolist())
    if not all_ts:
        raise RuntimeError("no timestamps in any raw source")
    idx = pd.DatetimeIndex(sorted(set(all_ts)))

    wide = pd.DataFrame(index=idx)
    wide.index.name = "t"
    published_max = pd.Series(index=idx, dtype="datetime64[ns]")
    scraped_max = pd.Series(index=idx, dtype="datetime64[ns]")
    target_event_time = pd.Series(index=idx, dtype="datetime64[ns]")

    for source_name, df in raw.items():
        df = df.copy()
        df["timestamp"] = _bucket(df["timestamp"])
        df["source_published_at"] = pd.to_datetime(df["source_published_at"])
        df["scraped_at"] = pd.to_datetime(df["scraped_at"])

        # Update published/scraped maxima
        per_day_pub = df.groupby("timestamp")["source_published_at"].max()
        per_day_scr = df.groupby("timestamp")["scraped_at"].max()
        published_max = published_max.combine(per_day_pub, lambda a, b: max(a, b) if pd.notna(a) and pd.notna(b) else (a if pd.notna(a) else b))
        scraped_max = scraped_max.combine(per_day_scr, lambda a, b: max(a, b) if pd.notna(a) and pd.notna(b) else (a if pd.notna(a) else b))

        # Numeric source -> pivot
        numeric_cols = [c for c in df.columns if c.startswith("num__")]
        for col in numeric_cols:
            series = df.groupby("timestamp")[col].last().reindex(idx)
            wide[col[len("num__"):]] = series

        # Text sources: forward `text` and any `text__<topic>` columns. Multiple text rows
        # per bucket are newline-joined so the text block builder sees them all.
        text_cols = [c for c in df.columns if c == "text" or c.startswith("text__")]
        for text_col in text_cols:
            joined = df.groupby("timestamp")[text_col].apply(
                lambda parts: "\n".join(str(p) for p in parts if pd.notna(p) and str(p).strip())
            )
            joined = joined.reindex(idx)
            if text_col in wide.columns:
                # If multiple sources contribute to the same text column (rare), concat.
                wide[text_col] = (wide[text_col].fillna("").astype(str) + "\n" + joined.fillna("").astype(str)).str.strip()
            else:
                wide[text_col] = joined.fillna("")

        # Target event time: any source can mark a target_event_time per row. We take min.
        if "target_event_time" in df.columns:
            te = df.groupby("timestamp")["target_event_time"].min()
            te = pd.to_datetime(te).reindex(idx)
            target_event_time = target_event_time.combine(te, lambda a, b: min(a, b) if pd.notna(a) and pd.notna(b) else (a if pd.notna(a) else b))

        # Realized outcome (used for labels)
        if "y_realized" in df.columns:
            y = df.groupby("timestamp")["y_realized"].last().reindex(idx)
            wide["y_realized"] = y

        # Market-side probability + decimal odds, used by the decision rule downstream.
        if "p_market" in df.columns:
            pm = df.groupby("timestamp")["p_market"].last().reindex(idx)
            wide["p_market"] = pm
        if "decimal_odds" in df.columns:
            od = df.groupby("timestamp")["decimal_odds"].last().reindex(idx)
            wide["decimal_odds"] = od

    wide["source_published_at_max"] = published_max
    wide["scraped_at_max"] = scraped_max
    wide["target_event_time"] = target_event_time
    return wide
