"""Stage A.5 adapter: per-event structured feature extraction via codex agents.

Reads:
- the event list from a companion adapter (e.g. `football_data` parquet)
- the articles dump from a companion adapter (e.g. `news_scraper` _news directory)

Batches events, spawns one codex agent per batch with the `extract-features` skill.
Each agent reads in-window articles per event and emits a strict schema of numeric
features as JSON lines.

Output rows (one per event) have:
- timestamp = target_event_time
- source_published_at, scraped_at: copied from the event's anchor
- source_url, source_type
- num__<feature_name> columns per the yaml-declared schema

The agent only sees articles with `published_at < event_time` (temporal-leak guard
enforced in the skill brief + verified by the property-based leak test downstream).

source_params schema:
  feature_schema: [{name, description, range: [lo, hi]}]
  articles_source: news_scraper           # adapter name that produced the article dump
  event_source: football_data             # adapter that produced the event list
  window_days: 7                          # how far back per event to consider
  batch_size: 20                          # events per codex invocation
  concurrency: 3
  default_value: 0.0                      # value when extraction fails or no articles
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant.agents.runner import Invocation, run_agents_parallel
from quant.config import MarketConfig


class AgentFeatureExtractorAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        feature_schema = list(source_params.get("feature_schema") or [])
        if not feature_schema:
            raise ValueError("agent_feature_extractor requires source_params['feature_schema']")
        for f in feature_schema:
            if "name" not in f:
                raise ValueError(f"feature_schema entry missing 'name': {f}")

        articles_source = source_params.get("articles_source", "news_scraper")
        event_source = source_params.get("event_source", "football_data")
        window_days = int(source_params.get("window_days", 7))
        batch_size = int(source_params.get("batch_size", 20))
        concurrency = int(source_params.get("concurrency", 3))
        default_value = float(source_params.get("default_value", 0.0))

        gated = os.environ.get("QUANT_ENABLE_AGENTS") == "1"

        # Locate the companion adapters' outputs.
        articles_dir = market.raw_dir() / "_news"
        event_parquet = market.raw_dir() / f"{event_source}.parquet"
        extracted_dir = market.raw_dir() / "_extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if not event_parquet.exists():
            raise RuntimeError(
                f"agent_feature_extractor needs the event source '{event_source}' to have run first; "
                f"missing {event_parquet}"
            )
        if not articles_dir.exists() or not any(articles_dir.iterdir()):
            raise RuntimeError(
                f"agent_feature_extractor needs scraped articles in {articles_dir}; "
                f"run news_scraper first (QUANT_ENABLE_AGENTS=1)"
            )

        events = _build_event_list(event_parquet)
        if events.empty:
            raise RuntimeError(f"no events in {event_parquet}")

        feature_names = [f["name"] for f in feature_schema]

        # Batch events for codex invocations. Idempotent via per-batch _done markers.
        batches = _chunk_events(events, batch_size)
        invocations: list[Invocation] = []
        for batch_idx, batch_df in enumerate(batches):
            batch_path = extracted_dir / f"batch_{batch_idx:04d}_input.jsonl"
            output_path = extracted_dir / f"batch_{batch_idx:04d}_output.jsonl"
            if output_path.exists():
                continue  # cached; skip
            _write_batch_input(batch_path, batch_df)
            invocations.append(Invocation(
                skill="extract-features",
                params={
                    "market_slug": market.slug,
                    "batch_path": str(batch_path.absolute()),
                    "articles_dir": str(articles_dir.absolute()),
                    "output_path": str(output_path.absolute()),
                    "feature_schema": feature_schema,
                    "window_days": window_days,
                },
                log_subdir=f"{market.slug}/extract-features/batch_{batch_idx:04d}",
            ))

        if invocations:
            if not gated:
                # Fall back to default-only mode: emit one row per event with neutral defaults.
                # Allows the pipeline to complete without spending tokens during development.
                return _default_rows(events, feature_schema, default_value)
            run_agents_parallel(invocations, concurrency=concurrency)

        # Aggregate batch outputs + any cached batches.
        all_rows = _aggregate_outputs(events, extracted_dir, feature_names, default_value)
        return all_rows


def _build_event_list(event_parquet: Path) -> pd.DataFrame:
    """Build the one-row-per-unique-event-time list the extractor batches against.

    The companion adapter (e.g. football_data) emits one row PER MATCH, but the
    downstream `structure.py` pivots by `target_event_time` (often a daily bucket),
    so many matches share the same event-time row in the structured frame. We
    extract features per UNIQUE event-time, then those features broadcast to all
    matches sharing that timestamp. This saves an order of magnitude of agent calls
    for sports markets where many matches play on the same day.
    """
    df = pd.read_parquet(event_parquet)
    if "target_event_time" not in df.columns:
        raise RuntimeError(f"{event_parquet} missing 'target_event_time'")
    events = df.dropna(subset=["target_event_time"]).copy()
    events["target_event_time"] = pd.to_datetime(events["target_event_time"])
    # Aggregate per unique event time: keep the first row's context columns. The agent
    # reads articles for the day anyway, not per-match, so the home/away team of the
    # representative row is sufficient context.
    events = (
        events.groupby("target_event_time", as_index=False)
        .first()
        .sort_values("target_event_time")
        .reset_index(drop=True)
    )
    # event_id is the integer position in time-sorted order; deterministic across runs.
    events["event_id"] = events.index
    return events


def _chunk_events(events: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [events.iloc[i : i + batch_size].copy() for i in range(0, len(events), batch_size)]


def _write_batch_input(path: Path, batch: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for _, row in batch.iterrows():
        # Pull a small per-event "context" subset of useful adapter-emitted hints.
        context = {}
        for key in ("home_team", "away_team", "Tournament", "Surface", "_league"):
            if key in row and pd.notna(row[key]):
                context[key] = str(row[key])
        # Football-data-specific: parse home/away from source_url is annoying; rely on source_type.
        if "source_type" in row and pd.notna(row["source_type"]):
            context["source_type"] = str(row["source_type"])
        line = {
            "event_id": int(row["event_id"]),
            "target_event_time": pd.Timestamp(row["target_event_time"]).isoformat(),
            "context": context,
        }
        lines.append(json.dumps(line, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")


def _aggregate_outputs(
    events: pd.DataFrame, extracted_dir: Path, feature_names: list[str], default_value: float
) -> pd.DataFrame:
    # Build a lookup event_id -> {feature: value}
    by_event: dict[int, dict[str, float]] = {}
    for out_path in sorted(extracted_dir.glob("batch_*_output.jsonl")):
        try:
            for raw in out_path.read_text().splitlines():
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                if obj.get("_batch_summary"):
                    continue
                eid = int(obj.get("event_id"))
                feats = obj.get("features") or {}
                by_event[eid] = {fn: float(feats.get(fn, default_value)) for fn in feature_names}
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    rows: list[dict] = []
    scraped_at = pd.Timestamp.utcnow().tz_localize(None)
    for _, row in events.iterrows():
        eid = int(row["event_id"])
        feats = by_event.get(eid)
        if feats is None:
            feats = {fn: default_value for fn in feature_names}
        rec = {
            "timestamp": pd.Timestamp(row["target_event_time"]),
            "source_published_at": pd.Timestamp(row["target_event_time"]),
            "scraped_at": scraped_at,
            "source_url": f"agent_feature_extractor://event/{eid}",
            "source_type": "agent_feature_extractor",
            "target_event_time": pd.Timestamp(row["target_event_time"]),
        }
        for fn, val in feats.items():
            rec[f"num__{fn}"] = float(np.clip(val, -8.0, 8.0))
        rows.append(rec)

    if not rows:
        raise RuntimeError("agent_feature_extractor produced 0 rows")
    return pd.DataFrame(rows)


def _default_rows(
    events: pd.DataFrame, feature_schema: list[dict], default_value: float
) -> pd.DataFrame:
    """Fallback when QUANT_ENABLE_AGENTS!=1: emit one row per event with neutral defaults.

    Lets the rest of the pipeline (structure, state, train, backtest) still run for
    development / CI without spending tokens. The downstream model just sees a constant
    feature column, which won't hurt and won't help.
    """
    rows: list[dict] = []
    scraped_at = pd.Timestamp.utcnow().tz_localize(None)
    feature_names = [f["name"] for f in feature_schema]
    for _, row in events.iterrows():
        eid = int(row["event_id"])
        rec = {
            "timestamp": pd.Timestamp(row["target_event_time"]),
            "source_published_at": pd.Timestamp(row["target_event_time"]),
            "scraped_at": scraped_at,
            "source_url": f"agent_feature_extractor://event/{eid}",
            "source_type": "agent_feature_extractor.default",
            "target_event_time": pd.Timestamp(row["target_event_time"]),
        }
        for fn in feature_names:
            rec[f"num__{fn}"] = float(default_value)
        rows.append(rec)
    return pd.DataFrame(rows)
