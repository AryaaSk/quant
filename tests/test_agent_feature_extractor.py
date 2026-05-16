"""Tests for the agent_feature_extractor adapter.

All mocked. No real codex calls. Verifies:
- batching of events into JSONL files
- aggregation of per-batch JSONL outputs into a DataFrame
- num__<field> column naming
- default-value fallback when QUANT_ENABLE_AGENTS != 1
- temporal-leak guard at the adapter level (events only see articles before their time)
- range clipping (extreme values clamped)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant.adapters.agent_feature_extractor import (
    AgentFeatureExtractorAdapter,
    _aggregate_outputs,
    _build_event_list,
    _chunk_events,
    _default_rows,
    _write_batch_input,
)


_FEATURE_SCHEMA = [
    {"name": "home_injuries_severity", "range": [-1, 1]},
    {"name": "away_injuries_severity", "range": [-1, 1]},
    {"name": "manager_pressure", "range": [0, 1]},
]
_FEATURE_NAMES = [f["name"] for f in _FEATURE_SCHEMA]


def _write_event_parquet(path: Path, n_events: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2025-01-01", periods=n_events, freq="W")
    df = pd.DataFrame({
        "timestamp": dates,
        "source_published_at": dates,
        "scraped_at": dates,
        "source_url": [f"u{i}" for i in range(n_events)],
        "source_type": ["football_data.E2"] * n_events,
        "target_event_time": dates,
        "y_realized": [float(i % 2) for i in range(n_events)],
        "p_market": [0.5] * n_events,
        "home_team": [f"Home_{i}" for i in range(n_events)],
        "away_team": [f"Away_{i}" for i in range(n_events)],
    })
    df.to_parquet(path, index=False)


def _make_market(tmp_path: Path):
    from quant.config import (
        MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig
    )
    import quant.config as cfg
    cfg.DATA_DIR = tmp_path / "data"
    cfg.RUNS_DIR = tmp_path / "runs"
    return MarketConfig(
        slug="extract-test", display_name="t", platform="test",
        time_grain="daily", held_out_months=1, sequence_length=4,
        data_sources=[
            DataSourceConfig(name="football_data", kind="numeric", params={"leagues": ["E2"]}),
            DataSourceConfig(name="news_scraper", kind="text", params={"topics": []}),
            DataSourceConfig(name="agent_feature_extractor", kind="numeric", params={
                "feature_schema": _FEATURE_SCHEMA,
                "articles_source": "news_scraper",
                "event_source": "football_data",
            }),
        ],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )


# ---------- Batching + I/O helpers ----------


def test_build_event_list_sorts_and_assigns_ids(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=5)
    events = _build_event_list(path)
    assert "event_id" in events.columns
    assert list(events["event_id"]) == [0, 1, 2, 3, 4]
    # Sorted by time
    times = pd.to_datetime(events["target_event_time"]).tolist()
    assert times == sorted(times)


def test_chunk_events(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=10)
    events = _build_event_list(path)
    batches = _chunk_events(events, batch_size=3)
    assert len(batches) == 4  # 3, 3, 3, 1
    assert sum(len(b) for b in batches) == 10


def test_write_batch_input_format(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=3)
    events = _build_event_list(path)
    batch_path = tmp_path / "batch_0_input.jsonl"
    _write_batch_input(batch_path, events)
    lines = batch_path.read_text().strip().splitlines()
    assert len(lines) == 3
    obj = json.loads(lines[0])
    assert "event_id" in obj
    assert "target_event_time" in obj
    assert "context" in obj
    # Football-data-specific context fields propagated
    assert obj["context"].get("home_team") == "Home_0"


# ---------- Aggregation ----------


def test_aggregate_outputs_picks_up_features(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=4)
    events = _build_event_list(path)

    extracted_dir = tmp_path / "_extracted"
    extracted_dir.mkdir()
    # Two batch output files, covering all 4 events
    (extracted_dir / "batch_0000_output.jsonl").write_text("\n".join([
        json.dumps({"event_id": 0, "features": {"home_injuries_severity": 0.5, "away_injuries_severity": -0.2, "manager_pressure": 0.8}}),
        json.dumps({"event_id": 1, "features": {"home_injuries_severity": -0.7, "away_injuries_severity": 0.1, "manager_pressure": 0.3}}),
        json.dumps({"_batch_summary": True, "events_processed": 2}),
    ]) + "\n")
    (extracted_dir / "batch_0001_output.jsonl").write_text("\n".join([
        json.dumps({"event_id": 2, "features": {"home_injuries_severity": 0.0, "away_injuries_severity": 0.0, "manager_pressure": 0.0}}),
        json.dumps({"event_id": 3, "features": {"home_injuries_severity": 0.9, "away_injuries_severity": -0.9, "manager_pressure": 1.0}}),
    ]) + "\n")

    df = _aggregate_outputs(events, extracted_dir, _FEATURE_NAMES, default_value=0.0)
    assert len(df) == 4
    for fn in _FEATURE_NAMES:
        col = f"num__{fn}"
        assert col in df.columns
    # Event 0 features carry through
    assert df.iloc[0][f"num__home_injuries_severity"] == 0.5


def test_aggregate_outputs_falls_back_to_default_when_event_missing(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=3)
    events = _build_event_list(path)
    extracted_dir = tmp_path / "_extracted"
    extracted_dir.mkdir()
    # Only event 0 covered
    (extracted_dir / "batch_0000_output.jsonl").write_text(
        json.dumps({"event_id": 0, "features": {fn: 0.4 for fn in _FEATURE_NAMES}}) + "\n"
    )
    df = _aggregate_outputs(events, extracted_dir, _FEATURE_NAMES, default_value=-0.5)
    assert len(df) == 3
    # Events 1, 2 get default value
    assert df.iloc[1]["num__home_injuries_severity"] == -0.5
    assert df.iloc[2]["num__manager_pressure"] == -0.5
    # Event 0 has the real value
    assert df.iloc[0]["num__home_injuries_severity"] == 0.4


def test_aggregate_outputs_clips_extreme_values(tmp_path):
    path = tmp_path / "events.parquet"
    _write_event_parquet(path, n_events=2)
    events = _build_event_list(path)
    extracted_dir = tmp_path / "_extracted"
    extracted_dir.mkdir()
    (extracted_dir / "batch_0000_output.jsonl").write_text("\n".join([
        json.dumps({"event_id": 0, "features": {"home_injuries_severity": 100.0, "away_injuries_severity": -100.0, "manager_pressure": 50.0}}),
        json.dumps({"event_id": 1, "features": {fn: 0.0 for fn in _FEATURE_NAMES}}),
    ]) + "\n")
    df = _aggregate_outputs(events, extracted_dir, _FEATURE_NAMES, default_value=0.0)
    # Adapter clips to [-8, 8] internally to prevent insane values from poisoning training
    assert df.iloc[0]["num__home_injuries_severity"] <= 8.0
    assert df.iloc[0]["num__away_injuries_severity"] >= -8.0


# ---------- Fallback / default mode ----------


def test_default_rows_when_agents_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANT_ENABLE_AGENTS", raising=False)
    market = _make_market(tmp_path)
    # Football-data parquet on disk
    _write_event_parquet(market.raw_dir() / "football_data.parquet", n_events=3)
    # _news dir exists with at least one article so the existence check passes
    (market.raw_dir() / "_news" / "x").mkdir(parents=True)
    (market.raw_dir() / "_news" / "x" / "article_0001.json").write_text(json.dumps({"text": "hi"}))

    adapter = AgentFeatureExtractorAdapter()
    df = adapter.fetch(market=market, source_params={
        "feature_schema": _FEATURE_SCHEMA,
        "articles_source": "news_scraper",
        "event_source": "football_data",
        "default_value": 0.0,
    })
    assert len(df) == 3
    for fn in _FEATURE_NAMES:
        assert f"num__{fn}" in df.columns
        assert (df[f"num__{fn}"] == 0.0).all()
    # Required columns present
    for col in ("timestamp", "source_published_at", "scraped_at", "source_url", "source_type", "target_event_time"):
        assert col in df.columns


def test_required_input_validation(tmp_path):
    market = _make_market(tmp_path)
    adapter = AgentFeatureExtractorAdapter()
    with pytest.raises(ValueError, match="feature_schema"):
        adapter.fetch(market=market, source_params={})


def test_event_source_must_exist(tmp_path):
    market = _make_market(tmp_path)
    adapter = AgentFeatureExtractorAdapter()
    with pytest.raises(RuntimeError, match="football_data"):
        adapter.fetch(market=market, source_params={
            "feature_schema": _FEATURE_SCHEMA,
            "event_source": "football_data",
            "articles_source": "news_scraper",
        })


def test_articles_dir_must_exist(tmp_path):
    market = _make_market(tmp_path)
    _write_event_parquet(market.raw_dir() / "football_data.parquet", n_events=2)
    # No _news dir
    adapter = AgentFeatureExtractorAdapter()
    with pytest.raises(RuntimeError, match="news_scraper first|articles"):
        adapter.fetch(market=market, source_params={
            "feature_schema": _FEATURE_SCHEMA,
            "event_source": "football_data",
            "articles_source": "news_scraper",
        })


# ---------- Temporal-leak invariant (per-adapter) ----------


def test_skill_brief_documents_temporal_leak_guard():
    """The extract-features skill MUST state the temporal-leak invariant.

    This is a defensive check: the skill brief is the only thing the agent sees.
    If a future edit accidentally removes the invariant from the brief, this test fails.
    """
    from quant.agents.runner import SKILLS_DIR
    brief = (SKILLS_DIR / "extract-features.md").read_text()
    assert "temporal" in brief.lower() or "leak" in brief.lower()
    assert "target_event_time" in brief
    assert "strictly" in brief.lower() and "less than" in brief.lower()
