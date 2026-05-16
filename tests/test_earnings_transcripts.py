"""Smoke tests for earnings_transcripts adapter (Motley Fool mirror).

All mocked. No real HTTP requests to fool.com.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from quant.adapters.earnings_transcripts import EarningsTranscriptsAdapter


def _make_market(tmp_path, slug="trans-test"):
    from quant.config import (
        MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig,
    )
    return MarketConfig(
        slug=slug, display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[
            DataSourceConfig(name="stock_earnings", kind="numeric", params={}),
            DataSourceConfig(name="earnings_transcripts", kind="text",
                             params={"event_source": "stock_earnings", "entity_column": "ticker"}),
        ],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )


def test_adapter_raises_when_event_source_missing(tmp_path, monkeypatch):
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)
    adapter = EarningsTranscriptsAdapter()
    with pytest.raises(RuntimeError, match="event_source"):
        adapter.fetch(market=market, source_params={"event_source": "stock_earnings"})


def test_adapter_picks_prior_transcript_strictly_before_event(tmp_path, monkeypatch):
    """If we have transcripts at 2024-01-30 and 2024-04-25, an event on 2024-04-01
    should pick the 2024-01-30 transcript (prior), not the 2024-04-25 one (post-event)."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)

    raw_dir = market.raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-04-01"]),
        "target_event_time": pd.to_datetime(["2024-04-01"]),
        "ticker": ["AAPL"],
        "y_realized": [1.0],
        "p_market": [0.55],
    })
    events.to_parquet(raw_dir / "stock_earnings.parquet", index=False)

    fake_idx = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-30", "2024-04-25"]),
        "url": ["https://fool.com/2024/01/30/aapl-q1/", "https://fool.com/2024/04/25/aapl-q2/"],
        "title": ["AAPL Q1", "AAPL Q2"],
    })

    def fake_index(ticker, headers, cache_dir):
        return fake_idx.copy()

    def fake_body(url, headers, cache_dir):
        if "2024/01/30" in url:
            return "Q1 transcript: forward guidance OK."
        return "Q2 transcript: should NOT be selected for the 2024-04-01 event."

    with patch("quant.adapters.earnings_transcripts._fetch_transcript_index", fake_index), \
         patch("quant.adapters.earnings_transcripts._fetch_transcript_body", fake_body):
        adapter = EarningsTranscriptsAdapter()
        df = adapter.fetch(market=market, source_params={
            "event_source": "stock_earnings", "entity_column": "ticker",
        })

    assert len(df) == 1
    row = df.iloc[0]
    assert "Q1 transcript" in row["text__transcript"]
    assert "should NOT" not in row["text__transcript"]
    assert row["num__has_transcript"] == 1.0


def test_adapter_emits_empty_row_when_no_prior_transcript(tmp_path, monkeypatch):
    """If the only available transcript is post-event, the row is emitted with empty text."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)

    raw_dir = market.raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-04-01"]),
        "target_event_time": pd.to_datetime(["2024-04-01"]),
        "ticker": ["AAPL"],
        "y_realized": [1.0],
        "p_market": [0.55],
    })
    events.to_parquet(raw_dir / "stock_earnings.parquet", index=False)

    fake_idx = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-04-25"]),
        "url": ["https://fool.com/2024/04/25/aapl-q2/"],
        "title": ["AAPL Q2"],
    })

    def fake_index(ticker, headers, cache_dir):
        return fake_idx.copy()

    with patch("quant.adapters.earnings_transcripts._fetch_transcript_index", fake_index):
        adapter = EarningsTranscriptsAdapter()
        df = adapter.fetch(market=market, source_params={
            "event_source": "stock_earnings", "entity_column": "ticker",
        })

    assert len(df) == 1
    assert df.iloc[0]["text__transcript"] == ""
    assert df.iloc[0]["num__has_transcript"] == 0.0
