"""Smoke tests for sec_filings adapter.

All mocked. No real HTTP requests to SEC.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quant.adapters.sec_filings import SECFilingsAdapter, _html_to_clean_text


def _make_market(tmp_path, slug="sec-test"):
    from quant.config import (
        MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig,
    )
    return MarketConfig(
        slug=slug, display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[
            DataSourceConfig(name="stock_earnings", kind="numeric", params={}),
            DataSourceConfig(name="sec_filings", kind="text",
                             params={"event_source": "stock_earnings", "entity_column": "ticker"}),
        ],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )


def test_html_to_clean_text_strips_scripts_and_tables():
    html = """
    <html><head><script>bad();</script></head>
    <body>
      <table><tr><td>XBRL noise</td></tr></table>
      <p>Hello world.</p>
      <p>Some  whitespace   here.</p>
    </body></html>
    """
    text = _html_to_clean_text(html)
    assert "Hello world." in text
    assert "XBRL noise" not in text
    assert "bad()" not in text
    # whitespace collapsed
    assert "  " not in text


def test_adapter_raises_when_event_source_missing(tmp_path, monkeypatch):
    """If stock_earnings parquet doesn't exist, sec_filings should raise clearly."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)
    adapter = SECFilingsAdapter()
    with pytest.raises(RuntimeError, match="event_source"):
        adapter.fetch(market=market, source_params={"event_source": "stock_earnings"})


def test_adapter_produces_aligned_rows_when_event_source_present(tmp_path, monkeypatch):
    """Adapter joins to event_source on target_event_time. Mocks SEC HTTP calls."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)

    # Seed a fake stock_earnings.parquet
    raw_dir = market.raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-04-01", "2024-07-01"]),
        "target_event_time": pd.to_datetime(["2024-04-01", "2024-07-01"]),
        "ticker": ["AAPL", "AAPL"],
        "y_realized": [1.0, 0.0],
        "p_market": [0.55, 0.55],
    })
    events.to_parquet(raw_dir / "stock_earnings.parquet", index=False)

    # Mock the SEC HTTP layer: ticker→CIK map + filings index + filing body
    fake_ticker_map = {"AAPL": "0000320193"}
    fake_filings_df = pd.DataFrame({
        "form": ["10-Q", "8-K"],
        "filed_at": pd.to_datetime(["2024-02-15", "2024-03-20"]),
        "accession_no": ["0000320193-24-000001", "0000320193-24-000002"],
        "primary_doc": ["aapl-10q.htm", "aapl-8k.htm"],
    })

    def fake_load_or_fetch_map(cache_dir, headers, rate):
        return fake_ticker_map

    def fake_filings(cik, headers, cache_dir, rate):
        return fake_filings_df.copy()

    def fake_body(cik, acc, doc, headers, cache_dir, rate):
        return f"[mock filing body for {acc}]"

    with patch("quant.adapters.sec_filings._load_or_fetch_ticker_cik_map", fake_load_or_fetch_map), \
         patch("quant.adapters.sec_filings._fetch_filings_for_cik", fake_filings), \
         patch("quant.adapters.sec_filings._fetch_filing_text", fake_body):
        adapter = SECFilingsAdapter()
        df = adapter.fetch(market=market, source_params={
            "event_source": "stock_earnings", "entity_column": "ticker",
        })

    assert len(df) == 2, "should emit one row per (ticker, event) in event_source"
    # Both rows should have text content from the mocked filings
    assert all("[mock filing body" in t for t in df["text__sec"])
    # Each row should be tied to the unique target_event_time of stock_earnings
    assert set(df["target_event_time"]) == set(events["target_event_time"])
    # Filing count >= 1 per row
    assert (df["num__sec_filing_count"] >= 1.0).all()
