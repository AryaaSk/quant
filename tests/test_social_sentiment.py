"""Smoke tests for social_sentiment adapter (Reddit + StockTwits).

All mocked. No real network calls.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from quant.adapters.social_sentiment import (
    SocialSentimentAdapter, _reddit_features, _stocktwits_features,
)


def _make_market(tmp_path, slug="social-test"):
    from quant.config import (
        MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig,
    )
    return MarketConfig(
        slug=slug, display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[
            DataSourceConfig(name="stock_earnings", kind="numeric", params={}),
            DataSourceConfig(name="social_sentiment", kind="numeric",
                             params={"event_source": "stock_earnings", "entity_column": "ticker"}),
        ],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )


def test_reddit_features_window_filter():
    """Only posts inside [t_event - window, t_event) should be counted."""
    df = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2024-03-15", "2024-03-25", "2024-04-05", "2024-05-01",
        ]),
        "score": [10, 20, 30, 40],
        "n_comments": [1, 2, 3, 4],
    })
    t_event = pd.Timestamp("2024-04-01")
    out = _reddit_features(df, t_event, window_days=30)
    # 2024-03-15 (16 days before) and 2024-03-25 (7 days before) are in window
    # 2024-04-05 is POST event, must be excluded
    # 2024-05-01 is well outside window
    assert out["num__reddit_mention_count_30d"] == 2.0
    assert out["num__reddit_comment_count_30d"] == 3.0  # 1+2
    assert out["num__reddit_avg_score_30d"] == 15.0  # mean(10, 20)


def test_reddit_features_empty_returns_zeros():
    out = _reddit_features(pd.DataFrame(), pd.Timestamp("2024-04-01"), 30)
    assert out["num__reddit_mention_count_30d"] == 0.0
    assert out["num__reddit_avg_score_30d"] == 0.0


def test_stocktwits_features_bullish_ratio():
    df = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31",
        ]),
        "sentiment": ["bullish", "bullish", "bearish", "neutral"],
    })
    t_event = pd.Timestamp("2024-04-01")
    out = _stocktwits_features(df, t_event, window_days=7)
    # All 4 in window. 2 bull, 1 bear, 1 neutral. Tagged total = 3. Ratio = 2/3.
    assert out["num__stocktwits_msg_count_7d"] == 4.0
    assert abs(out["num__stocktwits_bullish_ratio"] - (2 / 3)) < 1e-6


def test_stocktwits_features_empty_returns_neutral():
    """Missing data returns 0.5 bullish ratio (neutral), not NaN."""
    out = _stocktwits_features(pd.DataFrame(), pd.Timestamp("2024-04-01"), 7)
    assert out["num__stocktwits_msg_count_7d"] == 0.0
    assert out["num__stocktwits_bullish_ratio"] == 0.5


def test_adapter_emits_one_row_per_event_even_without_reddit_auth(tmp_path, monkeypatch):
    """Without REDDIT_CLIENT_ID set, Reddit half degrades to zeros but rows still emit."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    market = _make_market(tmp_path)

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

    def fake_stocktwits(ticker, cache_dir):
        return pd.DataFrame(columns=["created_at", "sentiment"])

    with patch("quant.adapters.social_sentiment._fetch_stocktwits_messages", fake_stocktwits):
        adapter = SocialSentimentAdapter()
        df = adapter.fetch(market=market, source_params={
            "event_source": "stock_earnings", "entity_column": "ticker",
        })

    assert len(df) == 2
    # Without Reddit auth, mention count is 0
    assert (df["num__reddit_mention_count_30d"] == 0.0).all()
    # Without StockTwits data, bullish ratio defaults to 0.5
    assert (df["num__stocktwits_bullish_ratio"] == 0.5).all()
    # target_event_time alignment preserved
    assert set(df["target_event_time"]) == set(events["target_event_time"])


def test_adapter_raises_when_event_source_missing(tmp_path, monkeypatch):
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    market = _make_market(tmp_path)
    adapter = SocialSentimentAdapter()
    with pytest.raises(RuntimeError, match="event_source"):
        adapter.fetch(market=market, source_params={"event_source": "stock_earnings"})
