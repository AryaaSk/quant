"""Tests for news_scraper adapter and the text__<topic> structure-frame path.

All mocked. No real subprocess calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quant.adapters.news_scraper import NewsScraperAdapter, _articles_to_rows


def _write_article(dir_path: Path, idx: int, *, topic: str, text: str, published: str | None = "2026-01-15T12:00:00Z"):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"article_{idx:04d}.json").write_text(json.dumps({
        "url": f"https://example.com/{topic}/{idx}",
        "title": f"{topic} article {idx}",
        "published_at": published,
        "scraped_at": "2026-05-15T10:00:00Z",
        "text": text,
        "topic": topic,
        "source_type": f"news_scraper.{topic}",
    }))


def test_articles_to_rows_handles_two_topics(tmp_path):
    news_root = tmp_path / "_news"
    _write_article(news_root / "oil_transport", 1, topic="oil_transport", text="Tanker rates rose 8% this week")
    _write_article(news_root / "oil_transport", 2, topic="oil_transport", text="Suez Canal traffic recovers")
    _write_article(news_root / "saudi_arabia", 1, topic="saudi_arabia", text="Saudi Arabia signals output cuts")

    # Mock MarketConfig: only raw_dir() and slug used
    class FakeMarket:
        slug = "test"
        def raw_dir(self): return tmp_path

    df = _articles_to_rows(news_root, FakeMarket())
    assert len(df) == 3
    assert "text__oil_transport" in df.columns
    assert "text__saudi_arabia" in df.columns
    # Each row only fills one topic column; others are NaN
    oil_rows = df[df["source_type"] == "news_scraper.oil_transport"]
    saudi_rows = df[df["source_type"] == "news_scraper.saudi_arabia"]
    assert len(oil_rows) == 2 and len(saudi_rows) == 1
    assert oil_rows["text__oil_transport"].notna().all()
    assert oil_rows["text__saudi_arabia"].isna().all()


def test_articles_to_rows_drops_empty_text(tmp_path):
    news_root = tmp_path / "_news"
    _write_article(news_root / "x", 1, topic="x", text="   ")  # whitespace only
    _write_article(news_root / "x", 2, topic="x", text="real content")

    class FakeMarket:
        slug = "t"
        def raw_dir(self): return tmp_path

    df = _articles_to_rows(news_root, FakeMarket())
    assert len(df) == 1
    assert df.iloc[0]["text__x"] == "real content"


def test_articles_to_rows_uses_scraped_at_when_published_null(tmp_path):
    news_root = tmp_path / "_news"
    _write_article(news_root / "x", 1, topic="x", text="hi", published=None)

    class FakeMarket:
        slug = "t"
        def raw_dir(self): return tmp_path

    df = _articles_to_rows(news_root, FakeMarket())
    assert len(df) == 1
    # Falls back to scraped_at when published_at is null
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2026-05-15T10:00:00")


def test_adapter_without_agents_falls_back_to_cached(tmp_path, monkeypatch):
    """If QUANT_ENABLE_AGENTS is not set, the adapter reads existing cache instead of spawning."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("QUANT_ENABLE_AGENTS", raising=False)

    from quant.config import MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig
    market = MarketConfig(
        slug="testfb", display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[DataSourceConfig(name="news_scraper", kind="text", params={"topics": [{"name": "x", "queries": []}]})],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )
    _write_article(market.raw_dir() / "_news" / "x", 1, topic="x", text="hello")

    adapter = NewsScraperAdapter()
    df = adapter.fetch(market=market, source_params={"topics": [{"name": "x", "queries": []}]})
    assert len(df) == 1
    assert df.iloc[0]["text__x"] == "hello"


def test_adapter_without_agents_and_no_cache_raises(tmp_path, monkeypatch):
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("QUANT_ENABLE_AGENTS", raising=False)

    from quant.config import MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig
    market = MarketConfig(
        slug="testfbno", display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[DataSourceConfig(name="news_scraper", kind="text", params={"topics": [{"name": "x", "queries": []}]})],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )
    adapter = NewsScraperAdapter()
    with pytest.raises(RuntimeError, match="QUANT_ENABLE_AGENTS"):
        adapter.fetch(market=market, source_params={"topics": [{"name": "x", "queries": []}]})


def test_structure_forwards_text_topic_columns(tmp_path, monkeypatch):
    """structure.py forwards text__<topic> columns through pivoting (5-line patch)."""
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")

    from quant.config import MarketConfig, LabelConfig, StateVectorConfig, StateBlockConfig, DataSourceConfig
    market = MarketConfig(
        slug="ts-text-forward", display_name="t", platform="test", time_grain="daily",
        held_out_months=1, sequence_length=4,
        data_sources=[DataSourceConfig(name="news_scraper", kind="text", params={"topics": [
            {"name": "oil", "queries": []}, {"name": "saudi", "queries": []}
        ]})],
        label=LabelConfig(kind="binary", target_event="t"),
        state_vector=StateVectorConfig(dim=16, blocks=[
            StateBlockConfig(name="cal", slots="0:16", builder="calendar")
        ]),
    )

    # Hand-build a raw parquet with two topic columns + required timestamps.
    market.raw_dir().mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01"), "source_published_at": pd.Timestamp("2026-01-01"),
         "scraped_at": pd.Timestamp("2026-01-02"), "source_url": "u1", "source_type": "news_scraper.oil",
         "text__oil": "oil text day 1"},
        {"timestamp": pd.Timestamp("2026-01-01"), "source_published_at": pd.Timestamp("2026-01-01"),
         "scraped_at": pd.Timestamp("2026-01-02"), "source_url": "u2", "source_type": "news_scraper.saudi",
         "text__saudi": "saudi text day 1"},
        {"timestamp": pd.Timestamp("2026-01-02"), "source_published_at": pd.Timestamp("2026-01-02"),
         "scraped_at": pd.Timestamp("2026-01-03"), "source_url": "u3", "source_type": "news_scraper.oil",
         "text__oil": "oil text day 2"},
    ])
    raw_df.to_parquet(market.raw_dir() / "news_scraper.parquet", index=False)

    from quant.pipeline.structure import structure
    structured_path = structure(market)
    wide = pd.read_parquet(structured_path)

    assert "text__oil" in wide.columns
    assert "text__saudi" in wide.columns
    # 2 distinct days
    assert len(wide) == 2
    assert "oil text day 1" in wide.loc[pd.Timestamp("2026-01-01"), "text__oil"]
    assert "saudi text day 1" in wide.loc[pd.Timestamp("2026-01-01"), "text__saudi"]
    assert "oil text day 2" in wide.loc[pd.Timestamp("2026-01-02"), "text__oil"]
