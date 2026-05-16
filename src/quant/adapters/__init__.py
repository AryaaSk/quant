"""Data source adapters. Each adapter has a `fetch(market, source_params) -> DataFrame`.

Adapters must emit rows with: timestamp, source_published_at, scraped_at, source_url,
source_type. Numeric series go under `num__<name>` columns; text goes under `text`;
target labels go under `target_event_time` + `y_realized` + optionally `p_market`.

Real-world adapters (Kalshi, FRED, Betfair, ...) ship as separate modules; the registry
below maps source names to their adapter implementation. The `synthetic` adapter is for
the E2E smoke test and CI.
"""
from __future__ import annotations

from importlib import import_module
from typing import Protocol

import pandas as pd

from quant.config import MarketConfig


class Adapter(Protocol):
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame: ...


_REGISTRY: dict[str, str] = {
    "synthetic": "quant.adapters.synthetic:SyntheticAdapter",
    "fred": "quant.adapters.fred:FREDAdapter",
    "football_data": "quant.adapters.football_data:FootballDataAdapter",
    "tennis_data": "quant.adapters.tennis_data:TennisDataAdapter",
    "kalshi_contracts": "quant.adapters.kalshi:KalshiContractsAdapter",
    "polymarket_contracts": "quant.adapters.polymarket_contracts:PolymarketContractsAdapter",
    "sec_filings": "quant.adapters.sec_filings:SECFilingsAdapter",
    "earnings_transcripts": "quant.adapters.earnings_transcripts:EarningsTranscriptsAdapter",
    "social_sentiment": "quant.adapters.social_sentiment:SocialSentimentAdapter",
    # Stubs below: implement when needed for that market.
    "sofascore": "quant.adapters._stubs:SofascoreAdapter",
    "sec_edgar": "quant.adapters._stubs:SECEdgarAdapter",
    "clinicaltrials": "quant.adapters._stubs:ClinicalTrialsAdapter",
    "fed_speakers": "quant.adapters._stubs:FedSpeakersAdapter",
    "hyperliquid": "quant.adapters.hyperliquid:HyperliquidAdapter",
    "news_scraper": "quant.adapters.news_scraper:NewsScraperAdapter",
    "agent_feature_extractor": "quant.adapters.agent_feature_extractor:AgentFeatureExtractorAdapter",
    "stock_earnings": "quant.adapters.stock_earnings:StockEarningsAdapter",
}


def get_adapter(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"no adapter registered for source '{name}'; known: {sorted(_REGISTRY)}")
    module_name, cls_name = _REGISTRY[name].split(":")
    mod = import_module(module_name)
    cls = getattr(mod, cls_name)
    return cls()
