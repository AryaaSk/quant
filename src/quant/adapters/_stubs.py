"""Stub adapters: implement loudly-failing skeletons for all real data sources.

Each stub raises a clear `AdapterNotImplementedError` explaining (1) what the adapter is
supposed to fetch, (2) where to find the historical archive, and (3) which env var or
credential gates it. Implementing one is straightforward: subclass, override `fetch`.

The point of having these stubs registered is so the dispatcher in `adapters/__init__.py`
gives a useful error instead of "unknown adapter," and so the screening pass can iterate
every market yaml without ImportError.
"""
from __future__ import annotations

import pandas as pd

from quant.config import MarketConfig


class AdapterNotImplementedError(NotImplementedError):
    pass


def _raise(name: str, hint: str) -> None:
    raise AdapterNotImplementedError(
        f"adapter '{name}' is not implemented yet.\n  Source role: {hint}\n  "
        f"Implement quant/adapters/{name.lower()}.py with class signature `fetch(market, source_params) -> DataFrame`."
    )


class FREDAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "FRED",
            "macroeconomic time series (CPI, payrolls, ...) via https://fred.stlouisfed.org/docs/api/ "
            "(free key; env FRED_API_KEY). source_params['series'] = list of FRED series IDs.",
        )


class KalshiContractsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Kalshi",
            "Kalshi event contract closing prices via https://trading-api.readme.io/reference/getmarkets "
            "(env KALSHI_EMAIL/PASSWORD) or third-party predictiondata.dev archive.",
        )


class PolymarketContractsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Polymarket",
            "Polymarket on-chain market history via the Polymarket CLOB API + Subgraph "
            "(https://docs.polymarket.com).",
        )


class FootballDataAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Football-Data",
            "historical European football match results + closing odds from https://www.football-data.co.uk/ "
            "(free CSV per season). source_params['leagues'] = list of league codes.",
        )


class TennisDataAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Tennis-Data",
            "historical ATP/WTA/ITF match results + odds from http://www.tennis-data.co.uk/ "
            "or Jeff Sackmann's tennis-data repos (https://github.com/JeffSackmann).",
        )


class SofascoreAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Sofascore",
            "Sofascore match details (lineups, stats, injuries) via the public web API. "
            "Be polite: rate-limit + cache. Source: https://www.sofascore.com/api/v1/.",
        )


class SECEdgarAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "SEC EDGAR",
            "SEC company filings via https://www.sec.gov/edgar.shtml. source_params['ciks'] = list of CIKs.",
        )


class ClinicalTrialsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "ClinicalTrials.gov",
            "trial registrations + status via https://clinicaltrials.gov/api/. source_params['nct_ids'] optional.",
        )


class FedSpeakersAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Fed speakers",
            "Fed speaker transcripts (FOMC, Beige Book, regional Feds) via https://www.federalreserve.gov/.",
        )


class HyperliquidAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "Hyperliquid",
            "perpetual futures ticks + funding via https://api.hyperliquid.xyz/info "
            "(public; rate-limited). source_params['coins'] = list of symbols.",
        )


class NewsScraperAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        _raise(
            "News scraper",
            "generic article scraper used by agent orchestration (Claude/Codex). "
            "source_params['urls'] or ['queries']. Outputs (timestamp, text, source_url) rows.",
        )
