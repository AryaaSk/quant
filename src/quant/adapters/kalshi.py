"""Kalshi public-API adapter (no authentication required).

Source: https://api.elections.kalshi.com/trade-api/v2/

Endpoints used (all public, no key):
- /series?category=Economics                                 list series in a category
- /events?series_ticker={s}&status=settled                  list settled events under a series
- /markets/{ticker}                                          single market detail
- /series/{s}/markets/{m}/candlesticks?start_ts=&end_ts=    price history per market

For each market under a settled event, we emit:
  timestamp = settle_time (date)
  target_event_time = settle_time
  y_realized = 1 if result == 'yes' else 0
  p_market = mid of last bid/ask candlestick BEFORE settle time
  num__yes_price_at_open, num__volume_total, num__open_interest_max as features
  num__threshold (the bucket strike, e.g. 2.3 for CPI YoY > 2.3%)

Picking "the" market per event:
- We collect ALL markets per event (typically 5-15 threshold buckets per event)
- Each becomes an independent prediction (event x bucket)
- The model learns to predict any threshold; the backtest will choose the highest-edge bucket

Caveats:
- KXCPIYOY only goes back ~2 years on the public API (Kalshi economics expanded in 2024).
  Older history would need PredictionData.dev (paid) or kalshi.com archive scraping.
- Without auth we cannot pull live orderbook depth; slippage in the yaml is a flat estimate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiContractsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        series_tickers = list(source_params.get("series", []))
        if not series_tickers:
            raise ValueError("kalshi_contracts requires source_params['series'] = list of Kalshi series tickers")
        max_events_per_series = int(source_params.get("max_events_per_series", 200))
        candle_interval = int(source_params.get("candle_interval", 1440))  # 1 day in minutes

        cache_dir = market.raw_dir() / "_kalshi_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        rows: list[dict] = []
        for series_ticker in series_tickers:
            events = self._fetch_events(series_ticker, max_events_per_series, cache_dir)
            for ev in events:
                event_ticker = ev.get("event_ticker")
                markets = ev.get("markets", []) or []
                for m in markets:
                    # Kalshi uses "finalized" or "settled" for resolved markets.
                    if m.get("status") not in ("settled", "finalized"):
                        continue
                    if not m.get("result"):
                        continue
                    mt = m.get("ticker")
                    settle_time = m.get("expected_expiration_time") or m.get("close_time") or m.get("settle_time")
                    if not settle_time:
                        continue
                    settle_ts = pd.Timestamp(settle_time)
                    candles = self._fetch_candles(series_ticker, mt, settle_ts, candle_interval, cache_dir)
                    p_market = _last_mid_price(candles, settle_ts)
                    if p_market is None:
                        continue
                    rows.append(_market_to_row(
                        series_ticker, event_ticker, m, candles, settle_ts, p_market, scraped_at,
                    ))

        if not rows:
            raise RuntimeError(f"kalshi_contracts returned no usable settled markets for {series_tickers}")
        return pd.DataFrame(rows)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _fetch_events(self, series_ticker: str, max_events: int, cache_dir: Path) -> list[dict]:
        cache_path = cache_dir / f"events_{series_ticker}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        all_events: list[dict] = []
        cursor: str | None = None
        while len(all_events) < max_events:
            params = {
                "series_ticker": series_ticker,
                "with_nested_markets": "true",
                "status": "settled",
                "limit": 50,
            }
            if cursor:
                params["cursor"] = cursor
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(f"{KALSHI_BASE}/events", params=params)
                resp.raise_for_status()
                data = resp.json()
            events = data.get("events", [])
            if not events:
                break
            all_events.extend(events)
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(0.4)
        cache_path.write_text(json.dumps(all_events))
        return all_events

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _fetch_candles(self, series_ticker: str, market_ticker: str, settle_ts: pd.Timestamp,
                       candle_interval: int, cache_dir: Path) -> list[dict]:
        cache_path = cache_dir / f"candles_{market_ticker}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)
        end_ts = int(settle_ts.timestamp())
        start_ts = end_ts - 365 * 24 * 3600  # up to one year before settle
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": candle_interval}
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{KALSHI_BASE}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
                    params=params,
                )
            if resp.status_code != 200:
                return []
            candles = resp.json().get("candlesticks", [])
        except httpx.HTTPError:
            candles = []
        cache_path.write_text(json.dumps(candles))
        time.sleep(0.2)
        return candles


def _last_mid_price(candles: list[dict], before_ts: pd.Timestamp) -> float | None:
    if not candles:
        return None
    before_unix = int(before_ts.timestamp())
    # Pick the last candle whose end_period_ts is strictly before settlement.
    valid = [c for c in candles if int(c.get("end_period_ts", 0)) < before_unix]
    if not valid:
        valid = candles[:-1] if len(candles) > 1 else candles
    if not valid:
        return None
    last = valid[-1]
    bid_close = float(last.get("yes_bid", {}).get("close_dollars", "0") or 0)
    ask_close = float(last.get("yes_ask", {}).get("close_dollars", "1") or 1)
    if bid_close == 0 and ask_close == 1:
        prev = last.get("price", {}).get("previous_dollars")
        if prev:
            return max(0.01, min(0.99, float(prev)))
        return None
    mid = (bid_close + ask_close) / 2
    return max(0.01, min(0.99, mid))


def _market_to_row(series_ticker: str, event_ticker: str, m: dict, candles: list[dict],
                   settle_ts: pd.Timestamp, p_market: float, scraped_at: pd.Timestamp) -> dict:
    result = m.get("result", "")
    y = 1.0 if result == "yes" else 0.0

    # Threshold parse: market tickers look like KXCPIYOY-26APR-T2.3 (T = threshold).
    market_ticker = m.get("ticker", "")
    threshold = 0.0
    try:
        parts = market_ticker.split("-T")
        if len(parts) == 2:
            threshold = float(parts[1])
    except (ValueError, IndexError):
        threshold = 0.0

    open_yes = 0.5
    total_volume = 0.0
    max_oi = 0.0
    if candles:
        first = candles[0]
        open_yes = float(first.get("yes_ask", {}).get("open_dollars", 0.5) or 0.5)
        total_volume = sum(float(c.get("volume_fp", 0) or 0) for c in candles)
        max_oi = max((float(c.get("open_interest_fp", 0) or 0) for c in candles), default=0.0)

    # Each (event, threshold) market is an independent prediction. Make target_event_time
    # unique per market by adding a deterministic hash-offset in seconds, otherwise the
    # structure step collapses all markets sharing a settle date into one row.
    offset_seconds = (abs(hash(market_ticker)) % 86400) - 43200  # +/- 12 hours
    unique_target = settle_ts.normalize() + pd.Timedelta(seconds=offset_seconds)
    unique_ts = settle_ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(seconds=offset_seconds)

    return {
        "timestamp": unique_ts,
        "source_published_at": settle_ts,
        "scraped_at": scraped_at,
        "source_url": f"{KALSHI_BASE}/markets/{market_ticker}",
        "source_type": f"kalshi.{series_ticker}",
        "target_event_time": unique_target,
        "y_realized": y,
        "p_market": p_market,
        "decimal_odds": 1.0 / p_market if p_market > 0 else 100.0,
        "num__threshold": threshold,
        "num__open_yes_price": open_yes,
        "num__total_volume": total_volume,
        "num__max_open_interest": max_oi,
        "num__days_open": float((settle_ts.normalize() - pd.Timestamp(m.get("open_time", settle_ts)).normalize()).days),
    }
