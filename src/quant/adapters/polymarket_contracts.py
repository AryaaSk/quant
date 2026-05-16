"""Polymarket Gamma API adapter (public, no auth).

Fetches resolved (closed) prediction markets via https://gamma-api.polymarket.com/.
For each market: pulls closing price (final implied probability before resolution),
realized outcome, and useful metadata.

source_params:
  tag_slugs: [oscars, emmys, golden-globes, grammys, baftas, sag-awards, ...]
  closed: true   # only resolved markets
  limit_per_tag: 200  # cap per tag (Polymarket caps at 500)
  start_date: 2020-01-01  # filter markets by endDate
  end_date:   2026-04-01

Emits one row per market with:
  timestamp = endDate (resolution date)
  target_event_time = endDate (made unique by adding ms offset)
  y_realized = 1 if outcome=Yes resolved 1, else 0
  p_market = the Yes-outcome price near close (proxy: last known price or 'lastTradePrice')
  decimal_odds = 1/p_market
  num__market_volume, num__market_liquidity, num__days_open
  text__question = the market question (short, can be Voyage-embedded)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


GAMMA_BASE = "https://gamma-api.polymarket.com"


class PolymarketContractsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        tag_slugs = list(source_params.get("tag_slugs") or ["oscars"])
        limit_per_tag = int(source_params.get("limit_per_tag", 200))
        start_date = pd.Timestamp(source_params.get("start_date", "2020-01-01"))
        end_date = pd.Timestamp(source_params.get("end_date", "2026-04-01"))

        cache_dir = market.raw_dir() / "_polymarket_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        all_events: list[dict] = []
        for tag in tag_slugs:
            events = self._fetch_events_by_tag(tag, limit_per_tag, cache_dir)
            for ev in events:
                ev["_source_tag"] = tag
            all_events.extend(events)

        rows: list[dict] = []
        for ev in all_events:
            event_title = ev.get("title", "")
            event_slug = ev.get("slug", "")
            source_tag = ev.get("_source_tag", "")
            for m in (ev.get("markets") or []):
                try:
                    row = _market_to_row(m, event_title, event_slug, source_tag, scraped_at, start_date, end_date)
                except Exception:
                    continue
                if row is not None:
                    rows.append(row)

        if not rows:
            raise RuntimeError(f"polymarket_contracts produced 0 rows for tags={tag_slugs}")
        return pd.DataFrame(rows)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _fetch_events_by_tag(self, tag: str, limit: int, cache_dir: Path) -> list[dict]:
        cache_path = cache_dir / f"events_{tag}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        url = f"{GAMMA_BASE}/events"
        params = {
            "tag_slug": tag,
            "closed": "true",
            "limit": min(limit, 500),
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError:
            data = []
        if not isinstance(data, list):
            data = []
        cache_path.write_text(json.dumps(data))
        time.sleep(0.3)
        return data


def _market_to_row(
    m: dict, event_title: str, event_slug: str, source_tag: str,
    scraped_at: pd.Timestamp, start_date: pd.Timestamp, end_date: pd.Timestamp,
) -> dict | None:
    end_date_str = m.get("endDate") or m.get("closedTime")
    if not end_date_str:
        return None
    end_dt = pd.Timestamp(end_date_str)
    if end_dt.tzinfo is not None:
        end_dt = end_dt.tz_localize(None)
    if not (start_date <= end_dt <= end_date):
        return None

    outcomes = m.get("outcomes", "[]")
    outcome_prices = m.get("outcomePrices", "[]")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            return None
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            return None
    if not outcomes or not outcome_prices:
        return None

    # Find Yes index (case-insensitive)
    yes_idx = None
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() in ("yes", "y", "true"):
            yes_idx = i
            break
    if yes_idx is None:
        yes_idx = 0
    try:
        y_realized = float(outcome_prices[yes_idx])
    except (ValueError, TypeError, IndexError):
        return None
    if y_realized not in (0.0, 1.0):
        # Market not cleanly resolved; skip
        return None

    # p_market: use last known price (lastTradePrice or similar). Fall back to 0.5.
    last_price = m.get("lastTradePrice")
    if last_price is None:
        last_price = m.get("bestBid")
    try:
        p_market = float(last_price) if last_price is not None else 0.5
    except (ValueError, TypeError):
        p_market = 0.5
    p_market = float(np.clip(p_market, 0.02, 0.98))

    # Make target_event_time unique per market via ms-offset
    offset_ms = abs(hash(m.get("id") or m.get("conditionId") or m.get("question", ""))) % 86_400_000
    unique_event_time = end_dt.normalize() + pd.Timedelta(milliseconds=offset_ms)

    start_dt = pd.Timestamp(m.get("startDate") or end_dt - pd.Timedelta(days=30))
    if start_dt.tzinfo is not None:
        start_dt = start_dt.tz_localize(None)
    days_open = max(1, (end_dt - start_dt).days)

    return {
        "timestamp": unique_event_time,
        "source_published_at": end_dt,
        "scraped_at": scraped_at,
        "source_url": f"polymarket://{m.get('slug', m.get('id', 'unknown'))}",
        "source_type": f"polymarket.{source_tag}",
        "target_event_time": unique_event_time,
        "y_realized": y_realized,
        "p_market": p_market,
        "decimal_odds": 1.0 / p_market,
        "event_title": event_title,
        "event_slug": event_slug,
        "market_question": m.get("question", "")[:500],
        "market_slug": m.get("slug", ""),
        "tag": source_tag,
        "num__market_volume": float(m.get("volume", 0) or 0),
        "num__market_liquidity": float(m.get("liquidity", 0) or 0),
        "num__days_open": float(days_open),
        "num__yes_price_close": p_market,
        "text__question": m.get("question", "")[:500],
    }
