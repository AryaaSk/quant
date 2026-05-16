"""FRED (Federal Reserve Economic Data) adapter.

Fetches macroeconomic time series from https://fred.stlouisfed.org/docs/api/
Requires a free API key (env FRED_API_KEY). Emits one row per (series, date) with
the value in a column named `num__<series_id>`. All series are merged onto a daily
calendar by the structure step.

Reliability: rate-limited at 120 req/min for the free tier. We cache responses to
`data/raw/<slug>/_fred_cache/` so repeated runs do not re-hit the API.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig, env


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FREDAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        api_key = env("FRED_API_KEY")
        series_ids = list(source_params.get("series", []))
        if not series_ids:
            raise ValueError("FRED adapter requires source_params['series'] = list of series IDs")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY is not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and add to .env."
            )

        cache_dir = market.raw_dir() / "_fred_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        scraped_at = pd.Timestamp.utcnow().tz_localize(None)
        rows: list[dict] = []

        for series_id in series_ids:
            obs = self._fetch_series(series_id, api_key, cache_dir)
            for o in obs:
                date_str = o.get("date")
                value_str = o.get("value")
                if not date_str or value_str in ("", ".", None):
                    continue
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                ts = pd.Timestamp(date_str)
                rows.append({
                    "timestamp": ts,
                    "source_published_at": ts,  # FRED releases on the observation date (best available)
                    "scraped_at": scraped_at,
                    "source_url": f"{FRED_BASE}?series_id={series_id}",
                    "source_type": f"fred.{series_id}",
                    f"num__{series_id}": value,
                })

        if not rows:
            raise RuntimeError(f"FRED returned 0 observations for {series_ids}")

        df = pd.DataFrame(rows)
        # Merge by (timestamp, series): one row per (date, series_id). Multiple series produce multiple rows
        # per date with different num__ columns; structure.py pivots them onto one daily row.
        return df

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _fetch_series(self, series_id: str, api_key: str, cache_dir: Path) -> list[dict]:
        cache_path = cache_dir / f"{series_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())["observations"]
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(FRED_BASE, params=params)
            resp.raise_for_status()
            payload = resp.json()
        cache_path.write_text(json.dumps(payload))
        time.sleep(0.6)  # gentle pacing well under 120/min
        return payload["observations"]
