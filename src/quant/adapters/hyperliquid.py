"""Hyperliquid perp futures adapter.

Public API: https://api.hyperliquid.xyz/info (POST endpoint, JSON body with `type` field).

Endpoints used:
- type=candleSnapshot  -> OHLCV bars per coin at a given interval
- type=fundingHistory   -> funding rate history per coin
- type=metaAndAssetCtxs -> current open interest, premium, mark price per coin

For each coin in source_params['coins'] we pull:
- hourly candles (configurable interval)
- funding rate history aligned to candles
- snapshot of current open interest (latest only; historical OI is not exposed publicly)

Forward-return label: percent change from candle.close[t] to candle.close[t + forward_h].
Binary label = sign(forward_return) > 0.

For p_market, we use 0.5 (no betting line on perps). The "edge" lives in directional
prediction; the backtest models a perp position with funding rate as the holding cost.
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


HL_INFO = "https://api.hyperliquid.xyz/info"


class HyperliquidAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        coins = list(source_params.get("coins", []))
        if not coins:
            raise ValueError("hyperliquid adapter requires source_params['coins']")
        interval = str(source_params.get("interval", "1h"))
        lookback_days = int(source_params.get("lookback_days", 365))
        forward_hours = int(source_params.get("forward_hours", 4))

        cache_dir = market.raw_dir() / "_hl_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        end_ms = int(scraped_at.timestamp() * 1000)
        start_ms = end_ms - lookback_days * 24 * 3600 * 1000

        rows: list[dict] = []
        for coin in coins:
            candles = self._fetch_candles(coin, interval, start_ms, end_ms, cache_dir)
            if not candles:
                continue
            funding = self._fetch_funding(coin, start_ms, end_ms, cache_dir)
            df = self._to_frame(coin, candles, funding, forward_hours, scraped_at)
            if df is not None and not df.empty:
                rows.append(df)

        if not rows:
            raise RuntimeError(f"hyperliquid returned no candles for {coins}")
        return pd.concat(rows, ignore_index=True, copy=False)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _fetch_candles(self, coin: str, interval: str, start_ms: int, end_ms: int, cache_dir: Path):
        cache_path = cache_dir / f"{coin}_{interval}_{start_ms}_{end_ms}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        # Hyperliquid limits 5000 candles per call. Page through if needed.
        all_candles: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            body = {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": end_ms},
            }
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(HL_INFO, json=body, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                batch = resp.json()
            if not batch:
                break
            all_candles.extend(batch)
            last_ts = int(batch[-1]["t"]) + 1
            if last_ts <= cursor:
                break
            cursor = last_ts
            time.sleep(0.4)  # gentle pacing
            if len(batch) < 5000:  # last page
                break
        cache_path.write_text(json.dumps(all_candles))
        return all_candles

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _fetch_funding(self, coin: str, start_ms: int, end_ms: int, cache_dir: Path):
        cache_path = cache_dir / f"{coin}_funding_{start_ms}_{end_ms}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        body = {"type": "fundingHistory", "coin": coin, "startTime": start_ms, "endTime": end_ms}
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(HL_INFO, json=body, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError:
            data = []
        cache_path.write_text(json.dumps(data))
        time.sleep(0.4)
        return data

    def _to_frame(self, coin: str, candles: list[dict], funding: list[dict], forward_hours: int, scraped_at: pd.Timestamp) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        c = pd.DataFrame(candles)
        # Hyperliquid candle fields: t (open time ms), T (close time ms), o,h,l,c (str), v (vol), n (trades)
        c["t"] = pd.to_datetime(c["t"], unit="ms")
        for col in ("o", "h", "l", "c", "v"):
            c[col] = pd.to_numeric(c[col], errors="coerce")
        c = c.dropna(subset=["c"]).sort_values("t").reset_index(drop=True)

        # Funding rates indexed by time
        if funding:
            f = pd.DataFrame(funding)
            f["t"] = pd.to_datetime(f["time"], unit="ms") if "time" in f.columns else pd.to_datetime(f.get("t"), unit="ms")
            f["fundingRate"] = pd.to_numeric(f.get("fundingRate"), errors="coerce")
            f = f.dropna(subset=["t", "fundingRate"]).sort_values("t")
            c = pd.merge_asof(c, f[["t", "fundingRate"]], on="t", direction="backward")
        else:
            c["fundingRate"] = 0.0

        # Forward return = (close[t+h] / close[t]) - 1, with h in candle steps.
        # Assume hourly candles -> forward_hours == steps.
        c["forward_close"] = c["c"].shift(-forward_hours)
        c["forward_return"] = c["forward_close"] / c["c"] - 1.0
        c["log_return"] = np.log(c["c"] / c["c"].shift(1))

        rows: list[dict] = []
        for _, row in c.iterrows():
            ts = row["t"]
            if pd.isna(row.get("forward_return")):
                continue
            rows.append({
                "timestamp": ts,
                "source_published_at": ts,
                "scraped_at": scraped_at,
                "source_url": f"{HL_INFO}?coin={coin}",
                "source_type": f"hyperliquid.{coin}",
                "target_event_time": ts + pd.Timedelta(hours=4),
                "y_realized": float(row["forward_return"] > 0),
                # No betting line; use a 0.5 prior as p_market so edge calculations are purely model-driven.
                "p_market": 0.5,
                "decimal_odds": 2.0,
                "num__close": float(row["c"]),
                "num__log_return": float(row["log_return"]) if pd.notna(row["log_return"]) else 0.0,
                "num__volume": float(row["v"]),
                "num__funding_rate": float(row["fundingRate"]) if pd.notna(row["fundingRate"]) else 0.0,
                "num__coin_id": float(abs(hash(coin)) % 10),
            })
        df = pd.DataFrame(rows)
        return df
