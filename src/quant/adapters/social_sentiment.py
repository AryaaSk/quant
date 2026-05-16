"""Reddit + StockTwits social sentiment adapter (numeric features only).

For each (ticker, earnings_date) in the event_source parquet, computes a small block
of numeric features summarising social-media chatter in the 30 days before earnings:

  num__reddit_mention_count_30d        Reddit posts mentioning the ticker
  num__reddit_comment_count_30d        Reddit comments in those threads
  num__reddit_avg_score_30d            Average upvotes of posts that mentioned it
  num__stocktwits_msg_count_7d         StockTwits messages in last 7 days before earnings
  num__stocktwits_bullish_ratio        Fraction of bullish-tagged StockTwits msgs
  num__social_concentration            Std-dev of daily mention count (volatility of buzz)

This is pure NUMERIC, not text. The text-block contribution lives in the news_scraper.
Why split: Reddit/StockTwits text is short, noisy, and meme-laden — counts and bullish-bearish
ratios encode the signal more cleanly than mean-pooled Voyage embeddings of "🚀🚀🚀".

source_params:
  event_source: stock_earnings
  entity_column: ticker
  reddit_subreddits: [wallstreetbets, stocks, investing, smallstreetbets]
  window_days_reddit: 30
  window_days_stocktwits: 7

Auth:
  Reddit needs `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` env vars. If missing, the
  Reddit half degrades to zeros (still emits rows so the state vector keyspace works).
  StockTwits public API needs no auth.

Note: PRAW (Reddit's Python wrapper) is optional. If not installed or auth missing,
this adapter still runs and emits zero features for Reddit.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from quant.config import MarketConfig


class SocialSentimentAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        event_source = source_params["event_source"]
        entity_column = source_params.get("entity_column", "ticker")
        subreddits = list(source_params.get("reddit_subreddits") or [
            "wallstreetbets", "stocks", "investing", "smallstreetbets"
        ])
        window_days_reddit = int(source_params.get("window_days_reddit", 30))
        window_days_st = int(source_params.get("window_days_stocktwits", 7))

        event_parquet = market.raw_dir() / f"{event_source}.parquet"
        if not event_parquet.exists():
            raise RuntimeError(
                f"social_sentiment needs event_source '{event_source}'; missing {event_parquet}"
            )
        events = pd.read_parquet(event_parquet)
        if entity_column not in events.columns:
            raise RuntimeError(f"event_source missing column '{entity_column}'")
        events["target_event_time"] = pd.to_datetime(events["target_event_time"])
        if events["target_event_time"].dt.tz is not None:
            events["target_event_time"] = events["target_event_time"].dt.tz_localize(None)

        cache_dir = market.raw_dir() / "_social_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        tickers = sorted(events[entity_column].dropna().unique().tolist())
        # Build per-ticker Reddit aggregates (one fetch per ticker covers the whole history)
        reddit_idx: dict[str, pd.DataFrame] = {}
        if _reddit_auth_present():
            reddit = _init_reddit()
            for ticker in tickers:
                try:
                    reddit_idx[ticker] = _fetch_reddit_mentions(reddit, ticker, subreddits, cache_dir)
                except Exception:
                    reddit_idx[ticker] = pd.DataFrame(columns=["created_at", "score", "n_comments"])

        # Build per-ticker StockTwits aggregates (free public API)
        stocktwits_idx: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                stocktwits_idx[ticker] = _fetch_stocktwits_messages(ticker, cache_dir)
            except Exception:
                stocktwits_idx[ticker] = pd.DataFrame(columns=["created_at", "sentiment"])

        out_rows: list[dict] = []
        for ticker in tickers:
            r = reddit_idx.get(ticker, pd.DataFrame())
            s = stocktwits_idx.get(ticker, pd.DataFrame())
            entity_events = events[events[entity_column] == ticker].copy()
            for _, ev in entity_events.iterrows():
                t_event = pd.Timestamp(ev["target_event_time"])
                if t_event.tzinfo is not None:
                    t_event = t_event.tz_localize(None)
                r_features = _reddit_features(r, t_event, window_days_reddit)
                s_features = _stocktwits_features(s, t_event, window_days_st)
                out_rows.append({
                    "timestamp": t_event,
                    "source_published_at": t_event,
                    "scraped_at": scraped_at,
                    "source_url": f"social_sentiment://{ticker}/{t_event.isoformat()}",
                    "source_type": f"social_sentiment.{ticker}",
                    "target_event_time": t_event,
                    **r_features,
                    **s_features,
                })

        if not out_rows:
            raise RuntimeError("social_sentiment produced 0 rows")
        return pd.DataFrame(out_rows)


def _reddit_auth_present() -> bool:
    return bool(os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"))


def _init_reddit():
    """Initialize PRAW; raises if praw isn't installed."""
    try:
        import praw
    except ImportError as e:
        raise ImportError("PRAW required for Reddit auth: pip install praw") from e
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="QuantPoC/1.0 (research)",
        check_for_async=False,
    )


def _fetch_reddit_mentions(reddit, ticker: str, subreddits: list[str],
                           cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"reddit_{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = []
    for sub in subreddits:
        try:
            for submission in reddit.subreddit(sub).search(
                ticker, sort="new", time_filter="all", limit=500
            ):
                rows.append({
                    "created_at": pd.to_datetime(submission.created_utc, unit="s"),
                    "score": int(submission.score or 0),
                    "n_comments": int(submission.num_comments or 0),
                })
        except Exception:
            continue
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    return df


def _fetch_stocktwits_messages(ticker: str, cache_dir: Path) -> pd.DataFrame:
    """StockTwits free symbol-stream API: ~30 most recent messages per call.

    The free tier only exposes recent messages, so for historical events the data is sparse.
    For events older than ~30 days, the StockTwits half will degrade to zeros. That's OK —
    Reddit covers the historical window.
    """
    cache_path = cache_dir / f"stocktwits_{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                df = pd.DataFrame(columns=["created_at", "sentiment"])
                df.to_parquet(cache_path, index=False)
                return df
            data = resp.json()
    except httpx.HTTPError:
        df = pd.DataFrame(columns=["created_at", "sentiment"])
        df.to_parquet(cache_path, index=False)
        return df
    rows = []
    for msg in data.get("messages", []):
        ts = msg.get("created_at")
        sentiment = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic") or ""
        rows.append({
            "created_at": pd.to_datetime(ts, errors="coerce"),
            "sentiment": str(sentiment).lower(),
        })
    df = pd.DataFrame(rows)
    if not df.empty and df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    df.to_parquet(cache_path, index=False)
    time.sleep(0.3)
    return df


def _reddit_features(df: pd.DataFrame, t_event: pd.Timestamp, window_days: int) -> dict:
    if df.empty:
        return {
            "num__reddit_mention_count_30d": 0.0,
            "num__reddit_comment_count_30d": 0.0,
            "num__reddit_avg_score_30d": 0.0,
            "num__social_concentration": 0.0,
        }
    df = df.copy()
    if df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    window_start = t_event - pd.Timedelta(days=window_days)
    mask = (df["created_at"] >= window_start) & (df["created_at"] < t_event)
    win = df[mask]
    if win.empty:
        return {
            "num__reddit_mention_count_30d": 0.0,
            "num__reddit_comment_count_30d": 0.0,
            "num__reddit_avg_score_30d": 0.0,
            "num__social_concentration": 0.0,
        }
    daily = win.groupby(win["created_at"].dt.date).size()
    return {
        "num__reddit_mention_count_30d": float(len(win)),
        "num__reddit_comment_count_30d": float(win["n_comments"].sum()),
        "num__reddit_avg_score_30d": float(win["score"].mean()),
        "num__social_concentration": float(daily.std()) if len(daily) > 1 else 0.0,
    }


def _stocktwits_features(df: pd.DataFrame, t_event: pd.Timestamp, window_days: int) -> dict:
    if df.empty:
        return {
            "num__stocktwits_msg_count_7d": 0.0,
            "num__stocktwits_bullish_ratio": 0.5,
        }
    df = df.copy()
    if df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    window_start = t_event - pd.Timedelta(days=window_days)
    mask = (df["created_at"] >= window_start) & (df["created_at"] < t_event)
    win = df[mask]
    if win.empty:
        return {
            "num__stocktwits_msg_count_7d": 0.0,
            "num__stocktwits_bullish_ratio": 0.5,
        }
    n_bull = (win["sentiment"] == "bullish").sum()
    n_bear = (win["sentiment"] == "bearish").sum()
    total_tagged = n_bull + n_bear
    ratio = float(n_bull / total_tagged) if total_tagged > 0 else 0.5
    return {
        "num__stocktwits_msg_count_7d": float(len(win)),
        "num__stocktwits_bullish_ratio": ratio,
    }
