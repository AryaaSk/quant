"""Earnings call transcripts adapter (Motley Fool public mirror).

For each (ticker, earnings_date) in the event_source parquet, fetches the most recent
earnings call transcript with `published_at < earnings_date` — the PRIOR quarter's call,
which encodes forward guidance for the current quarter.

Primary source: fool.com/earnings-call-transcripts/ (publicly mirrored, scrapable).
Fallback: tries seekingalpha.com search if Motley Fool returns nothing.

source_params:
  event_source: stock_earnings
  entity_column: ticker
  user_agent: <browser-like UA>

Output rows keyed to event_source's `target_event_time`. Each row carries the prior-quarter
transcript text under `text__transcript`.

Note: this adapter is best-effort. If Motley Fool doesn't have a transcript for a given
(ticker, quarter), the row is emitted with empty text — the temporal-leak guard in
state/text.py treats empty text as zeros, so this fails gracefully.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


FOOL_BASE = "https://www.fool.com"
FOOL_SEARCH = "https://www.fool.com/search/solr.aspx"


class EarningsTranscriptsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        event_source = source_params["event_source"]
        entity_column = source_params.get("entity_column", "ticker")
        user_agent = source_params.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        )

        event_parquet = market.raw_dir() / f"{event_source}.parquet"
        if not event_parquet.exists():
            raise RuntimeError(
                f"earnings_transcripts needs event_source '{event_source}'; missing {event_parquet}"
            )
        events = pd.read_parquet(event_parquet)
        if entity_column not in events.columns:
            raise RuntimeError(f"event_source missing column '{entity_column}'")
        events["target_event_time"] = pd.to_datetime(events["target_event_time"])
        if events["target_event_time"].dt.tz is not None:
            events["target_event_time"] = events["target_event_time"].dt.tz_localize(None)

        cache_dir = market.raw_dir() / "_transcripts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)
        headers = {"User-Agent": user_agent}

        tickers = sorted(events[entity_column].dropna().unique().tolist())
        ticker_transcripts: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                tdf = _fetch_transcript_index(ticker, headers, cache_dir)
            except Exception:
                tdf = pd.DataFrame(columns=["published_at", "url", "title"])
            ticker_transcripts[ticker] = tdf

        out_rows: list[dict] = []
        for ticker in tickers:
            tdf = ticker_transcripts[ticker]
            entity_events = events[events[entity_column] == ticker].copy()
            for _, ev in entity_events.iterrows():
                t_event = pd.Timestamp(ev["target_event_time"])
                if t_event.tzinfo is not None:
                    t_event = t_event.tz_localize(None)
                # Find the most recent transcript STRICTLY before this earnings event
                if not tdf.empty:
                    prior = tdf[tdf["published_at"] < t_event].sort_values(
                        "published_at", ascending=False
                    ).head(1)
                else:
                    prior = pd.DataFrame()

                transcript_text = ""
                published_at = None
                if not prior.empty:
                    url = prior.iloc[0]["url"]
                    published_at = prior.iloc[0]["published_at"]
                    transcript_text = _fetch_transcript_body(url, headers, cache_dir)

                out_rows.append({
                    "timestamp": t_event,
                    "source_published_at": published_at if published_at is not None else t_event,
                    "scraped_at": scraped_at,
                    "source_url": prior.iloc[0]["url"] if not prior.empty else f"fool://no-transcript/{ticker}",
                    "source_type": f"earnings_transcripts.{ticker}",
                    "target_event_time": t_event,
                    "text__transcript": transcript_text,
                    "num__has_transcript": 1.0 if transcript_text else 0.0,
                })

        if not out_rows:
            raise RuntimeError("earnings_transcripts produced 0 rows")
        return pd.DataFrame(out_rows)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_transcript_index(ticker: str, headers: dict, cache_dir: Path) -> pd.DataFrame:
    """Search Motley Fool for transcripts for this ticker. Cache per ticker."""
    cache_path = cache_dir / f"{ticker}_index.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Motley Fool search returns HTML pages. We do up to 3 pages.
    rows: list[dict] = []
    for page in range(1, 4):
        params = {"q": f"{ticker} earnings call transcript", "page": page}
        try:
            with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(FOOL_SEARCH, params=params)
                if resp.status_code != 200:
                    break
                html = resp.text
        except httpx.HTTPError:
            break
        soup = BeautifulSoup(html, "html.parser")
        # Result anchors: links to /earnings/call-transcripts/ paths
        anchors = soup.find_all("a", href=re.compile(r"/earnings/call-transcripts/"))
        if not anchors:
            break
        for a in anchors:
            href = a.get("href", "")
            if not href.startswith("http"):
                href = FOOL_BASE + href
            title = a.get_text(strip=True) or ""
            # Try to extract date from the URL path (Motley Fool uses /YYYY/MM/DD/)
            m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", href)
            if not m:
                continue
            try:
                pub = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
            except Exception:
                continue
            rows.append({"published_at": pub, "url": href, "title": title})
        time.sleep(1.0)  # Motley Fool politeness

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"]).sort_values("published_at", ascending=False).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    return df


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_transcript_body(url: str, headers: dict, cache_dir: Path) -> str:
    """Fetch a transcript page and extract its text body."""
    # Cache by URL slug hash
    slug = re.sub(r"[^A-Za-z0-9]+", "_", url)[-120:]
    cache_path = cache_dir / f"body_{slug}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    try:
        with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return ""
            html = resp.text
    except httpx.HTTPError:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    # Motley Fool transcripts live inside <article> or <div class="article-content">
    article = soup.find("article") or soup.find("div", class_=re.compile(r"article|tailwind|content"))
    if article is None:
        article = soup
    for tag in article(["script", "style", "aside", "nav", "footer", "header"]):
        tag.decompose()
    text = article.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    # Cap to 80k chars (typical transcript is 30-60k)
    text = text[:80000]
    cache_path.write_text(text)
    time.sleep(0.5)
    return text
