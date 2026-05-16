"""SEC EDGAR full-text filings adapter.

For each (ticker, earnings_date) in the event_source parquet, fetches:
  - The most recent 10-Q with filed_at < earnings_date (prior quarter's filing)
  - Any 8-K filings in the 90 days before earnings (often contain forward-looking commentary)

Output rows are keyed to the same `target_event_time` as the event_source so structure.py's
pivot aligns them. Each row carries the concatenated filing text under `text__sec`.

source_params:
  event_source: stock_earnings        # required; parquet to read events from
  entity_column: ticker               # default 'ticker'
  filing_types: [10-Q, 8-K]           # default
  days_before_earnings_8k: 90         # how far back to look for 8-K filings
  user_agent: <REQUIRED by SEC>       # default "QuantPoC research@example.com"
  rate_limit_rps: 8                   # SEC rate limit is 10 req/s; default conservative

Cache: per-(cik, accession_no) raw HTML in data/raw/<slug>/_sec_cache/
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


SEC_BASE = "https://data.sec.gov"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index"


class SECFilingsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        event_source = source_params["event_source"]
        entity_column = source_params.get("entity_column", "ticker")
        filing_types = list(source_params.get("filing_types") or ["10-Q", "8-K"])
        days_before_8k = int(source_params.get("days_before_earnings_8k", 90))
        user_agent = source_params.get("user_agent", "QuantPoC research@example.com")
        rate_limit_rps = float(source_params.get("rate_limit_rps", 8.0))

        event_parquet = market.raw_dir() / f"{event_source}.parquet"
        if not event_parquet.exists():
            raise RuntimeError(
                f"sec_filings needs event_source '{event_source}' to run first; missing {event_parquet}"
            )
        events = pd.read_parquet(event_parquet)
        if entity_column not in events.columns:
            raise RuntimeError(f"event_source missing column '{entity_column}'")
        events["target_event_time"] = pd.to_datetime(events["target_event_time"])
        if events["target_event_time"].dt.tz is not None:
            events["target_event_time"] = events["target_event_time"].dt.tz_localize(None)

        cache_dir = market.raw_dir() / "_sec_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

        # Resolve CIKs once per unique ticker
        tickers = sorted(events[entity_column].dropna().unique().tolist())
        cik_map = _load_or_fetch_ticker_cik_map(cache_dir, headers, rate_limit_rps)
        ticker_to_cik = {t: cik_map.get(t.upper()) for t in tickers}
        # Drop tickers with no CIK (delisted, foreign, etc.)
        ticker_to_cik = {t: c for t, c in ticker_to_cik.items() if c}

        out_rows: list[dict] = []
        for ticker, cik in ticker_to_cik.items():
            filings = _fetch_filings_for_cik(cik, headers, cache_dir, rate_limit_rps)
            if filings is None or filings.empty:
                continue
            entity_events = events[events[entity_column] == ticker].copy()
            for _, ev in entity_events.iterrows():
                t_event = pd.Timestamp(ev["target_event_time"])
                if t_event.tzinfo is not None:
                    t_event = t_event.tz_localize(None)
                t_norm = t_event.normalize()

                texts: list[str] = []
                published_max: pd.Timestamp | None = None
                n_filings = 0

                # Most recent 10-Q strictly before earnings (with at least 1 day buffer to avoid same-day leak)
                if "10-Q" in filing_types:
                    qmask = (filings["form"] == "10-Q") & (filings["filed_at"] < t_norm)
                    q_candidates = filings[qmask].sort_values("filed_at", ascending=False).head(1)
                    for _, fr in q_candidates.iterrows():
                        body = _fetch_filing_text(cik, fr["accession_no"], fr["primary_doc"],
                                                  headers, cache_dir, rate_limit_rps)
                        if body:
                            texts.append(f"[10-Q filed {fr['filed_at'].date()}]\n{body[:60000]}")
                            n_filings += 1
                            if published_max is None or fr["filed_at"] > published_max:
                                published_max = fr["filed_at"]

                # 8-K filings in window [t_event - days_before_8k, t_event)
                if "8-K" in filing_types:
                    kmask = (
                        (filings["form"] == "8-K")
                        & (filings["filed_at"] < t_norm)
                        & (filings["filed_at"] >= t_norm - pd.Timedelta(days=days_before_8k))
                    )
                    k_candidates = filings[kmask].sort_values("filed_at", ascending=False).head(5)
                    for _, fr in k_candidates.iterrows():
                        body = _fetch_filing_text(cik, fr["accession_no"], fr["primary_doc"],
                                                  headers, cache_dir, rate_limit_rps)
                        if body:
                            texts.append(f"[8-K filed {fr['filed_at'].date()}]\n{body[:20000]}")
                            n_filings += 1
                            if published_max is None or fr["filed_at"] > published_max:
                                published_max = fr["filed_at"]

                if not texts and n_filings == 0:
                    # Still emit row so structure.py keyspace aligns; text empty
                    pass
                out_rows.append({
                    "timestamp": t_event,
                    "source_published_at": published_max if published_max is not None else t_event,
                    "scraped_at": scraped_at,
                    "source_url": f"sec_edgar://{ticker}/{t_event.isoformat()}",
                    "source_type": f"sec_filings.{ticker}",
                    "target_event_time": t_event,
                    "text__sec": "\n\n".join(texts) if texts else "",
                    "num__sec_filing_count": float(n_filings),
                })

        if not out_rows:
            raise RuntimeError("sec_filings produced 0 rows; check ticker→CIK mapping + EDGAR reachability")
        return pd.DataFrame(out_rows)


def _load_or_fetch_ticker_cik_map(cache_dir: Path, headers: dict, rate_limit_rps: float) -> dict:
    cache_path = cache_dir / "_ticker_cik_map.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    # SEC publishes a JSON file of all tickers→CIK at this stable URL
    url = "https://www.sec.gov/files/company_tickers.json"
    with httpx.Client(timeout=30.0, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    # Format: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    mapping = {}
    for v in data.values():
        ticker = str(v.get("ticker", "")).upper()
        cik = str(v.get("cik_str", "")).zfill(10)
        if ticker and cik != "0000000000":
            mapping[ticker] = cik
    cache_path.write_text(json.dumps(mapping))
    time.sleep(1.0 / rate_limit_rps)
    return mapping


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_filings_for_cik(cik: str, headers: dict, cache_dir: Path,
                           rate_limit_rps: float) -> pd.DataFrame | None:
    """Fetch the filing index for a CIK (all 10-Q + 8-K filings)."""
    cache_path = cache_dir / f"{cik}_filings.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return None
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return None
    df = pd.DataFrame({
        "form": recent.get("form", []),
        "filed_at": pd.to_datetime(recent.get("filingDate", []), errors="coerce"),
        "accession_no": recent.get("accessionNumber", []),
        "primary_doc": recent.get("primaryDocument", []),
    })
    df = df.dropna(subset=["filed_at"])
    df = df[df["form"].isin(["10-Q", "10-K", "8-K"])]  # 10-K could be substituted at year-end
    df.to_parquet(cache_path, index=False)
    time.sleep(1.0 / rate_limit_rps)
    return df


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_filing_text(cik: str, accession_no: str, primary_doc: str,
                       headers: dict, cache_dir: Path, rate_limit_rps: float) -> str:
    """Fetch + extract text from a single filing's primary document."""
    accession_clean = accession_no.replace("-", "")
    cache_path = cache_dir / f"{cik}_{accession_clean}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
    try:
        with httpx.Client(timeout=60.0, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError:
        return ""
    text = _html_to_clean_text(html)
    cache_path.write_text(text)
    time.sleep(1.0 / rate_limit_rps)
    return text


def _html_to_clean_text(html: str) -> str:
    """Strip HTML/XBRL noise and return plain text. Cap length to avoid bloat."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):  # tables are XBRL-heavy + noisy
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Strip page footers / repeated boilerplate
    text = re.sub(r"Page \d+ of \d+", "", text)
    return text[:200000]  # 200k char cap per filing
