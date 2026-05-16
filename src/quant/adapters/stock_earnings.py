"""Stock earnings adapter (yfinance-based).

For each ticker in source_params['tickers'], pulls historical earnings releases
with EPS estimate vs reported EPS (surprise), plus the surrounding price action.

For each earnings event, emits a row with:
- target_event_time = earnings release datetime
- y_realized = 1 if reported EPS > estimate (beat), else 0
- p_market = baseline beat rate prior (approx 0.6 for sandbagging effect; per-ticker tunable)
- decimal_odds = 1/p_market for compat
- num__<feature>: prior-N-quarter surprises, revenue trends, stock returns N days back,
  sector ETF return, days since prior earnings

source_params:
  tickers: [PLTR, COIN, RBLX, ...]
  start_date: 2023-01-01    # earliest earnings date to keep
  end_date: 2025-01-01      # latest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import MarketConfig


# Default ticker universe: ~200 small/mid-cap retail-narrative stocks where
# analyst consensus is weak and text signal (SEC filings / transcripts / news / Reddit) plausibly
# adds information. Avoids mega-caps. Mix of AI, crypto-miners, EV, fintech, biotech, meme,
# cannabis, space, semiconductors, growth tech — domains where retail narrative dominates and
# the text-feature thesis has the highest chance of beating numerical baselines.
DEFAULT_TICKERS = [
    # AI / data infra
    "BBAI", "SOUN", "INOD", "AI", "PLTR", "GTLB", "ESTC", "MDB", "DDOG", "NET", "PATH", "MLNK",
    # EV / autonomous / clean transport
    "RIVN", "LCID", "MULN", "GOEV", "NKLA", "CHPT", "BLNK", "EVGO", "XPEV", "NIO", "LI",
    "FSR", "QS", "PSNY",
    # Crypto miners / crypto-exposed
    "CLSK", "RIOT", "MARA", "HUT", "IREN", "BITF", "WULF", "CIFR", "BTBT", "HIVE", "BTDR",
    "COIN", "HOOD",
    # Fintech (BNPL / lending / payments)
    "SOFI", "UPST", "AFRM", "OPEN", "ROOT", "DAVE", "LMND", "OPRT", "MQ", "PYPL",
    # Meme / retail-heavy small/mid caps
    "GME", "AMC", "BB", "KOSS", "NVDQ", "BBAI", "ATER", "PRTY", "EXPR",
    # Cannabis
    "TLRY", "CGC", "SNDL", "ACB", "CRON", "GTBIF", "CURLF", "GRWG", "OGI", "VFF",
    # Space / defense / aerospace small caps
    "RKLB", "ASTR", "PL", "SPCE", "ASTS", "MAXR", "BKSY", "JOBY", "ACHR",
    # Biotech / pharma small caps (FDA exposure, text-driven)
    "OCGN", "SAVA", "NVAX", "VKTX", "TLRY", "BNGO", "CRSP", "EDIT", "BEAM", "NTLA",
    "PACB", "TWST", "VIR", "ACAD", "SRPT",
    # Solar / clean energy
    "RUN", "ENPH", "SEDG", "FSLR", "PLUG", "BE", "BLDP", "FCEL", "BLD", "ARRY", "SHLS",
    # Semiconductors small/mid
    "AMD", "MRVL", "MU", "NVTS", "ON", "WOLF", "ALGM", "SITM", "INDI", "AEHR", "POWI",
    # Growth tech / SaaS small/mid
    "BMBL", "U", "PINS", "SNAP", "ROKU", "DASH", "ABNB", "TDOC", "PTON", "Z", "RDFN",
    "UPST", "ETSY", "WISH", "ENVX", "RBLX",
    # Other small caps with notable retail / narrative volatility
    "WKHS", "NEGG", "MMAT", "PHUN", "ATER", "MULN", "BIG", "TRKA", "BBBYQ", "FAMI",
    "SPRT", "GREE", "DWAC",
    # E-commerce / DTC
    "CHWY", "FIGS", "ALLO", "REAL", "RVLV", "WOOF", "VRA",
]
# De-duplicate while preserving order (some tickers appear in multiple buckets).
DEFAULT_TICKERS = list(dict.fromkeys(DEFAULT_TICKERS))


class StockEarningsAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as e:
            raise ImportError("yfinance is required: pip install yfinance") from e

        tickers = list(source_params.get("tickers") or DEFAULT_TICKERS)
        start_date = pd.Timestamp(source_params.get("start_date", "2023-01-01"))
        end_date = pd.Timestamp(source_params.get("end_date", "2025-01-01"))
        beat_prior = float(source_params.get("beat_rate_prior", 0.60))

        cache_dir = market.raw_dir() / "_yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        rows: list[dict] = []
        sector_etf = self._fetch_price_history("SPY", cache_dir)  # market baseline

        for ticker in tickers:
            ticker_rows = self._process_ticker(
                ticker, start_date, end_date, beat_prior, scraped_at, sector_etf, cache_dir
            )
            rows.extend(ticker_rows)

        if not rows:
            raise RuntimeError(f"stock_earnings produced 0 rows for {tickers}")
        df = pd.DataFrame(rows)
        return df

    def _fetch_price_history(self, ticker: str, cache_dir: Path) -> pd.DataFrame:
        """Get daily Close + Volume for the ticker, cached to disk.

        Period extended to 'max' so the 2014-2024 window is reachable for 200-ticker universe.
        """
        import yfinance as yf
        cache_path = cache_dir / f"{ticker}_prices.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        try:
            hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        except Exception:
            return pd.DataFrame()
        if hist.empty:
            return pd.DataFrame()
        keep = [c for c in ["Close", "Volume"] if c in hist.columns]
        hist = hist[keep].copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist.to_parquet(cache_path)
        time.sleep(0.3)  # yfinance rate-limit politeness
        return hist

    def _fetch_earnings_dates(self, ticker: str, cache_dir: Path) -> pd.DataFrame:
        import yfinance as yf
        cache_path = cache_dir / f"{ticker}_earnings.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        try:
            ed = yf.Ticker(ticker).earnings_dates
        except Exception:
            return pd.DataFrame()
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)
        ed = ed.reset_index().rename(columns={"Earnings Date": "earnings_date", "index": "earnings_date"})
        # Some yfinance versions return col named 'Earnings Date'; normalize
        if "earnings_date" not in ed.columns:
            ed = ed.rename(columns={ed.columns[0]: "earnings_date"})
        ed = ed.rename(columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "surprise_pct",
        })
        ed.to_parquet(cache_path, index=False)
        time.sleep(0.3)
        return ed

    def _process_ticker(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, beat_prior: float,
        scraped_at: pd.Timestamp, sector_etf: pd.DataFrame, cache_dir: Path,
    ) -> list[dict]:
        prices = self._fetch_price_history(ticker, cache_dir)
        ed = self._fetch_earnings_dates(ticker, cache_dir)
        if ed.empty or prices.empty:
            return []

        # Keep only events in window with both estimate AND reported available
        ed = ed.dropna(subset=["eps_estimate", "reported_eps"]).copy()
        ed = ed[(ed["earnings_date"] >= start) & (ed["earnings_date"] <= end)]
        if ed.empty:
            return []
        ed = ed.sort_values("earnings_date").reset_index(drop=True)

        rows = []
        for i, row in ed.iterrows():
            event_dt = pd.Timestamp(row["earnings_date"]).tz_localize(None) if pd.Timestamp(row["earnings_date"]).tzinfo else pd.Timestamp(row["earnings_date"])
            event_date = event_dt.normalize()
            est = float(row["eps_estimate"])
            rep = float(row["reported_eps"])
            beat = 1.0 if rep > est else 0.0

            # Prior surprises (use rows BEFORE this index; strictly past-only).
            # Cap percentage surprises at +/-200 to prevent near-zero-EPS blowups (e.g.
            # estimate=-0.005, actual=-0.01 -> 100% surprise for a tiny absolute miss).
            prior_surprises = ed.iloc[:i]["surprise_pct"].astype(float).dropna().clip(-200, 200)
            avg_surp_4q = float(prior_surprises.tail(4).mean()) if len(prior_surprises) >= 1 else 0.0
            beat_rate_prior_4q = float((prior_surprises.tail(4) > 0).mean()) if len(prior_surprises) >= 1 else beat_prior

            # Stock returns ending day BEFORE earnings (no leak)
            cutoff = event_date - pd.Timedelta(days=1)
            ticker_returns = _trailing_returns(prices, cutoff, lookbacks=(7, 30, 60, 90, 180))
            sector_returns = _trailing_returns(sector_etf, cutoff, lookbacks=(30, 90)) if not sector_etf.empty else {}

            # Days since last earnings
            days_since = int((event_date - pd.Timestamp(ed.iloc[i - 1]["earnings_date"]).normalize()).days) if i > 0 else 90

            # Realized vol last 30 days
            real_vol = _realized_vol(prices, cutoff, lookback_days=30)
            # IV-rank proxy: where does current 30d realized vol sit vs trailing-year distribution?
            # Range 0-1. High value = elevated vol regime (often pre-earnings flag).
            iv_rank = _iv_rank_proxy(prices, cutoff)
            # Earnings-window vol: vol in the 5 trading days leading into the event.
            recent_vol = _realized_vol(prices, cutoff, lookback_days=5)
            # Volume shock: ratio of last-5d avg volume to last-60d avg.
            vol_shock = _volume_shock(prices, cutoff)
            # Gap risk: drift between most recent 7d return and 90d return (momentum extremes).
            mom_diff = (ticker_returns.get(7, 0.0) - ticker_returns.get(90, 0.0))

            # Make target_event_time unique per (ticker, date) so multiple earnings on the
            # same day don't collide during structure.py's pivot. Offset by deterministic
            # per-ticker hash in seconds (within +/- 12 hours of the actual date).
            offset_seconds = (abs(hash(ticker)) % 86400) - 43200
            unique_event_time = event_date + pd.Timedelta(seconds=offset_seconds)
            # p_market is the historical base-rate proxy. Clip to [0.10, 0.90] so the
            # backtest's Kelly sizing doesn't see inf odds when prior_4q has no beats
            # (tickers with all-miss history would otherwise blow up the equity curve).
            raw_prior = beat_rate_prior_4q if not np.isnan(beat_rate_prior_4q) else beat_prior
            p_market_clipped = float(np.clip(raw_prior, 0.10, 0.90))
            rec = {
                "timestamp": unique_event_time,
                "source_published_at": event_date,
                "scraped_at": scraped_at,
                "source_url": f"yfinance://{ticker}/{event_date.date()}",
                "source_type": f"stock_earnings.{ticker}",
                # target_event_time must be UNIQUE per (ticker, date) so that downstream
                # adapters (especially news_scraper per_entity_mode) can use it as a join
                # key to emit aligned per-event rows. The plain calendar date alone is not
                # unique when multiple tickers earn on the same day.
                "target_event_time": unique_event_time,
                "y_realized": beat,
                "p_market": p_market_clipped,
                "decimal_odds": 1.0 / p_market_clipped,
                "ticker": ticker,
                "earnings_date": event_date,
                "num__eps_estimate": est,
                "num__avg_surprise_4q": avg_surp_4q if not np.isnan(avg_surp_4q) else 0.0,
                "num__beat_rate_prior_4q": beat_rate_prior_4q if not np.isnan(beat_rate_prior_4q) else beat_prior,
                "num__days_since_last_earnings": float(days_since),
                "num__realized_vol_30d": real_vol,
                "num__return_7d": ticker_returns.get(7, 0.0),
                "num__return_30d": ticker_returns.get(30, 0.0),
                "num__return_60d": ticker_returns.get(60, 0.0),
                "num__return_90d": ticker_returns.get(90, 0.0),
                "num__return_180d": ticker_returns.get(180, 0.0),
                "num__sector_return_30d": sector_returns.get(30, 0.0),
                "num__sector_return_90d": sector_returns.get(90, 0.0),
                "num__iv_rank_proxy": iv_rank,
                "num__recent_vol_5d": recent_vol,
                "num__volume_shock_5_60": vol_shock,
                "num__momentum_diff_7_90": mom_diff,
            }
            rows.append(rec)
        return rows


def _trailing_returns(prices: pd.DataFrame, cutoff: pd.Timestamp, lookbacks: tuple) -> dict:
    if prices.empty:
        return {}
    p = prices.loc[prices.index <= cutoff]
    if p.empty:
        return {}
    latest = float(p.iloc[-1]["Close"])
    out: dict = {}
    for d in lookbacks:
        target = cutoff - pd.Timedelta(days=d)
        prior = p.loc[p.index <= target]
        if prior.empty:
            out[d] = 0.0
        else:
            prev = float(prior.iloc[-1]["Close"])
            out[d] = (latest / prev) - 1.0 if prev > 0 else 0.0
    return out


def _realized_vol(prices: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int = 30) -> float:
    if prices.empty:
        return 0.0
    p = prices.loc[prices.index <= cutoff]
    if len(p) < 5:
        return 0.0
    window = p.tail(lookback_days)
    rets = window["Close"].pct_change().dropna()
    if rets.empty:
        return 0.0
    return float(rets.std() * np.sqrt(252))


def _iv_rank_proxy(prices: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    """Rank current 30d realized vol vs trailing-252d distribution of 30d-vol values.

    Returns a value in [0, 1]. 1.0 = current 30d vol is the highest in the past year.
    Used as a proxy for option IV-rank since yfinance doesn't expose historical option IVs.
    """
    if prices.empty:
        return 0.5
    p = prices.loc[prices.index <= cutoff]
    if len(p) < 60:
        return 0.5
    p_year = p.tail(252)
    rolling = p_year["Close"].pct_change().rolling(30).std().dropna()
    if rolling.empty:
        return 0.5
    current = float(rolling.iloc[-1])
    rank = float((rolling <= current).sum()) / float(len(rolling))
    return float(np.clip(rank, 0.0, 1.0))


def _volume_shock(prices: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    """Ratio of last-5d average volume to last-60d average volume.

    >1 means recent volume is elevated (pre-earnings positioning or speculation).
    Returns 1.0 if Volume column absent or insufficient data.
    """
    if prices.empty or "Volume" not in prices.columns:
        return 1.0
    p = prices.loc[prices.index <= cutoff]
    if len(p) < 60:
        return 1.0
    recent = float(p["Volume"].tail(5).mean())
    baseline = float(p["Volume"].tail(60).mean())
    if baseline <= 0:
        return 1.0
    return float(np.clip(recent / baseline, 0.0, 20.0))
