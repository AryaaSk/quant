"""Scrape remaining (not-yet-_done) tickers for earnings-flagship-B at high concurrency.

Quick utility script for finishing a partially-completed news_scraper run after
quota exhaustion / restart. Skips any ticker that already has _done marker.

Usage:
  QUANT_ENABLE_AGENTS=1 python scripts/03_scrape_remaining.py [--concurrency 16]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.agents.runner import Invocation, run_agents_parallel  # noqa: E402
from quant.config import load_market  # noqa: E402

BASE_SLUG = "earnings-flagship-B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of parallel haiku agents")
    parser.add_argument("--max-articles", type=int, default=30)
    args = parser.parse_args()

    if os.environ.get("QUANT_ENABLE_AGENTS") != "1":
        print("ERROR: QUANT_ENABLE_AGENTS=1 required", file=sys.stderr)
        return 1

    market = load_market(BASE_SLUG)
    events = pd.read_parquet(market.raw_dir() / "stock_earnings.parquet")
    events["target_event_time"] = pd.to_datetime(events["target_event_time"])
    news_root = market.raw_dir() / "_news"

    all_tickers = sorted(events["ticker"].dropna().unique().tolist())
    todo = [t for t in all_tickers if not (news_root / t / "_done").exists()]
    print(f"all={len(all_tickers)}  todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    invocations = []
    for t in todo:
        ev = events[events["ticker"] == t]
        dates = sorted(pd.to_datetime(ev["target_event_time"]).dt.normalize().unique())
        queries = [
            f"{t} earnings preview {d.year} Q{((d.month - 1) // 3) + 1}"
            for d in dates
        ]
        queries += [
            f"{t} stock news preview",
            f"{t} guidance outlook quarterly results",
            f"{t} analyst expectations",
            f"{t} retail investor sentiment",
        ]
        out = news_root / t
        out.mkdir(parents=True, exist_ok=True)
        invocations.append(Invocation(
            skill="scrape-topic",
            params={
                "market_slug": market.slug,
                "topic": t,
                "queries": queries,
                "output_dir": str(out.absolute()),
                "max_articles": args.max_articles,
            },
            log_subdir=f"{market.slug}/scrape-topic/{t}",
        ))

    print(f"firing {len(invocations)} agents at concurrency {args.concurrency}", flush=True)
    results = run_agents_parallel(invocations, concurrency=args.concurrency)
    ok = sum(1 for r in results if r.ok)
    print(f"done: {ok}/{len(results)} ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
