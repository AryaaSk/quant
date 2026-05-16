"""Agent-driven news-scraper adapter.

For each `topic` in `source_params['topics']`, spawn one scraper agent via the `claude -p`
backend with the `skills/scrape-topic.md` brief. The agent writes per-article JSON files
into `data/raw/<slug>/_news/<topic>/`. After all agents finish, this adapter reads those
files and emits structured rows with one `text__<topic>` column per topic.

source_params schema:
  topics:
    - {name: oil_transport, queries: [global oil shipping rates, tanker traffic Suez]}
    - {name: saudi_arabia,  queries: [Saudi OPEC production, MBS oil policy]}
    ...
  concurrency: 3                    # default 3
  max_articles_per_topic: 30        # default 30
  max_budget_per_topic_usd: 1.0     # default 1.0

Output rows (one per article):
  timestamp                = published_at (or scraped_at if null)
  source_published_at      = published_at (or scraped_at if null)
  scraped_at               = scraped_at
  source_url               = article url
  source_type              = "news_scraper.<topic>"
  text__<topic>            = article text (the matching column is set; other topic columns NaN)

The downstream `structure.py` is patched to forward every `text__<topic>` column.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from quant.agents.runner import Invocation, run_agents_parallel
from quant.config import MarketConfig


class NewsScraperAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        # Two modes:
        #   1. topics mode (legacy): one agent per topic, each scrapes from a query list,
        #      output is one row per article keyed by article published_at.
        #   2. per_entity mode: one agent per entity (e.g. ticker), each scrapes news
        #      around per-entity event dates. Output is one row PER EVENT (aligned to
        #      the event_source's unique target_event_time) with a single text__news
        #      column holding the per-event aggregated text.
        per_entity = bool(source_params.get("per_entity_mode", False))
        if per_entity:
            return self._fetch_per_entity(market=market, source_params=source_params)

        topics = list(source_params.get("topics") or [])
        if not topics:
            raise ValueError(
                "news_scraper requires source_params['topics'] (or per_entity_mode=true)"
            )

        concurrency = int(source_params.get("concurrency", 3))
        max_articles = int(source_params.get("max_articles_per_topic", 30))

        gated = os.environ.get("QUANT_ENABLE_AGENTS") == "1"
        news_root = market.raw_dir() / "_news"
        news_root.mkdir(parents=True, exist_ok=True)

        if not gated:
            existing = list(news_root.rglob("article_*.json"))
            if not existing:
                raise RuntimeError(
                    "news_scraper requires QUANT_ENABLE_AGENTS=1 to spawn agents, and no cached "
                    f"articles exist in {news_root}. Set the env var to enable agent calls."
                )
            return _articles_to_rows(news_root, market)

        invocations = []
        for topic_spec in topics:
            topic_name = topic_spec["name"]
            output_dir = news_root / topic_name
            output_dir.mkdir(parents=True, exist_ok=True)
            invocations.append(Invocation(
                skill="scrape-topic",
                params={
                    "market_slug": market.slug,
                    "topic": topic_name,
                    "queries": list(topic_spec.get("queries") or []),
                    "output_dir": str(output_dir.absolute()),
                    "max_articles": max_articles,
                },
                log_subdir=f"{market.slug}/scrape-topic/{topic_name}",
            ))

        results = run_agents_parallel(invocations, concurrency=concurrency)
        failures = [r for r in results if not r.ok]
        if failures and not list(news_root.rglob("article_*.json")):
            errors = "; ".join(f"{r.skill}: rc={r.returncode} err={r.error or r.stderr[:80]}" for r in failures)
            raise RuntimeError(f"news_scraper agents failed and no fallback files: {errors}")

        return _articles_to_rows(news_root, market)

    # ------------------------------------------------------------------
    # per_entity mode: scrape per ticker, then aggregate per event.
    # ------------------------------------------------------------------

    def _fetch_per_entity(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        event_source = source_params["event_source"]               # e.g. "stock_earnings"
        entity_column = source_params.get("entity_column", "ticker")
        window_days = int(source_params.get("window_days", 30))
        articles_per_entity = int(source_params.get("articles_per_entity", 60))
        concurrency = int(source_params.get("concurrency", 4))
        # Optional: custom query templates for non-earnings domains. Each string may
        # contain "{entity}" and "{date}" placeholders. If omitted, defaults to the
        # stock-earnings query set below.
        query_templates: list[str] = list(source_params.get("query_templates") or [])
        # Optional: extra static queries that don't reference per-date info.
        extra_queries: list[str] = list(source_params.get("extra_queries") or [])

        event_parquet = market.raw_dir() / f"{event_source}.parquet"
        if not event_parquet.exists():
            raise RuntimeError(
                f"per_entity_mode needs event_source '{event_source}' to have run first; "
                f"missing {event_parquet}"
            )
        events = pd.read_parquet(event_parquet)
        if entity_column not in events.columns:
            raise RuntimeError(f"event_source missing column '{entity_column}'")
        events["target_event_time"] = pd.to_datetime(events["target_event_time"]).dt.tz_localize(None) if pd.to_datetime(events["target_event_time"]).dt.tz is not None else pd.to_datetime(events["target_event_time"])

        entity_top_n = source_params.get("entity_top_n")
        if entity_top_n is not None:
            # Rank entities by event count, take top-N for scrape efficiency.
            counts = events[entity_column].value_counts()
            top = counts.head(int(entity_top_n)).index.tolist()
            entities = [e for e in counts.index if e in set(top)]
        else:
            entities = sorted(events[entity_column].dropna().unique().tolist())

        gated = os.environ.get("QUANT_ENABLE_AGENTS") == "1"
        news_root = market.raw_dir() / "_news"
        news_root.mkdir(parents=True, exist_ok=True)

        if gated:
            invocations: list[Invocation] = []
            for entity in entities:
                entity_events = events[events[entity_column] == entity].copy()
                entity_dates = sorted(pd.to_datetime(entity_events["target_event_time"]).dt.normalize().unique())
                if query_templates:
                    queries = []
                    for tmpl in query_templates:
                        if "{date}" in tmpl:
                            for d in entity_dates:
                                queries.append(tmpl.format(entity=entity, date=d.strftime("%Y-%m-%d"), year=d.year))
                        else:
                            queries.append(tmpl.format(entity=entity))
                    queries += [q.format(entity=entity) if "{entity}" in q else q for q in extra_queries]
                else:
                    # Default: stock-earnings query patterns.
                    queries = [f"{entity} earnings preview {d.strftime('%Y Q%q')}".replace("Q%q", f"Q{((d.month-1)//3)+1}") for d in entity_dates]
                    queries += [
                        f"{entity} stock news preview",
                        f"{entity} guidance outlook quarterly results",
                        f"{entity} analyst expectations",
                        f"{entity} retail investor sentiment",
                    ]
                output_dir = news_root / str(entity)
                output_dir.mkdir(parents=True, exist_ok=True)
                invocations.append(Invocation(
                    skill="scrape-topic",
                    params={
                        "market_slug": market.slug,
                        "topic": str(entity),
                        "queries": queries,
                        "output_dir": str(output_dir.absolute()),
                        "max_articles": articles_per_entity,
                    },
                    log_subdir=f"{market.slug}/scrape-topic/{entity}",
                ))
            results = run_agents_parallel(invocations, concurrency=concurrency)
            # Don't raise on partial failures; just log how many entities have any articles
            existing_entities = sum(1 for e in entities if any((news_root / str(e)).glob("article_*.json")))
            if existing_entities == 0:
                errors = "; ".join(f"{r.skill}: rc={r.returncode} err={r.error or r.stderr[:80]}" for r in results if not r.ok)[:600]
                raise RuntimeError(f"per_entity scrape produced no articles. Errors: {errors}")

        # Build per-event rows aligned to the event_source's unique target_event_time.
        # Each row's text__news column = concatenated articles ABOUT that ticker, published
        # within [target_event_time - window_days, target_event_time).
        scraped_at_fallback = pd.Timestamp.utcnow().tz_localize(None)
        out_rows: list[dict] = []
        for _, ev in events.iterrows():
            entity = ev[entity_column]
            t_event = pd.Timestamp(ev["target_event_time"])
            if t_event.tzinfo is not None:
                t_event = t_event.tz_localize(None)
            entity_dir = news_root / str(entity)
            texts: list[str] = []
            scraped_max = None
            published_max = None
            n_articles = 0
            for art_path in sorted(entity_dir.glob("article_*.json")) if entity_dir.exists() else []:
                try:
                    data = json.loads(art_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                published_at = _parse_ts(data.get("published_at"))
                scraped_at = _parse_ts(data.get("scraped_at")) or scraped_at_fallback
                effective = published_at if published_at is not None else scraped_at
                # Temporal-leak guard: only use articles with effective < event_time
                # AND within window.
                if effective is None:
                    continue
                if not (t_event - pd.Timedelta(days=window_days) <= effective < t_event):
                    continue
                texts.append(text[:4000])  # cap per-article length
                n_articles += 1
                if scraped_max is None or scraped_at > scraped_max:
                    scraped_max = scraped_at
                if published_max is None or (published_at is not None and (published_max is None or published_at > published_max)):
                    published_max = published_at
            scraped_max = scraped_max or scraped_at_fallback
            published_max = published_max or t_event
            out_rows.append({
                "timestamp": t_event,
                "source_published_at": published_max,
                "scraped_at": scraped_max,
                "source_url": f"news_scraper://per_event/{entity}/{t_event.isoformat()}",
                "source_type": f"news_scraper.per_event.{entity}",
                "target_event_time": t_event,
                "text__news": "\n\n".join(texts) if texts else "",
                "num__news_article_count": float(n_articles),
            })

        if not out_rows:
            raise RuntimeError("per_entity_mode produced 0 event rows")
        return pd.DataFrame(out_rows)


def _articles_to_rows(news_root: Path, market: MarketConfig) -> pd.DataFrame:
    rows: list[dict] = []
    scraped_at_fallback = pd.Timestamp.utcnow().tz_localize(None)
    for topic_dir in sorted(p for p in news_root.iterdir() if p.is_dir()):
        topic = topic_dir.name
        for article_path in sorted(topic_dir.glob("article_*.json")):
            try:
                data = json.loads(article_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            text = (data.get("text") or "").strip()
            if not text:
                continue
            published_at = _parse_ts(data.get("published_at"))
            scraped_at = _parse_ts(data.get("scraped_at")) or scraped_at_fallback
            ts = published_at if published_at is not None else scraped_at
            row = {
                "timestamp": ts,
                "source_published_at": published_at if published_at is not None else scraped_at,
                "scraped_at": scraped_at,
                "source_url": data.get("url", str(article_path)),
                "source_type": f"news_scraper.{topic}",
                f"text__{topic}": text,
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(f"news_scraper produced 0 articles in {news_root}")
    return pd.DataFrame(rows)


def _parse_ts(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts
