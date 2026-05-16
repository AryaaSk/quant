## What advances to v2 and what is shelved

A market advances to deep iteration if it shows `brier_improvement > +0.01` AND `net_return > 0` simultaneously. None have cleared that bar yet. Honest verdicts:

- **`kalshi-cpi`**: inconclusive (~28 events via public Kalshi API, no FRED key, hand-tuned features). Highest a-priori promise per the Fed paper. **Top priority for next session** once FRED key + longer Kalshi archive are wired.
- **`betfair-eng-l1l2`**: now on v3 (Voyage embeddings + enriched football_data + agent-scraped news + H200 compute). Result byte-for-byte matches v2 because all scraped articles are dated 2026 and all events are 2015-2025 — temporal-leak guard correctly excluded every article. **Pivot to prospective testing**: scrape weekly, predict the upcoming weekend, score after resolution. See `research/11-temporal-mismatch-and-voyage.md`.
- **`hyperliquid-midcap`**: structured-features-only crypto has no edge (Brier ~0 vs 0.5 prior). Next: sentiment + on-chain whale flows.
- **`betfair-itf-challenger`**: ATP main substitute (tennis-data doesn't cover ITF). Next: Jeff Sackmann's `tennis_atp_challenger` repo + an ITF odds source.
- **`_synthetic`**: smoke test; meaningless for ranking.

See `research/per-market/<slug>.md` for full per-market write-ups.

## What this project now has

1. **End-to-end pipeline running on H200 GPU 6**. 81-second wall clock for `state build → train → backtest → report` via `bin/h200 run scripts/run_remote_pipeline.py <slug> --device cuda`. Mac is only used for scraping (claude with OAuth) and viewing results.
2. **Agent-driven data flow**. Stage A: claude scrapers per topic. Stage A.5: codex feature extractors (when codex quota is available). Stage B: codex schema designer. All gated behind `QUANT_ENABLE_AGENTS=1` with documented budget caps.
3. **Voyage AI integration**. Top-of-MTEB text embeddings (`voyage-3-large`, 1024-dim) with disk caching per (model, text) sha256. Replaces MPNet entirely. Configured via `mode: voyage` in any yaml's text block.
4. **Enriched football_data adapter**. 27 raw numeric features per match (up from 9), all strictly past-only with rest_days + season-to-date counters.
5. **Strict backtest hygiene**. Label-shuffle sanity test, temporal-leak property test, held-out fence locked write-once. The v3 run's "no signal" verdict is the leak guard correctly excluding post-hoc articles — a weaker pipeline would have fabricated a positive result.

## What this project still does NOT have

- Historical article corpora. Current scrapes can't backtest historical events without one.
- FRED API key wired. Blocks `kalshi-cpi` from being a real experiment.
- Sofascore or news-archive adapter for prospective sports text.
- Multi-market pretraining (still on the roadmap; needs at least one market to clear the gate first).

See `research/11-temporal-mismatch-and-voyage.md` for the methodology pivot and `research/07-h200-window-result.md` for prior-iteration context.
