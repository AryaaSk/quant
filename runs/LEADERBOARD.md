# Leaderboard

Ranked by net_return after commission, slippage, and safety margin. Brier improvement is computed against market closing implied probability. All metrics include the label-shuffle sanity check (`tests/test_backtest.py::test_label_shuffle_collapses_brier_improvement`).

slug | net_return | brier_improvement | trades | hit_rate | sharpe | max_drawdown | run
---|---|---|---|---|---|---|---
`polymarket-entertainment-A` | 295398.76% | 0.1190 | 24 | 58.33% | 1.66 | -7.86% | `20260515T202227`
`polymarket-entertainment-B` | 268802.44% | 0.1103 | 24 | 54.17% | 1.65 | -8.62% | `20260515T211843`
`earnings-flagship-B-no-sec` | 706.45% | -0.0235 | 206 | 35.92% | 2.42 | -30.74% | `20260516T090936`
`earnings-flagship-B-70K` | 706.09% | -0.0213 | 206 | 35.92% | 2.42 | -30.74% | `20260516T092741`
`earnings-flagship-B-33M` | 705.99% | -0.0226 | 206 | 35.92% | 2.42 | -30.74% | `20260516T085740`
`earnings-flagship-B-11M` | 705.85% | -0.0226 | 206 | 35.92% | 2.42 | -30.74% | `20260516T085133`
`earnings-flagship-B-1M` | 705.51% | -0.0208 | 206 | 35.92% | 2.42 | -30.74% | `20260516T092243`
`earnings-flagship-B-117M` | 705.38% | -0.0216 | 206 | 35.92% | 2.42 | -30.74% | `20260516T085414`
`earnings-flagship-B-160K` | 705.21% | -0.0205 | 206 | 35.92% | 2.42 | -30.74% | `20260516T092613`
`earnings-flagship-B-203M` | 705.12% | -0.0211 | 206 | 35.92% | 2.42 | -30.74% | `20260516T090014`
`earnings-flagship-B-4M` | 701.39% | -0.0234 | 206 | 35.92% | 2.42 | -30.74% | `20260516T085914`
`earnings-flagship-B-400K` | 692.22% | -0.0191 | 206 | 35.92% | 2.41 | -30.74% | `20260516T092503`
`earnings-flagship-B-600K` | 635.62% | -0.0274 | 204 | 35.29% | 2.35 | -31.33% | `20260516T092352`
`_synthetic` | 127.28% | -0.1160 | 33 | 45.45% | 1.98 | -8.30% | `20260515T123806`
`earnings-beatmiss-A` | 106.60% | -0.0075 | 27 | 51.85% | 1.83 | -9.12% | `20260515T201117`
`earnings-flagship-A` | 79.59% | -0.1313 | 204 | 48.04% | 1.05 | -24.08% | `20260515T233324`
`earnings-beatmiss-B` | 37.45% | -0.0612 | 31 | 61.29% | 1.57 | -7.61% | `20260515T204919`
`kalshi-cpi` | 0.00% | 0.0001 | 0 | 0.00% | 0.00 | 0.00% | `20260515T135251`
`betfair-eng-l1l2` | -10.79% | -0.0392 | 13 | 23.08% | -1.32 | -13.42% | `20260515T180709`
`hyperliquid-midcap` | -14.66% | -0.0012 | 734 | 48.91% | -0.76 | -40.26% | `20260515T124514`
`betfair-itf-challenger` | -26.68% | -0.1252 | 134 | 38.81% | -1.25 | -29.19% | `20260515T133350`

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
