# betfair-eng-l1l2 — v3 result (text features via agent swarm)

**Status: in-flight as of 2026-05-15.** This document will be filled in after the v3 backtest completes.

## Setup

- **Adapters**: `football_data` (E2 + E3, 11 seasons, ~11k matches with closing odds) PLUS `news_scraper` (3 parallel claude scraper agents for topics: `lineups`, `injuries`, `manager`).
- **Label**: home_win (binary). Same target as v1 and v2; clean three-way comparison.
- **State vector**: 1024 dim. New blocks vs v2:
  - `form_numeric` 0:240 (unchanged from v2)
  - `market_state` 240:368 (unchanged)
  - `text_lineups` 368:560 — 192-dim, MPNet pooled over past 4 days of scraped lineup text
  - `text_injuries` 560:752 — 192-dim, MPNet pooled over past 7 days of injury text
  - `text_manager` 752:1008 — 256-dim, MPNet pooled over past 7 days of manager-press text
  - `calendar` 1008:1024 (unchanged)
- **Model**: time-series transformer, 10 layers, model_dim 384, 8 heads. ~25M params total. Trained on H200 GPU 6.
- **Held-out fence**: last 6 months (same as v1 / v2 for direct comparison).

## Stage A scrape (claude-driven)

- Concurrency: 3 (one agent per topic)
- Budget cap per topic: $1.00 (skill frontmatter)
- Aggregate budget cap: $5.00 (`QUANT_CLAUDE_BUDGET_USD`)
- Articles target: 80 per topic; expected total ~150-240 articles after dedup

Results (TBD):
- Articles saved: TBD per topic
- Total claude spend: TBD
- Date range covered: TBD

## Backtest result (TBD)

| metric | v1 GBDT | v2 transformer | v3 transformer + text | delta v3-v2 |
|---|---|---|---|---|
| net_return | -7.91% | -10.79% | TBD | TBD |
| Brier model | 0.276 | 0.236 | TBD | TBD |
| Brier market | 0.236 | 0.197 | TBD | TBD |
| **Brier improvement** | **-0.160** | **-0.039** | **TBD** | **TBD** |
| hit rate | 46.43% | 23.08% | TBD | TBD |
| Sharpe | -0.72 | -1.32 | TBD | TBD |
| max drawdown | -9.24% | -13.42% | TBD | TBD |
| trades | 28 | 13 | TBD | TBD |

## Interpretation (TBD)

To be written based on result. Three possible verdicts:

- **Brier improvement crosses zero (> 0)**: text features add real signal. Advance to v4 with Qwen3-Embedding-8B on H200 for better text representation. This validates the entire agent-army edge thesis.
- **Brier improvement stays negative but closer to zero than v2**: text adds noise reduction but not edge. Try per-team scraping in v3b for tighter team-news mapping.
- **Brier improvement at or below v2 (-0.04)**: text at topic-level doesn't help. Shelve eng-l1l2; pivot to a market where text features are denser per event (kalshi-cpi with FRED + Fed speakers).

## Verdict (TBD)

To be filled in.
