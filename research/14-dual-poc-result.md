# 14. Dual A-vs-B PoC result (2026-05-15)

The methodology test: do agent-scraped + Voyage-embedded text features outperform numerical-only features on *retail-relevant* prediction markets?

Two PoCs run in parallel in the same session to isolate the text-feature contribution under different domain conditions.

## PoC 1 — Stock earnings beat/miss (PoC primary)

- **Universe**: 24 mid/small-cap retail-narrative tickers (BBAI, RIVN, LCID, CLSK, RIOT, SOFI, GME, AMC, etc.)
- **Sample**: 253 quarterly earnings events 2022-01-01 → 2025-01-01
- **Label**: `y = 1 if reported_eps > consensus_estimate`
- **Held-out**: last 6 months (28 events)
- **Class balance**: 56% beat rate (clean)
- **Model class**: GBDT for both A and B (n=253 is GBDT territory)
- **Text source (B only)**: haiku scraper per ticker × earnings event, ~25-80 articles per event, Voyage-embedded into 240 slots

Result table:

| variant | features | brier_improvement | net_return | trades | hit_rate |
|---|---|---|---|---|---|
| A | numerical only | -0.0075 | +106.60% | 27 | 51.85% |
| **B** | A + Voyage(haiku news) | **-0.0612** | **+37.45%** | **31** | **61.29%** |

**Verdict: B regressed.** 240-slot text block on 253 events ran into the classic dimensionality-overfit regime. Methodology was sound (proper temporal alignment, real article quality), dataset is just too small. Need 10-100x scale to validate.

## PoC 2 — Polymarket entertainment unified

- **Universe**: ALL resolved Polymarket entertainment markets across awards (Oscars, Emmys, Globes, Grammys, BAFTAs, SAG, Critics' Choice), film festivals (Cannes, Venice, Sundance), box office milestones, and TV/streaming events.
- **Sample**: **5,271 binary yes/no markets** across **411 unique event_slugs** (2024-01-16 → 2026-03-31)
- **Label**: `y = 1 if market resolved Yes`
- **Held-out**: last 6 months (~280 events)
- **Class balance**: 10.5% positive rate (one winner per category-year ⇒ imbalanced)
- **Model class**: GBDT for both A and B (5,271 is at the lower edge of transformer territory; this PoC focuses on the feature-set delta, not the model class)
- **Text source (B only)**: haiku scraper per `event_slug`, top-60 events by market count (covering ~70% of data), 25 articles per event, Voyage-embedded into 240 slots

Result table:

| variant | features | brier_improvement | net_return | trades | hit_rate |
|---|---|---|---|---|---|
| A | numerical only | +0.1190 | +295,398%* | 24 | 58.33% |
| **B** | A + Voyage(haiku commentary, 23/60 entities scraped) | **+0.1103** | **+268,802%*** | **24** | **54.17%** |

**Verdict: B marginally regressed (-0.009 Brier).** Partial text coverage (23/60 entities scraped before claude usage exhausted; 70% of rows have empty `text__news`) made the text block act more as noise than signal. Methodology not invalidated; coverage is the confound.

*The huge `net_return` for entertainment-A is a Kelly-on-longshot artefact — many Polymarket markets are priced at < 0.10, so even a small edge × ~10x payoff compounded over 24 trades produces astronomical returns. The headline metric for entertainment is `brier_improvement` (calibration), not return.

## Apparatus reused across both PoCs

- `bin/h200` deploy helper (Voyage key forwarded via env_prefix)
- `scripts/02_collect.py` raw + structure + state in one pass
- `scripts/run_remote_pipeline.py` (state + train + backtest + report on H200, used for stock-earnings-B because GBDT is fast enough on Mac)
- `src/quant/agents/runner.py` — claude `-p` subprocess wrapper, supports `--model haiku|sonnet|opus`
- `src/quant/adapters/news_scraper.py` — `per_entity_mode` + `entity_top_n` + custom `query_templates`
- `src/quant/text/voyage_embedder.py` — voyage-3-large with disk caching by sha256
- `src/quant/state/text.py` `mode: voyage`
- `skills/scrape-topic.md` (haiku, $1.0 per-agent cap)
- Backtest hygiene: temporal-leak property test + label-shuffle sanity (`tests/test_backtest.py`)

## Methodology lessons (from this session)

1. **Unique `target_event_time` per event row is non-negotiable.** Without per-(entity, date) uniqueness, the `news_scraper` per_entity_mode cannot align text rows to event rows, and `structure.py`'s `groupby(timestamp)` collapses multi-entity-same-date rows. Both `stock_earnings` and `polymarket_contracts` adapters apply a deterministic per-entity hash offset to the resolution date.

2. **Temporal-leak guard must be strict, not advisory.** The earlier `betfair-eng-l1l2 v3` fumble (scraped 2026 news for 2015-2025 events) was correctly caught by the news_scraper's `published_at < target_event_time` filter — every article was excluded, the text block returned zeros, and the result was bit-identical to v2. Pick markets where the temporal contract is naturally satisfied: stock earnings (news published before earnings) and Polymarket entertainment (commentary published before ceremony / box-office release).

3. **Model routing**: claude **haiku** for bulk article scraping (cheap, plenty for tool-use loops), **sonnet** for nuanced per-event feature extraction, **opus** for one-shot schema design. Codex deferred (daily quota cap unreliable across sessions).

4. **Scope control matters at scale**. Polymarket entertainment has 411 unique event_slugs. At ~250s per haiku agent and concurrency 6, a full scrape would take 5+ hours. `entity_top_n: 60` reduces wall time to ~40 minutes while keeping 70% data coverage (top events have most markets).

5. **State vector philosophy**: mix everything useful, name every slot range. Numeric (rolling z-score), market_state (price + velocity lags), text (Voyage mean-pooled into a fixed slot region), calendar. Per-market schema declares slot ranges; composer assembles the fixed-dim state vector deterministically.

## What advances after this session

- Either or both Bs clear the gate (Brier improvement +0.02 over their A baseline) → text-feature thesis confirmed for that domain.
- Both Bs flat → text features add noise more than signal at GBDT scale. Next move: scale to transformer + more events, or move to a domain with denser unique text per event (e.g. FDA approval clinical trial filings — high signal per event but tiny event count).
- Earnings B wins, entertainment B doesn't (or vice versa) → text contribution is domain-dependent. Both directions are interesting; pick the winner for v2 deep iteration.

(Result tables will be filled in once both Bs complete training.)
