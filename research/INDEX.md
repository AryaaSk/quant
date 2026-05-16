# Research index

Every meaningful design decision, market hypothesis, screening result, and post-mortem lands here. Convention: short, dated entries; conclusions first, supporting evidence below.

## Living docs (kept current)

- [00-context.md](00-context.md) — why this project, the edge thesis, what makes it different
- [01-market-universe.md](01-market-universe.md) — the candidate universe with thesis per market, ranked tiers
- [02-pipeline-design.md](02-pipeline-design.md) — state-vector design, per-market schemas, transformer choice
- [03-backtest-hygiene.md](03-backtest-hygiene.md) — non-negotiable rules + why each exists
- [04-screening-method.md](04-screening-method.md) — how Phase 1 works, scoring rubric
- [05-phase1-result.md](05-phase1-result.md) — actual screening result + which adapters are still stubbed
- [06-h200-gate.md](06-h200-gate.md) — H200 gate status + how to unblock when GPU 6 frees
- [07-h200-window-result.md](07-h200-window-result.md) — outcomes of the 3-hour H200 window: 4 new markets, 2 transformer trainings on GPU 6, honest verdicts
- [08-architecture-vision.md](08-architecture-vision.md) — user-articulated two-stage agent-swarm vision: Stage A (scraper agents dump raw), Stage B (schema-design agents propose yaml + encoding), Stage C (transformer + unembedding); maps onto current code
- [09-agent-swarm-implementation.md](09-agent-swarm-implementation.md) — the actual implementation: dual-runtime (claude for Stage A web scrape, codex for Stage B local synthesis), skills/, runner.py, news_scraper adapter, propose-schema script, 23 mocked tests + 2 real-subprocess smoke tests
- [10-stage-a5-feature-extraction.md](10-stage-a5-feature-extraction.md) — Stage A.5 inserted between scrape and structure: codex agents read in-window articles per event and emit a strict schema of named numeric features. Complementary to the MPNet text embeddings; both flow into the same state vector. 11 mocked tests.
- [11-temporal-mismatch-and-voyage.md](11-temporal-mismatch-and-voyage.md) — Voyage AI integration + heavy-compute-on-H200 split + the temporal-mismatch insight: current scraped news can't backtest historical events. Methodology pivot toward prospective testing or historical corpora.
- [12-session-lessons-2026-05-15.md](12-session-lessons-2026-05-15.md) — **READ FIRST IF CONTINUING THIS PROJECT.** End-of-session lessons doc covering: the actual goal, the fumbles to avoid, debugging traps (claude/codex/pipeline/compute), what infrastructure is reusable, how to start a new market.
- [13-broader-entertainment-poc.md](13-broader-entertainment-poc.md) — **PoC 2 design** (broader entertainment Polymarket): union ALL entertainment markets (Oscars + Emmys + Globes + festivals + box office + streaming) into a transformer-scale 3k-8k event dataset. Methodologically stronger than just-Oscars due to transfer learning and balanced labels. Pipeline + concrete adapters specified.

## Per-market deep dives

After Phase 1 screening, each surviving market gets a dedicated file under [per-market/](per-market/) documenting:

- the information-asymmetry thesis in detail
- which adapters fetch what
- state vector schema (block-by-block rationale)
- data quality issues encountered
- backtest result + interpretation
- next-iteration ideas

## Conventions

- No em dashes. Use commas, periods, parens.
- No emoji unless explicitly requested.
- Date stamps for time-sensitive claims (e.g. "as of 2026-05-15, Kalshi 30-day volume = $X").
- Link liberally between docs.
- If a decision is reversed, do not delete the old text. Strike through and add a dated note explaining why.

## When to write

- Before starting a new phase: outline what success looks like.
- After landing a meaningful change: write down the decision and the trade-off.
- After a market is screened: log the verdict and reasoning.
- After a backtest: log the result, the realistic-cost adjustments applied, and whether it advances to iteration 2.
