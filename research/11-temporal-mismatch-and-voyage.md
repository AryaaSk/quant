# 11. Temporal mismatch + Voyage integration (2026-05-15)

Two big infrastructure stories from the eng-l1l2 v3 run, plus a methodology insight that changes how we evaluate any market with scraped text.

## What shipped

### Voyage AI as the text encoder (paid, top-of-MTEB)

- `src/quant/text/voyage_embedder.py` — client for `https://api.voyageai.com/v1/embeddings` with disk caching per (model, text) sha256 hash.
- Two models supported: `voyage-3-large` (general) and `voyage-finance-2` (finance-tuned).
- `state/text.py` gains `mode: voyage`. Yaml selects model via `encoder_name`.
- Caching path is per-market: `data/raw/<slug>/_voyage_cache/<sha256>.npy`. Reruns hit cache; only new strings pay.
- API call: ~0.5s for 2 articles. Cached rerun: ~1ms.

### Heavy compute moved to H200

Mac's role is now strictly: scraping (claude with OAuth/keychain creds) + viewing results. Everything else runs on the H200 box.

- `scripts/02b_build_state.py` — standalone state-vector builder.
- `scripts/run_remote_pipeline.py` — runs state build → train → backtest → report in sequence with stage-level timing logs.
- `bin/h200 sync` now also pushes `data/raw/<slug>/` (excluding heavy local caches like `_fd_cache`, `_td_cache`, `_hl_cache`, `_fred_cache`, `_kalshi_cache`) and `skills/`.
- `bin/h200 run` forwards `VOYAGE_API_KEY` (plus `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `FRED_API_KEY`, `THE_ODDS_API_KEY`) to the remote process for the lifetime of the SSH command.
- `state/text.py` auto-detects `cuda` / `mps` / `cpu` for sentence-transformers when in `encoder` mode (Voyage doesn't need this since it's API-based).
- Verified: `bin/h200 run scripts/run_remote_pipeline.py betfair-eng-l1l2 --device cuda` does the full state build + train + backtest + report in 81 seconds on GPU 6.

### Enriched football_data adapter

`football_data.py` now emits 27 numeric features per match instead of 9. Added: rolling-5 shots / shots on target / corners / yellows / clean sheets, rest_days, season-to-date points / matches played / reds. All strictly past-only.

## What we learned

### Current-scraped news cannot backtest historical events

The eng-l1l2 v3 backtest was numerically identical to v2 (same Brier improvement, same trades, same return). Reason:

- All 24 scraped articles were dated 2026-04-01 to 2026-05-15 (current).
- All events in football_data are 2015-2025.
- Temporal-leak guard at `state/text.py::build()` enforces `published_at < event_time`.
- Therefore every event's text window was empty, every text block returned zeros, Voyage was never called.

The leak guard is correct. A weaker pipeline would have leaked the post-hoc text into the state vector and produced a glossy fake positive result. Better to ship an honest "no signal" than an inflated artifact.

### Implication for the agent-army thesis

The text-feature edge needs ONE of:

1. **Historical article corpora** (scraping with date filters from Common Crawl, news API archives, Wayback Machine). Hard. Wayback is rate-limited and not all sites are indexed. Paid news APIs are pricey but offer date-filtered search.
2. **Prospective testing** — scrape articles weekly, predict the next week's events, score after resolution. Slow feedback but the natural deployment shape. For Betfair football this means starting now and collecting one round per weekend.

For markets where text is dense and naturally archived (macro nowcasts, Fed speeches, SEC filings), the historical-corpus path is tractable. For breaking team news in lower-division football, prospective testing is more realistic.

### Recommendation: pivot the text-edge experiment to `kalshi-cpi`

Kalshi macro markets are a better fit for historical-text-backtest because:
- Atlanta Fed GDPNow, ADP, Truflation, Redbook all maintain dated daily/weekly archives with stable URLs.
- The text is structurally about KNOWN release dates (CPI releases on the 13th of each month etc), so we can scrape "as of t-7 days" snapshots cleanly.
- FRED API gives the actual macro time series with publish dates.
- Fed paper (NBER 2026) empirically validated that Kalshi macro contracts have measurable inefficiency vs Bloomberg consensus.

Need a FRED API key to make this real.

## Why v3 still ran in 81 seconds despite text being zero'd

The on-H200 pipeline timing breakdown:

- build_state: 68.4s (composer iterates 904 events × 32 timesteps; for each timestep it builds 6 block features; even though text blocks return zeros, the path through them still costs Python time)
- train: 11.7s (8-layer transformer × 30 epochs × 904 events on H200 GPU 6, well under 1% utilization of the GPU)
- backtest: 0.1s
- report: 0.6s

The 68s build_state is dominated by Python-loop overhead (904 × 32 = 28,928 composer calls). The actual Voyage API calls would have added milliseconds per cached embedding lookup; the cache had zero entries this run.

When text is populated in a future run, expect build_state to grow by maybe ~10-30s for ~100 articles × Voyage API roundtrip (sub-second per batch of 64). Still well under 2 minutes total end-to-end on H200.

## Cost summary for the eng-l1l2 v3 cycle

| stage | tool | cost |
|---|---|---|
| Stage A scrape | claude -p (3 parallel agents, 24 articles total) | ~$3.47 |
| Stage A.5 extraction | codex exec (attempted, ~46 batches planned) | hit usage limit; fell back to default 0.0 features; dropped from yaml |
| Stage B propose-schema | codex exec | hit usage limit; hand-designed instead |
| State build (Voyage embed) | Voyage API | $0 (zero articles passed the leak guard) |
| Training | H200 GPU 6 | ~12 GPU-seconds |
| Backtest | local CPU | negligible |
| **Total Mac-only spend** | — | **~$3.47 claude** |
| **Codex subscription used** | — | exhausted daily cap on 2 calls before usage hit limit |

Voyage spend remained zero on this run because of the temporal mismatch; on a real (prospective) v4 with current scrapes vs upcoming events, expect ~$0.05-0.20 per market run.

## Next steps

1. **kalshi-cpi v2**: implement FRED API scraping with proper publish-date archives. Voyage embeddings on the macro nowcast text. This is the right test of the agent-army thesis.
2. **Prospective eng-l1l2**: lock the v3 pipeline as the live evaluator. Scrape weekly, predict the next weekend's matches, score after resolution. Start collecting now.
3. **Article-history adapter**: explore Common Crawl + Wayback Machine + paid news APIs for historical EFL injury reports. Effort is high; only worth it if Kalshi v2 doesn't pan out.
