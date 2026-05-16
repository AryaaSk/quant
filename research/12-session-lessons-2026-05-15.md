# 12. Session lessons (2026-05-15) — for the next agent that picks this up

Written end-of-session before context exhaustion. **READ THIS FIRST** if you're continuing this project. It distills what works, what doesn't, and how not to repeat my mistakes.

## The user's actual goal (don't lose sight of it)

A proof of concept showing **agent-army-research-augmented transformer > structured-features-only transformer** on the same prediction task. Specifically:

- Pick a real market with archived prices.
- Train Model A on structured features only (price, basic stats).
- Train Model B on the same features PLUS agent-scraped/extracted text features.
- Show B > A on held-out Brier score or net return after costs.

That is the ENTIRE bet. Infrastructure is overhead. Don't optimize Voyage integration before checking if the data is even temporally coherent.

## The fumble I made (don't repeat)

I picked `betfair-eng-l1l2` because historical odds data was clean. I then scraped CURRENT (2026-05) news to predict HISTORICAL (2015-2025) matches. The temporal-leak guard correctly nuked every article, producing v3 = v2 identically. **Today's news cannot predict yesterday's events.** I wasted hours building Voyage + H200 routing without sanity-checking the data temporal alignment.

**The check that would have caught it**: before building anything, ask "is the text temporally aligned with the events?" If text dates are AFTER event dates, the experiment is broken before it starts.

## What markets work for this PoC (text temporally precedes events)

| Market | Why it works | Constraint |
|---|---|---|
| **Polymarket Oscars / Emmys** | Critic reviews dated months/weeks before ceremony; Polymarket history on-chain | Only ~20-25 categories per ceremony; need 3-5 ceremonies for sample size |
| **Kalshi macro (CPI/NFP)** | Nowcasts (Atlanta Fed GDPNow, Truflation, ADP) published daily with dated archives; Kalshi has dated contract prices | Need FRED API key + Kalshi historical scrape; ~12-24 events/year |
| **Single-stock earnings** | SEC filings, earnings call transcripts dated and archived; options market implied move pre-earnings | Need a price-data source (yfinance free, paid for tick-level) |
| **Polymarket geopolitics** | News dated by publish time; outcomes resolved on-chain | Risk of insider front-running on big events |

**Markets that DON'T work for this PoC**:
- Any sport with current-only news scrape (EFL football v3). Use ONLY for prospective testing.
- Crypto without sentiment archive (Twitter has daily archives but locked behind paid API).

## Concrete debugging traps I hit (don't waste hours)

### Subprocess agent traps
1. **`claude --bare` disables OAuth/keychain auth.** Subprocess invocations from a Claude Code session have no `ANTHROPIC_API_KEY` env. **Remove `--bare`** so claude uses keychain.
2. **`claude --add-dir <dirs...>` is variadic** and consumes any trailing positional arg as another directory. **Never pass prompt as a positional after `--add-dir`**. Use `subprocess.run(cmd, input=prompt, ...)` to pipe via stdin.
3. **`claude -p` has ~$0.13 floor cost per invocation** for system-prompt cache creation, even with cached prompts. Don't expect tiny calls to be free.
4. **Codex daily quota** caps out mid-session. Plan around: budget 50-100 codex calls max per day.
5. **Codex output is NDJSON event stream**, not a single JSON. Parse `item.completed.item.type == "agent_message"` for the actual response text. The last NDJSON line is `turn.completed` with usage stats, not the response.

### Pipeline traps
6. **Cached parquets must be invalidated when adapter logic changes.** `collect.py` skips an adapter if its parquet exists. After enriching `football_data.py` from 9 to 27 features, the OLD parquet was reused — invisible bug. Delete the parquet OR pass `--force`.
7. **Multiple events per `target_event_time`**: in sports, many matches share a date. `_build_event_list` MUST dedupe (`groupby(target_event_time).first()`) or the agent extractor over-batches by 10-100x.
8. **Python imports are cached at script start.** Editing source while a long-running script holds the old code makes for confusing "why isn't my fix taking effect" moments. Restart the process.
9. **Module-level cwd assumptions break under `bin/h200 run`.** The remote process's cwd is `~/Aryaa/quant` not `~/Desktop/quant`. Use absolute paths or relative-to-`REPO_ROOT`.

### Compute traps
10. **Don't run heavy compute on the Mac when an idle H200 sits at 0% utilization.** MPNet text encoding for 86k strings is 10-30 min on Mac CPU, 30-60 sec on H200. The pattern is: Mac for scraping+orchestration, H200 for state build + train + backtest.
11. **`bin/h200 sync` excludes are critical** for `_fd_cache/_td_cache/_hl_cache/_fred_cache` — these are local-only raw data caches that don't need to leave the Mac.
12. **Forward all API keys via `bin/h200 run`'s `env_prefix`**: ANTHROPIC, OPENAI, FRED, VOYAGE, THE_ODDS. SSH won't inherit Mac's keychain.

## What infrastructure is already built and ready to reuse

### Adapters (in `src/quant/adapters/`)
- `synthetic.py` — toy market for CI smoke test
- `football_data.py` — Football-Data.co.uk CSVs (27 numeric features per match; rolling-5 stats + season-to-date counters + closing odds)
- `tennis_data.py` — tennis-data.co.uk xlsx (ATP main draw)
- `fred.py` — FRED API client with disk cache (requires FRED_API_KEY)
- `kalshi.py` — Kalshi public API (no auth needed for `api.elections.kalshi.com`)
- `hyperliquid.py` — Hyperliquid public POST API for ohlcv + funding
- `news_scraper.py` — claude scrape orchestrator (3 parallel agents, $1/topic budget cap)
- `agent_feature_extractor.py` — codex per-event feature extractor (dedup-by-event-time, batched, idempotent)
- `_stubs.py` — placeholder stubs for unimplemented adapters

### Text encoders
- `state/text.py` supports `mode: voyage | encoder | handcrafted`
- `text/voyage_embedder.py` — Voyage AI client with disk caching per (model, text) sha256
- Auto-detects CUDA/MPS/CPU for sentence-transformers when mode=encoder

### Pipeline scripts
- `scripts/02_collect.py` — collect → structure → state on Mac
- `scripts/02b_build_state.py` — state build only (for H200)
- `scripts/run_remote_pipeline.py` — state build + train + backtest + report (for H200)
- `scripts/03_train.py` — train only
- `scripts/04_backtest.py` — backtest only
- `scripts/05_compare.py` — write LEADERBOARD.md (preserves prose footer)
- `scripts/06_propose_schema.py` — codex schema designer (gated by `QUANT_ENABLE_AGENTS=1`)

### Skills (markdown briefs for agents)
- `skills/scrape-topic.md` (backend: claude) — scrape open-web text for a topic
- `skills/extract-features.md` (backend: codex) — per-event structured feature extraction
- `skills/propose-schema.md` (backend: codex) — design state vector yaml

### bin/
- `bin/h200` — deploy + run helper. Now forwards Voyage/Anthropic/OpenAI/FRED/Odds keys. Syncs src + markets + scripts + data + skills.
- `bin/run-market <slug>` — end-to-end pipeline for a market
- `bin/screen-markets` — Phase 1 candidate ranking
- `bin/propose-schema` — codex schema designer wrapper

### Tests (48 passing, free)
- `test_agent_runner.py` (17): backend dispatch, prompt rendering, idempotency
- `test_news_scraper.py` (6): adapter plumbing
- `test_agent_feature_extractor.py` (11): batching + dedup + clipping + default fallback
- `test_bootstrap.py` (4): yaml parsing
- `test_temporal_leak.py` (3): property-based, Hypothesis
- `test_backtest.py` (2): label-shuffle sanity check
- `test_collection.py` (2): held-out fence
- `test_train_smoke.py` (2): training loop converges
- `test_leaderboard.py` (1): claims backed by metrics.json
- `test_e2e_smoke.py` (3): full pipeline on synthetic market
- `test_codex_smoke.py` (gated): real codex call
- `test_claude_smoke.py` (gated): real claude call

## How to write a NEW market (concrete steps)

1. **Pick the market**. Check temporal alignment of text vs events. If text is naturally archived with publish dates earlier than events, you can backtest. Otherwise, prospective-only.

2. **Build the price/outcome adapter** (or reuse existing). Adapter contract:
   - Returns a DataFrame with required columns: `timestamp`, `source_published_at`, `scraped_at`, `source_url`, `source_type`, `target_event_time`, `y_realized`, `p_market`, `decimal_odds`.
   - Numeric features as `num__<name>` columns.
   - Register in `src/quant/adapters/__init__.py` `_REGISTRY`.

3. **Build the text adapter** (or reuse `news_scraper`). Text columns as `text__<topic>`. CRITICAL: every text row must have a `published_at` that PRECEDES its target event time.

4. **Hand-design or codex-propose the yaml schema**. Slot allocation should match data density (don't allocate 1024 dims when you have 24 articles of sparse coverage).

5. **Run on H200**: `bin/h200 run scripts/run_remote_pipeline.py <slug> --device cuda`.

6. **Pull results back, update leaderboard, write per-market doc.**

## The Polymarket Oscars PoC (planned for this session, after this doc)

### Why Oscars
- Polymarket has dated on-chain history for ~5 ceremonies (2021-2025)
- ~20 categories × 5 years = ~100 events. Enough for stat-sig PoC.
- Each nominated film has dated critic reviews (Letterboxd, RottenTomatoes, Metacritic)
- Festival wins (Cannes, Venice, Telluride) have known historical dates
- Realistic edge thesis: handicapper models (Ben Zauzmer, GoldDerby) drive the closing line; agent research surfaces consumer sentiment shifts handicappers miss

### Concrete plan
1. Build `polymarket_oracle` adapter:
   - Fetch event tickers via Polymarket Gamma API (https://gamma-api.polymarket.com/events)
   - Filter to Oscars
   - For each market: pull closing price 14 days before resolution + actual outcome
2. Build `film_critics` adapter:
   - For each (year, nominated film): RT critic score + audience score + Metacritic score
   - At a snapshot date (14 days before ceremony)
3. Use existing `news_scraper` for agent research per film
4. Yaml `markets/polymarket-oscars.yaml`:
   - Model A schema: numeric block (critic scores, festival wins, p_market) + calendar
   - Model B schema: A + Voyage text block over scraped commentary + agent-extracted features
5. Train both on H200, compare Brier on held-out (most recent ceremony)
6. Report

### Gotchas to watch
- Polymarket Gamma API requires no auth but rate-limits. Cache responses.
- Critic data scraping: RT has a public website but may rate-limit; use `claude -p` with WebFetch as a fallback.
- Voyage Brier improvement should be visible: critic text is dense (hundreds of reviews per film), Voyage will get real signal.

## What's worth iterating in the next session (in priority order)

1. **Finish the Oscars PoC** — get to a clean Brier A vs Brier B number. This proves or disproves the thesis.
2. **If A vs B shows positive delta** → write the LinkedIn-shippable result, frame as "agent-augmented transformer beats market on award predictions."
3. **If A vs B is null** → debug: is the agent text temporally aligned? Is the model overfitting? Are critic scores already dominant signal?
4. **`kalshi-cpi` v2** with FRED key + dated nowcast archives (Truflation, Atlanta Fed GDPNow). Macro markets are the real prize because Fed-paper validated inefficiency.
5. **Stock earnings predictions** with SEC EDGAR + dated transcripts.
6. **Multi-market pretraining** on H200 once 2+ markets clear the gate.

## Files you can delete cleanly if rebuilding

- `data/raw/<slug>/` — all caches, can re-collect
- `data/structured/`, `data/state/`, `runs/` — all derived
- Don't delete `markets/*.yaml` — those are project source
- Don't delete `src/quant/` — actual code

## Final honest take

The infrastructure built this session is real and ready: agent runner, Voyage embedder, H200 pipeline, per-event extraction, enriched football adapter, propose-schema flow. It can support ANY temporally-coherent market. The PoC just needs the right MARKET — Oscars is the cleanest available.

The lesson is methodological: **infrastructure quality doesn't matter if the experiment design is broken**. Check data temporal alignment FIRST.
