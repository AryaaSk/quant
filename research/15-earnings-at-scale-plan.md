# 15. Earnings-at-scale execution plan (next session)

The flagship PoC. Scales the current 253-event earnings dataset 30-50x using only free data sources, runs a real transformer on H200, and gives Voyage-embedded SEC + transcripts + news the dataset size it needs to actually contribute.

Constraint check that drives the design:
- All text sources are open and free (SEC EDGAR, Motley Fool transcript mirror, Yahoo Finance, Reddit, StockTwits) — no paid odds, no bot-protected platforms
- Claude Max ($200/mo) covers the scrape budget completely
- H200 GPU justifies a transformer with ~50-150M params
- Deployment is via any global brokerage; no geo-fence

## Target dataset

| dim | current | target |
|---|---|---|
| tickers | 24 mid/small-cap | 200 mid/small-cap retail-narrative |
| date range | 2022-2024 (3 yrs) | 2014-2024 (10 yrs) |
| events per ticker | ~10 quarters | ~40 quarters |
| **total events** | **253** | **~8,000** |
| held-out | 28 (last 6 mo) | ~600 (last 18 mo) |
| text per event | 30 articles | 10-Q + transcript + 30 articles |
| text tokens per event | ~5,000 | ~80,000 |
| state-vector text block | 240 slots | 1024 slots |
| model | GBDT | transformer 50-150M params |

8,000 events × 1,024 text dims is finally in the safe regime (8:1 ratio) for the embedding block. And 80k tokens per event means Voyage-3-large has enough material to extract real signal, not just article skim.

## Phase 1: Deterministic scaffolding (no claude, no codex, 3-4 hrs)

All free; can be done with claude usage at 0%.

### 1a. Expand `stock_earnings` adapter

- Bump default `tickers` list to 200 mid/small-cap retail-narrative names. Universe selection criteria:
  - Market cap $200M-$5B (small + mid)
  - Retail-narrative-driven (subreddit > 50k subs, Twitter buzz, or meme history)
  - 10+ quarters of earnings data available on yfinance
  - Mix sectors: AI/data, EV, crypto miners, fintech, meme, cannabis, space, biotech (FDA exposed), small-cap industrials
- Extend window to 2014-01-01 → 2024-10-01 (allow held-out to be the last 18 months: 2023-04 → 2024-10)
- Add per-event option-implied move (yfinance options chain) as an additional feature

Files: `src/quant/adapters/stock_earnings.py` (extend DEFAULT_TICKERS + date range), `markets/earnings-flagship-A.yaml` (new), `markets/earnings-flagship-B.yaml` (new).

### 1b. Build `sec_filings` adapter

SEC EDGAR full-text search is open at `https://efts.sec.gov/LATEST/search-index`. For each `(ticker, earnings_date)`:
- Fetch the most recent 10-Q with `filed_at < earnings_date` (the prior quarter's filing)
- Fetch any 8-K filings in the 90 days before earnings (often contain forward-looking commentary)
- Cache raw HTML + extract clean text (BeautifulSoup; strip XBRL noise)

Output: `data/raw/earnings-flagship-B/sec_filings.parquet` with columns `ticker, earnings_date, filing_type, filed_at, text, source_url`. ~16,000 filings (200 tickers × 40 quarters × 2 filing types).

Cost: free. Estimated wall time: 2-3 hours with rate-limit politeness (SEC asks for 10 req/s and a User-Agent header).

Files: `src/quant/adapters/sec_filings.py` (new), `src/quant/adapters/__init__.py` (register).

### 1c. Build `earnings_transcripts` adapter

Motley Fool publicly archives earnings call transcripts at predictable URLs. For each `(ticker, earnings_date)`:
- Construct candidate URLs (Motley Fool, SeekingAlpha public mirror, Investing.com)
- Fetch the most recent transcript with `published_at < earnings_date` (the PRIOR quarter's call — encodes forward guidance for the current quarter)
- Extract speaker-tagged dialogue

Output: `data/raw/earnings-flagship-B/earnings_transcripts.parquet`. ~8,000 transcripts.

Files: `src/quant/adapters/earnings_transcripts.py` (new).

### 1d. Build `social_sentiment` adapter (Reddit + StockTwits)

For each ticker × week (per-ticker rolling window of mention counts + comment sentiment):
- Reddit: r/wallstreetbets, r/CryptoCurrency, r/stocks, r/investing, ticker-specific subreddits — via PRAW (free)
- StockTwits: free API, 200 calls/hour
- Output per ticker × earnings_date: number of mentions, mean sentiment, bullish/bearish ratio

Output: `data/raw/earnings-flagship-B/social_sentiment.parquet`. Numeric features per event.

Files: `src/quant/adapters/social_sentiment.py` (new).

### 1e. Verify temporal-leak property test passes on the new dataset

The existing `tests/test_temporal_leak.py` is a Hypothesis property test. It must pass on the new dataset before any text feature is trusted.

## Phase 2: Numerical baseline at scale (no agents, GBDT, ~1 hr)

Train `earnings-flagship-A` (numerical features only, GBDT) on the full 8,000-event dataset.

Features (96 slots, same shape as current A but on bigger data):
- EPS estimate, prior-4q surprise mean, prior-4q beat rate, days since last earnings
- Realized vol 30d, returns at 7/30/60/90/180d, sector ETF returns at 30/90d
- New: implied move from options chain, short interest, days-to-cover, IV rank
- New: social sentiment baseline (mean Reddit mentions + StockTwits bull ratio)

Expected output: a baseline that's stronger than the current A (more data = better-calibrated GBDT). Should land at Brier improvement +0.01 to +0.05 over the historical beat-rate prior.

Files: `markets/earnings-flagship-A.yaml` (new), `runs/earnings-flagship-A/<ts>/`.

## Phase 3: Text scraping at scale (~$100-150 claude budget, 4-6 hrs wall time)

For `earnings-flagship-B`, run haiku scrapers with per_entity_mode keyed by `ticker × quarter`. For each (ticker, quarter) bucket:
- Pull the 30 most-relevant news articles dated 7-30 days before earnings
- Article filters: real publish date present, English, source domain in allowlist (Yahoo, MarketWatch, Bloomberg, Reuters, CNBC, WSJ, FT, Seeking Alpha, Motley Fool, Benzinga, Barron's)
- Output: `data/raw/earnings-flagship-B/_news/<ticker>/<quarter>/article_*.json`

Total scrapes: 200 tickers × 40 quarters = 8,000 haiku calls × ~$0.03 each = **~$240 claude**.

That's at the edge of monthly budget. Two ways to reduce:
- **Per-ticker scrape** (not per-quarter): one haiku covers a ticker's last 10 years, returns articles tagged by which earnings they're for. 200 calls × ~$0.50 each = **$100**. Lower precision but cheaper.
- **Per-quarter scrape but with smaller article cap** (15 instead of 30): halves cost to **$120**.

Recommendation: start with per-ticker at $100, escalate to per-quarter if precision matters for the result.

Files: `scripts/02_collect.py earnings-flagship-B` (existing infra; just runs longer).

## Phase 4: Voyage embedding + state build (~1-2 hrs, ~$10-20 voyage)

State vector design for `earnings-flagship-B` (target dim 1664):

```yaml
state_vector:
  dim: 1664
  blocks:
    - {name: numeric_features, slots: "0:128",   builder: numeric, ...}     # all 1a features
    - {name: social_numeric,   slots: "128:160", builder: numeric, ...}     # Reddit + StockTwits
    - {name: market_state,     slots: "160:192", builder: market_state, ...} # price + velocity
    - {name: text_10q,         slots: "192:704",  builder: text, params: {text_column: text__sec, mode: voyage, window_days: 90}}
    - {name: text_transcript,  slots: "704:1216", builder: text, params: {text_column: text__transcript, mode: voyage}}
    - {name: text_news,        slots: "1216:1632", builder: text, params: {text_column: text__news, mode: voyage, window_days: 30}}
    - {name: calendar,         slots: "1632:1664", builder: calendar, params: {}}
```

Three separate text blocks because SEC filings, transcripts, and news encode different signals. The transformer's attention can learn relative weighting between them per event.

Voyage cost: 8,000 events × 3 text blocks × ~5,000 tokens averaged each = ~120M tokens × $0.06/1M = **~$7**. Cached by sha256, so reruns are free.

Files: `scripts/02_collect.py earnings-flagship-B` (existing infra), `markets/earnings-flagship-B.yaml`.

## Phase 5: Transformer training on H200 (~2-3 hrs)

Architecture for `earnings-flagship-B`:
- Input: sequence of state vectors `(batch, seq_len=12, state_dim=1664)`. seq_len=12 covers the last 12 quarterly snapshots of the ticker (3 years of history per event).
- Input projection: linear 1664 → 768
- Encoder: 12 layers, d_model 768, 12 heads, FFN ratio 4, RoPE, FlashAttention-2
- Total params: ~85M. Comfortable on a 141GB H200.
- Output head: sigmoid scalar (P(beat))
- Loss: binary cross-entropy
- Training: AdamW lr=2e-4 cosine warmup, 40 epochs, batch 32, val fraction 0.15
- Held-out: last 18 months (2023-04 → 2024-10), strict temporal split

Expected wall time on H200: 60-90 min for the full transformer; another 30 min for backtest + report. Total H200 budget: **~2 hours of one GPU**.

Files: `bin/h200 run scripts/run_remote_pipeline.py earnings-flagship-B --device cuda`, `markets/earnings-flagship-B.yaml`.

## Phase 6: Backtest + comparison + per-market doc (~1 hr)

Walk-forward backtest with proper retraining every 6 months. Compare A (GBDT numerical-only) vs B (transformer numerical+text):

| metric | A target | B target | gate to clear |
|---|---|---|---|
| Brier improvement | +0.01 to +0.05 | A + 0.02 | B > A AND B > 0 |
| Net return after costs | +20% to +100% | A or better | B ≥ A |
| Sharpe | > 1.0 | > 1.5 | B > A |
| Calibration (reliability plot) | flat | flat | both pass |

**Decision rule:** If B clears the +0.02 Brier gate over A, the methodology is proven. Write the result up, ship a research note, and move to either (a) live paper-trading deployment, or (b) Phase 7 below.

## Phase 7 (conditional): Multi-market pretraining (next-next session)

If Phase 6 succeeds: pretrain a unified backbone transformer on (earnings + polymarket-entertainment + crypto-perps) for transfer learning. ~200k events total, ~250M params. This is the actual flagship deliverable.

## Total cost + time budget

| phase | claude $ | voyage $ | H200 hours | wall time |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 3-4 hrs (deterministic, can be overnight) |
| 2 | 0 | 0 | 0 (local CPU) | 1 hr |
| 3 | $100-150 | 0 | 0 | 4-6 hrs (haiku scrapes in parallel) |
| 4 | 0 | $7-20 | 0 | 1-2 hrs (Voyage parallel) |
| 5 | 0 | 0 | 2 | 2-3 hrs |
| 6 | 0 | 0 | 0.5 | 1 hr |
| **Total** | **$100-150** | **$7-20** | **2.5** | **~12-17 hrs (mostly parallel)** |

That's ~$120-170 in API spend and one good working session of H200 time. Fits comfortably in your $200/mo Claude Max + H200 spot quota.

## Critical files to create (next-session checklist)

```
src/quant/adapters/stock_earnings.py             extend: 200-ticker default, 2014-2024 window, options-IV features
src/quant/adapters/sec_filings.py                new: SEC EDGAR client + HTML→text + per-event 10-Q + 8-K fetch
src/quant/adapters/earnings_transcripts.py       new: Motley Fool / SeekingAlpha mirror scraper
src/quant/adapters/social_sentiment.py           new: Reddit (PRAW) + StockTwits (free API) per-week aggregates
src/quant/adapters/__init__.py                   register sec_filings, earnings_transcripts, social_sentiment
markets/earnings-flagship-A.yaml                 new: 8k events, numerical-only GBDT
markets/earnings-flagship-B.yaml                 new: 8k events, numerical + 3 text blocks, transformer 85M params
scripts/03_train.py                              extend: --model transformer flag (probably already exists; verify)
tests/test_sec_filings.py                        new: mocked smoke test
tests/test_earnings_transcripts.py               new: mocked smoke test
tests/test_temporal_leak.py                      verify it passes on the new dataset
research/per-market/earnings-flagship.md         new: post-run write-up with A vs B table
runs/LEADERBOARD.md                              updated by 05_compare.py
```

## What this phase explicitly does NOT do

- No NBA player props (no paid odds, no X API)
- No crypto Discord/Telegram scraping (bot-protected)
- No live trading
- No Polymarket entertainment data (parking that one until we fix the lastTradePrice contamination issue separately)
- No multi-market pretraining (Phase 7, only if Phase 6 advances)

## Why this is the right next session

This is the cheapest, cleanest, most defensible PoC that genuinely uses both unique advantages — Claude Max for sustained agent-driven scraping of dense free public text (SEC + transcripts + news), and H200 for a real transformer at a scale where the math finally favors text features over numerical baselines. Every input is on the open internet, every output is a reproducible artifact, deployment is one brokerage account away. If text features beat numerical-only on this dataset, the methodology is proven at scale. If they don't, we've ruled out the "agent-army + text" thesis under the cleanest possible conditions and learned something real.
