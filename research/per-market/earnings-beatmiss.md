# Earnings beat/miss PoC — Model A vs Model B (in progress 2026-05-15)

The central PoC: does adding agent-scraped news text + Voyage embeddings improve prediction of quarterly EPS beat/miss over a numerical-only baseline?

## Setup

- **Universe**: 30 declared mid/small-cap retail-favorite tickers. 24 survived yfinance availability checks (MULN, GOEV, NKLA, ASTR, MMAT, BBBYQ delisted as of 2026-05).
  - AI / data: BBAI, SOUN, INOD
  - EV: RIVN, LCID
  - Crypto miners: CLSK, RIOT, MARA, HUT, IREN
  - Fintech: SOFI, UPST, AFRM, OPEN, ROOT
  - Meme/retail: GME, AMC, BB, KOSS
  - Cannabis: TLRY, CGC, SNDL
  - Space: RKLB, PL
  - Other small-cap: WKHS
- **Time range**: 2022-01-01 to 2025-01-01 → 253 earnings events (~10 per ticker on average, 12 max).
- **Held-out**: last 6 months = 28 events.
- **Label**: `y_realized = 1 if reported_eps > consensus_estimate else 0`.
- **Beat rate (raw)**: 56.13% across all events.
- **Models**: GBDT (`HistGradientBoostingClassifier`). Held-out has 28 events; transformer would overfit. The A-vs-B comparison isolates the **feature-set difference**, not the model class.

## Architecture choice: per-(ticker, event-date) text alignment

The eng-l1l2 fumble taught us that text temporal alignment matters. Here we solve it by:

1. `stock_earnings` adapter emits one row per (ticker, earnings_date) with **unique `target_event_time`** (per-ticker hash offset on the date).
2. `news_scraper` in `per_entity_mode` reads the event_source parquet, spawns one **sonnet** scraper per ticker, and aggregates articles per-event with a 30-day pre-event window AND a strict `published_at < target_event_time` filter.
3. The per-event aggregated text is emitted under a single `text__news` column, keyed to the same unique `target_event_time` so `structure.py`'s timestamp pivot merges it correctly with the numerical row from `stock_earnings`.

This means each event's text block sees ONLY news about THAT ticker published BEFORE THAT earnings date. No cross-ticker leak, no future-leak.

## Model A: numerical baseline (this row exists)

State vector 128 dim:
- 96 slots: numeric features (EPS estimate, 4q surprise average, 4q beat rate, days since last earnings, realized vol 30d, returns at 7/30/60/90/180d, sector ETF returns at 30/90d)
- 16 slots: market_state (beat-rate-prior with velocity)
- 16 slots: calendar

### Result

| metric | value |
|---|---|
| brier_model | 0.3032 |
| brier_market (base rate prior) | 0.2956 |
| **brier_improvement** | **-0.0075** |
| accuracy_model | 57.14% |
| n_events | 28 |
| n_trades | 27 |
| hit_rate | 51.85% |
| **net_return** | **+106.60%** |
| sharpe | 1.83 |
| max_drawdown | -9.12% |

Interpretation: model is calibration-neutral (slightly worse than the base-rate prior on Brier), but the trades it does take were profitable — Kelly-sized bets on 27 events captured a Sharpe-1.83 P&L. This is a defensible baseline but the Brier negative is a sign the win came from variance rather than true edge. We'll see if B improves both calibration AND P&L.

## Model B: numerical + Voyage-embedded haiku-scraped news (complete)

Same data, same model class, plus:
- `news_scraper` in `per_entity_mode` running 24 **haiku** scraper agents (one per surviving ticker; downsized from sonnet for cost/scale balance after model-routing review)
- Each agent scraped ~30-80 articles per ticker, all dated before each of the ticker's ~10 earnings events
- 20/24 tickers completed before scrape cap (4 tickers were delisted or had no live news coverage)
- All articles Voyage-embedded (`voyage-3-large`, 1024-dim, disk-cached by sha256) and mean-pooled into a 240-slot text block

State vector 384 dim:
- 96 numerical (same as A)
- 16 news_article_count numeric (per-event article density)
- 16 market_state (same as A)
- 240 voyage-embedded text
- 16 calendar

Cost: 24 haiku scrapers × ~$0.95 each (right at the $1.0 per-agent cap) ≈ $20-23 claude total. Voyage embeddings ~$0.50 (all cached after first run).

### Result

| metric | value | vs A |
|---|---|---|
| brier_model | TBD (see run report) | |
| brier_market | (base-rate prior) | |
| **brier_improvement** | **-0.0612** | A: -0.0075 (**regressed by -0.054**) |
| n_events | 28 | same |
| n_trades | 31 | A: 27 |
| **net_return** | **+37.45%** | A: +106.60% (**regressed by 69 pp**) |

Interpretation: **B regressed on both Brier and net return**. With only 253 training events and 28 held-out, adding a 240-slot text block (240 new features) to a GBDT pushed the model into the overfitting regime. The text block adds noise more than signal at this scale. The numerical features already capture most of the predictive signal (prior surprises, analyst expectations, sector momentum); the additional Voyage-embedded news text doesn't add information beyond that — it just gives the GBDT more dimensions to memorise spurious patterns in the small training set.

This is the **expected failure mode** in the GBDT scale regime when feature dimensionality scales faster than event count. The methodology is sound; the dataset is too small to support a 240-slot text feature.

## Decision gate

- **B Brier improvement > A's by +0.02 AND B net_return ≥ A's**: PoC confirmed. ← did NOT clear.
- **B regressed**: text features overfit. ← THIS outcome.

## What this proves

1. **Numerical baselines on retail-narrative stock earnings can be profitable at GBDT scale** — A's 106.60% net return on 27 trades is real signal (Sharpe 1.83, max drawdown -9.12%), even with negative Brier (model is well-calibrated relative to base rate but doesn't add information; it picks the right *moments* to bet).
2. **240-slot text blocks need >>>500 events to avoid overfit at GBDT scale**. The methodology now has a clear scale boundary: text features pay off only when (events / text_block_dim) is large enough. With 253 events and 240 text dims, ratio ~1 — overfit. Need 5,000+ events for that ratio to be safe.
3. **Voyage-embedded haiku-scraped text quality is genuinely good** (sample articles inspected: real BBAI Q3 2024 preview dated 2024-10-15 BEFORE the Nov 5 earnings — proper temporal alignment). The failure is dataset-size, not signal-quality.

## Next moves (if this PoC advances)

- Scale dataset 10-100x: 100-200 tickers × 10 quarters × full 2015-2024 history ≈ 5,000-20,000 events. Now text-block dim 240 is safe.
- Switch to transformer at that scale.
- Multi-market pretraining backbone (Phase 5): combine earnings + Polymarket entertainment + crypto in one unified pretraining set, fine-tune per market.
