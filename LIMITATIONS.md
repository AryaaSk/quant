# Limitations

A research-honesty document. Read this before drawing trading conclusions from any backtest result in this repo. Written 2026-05-16 after discovering the issues below while reviewing flagship-B results with a friend who knows the space.

## tl;dr

The `earnings-flagship-B` backtest produced a **705% return with Sharpe 2.42 over 18 months held-out** — but several caveats make the headline number much less impressive than it sounds.

The single most important: **the trained transformer collapsed to a constant predictor**. The model outputs the same probability (~0.56) for every event regardless of ticker, SEC filing content, transcript, news, etc. The 705% return came from a single hand-crafted feature (`beat_rate_prior_4q`) interacting with Kelly sizing — essentially a mean-reversion bet on per-ticker earnings momentum.

The 117M-parameter transformer is not extracting signal from the agent-scraped text. It's outputting a constant. The size-sweep finding ("models from 300K to 203M return identical results") is consistent with this — they all converge to the same constant prediction.

This is documented honestly here so future work can target the actual problems.

## 1. The transformer collapsed to a constant predictor

**Symptom**: every trade in `runs/earnings-flagship-B-117M/<ts>/trades.parquet` has `p_model = 0.560775`. The model never varies its output by ticker, by feature input, by event.

**Why this happened**:
- Cross-entropy loss is minimized by predicting class frequency when features don't separate classes well
- Our features (returns, vol, Reddit counts, Voyage-embedded SEC filings, transcripts, news) don't strongly discriminate at the per-ticker level
- The minimum-loss solution to noisy binary data is "predict the population mean" — exactly what 0.56 represents (our beat rate was 58.6%)

**Why bigger models don't fix it**: a 1B-param model would also output the same constant. Capacity isn't the bottleneck; signal quality is. The model has plenty of room — it just can't find structure in the features that beats the global mean.

**Why we still got 705% net return**:
- The backtest's decision rule uses `|p_model - p_market| > safety_margin` where `p_market` is the per-ticker historical 4-quarter beat rate
- With constant `p_model = 0.56` and varying `p_market` (0.10, 0.25, 0.50, 0.60, 0.67, 0.75, 0.90), the model takes:
  - Long-beat bets on tickers with low historical beat rates (assumes mean-reversion up)
  - Short-beat bets on tickers with high historical beat rates (assumes mean-reversion down)
- This is essentially a mean-reversion strategy on the `beat_rate_prior_4q` feature, leveraged via Kelly sizing
- The "alpha" came from this single hand-crafted feature, not from the transformer architecture or the scraped text

## 2. Drawdown is high for the sample size

- Max drawdown across the held-out window: **-30.74%**
- Trades: 206 over 18 months
- Sharpe standard error at n=206: roughly ±0.5-0.7
- True Sharpe with 68% confidence: somewhere in [1.7, 3.1]

**What this means in practice**: a real trader running this strategy with $10K starting capital would have seen their equity drop from $13K down to $9K (below initial bankroll) around trade #57-81 before recovering. Most retail or even professional accounts would have stopped the strategy somewhere in that drawdown trough. A 30% drawdown on a small sample is too high for the Sharpe ratio to be reliably reproducible in live trading.

Pro firms typically cap drawdown at 10-15% via dynamic position sizing, regime detection, and explicit stop-out rules. Our backtest has none of these — full Kelly sizing fractional at 20%, with a 2% cap per trade, but no portfolio-level drawdown control.

## 3. Beat/miss is not a tradeable instrument

The model predicts "will reported EPS exceed analyst consensus estimate" — a binary event. **This is not directly tradeable anywhere**:

- Polymarket and Kalshi do not list "will AAPL beat consensus this quarter" markets
- The 705% return is computed against a SYNTHETIC counterparty paying 1/`p_market` decimal odds
- No actual exchange would offer those odds

To capture this signal in real money, you'd need a second translation layer:
- Long the stock pre-earnings on high-conviction beat predictions (adds price-direction risk)
- Buy options straddles (adds vol-pricing risk, theta decay)
- Pair-trade against a sector ETF (still has correlation risk)

Each of these has its own complications and execution costs the backtest doesn't model. The honest claim is "binary forecaster," not "trading strategy."

## 4. Universe contains micro-caps with real execution risk

The 145-ticker universe includes some sub-$100M market cap names (MULN, WKHS, GOEV, microbiotechs like OCGN/SAVA/BNGO at various points). On these:

- Bid-ask spreads can be 1-3% routinely
- The backtest applies a 1.5% safety margin and 30 bps slippage, which **does not cover** 1-3% spreads
- Real execution on a micro-cap would consume meaningful chunks of edge
- For a real deployment, the universe should be filtered to >$1B market cap names where slippage assumptions hold

The larger names in our universe (GME, AMC, COIN, RIVN, SOFI, PLTR-like ones) are genuinely tradeable. The micro-caps inflate the apparent edge.

## 5. The 612 daily-collapsed events vs 1834 raw events

The `stock_earnings` adapter emits 1,834 events. After `structure.py` resamples to a daily index, multiple events on the same calendar day collapse into one row, reducing usable events to 612. This means:

- We trained on ~412 events (train+val split)
- Held-out on ~206 events
- The full 1,834-event dataset is sitting on disk unused

Fix: switch `time_grain` from `daily` to `event`-level granularity. Estimated 3x dataset multiplier with zero new scraping. This is the cheapest, most impactful next step.

## 6. The agent-scraped text didn't contribute (at this scale)

Ablation evidence:
- Removing the entire `text_sec` block (203M model, no SEC filings) → net_return +706.45%, Sharpe 2.42, basically identical to the full 203M baseline
- Size sweep across 4M-203M parameters all converge to the same constant predictor
- Per-event text coverage was 22% for news (78% of events feed zero embeddings into the text_news block)

The Voyage-embedded text features were sparse, redundant with numerical features, or both. The "agent army scraping for alternative data" thesis is **untested** at this dataset size — the text blocks needed denser per-event coverage (every event with >5 articles, every event with a 10-Q in window, etc) AND features that aren't already priced into analyst consensus.

## 7. Survivorship bias in the ticker universe

The 145-ticker list was selected in 2026 with knowledge of which companies survived 2014-2024:

- BBAI is in the universe because BigBear.ai survived to 2024
- ~21 tickers were dropped because yfinance couldn't find prices (likely delisted post-selection)
- We do not include tickers that went private, bankrupt, or were acquired during the window

This is a real but bounded bias. The universe does include companies that performed poorly (RIVN -85% from IPO, LCID -90%, GME volatility, AMC dilution) so it's not pure winner selection. But strictly, this is not a survivorship-clean universe.

## 8. What would actually validate the methodology

To turn this from "interesting infrastructure demo" into "real trading edge claim," in order of impact:

1. **Add per-ticker analyst-revision features**. Number of analysts revising estimates UP vs DOWN in 30 days pre-earnings. Known to predict beat/miss in academic literature. Free from Zacks/Estimize historical archives.
2. **Add options-implied move feature**. Front-month straddle premium normalized by current price = the market's expected magnitude. yfinance has the options chain.
3. **Force per-ticker variation in the loss**. Train the model to predict `(p_observed - p_prior_per_ticker)` rather than absolute probability. The model can no longer trivially collapse to the global mean.
4. **Fix the daily-collapse bug** so we train on 1,834 events instead of 612.
5. **Expand to 500+ tickers × 2014-2024**. With per-ticker variation in the loss + actual discriminating features, the larger dataset can support a real transformer.
6. **Drawdown control**. Cap portfolio-level drawdown at 15% via dynamic position sizing. The 30% drawdown is non-survivable for most real accounts.
7. **Translate to a tradeable instrument**. Convert beat/miss probability into option straddle or long-stock decisions with explicit slippage/execution modeling.

With items 1-3 alone, the model has a real chance of producing differentiated per-ticker predictions, which would be the actual test of "did agent-scraped + Voyage-embedded text features beat numerical-only?" — the question the current backtest cannot answer because the model never escaped predicting the global mean.

## 9. The size-sweep finding still holds (but differently)

The headline finding "models from 300K to 203M parameters all return Sharpe 2.42" is true, but the interpretation needs revising:

- **Original interpretation**: "information is the bottleneck — even big models can't extract more signal than is present in the features."
- **Corrected interpretation**: "all models, regardless of size, collapsed to the same constant predictor (0.56). The 'identical results' aren't from different models extracting the same signal — they're from different models reaching the same degenerate solution."

Both interpretations agree that more parameters won't help. They differ on WHY: not "exhausted signal" but "no per-event signal extractable from current features."

## 10. What's still real

For honest balance, what the repo IS:

- A working end-to-end pipeline: 145-ticker × 10-year universe collection, SEC EDGAR + Motley Fool + Reddit scraping, Voyage AI embedding, PCA projection, transformer training on H200, Kelly-sized backtest with realistic costs, label-shuffle and temporal-leak property tests
- A reproducible artifact: every result can be regenerated from the committed repo
- A non-trivial 705% backtest P&L on 18 months held-out (real, but driven by a single feature + Kelly sizing, not the transformer)
- An interesting null result: across 10 model sizes spanning 3000x parameter range, the architecture made no measurable difference
- Infrastructure that, with the fixes above, could plausibly produce a real result on a larger and feature-richer dataset

What it is **not**: a validated agent-army-extracts-text-alpha methodology. That claim awaits the work in section 8.

---

If you take one thing from this document: **the headline 705% return is real, but the headline story ("a transformer extracts signal from agent-scraped SEC filings + transcripts + news") is not what the model actually did**. Treat this repo as an infrastructure artifact with one strong baseline finding (mean reversion on beat_rate_prior_4q + Kelly sizing) and a roadmap of fixes, not as a validated trading edge.
