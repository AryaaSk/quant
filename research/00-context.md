# 00. Context

## What this project is

A backtest-only proof of concept that selects trading markets (from a wide candidate universe), scrapes proprietary feature data via LLM agents, trains transformers on an H200, and produces walk-forward backtests optimizing net profit % after realistic execution costs.

## The three edges we are trying to exploit

1. **Information asymmetry via agent army.** LLM agents (Claude, Codex) can synthesize unstructured text (forum posts, foreign-language press, regulatory filings, expert commentary) at scale that no median bettor and few quant firms invest in for the markets we target. Most quant firms ignore markets that are too small to deploy nine-figure positions in; most retail bettors do not have the engineering capacity to build agent-scraping pipelines.

2. **GPU capacity.** A 141 GB H200 lets us train transformers large enough to actually fuse heterogeneous features (numeric time series, text-derived features, market state, calendar) rather than relying on small models that under-fit. The architecture choice itself is not the moat (time-series transformers are well known); the moat is being able to train a sufficiently capable model on the dataset we built.

3. **Niche markets.** Our liquidity floor (a few hundred to a few thousand USD per trade) opens hundreds of markets that institutional quants ignore because they cannot deploy at scale there. We are explicitly NOT competing in oil futures or BTC/ETH perps.

## What this is NOT

- Not a real-money trading system. No order routing, no live risk management, no execution monitoring.
- Not a multi-market portfolio optimizer. Each market is evaluated independently.
- Not a deep-learning research project. Architecture choices are pragmatic.
- Not coupled to Zoral. Shares the H200 box but lives in a separate repo.

## Constraints

- 3 days of H200 access (GPU 6 reserved; GPUs 6+7 are also used by Zoral continual-learning work, so H200 calls are gated).
- Public + agent-scraped data only; no paid market-data vendors.
- Backtest-only POC. Paper trading and live execution are out of scope.

## What "done" looks like

`runs/LEADERBOARD.md` ranks every market actually attempted, with net profit % after realistic costs, Brier improvement vs market closing, walk-forward Sharpe, max drawdown, and per-market notes. Every claim in the leaderboard is backed by a `runs/<slug>/<timestamp>/metrics.json`. A fresh agent should reproduce the leaderboard by following `ONBOARDING.md`.

## Sources of truth

1. The plan: `/Users/aryaask/.claude/plans/market-selection-for-a-synchronous-prism.md`
2. The architecture: `../ARCHITECTURE.md`
3. The market yamls: `../markets/*.yaml`
4. This research dir
5. The code
