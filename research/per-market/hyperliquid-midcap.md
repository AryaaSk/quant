# hyperliquid-midcap — v1 result (2026-05-15)

## Setup

- **Adapter**: Hyperliquid public API (`api.hyperliquid.xyz/info`). Pulled hourly OHLCV + funding rate for SOL perp over the last ~7 months (capped at 5000 candles per API call; pagination needed for longer history, see "Open issues" below).
- **Label**: binary (next 4h forward return > 0).
- **State vector**: 256 dim. Numeric block (close, log_return, volume, funding_rate, coin_id z-scored over 168h rolling window) + market state (level + velocity at lags 1/4/24h) + calendar.
- **Model**: time-series transformer trained on H200 GPU 6, 8 layers, model_dim 256, 8 heads, 12 epochs, AdamW. Trained from scratch (no pre-training).
- **Held-out fence**: last 1 month (~734 events post-event-extraction).

## Backtest result

| metric | value |
|---|---|
| net_return | **-14.66%** |
| Brier model | 0.2517 |
| Brier market (0.5 prior) | 0.2505 |
| **Brier improvement** | **-0.0012** |
| hit rate | 48.91% |
| Sharpe | -0.76 |
| max drawdown | -40.26% |
| trades | 734 |

## Interpretation

The model is essentially indistinguishable from a 50/50 prior (Brier improvement near zero). The negative return comes from commission (5 bps) + slippage (25 bps) on 734 trades = ~22% in costs alone before any directional edge. Hit rate of 48.91% is also coin-flip.

This is the most honest "structured-features-only crypto has no edge" result we could ask for. Pure OHLCV + funding rate at hourly resolution is the most-modeled signal in crypto. Our pipeline does not produce a magical alpha when none exists.

## Why this is the wrong market for v1

We pulled price + volume + funding rate. We did NOT pull:
- Discord / Telegram / Twitter sentiment per coin
- On-chain whale flows
- GitHub commit velocity
- Cross-exchange basis / open interest deltas
- Token-unlock event schedules

The edge thesis for crypto is sentiment + on-chain flows feeding a transformer. We haven't tested that thesis. We tested "transformer on raw price data" which is widely known to lack edge.

## What this proves about the pipeline

- The H200 round-trip works: `bin/h200 sync && bin/h200 run scripts/03_train.py hyperliquid-midcap --confirm-gpu-free --device cuda` trains successfully, checkpoint pulls back, local backtest runs.
- Hourly time-grain works (the `structure.py` fix to honor `time_grain: hourly` lands here).
- Transformer training on 4500 events x 48 timesteps x 256 dim completes in seconds on the H200.

## Open issues

1. Only 7 months of SOL data because the Hyperliquid API returned 5000 candles per call and our pagination logic exited early. Fix: paginate by requesting smaller windows iteratively. Would yield ~12+ months of data.
2. Single coin only (SOL). To use the full universe, the pipeline needs per-(coin, time) row support in structure.py rather than per-time pivoting. Cleanest path: one yaml per coin.
3. No sentiment / on-chain data. Implementing those is the next-iteration priority.

## Verdict

**Shelved at v1.** Negative result is expected without text/sentiment features. To revisit, layer in scraped Discord/Twitter sentiment via a news_scraper adapter and on-chain whale-flow data via Etherscan / Solscan APIs.
