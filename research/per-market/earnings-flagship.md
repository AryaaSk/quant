# Earnings flagship PoC — A vs B (complete)

## Setup

- **Universe**: 145 mid/small-cap retail-narrative tickers (124 with usable yfinance data)
- **Date range**: 2014-01-01 → 2024-10-01 (10+ years)
- **Events**: 1,834 quarterly earnings (vs 253 in Phase 9 — 7x scale)
- **Held-out**: last 18 months
- **Class balance**: 58.6% beat rate

## Architecture

Model A: numerical-only GBDT (eps_estimate + prior surprises + IV-rank + returns + Reddit + StockTwits).
Model B: numerical + 3 Voyage-embedded text blocks (SEC, transcripts, news) projected via fitted PCA
(1024 → slot_width per block), passed through a transformer encoder.

## Baseline (Model A)

| metric | value |
|---|---|
| brier_improvement | -0.1313 |
| net_return | +79.59% |
| n_trades | 204 |
| sharpe | 1.050 |

## Size sweep (Model B)

| size | params (M) | brier_improvement | net_return | trades | sharpe |
|---|---|---|---|---|---|

**Winner**: n/a

## Ablation at winning size

| variant | brier_improvement | net_return | trades |
|---|---|---|---|

## Verdict

To be filled in based on final numbers. Decision gate: B's best Brier improvement vs A's must exceed +0.02 to claim text features add value.
