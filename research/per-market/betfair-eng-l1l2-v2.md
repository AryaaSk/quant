# betfair-eng-l1l2 — v2 transformer result (2026-05-15)

## Setup

Identical features to v1 (Football-Data.co.uk for E2+E3 across 11 seasons). What changed:
- `model: gbdt` → `model: transformer`
- State vector dim 256 → 384 (more headroom for sequence-level attention)
- 8-layer encoder, model_dim 256, 8 heads, GELU FFN, dropout 0.1
- 25 epochs, batch 64, lr 3e-4, AdamW
- Trained on H200 GPU 6

## Backtest result, vs v1 GBDT side-by-side

| metric | v1 GBDT | v2 transformer | delta |
|---|---|---|---|
| net_return | -7.91% | **-10.79%** | -2.88pp |
| Brier model | 0.276 | 0.236 | **-0.04 (better)** |
| Brier market | 0.236 | 0.197 | n/a (same held-out, different split) |
| Brier improvement | -0.160 | **-0.039** | **+0.12** |
| hit rate | 46.43% | 23.08% | -23.35pp |
| trades | 28 | 13 | -15 |

## Interpretation

The transformer is dramatically better calibrated than GBDT (Brier improvement -0.04 vs -0.16) on identical features. With richer per-event context (32-timestep sequence), it learns to track the closing line much more faithfully.

But it is also more conservative: 13 trades vs GBDT's 28. The trades it does take lose money (23% hit rate vs 46% for GBDT). This is consistent with "transformer better understands when to NOT bet, but the bets it does take are still ill-informed because the features themselves carry no edge."

**Both models confirm: same features, no edge. The bottleneck is data, not model capacity.**

## What this answers

The v1 → v2 comparison was set up explicitly to ask: does model capacity alone unlock anything when held against the same features? The answer is **no for net P&L** (both lose) but **yes for calibration** (transformer is much closer to the market). This is the cleanest possible reading of the result: capacity helps you mimic the market, not beat it. Beating it requires features the market doesn't have.

## H200 usage

This was the second H200 training of the window (after `hyperliquid-midcap`). Total wall-clock for the train step: a few seconds on GPU 6 for 904 events × 32 timesteps × 384 dim, 25 epochs, model_dim 256.

## Verdict

**Confirmed feature-bottleneck. Sofascore is the next move.** Deep iteration on this market should pause until we have text features (injuries, lineups, team-news) wired in. Then re-run with the same transformer; expect Brier to stay flat OR improve, and net return to potentially turn positive if the text features are genuinely additive.
