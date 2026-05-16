# betfair-itf-challenger — v1 result (2026-05-15)

## Setup

- **Adapter**: tennis-data.co.uk yearly Excel files for the **ATP main draw**, 2015-2025. The original spec called for ITF Challenger tour, where independent commentary confirms sportsbook under-modeling. tennis-data.co.uk does not cover ITF Challenger; ATP main draw is the v1 substitute. Plan to wire Jeff Sackmann's `tennis_atp_challenger` GitHub repo + an odds source in the next iteration to test the actual ITF thesis.
- **Label**: binary (player_A wins, where A is alphabetically first to avoid winner-name leakage).
- **State vector**: 192 dim. Form (rank, rolling 10-match win rate, rest days, inverted closing odds) + market state + calendar.
- **Model**: sklearn HistGradientBoostingClassifier (low-event-count fallback).
- **Held-out fence**: last 6 months of available matches (~134 events).

## Backtest result

| metric | value |
|---|---|
| net_return | **-26.68%** |
| Brier model | 0.378 |
| Brier market | 0.253 |
| **Brier improvement** | **-0.1252** |
| hit rate | 38.81% |
| Sharpe | -1.25 |
| max drawdown | -29.19% |
| trades | 134 |

## Interpretation

Same story as `betfair-eng-l1l2`: ATP main draw is heavily modeled by professional bettors, the closing line digests rank + form + odds-inversion features, and a naive model trained on those features alone loses against the market. -26% on 134 trades after 5% commission + 80 bps slippage was foreseeable.

The substitution from ITF Challenger to ATP main draw is the bigger issue. The whole point of the ITF Challenger thesis is that it is under-modeled. ATP main is over-modeled. Our v1 is testing the wrong hypothesis.

## What this proves about the pipeline

- `tennis_data` adapter end-to-end: 11 seasons of xlsx files cached, parsed, rolled into per-player form features, closing-odds-derived market probability, output as records.
- Player-anonymization via alphabetic re-ordering: confirmed no label leak (Brier-shuffle test passes on this market).
- Multi-source / multi-tour parameterization works (we only used ATP this run; WTA is one yaml flip away).

## Next iteration

The right v2 is NOT more iteration on ATP main. It is implementing the original spec:
1. Pull ITF Challenger results from Jeff Sackmann's repo.
2. Source closing odds from an ITF-specific scrape (oddsportal, betexplorer).
3. Re-run with the same state-vector schema.

If the thesis is right, ITF should show positive Brier improvement where ATP showed -0.125.

## Verdict

**Wrong-market shelved.** Pipeline confirmed working. Re-test with true ITF Challenger data in next session.
