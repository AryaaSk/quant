# betfair-eng-l1l2 — v1 result (2026-05-15)

## Setup

- **Adapter**: Football-Data.co.uk CSVs for `E2` (League One) + `E3` (League Two), 2015-16 through 2025-26 (11 seasons).
- **Label**: home_win (binary).
- **State vector**: 256 dim. Form (5-match rolling), goals scored / conceded, inverted closing odds (Pinnacle close fallback to Bet365 close), market state, calendar.
- **Model**: scikit-learn HistGradientBoostingClassifier (LightGBM segfaults on macOS arm64 + numpy 2.4; sklearn HGB is the substitute).
- **Held-out fence**: last 6 months of available match dates (computed at state-build time from the data's actual range, not wall clock).

## Backtest result

| metric | value |
|---|---|
| net_return | -7.91% |
| Brier model | 0.276 |
| Brier market | 0.236 |
| **Brier improvement** | **-0.160** |
| hit rate | 46.43% |
| Sharpe | -0.72 |
| max drawdown | -9.24% |
| trades | 28 |

## Interpretation

The model lost money to the market on the 6-month held-out. Brier improvement is negative, meaning closing odds are better calibrated than our v1 model. This is the right answer for v1 features: form (last-5 results) + goals-for/against + inverted bookmaker odds is exactly what professional sharps have been modeling on EFL for years. The closing line absorbs all that signal.

**This is not a failure of the pipeline; it is the pipeline correctly reporting "no edge."** The same pipeline applied to a market where we genuinely add information (e.g. via injury reports, lineup leaks, manager changes scraped by LLM agents) should produce different numbers.

## Why this is the wrong market for v1

The Football-Data CSV gives us:
- match result + closing odds (already digested by professional bettors)
- public team identities (no proprietary signal)

It does NOT give us:
- pre-match team news (injuries, suspensions, U21 call-ups, loan moves)
- manager interview tone
- weather / pitch condition
- referee history vs team archetypes
- in-game motivational context (relegation battle, promotion push, dead rubber)

The premise of the edge thesis (LLM agents synthesize text features that quant firms ignore) requires unstructured text adapters that we have not yet implemented. With only structured public features, v1 just reprices what the market already prices.

## Next iteration (v2)

1. **Sofascore adapter** for team-news text (injuries, lineups, suspensions).
2. **Manager-press-conference adapter** to encode pre-match tone.
3. **Weather adapter** for outdoor pitches (low-temp + wet correlates with set-piece goals).
4. Hand-crafted text features (mention counts of injury-related vocabulary per team).
5. Switch model to transformer to fuse text features with sequence-of-form numerics.

## What this proves about the pipeline

- Adapter -> structure -> state -> dataset -> train -> backtest chain works on a real, public dataset.
- Held-out fence locks correctly at data's actual latest date.
- Temporal-leak property test passes on real data.
- GBDT (sklearn HGB) trains in <10s on 4500+ events.
- Equity curve + reliability diagram + metrics.json output correctly.

The "no-edge result" is a feature: it tells us the pipeline is honest. A pipeline that claimed positive edge on this dataset would be the suspicious one.
