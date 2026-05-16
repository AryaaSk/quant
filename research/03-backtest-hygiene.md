# 03. Backtest hygiene (non-negotiable)

This is where most aspirational quant projects die. Each rule below is enforced in code, not in convention.

## 1. Held-out period locked Day 1 of collection

`data/held_out/<slug>/MANIFEST.json` is written at first collection and refuses to be overwritten. Subsequent collect runs honor the original fence. The walk-forward backtest evaluates exactly here and the model never sees these events during training.

Implementation: `pipeline/collect.py::_write_held_out_manifest`.

## 2. Strict temporal feature filtering

Every raw record carries `source_published_at` AND `scraped_at`. At inference time T, only data with `min(published_at, scraped_at) <= T` is usable. Adapters that cannot produce both timestamps must raise loudly (`collect.py::_validate_records`).

Property-based test in `tests/test_temporal_leak.py`: for 1000 randomly sampled (event, feature) pairs, no feature value is derived from data with timestamp after the target event time.

## 3. Walk-forward retraining

For backtest period, retrain monthly (or per event for low-cadence markets). Each retrain only sees data up to that point. The simplest default (single train then sequential predict) is acceptable for v1; full walk-forward is iteration 2.

## 4. Realistic execution cost model

Per-market yaml encodes `commission_bps`, `base_slippage_bps`, `safety_margin_bps`, `kelly_fraction`, `kelly_cap`. Backtest charges these in full. A trade only opens if `|edge| > (commission + slippage + safety) / 10_000`.

For Betfair markets, commission is on profit only (not stake), so the multiplier is applied to the realized P&L not the stake. (To-do: split commission into stake-vs-profit configurations in `models/decision.py` once we have a Betfair market actually wired up.)

## 5. Calibration is the primary metric

Brier score and reliability diagram against market closing implied probability. A model can have positive Brier and still lose money; the reverse (negative Brier, positive P&L) is almost always look-ahead bias and triggers a forensic audit.

## 6. Label-shuffle sanity check

`tests/test_backtest.py::test_label_shuffle_collapses_brier_improvement` runs the entire pipeline on synthetic data with labels randomly shuffled. The backtest must produce Brier improvement near zero. If it does not, the pipeline has a leak and CI halts.

## 7. No hyperparameter tuning against held-out

All sweeps use train + val from the pre-held-out window. Held-out is touched exactly once per market, at the end. Multiple runs against the same held-out window are allowed only when the change being tested is documented in `research/per-market/<slug>.md` as an iteration.

## 8. Realistic position sizing

Fractional Kelly (default 25% of full Kelly) with hard cap (default 2% of bankroll). Kelly is sensitive to model overconfidence; the safety margin is intentional.

## 9. Survivorship bias awareness (crypto)

Crypto Tier-C markets are the worst offender. Many "winning" backtests in published research used the current top-100 token universe, which dropped 58% of tokens that delisted post-2022. Our Tier-C scope is limited to currently-listed mid-caps; we acknowledge this is a survivorship-biased subsample and discount expected live performance accordingly (Coinbase research suggests 30-50% reduction is realistic).
