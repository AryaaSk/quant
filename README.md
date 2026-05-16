# quant

Multi-market transformer trading POC. Hand-designed state vectors, agent-driven data scraping, walk-forward backtests with realistic execution costs.

## What this is

For each candidate market (Kalshi, Polymarket, Betfair Exchange, Hyperliquid, ...) we:

1. Scrape rich features via LLM agents into a structured store.
2. Compose a per-market state vector (named numeric blocks + market state + text features + calendar) using `state/composer.py`.
3. Train a time-series transformer (or LightGBM fallback) over sequences of those state vectors.
4. Walk-forward backtest with commission and slippage from the market yaml.
5. Rank every market by net profit % after costs in `runs/LEADERBOARD.md`.

The transformer is NOT an LLM. It is a time-series transformer over our hand-designed state vectors. See `ARCHITECTURE.md`.

## Quick start

```bash
# 1. Create env and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy env template and fill keys
cp .env.example .env
$EDITOR .env

# 3. Verify pipeline end-to-end on synthetic data
pytest -v tests/test_e2e_smoke.py

# 4. Run market screening across the candidate universe
bin/screen-markets

# 5. Run a single market end-to-end (collect, structure, state, train, backtest, report)
bin/run-market kalshi-cpi

# 6. Compare across markets after multiple runs
python scripts/05_compare.py
cat runs/LEADERBOARD.md
```

## Layout

```
bin/                deploy + run helpers
markets/            one yaml per candidate market (data sources, label, state schema)
src/quant/          importable package
  pipeline/         collect, structure, state, dataset, train, backtest, report
  state/            block builders: numeric, market_state, text, calendar, composer
  adapters/         one module per data source (Kalshi, FRED, Betfair, ...)
  models/           transformer, LightGBM, decision rule, Kelly sizing
  eval/             metrics, plots
tests/              full coverage including property-based temporal-leak test
scripts/            CLI entry points 00_bootstrap.py through 05_compare.py
data/               gitignored: raw, structured, state, held_out
runs/               training output + LEADERBOARD.md
```

## H200 (Phase 4)

Training is gated on H200 GPU 6 being free (shared with Zoral). `bin/h200 --check` validates config offline. `bin/h200 sync && bin/h200 run scripts/03_train.py kalshi-cpi` deploys and trains. Phase 4 will not be invoked until the user confirms GPU 6 is free.

## Tests

```bash
pytest                                  # all tests
pytest tests/test_temporal_leak.py      # property-based leak prevention
pytest tests/test_e2e_smoke.py          # full pipeline on 200-event synthetic market
```

Every pipeline stage has a test. The label-shuffle sanity check in `tests/test_backtest.py` catches subtle data leaks: shuffling labels must collapse Brier improvement to zero. If it does not, the pipeline halts.

## Documentation

- `ARCHITECTURE.md`: pipeline diagram, state-vector design, why each stage exists
- `ONBOARDING.md`: walk a fresh agent through reproducing the leaderboard
- `markets/<slug>.yaml`: per-market thesis in the `notes` field
