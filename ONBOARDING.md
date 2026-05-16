# Onboarding

A fresh agent (or human) should be able to reproduce the leaderboard from a clean clone by following this guide.

## 1. Environment

```bash
git clone <repo> ~/Desktop/quant
cd ~/Desktop/quant
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
$EDITOR .env       # fill ANTHROPIC_API_KEY and OPENAI_API_KEY at minimum
```

## 2. Verify the pipeline works (smoke test)

```bash
pytest -v tests/test_e2e_smoke.py
```

This runs the entire pipeline on a synthetic 200-event market with a 1M-param toy transformer in under 5 minutes. If it fails, do not proceed.

## 3. Screen the candidate market universe

```bash
bin/screen-markets
cat runs/screening/leaderboard.md
```

Phase 1 ranks every market in `markets/*.yaml` by feasibility (data reachability, sample size, liquidity, info density). Markets below threshold are killed; survivors advance.

## 4. Run a single market end-to-end

```bash
bin/run-market kalshi-cpi
```

Runs Phases 2-5 for one market: collect data, lock held-out, build state vectors, train, walk-forward backtest. Writes to `runs/kalshi-cpi/<timestamp>/`.

For transformer markets, this halts before training and waits for `H200_CONFIRMED=1` env var to confirm the H200 GPU 6 is free. LightGBM markets train locally.

## 5. Compare across markets

```bash
python scripts/05_compare.py
cat runs/LEADERBOARD.md
```

Aggregates metrics from every `runs/<slug>/<timestamp>/metrics.json` and ranks markets by net profit % after costs. Each leaderboard claim is backed by a metrics file.

## 6. Iterate on winners

Top 3 markets get a second-iteration sweep: larger state vector, longer sequence length, more features. Update the yaml, re-run, compare.

## Tests must pass at each phase

- After Phase 0: `pytest tests/test_bootstrap.py tests/test_e2e_smoke.py`
- After Phase 1: `pytest tests/test_screening.py`
- After Phase 2: `pytest tests/test_collection.py`
- After Phase 3: `pytest tests/test_temporal_leak.py`
- After Phase 4: `pytest tests/test_train_smoke.py`
- After Phase 5: `pytest tests/test_backtest.py`
- After Phase 6: `pytest tests/test_leaderboard.py`

Or just `pytest -v` to run everything.

## Project sources of truth

1. The plan: `/Users/aryaask/.claude/plans/market-selection-for-a-synchronous-prism.md`
2. The architecture: `ARCHITECTURE.md`
3. The market table: `markets/*.yaml`
4. The code itself
