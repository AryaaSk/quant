# 05. Phase 1 screening result (2026-05-15)

## What ran

`scripts/01_screen_markets.py` ranks all 18 market yamls by feasibility. Only markets whose adapters are implemented can pass reachability and advance.

## Outcome

See `runs/screening/leaderboard.md` and per-market `runs/screening/<slug>.json`. Summary:

| Status | Markets |
|---|---|
| **Advance** (sources reachable) | `_synthetic`, `betfair-eng-l1l2` |
| **Stub-blocked** (adapter not implemented) | the remaining 16 |

`_synthetic` exists for the smoke test only. The first real market to advance is `betfair-eng-l1l2` because we implemented the `football_data` adapter.

## Adapters implemented in this iteration

- `synthetic`: deterministic generator for CI / smoke test.
- `fred`: FRED API client with disk cache. Free key required (env `FRED_API_KEY`).
- `football_data`: Football-Data.co.uk CSV downloader. No auth. 11+ seasons cached locally.

## Adapters still stubbed

| name | source | rough effort |
|---|---|---|
| `kalshi_contracts` | Kalshi REST API + PredictionData.dev archive | 1-2 hrs |
| `polymarket_contracts` | Polymarket CLOB API + Subgraph | 1-2 hrs |
| `tennis_data` | tennis-data.co.uk CSVs (very similar to football_data) | 30 min |
| `sofascore` | Sofascore web API (rate-limited; needs polite scraping) | 2-3 hrs |
| `sec_edgar` | SEC EDGAR full-text search + form filings | 1-2 hrs |
| `clinicaltrials` | ClinicalTrials.gov API | 1 hr |
| `fed_speakers` | Federal Reserve speeches + minutes (HTML scraping) | 1-2 hrs |
| `hyperliquid` | Hyperliquid public API | 1 hr |
| `news_scraper` | generic article scraper orchestrated by LLM agents | 3+ hrs |

Each stub in `src/quant/adapters/_stubs.py` raises an explanatory `AdapterNotImplementedError` so the failure is loud, not silent.

## What this enables today

- Pipeline runs cleanly end-to-end on real data (`bin/run-market betfair-eng-l1l2` produces a backtest report).
- Leaderboard reflects real results, not just synthetic.
- Each stage's test passes (`pytest -v`).

## Next-iteration priorities

To unlock the highest-priority markets per the plan:

1. **`tennis_data`** (smallest lift; unlocks `betfair-itf-challenger`).
2. **`hyperliquid`** (medium lift; unlocks the crypto tier).
3. **`kalshi_contracts`** (medium lift; unlocks the macro/CPI/NFP markets, our highest-confidence edge thesis per `research/01-market-universe.md`).
4. **`fred`** (already implemented, but the macro markets also need Kalshi contracts to be tradeable).

## Files referenced

- Leaderboard: `runs/screening/leaderboard.md` and `runs/LEADERBOARD.md`
- Per-market deep dive (first real market): `research/per-market/betfair-eng-l1l2.md`
- Stub list: `src/quant/adapters/_stubs.py`
