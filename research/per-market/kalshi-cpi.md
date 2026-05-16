# kalshi-cpi — v1 result (2026-05-15)

## Setup

- **Adapter**: Kalshi public API (`api.elections.kalshi.com/trade-api/v2`, no auth required). Series `KXCPIYOY`. Returned 56 settled threshold-markets across 3 CPI events (Mar/Apr/May 2026 releases). API only exposes ~2 years of CPI series under the current ticker.
- **Label**: binary (did CPI YoY exceed threshold X — `result == "yes"`).
- **State vector**: 128 dim. Kalshi market features (threshold, open yes-price, total volume, max open interest, days open) + market-state velocity + calendar. **No FRED macro features** because `FRED_API_KEY` is not set in `.env`; the FRED adapter is implemented and ready to enable once the key is added.
- **Model**: sklearn HistGradientBoostingClassifier.
- **Held-out fence**: last 1 month (~28 events because the API only had 3 distinct settle dates total).

## Backtest result

| metric | value |
|---|---|
| net_return | **0.00%** |
| Brier model | 0.250 |
| Brier market | 0.250 |
| **Brier improvement** | **+0.0001** |
| hit rate | 0.00% |
| Sharpe | 0.00 |
| max drawdown | 0.00% |
| trades | 0 |

## Interpretation

The model placed zero trades because the decision rule (`|p_model - p_market| > commission + slippage + safety_margin = 220 bps`) was never triggered. This is the cost-aware decision rule working correctly: without FRED leading indicators (Truflation, ADP, GDPNow) feeding the model, the model has nothing to add over the Kalshi closing price, and the trades it might take would lose to friction.

A Brier improvement of +0.0001 is statistically zero on 28 events; we cannot infer anything from it.

## What this proves about the pipeline

- Kalshi public API works without authentication (the new `api.elections.kalshi.com` host is open).
- The adapter handles Kalshi's `finalized`/`settled` status confusion, parses threshold from ticker (`KXCPIYOY-26APR-T2.3` → threshold 2.3), and reads price history via `/candlesticks` endpoint with day-bar granularity.
- The "unique-target-per-market" workaround (per-market hash-offset on `target_event_time`) is in place so multiple threshold markets per event do not collapse into one row at structure time.
- The decision rule correctly refuses to trade when the model has no edge over costs.

## Why we don't yet know if Kalshi has edge

Two limitations are entangled here:
1. **FRED unavailable** (no API key). The whole edge thesis for Kalshi CPI is that nowcasts diverge from market consensus. Without FRED, the model only sees Kalshi market features.
2. **Only 3 unique events in the public API** for this series. Whatever we infer is statistically uninformative.

To produce a real Kalshi-CPI result we need (a) a FRED key, and (b) longer historical data. The second point requires either pulling older series tickers (CPIYOY which appears to have older history but returned 0 events on `status=settled` in my probe) or a paid archive (predictiondata.dev).

## Verdict

**Inconclusive.** Pipeline plumbing is verified; substantive test deferred to a session with FRED access + a longer Kalshi history scrape. This is the most a-priori promising market in our universe (Fed paper), so it deserves the next deep iteration.
