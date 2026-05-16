# 04. Phase 1 screening method

## Goal

Avoid the trap of building beautiful pipeline for a market that has 30 events of usable history.

## Per-market screening procedure

For each `markets/<slug>.yaml`, spawn an LLM agent that does the following:

1. **Reachability check.** Attempt to fetch one sample from each data source declared in the yaml. If any source is unreachable or requires unavailable credentials, fail the row.
2. **Sample-size estimate.** Count distinct `target_event_time` values in 1 year of one representative source. Output `events_per_year`.
3. **Liquidity probe.** For platforms with live markets (Kalshi, Polymarket, Betfair), sample 5 currently-open events and record their matched volume / orderbook depth. Output `liquidity_p50_usd`.
4. **Information density.** For text-heavy sources, count distinct articles/posts per event window. Output `text_records_per_event`.
5. **Verdict.** Produce a 1-paragraph qualitative verdict + a 0-100 score weighting:
   - data reachability (binary, gates everything)
   - sample size (>500 events = full credit, >100 = partial, <100 = LightGBM-only)
   - liquidity (>$10k = full, >$1k = partial)
   - info density (>5 records/event = full, >1 = partial, 0 = numeric-only)
   - friction headroom (commission + slippage < 100bps = full, < 300bps = partial)

## Scoring rubric

```
score = 30 * data_reachable
      + 25 * min(events/500, 1)
      + 20 * min(liquidity / 10000, 1)
      + 15 * min(text_density / 5, 1)
      + 10 * max(0, 1 - friction_bps / 300)
```

Threshold: score >= 60 advances to full collection (Phase 2). Below 60 is documented in `runs/screening/leaderboard.md` with the reason and shelved.

## Output

`runs/screening/leaderboard.md` ranks all candidates with:
- score (0-100)
- verdict (1 paragraph)
- raw metrics (events/yr, liquidity, density, friction)
- killed / advanced flag

Plus per-market detail dropped to `runs/screening/<slug>.json` for machine consumption.

## Implementation

`scripts/01_screen_markets.py` orchestrates the agents. For now this is a Python script that spawns Claude/Codex agents via the Anthropic SDK. When the H200 is gated and we are doing local-only work, the screening can run from this Mac. No GPU needed.
