# 07. H200 window result (2026-05-15)

## Summary

3-hour H200 window. GPU 6 went from idle to two full transformer trainings + four new market backtests. The leaderboard at `runs/LEADERBOARD.md` grew from 2 entries (1 synthetic, 1 real) to 5 entries (1 synthetic, 4 real). All four real-market results are losses or zero-trade, which is the honest answer for v1 feature sets.

## Adapters implemented this window

| adapter | status | unlocks |
|---|---|---|
| `tennis_data` | implemented (xlsx via openpyxl, ATP main draw 2015-2025) | `betfair-itf-challenger` (with caveat: ATP main is the substitute for ITF Challenger pending Jeff Sackmann CSV merge) |
| `hyperliquid` | implemented (public POST API, hourly OHLCV + funding) | `hyperliquid-midcap` |
| `kalshi_contracts` | implemented (public `api.elections.kalshi.com`, no auth) | `kalshi-cpi`, `kalshi-nfp` (latter not run this window because only 2 NFP events were accessible) |

Still stubbed: `sofascore`, `polymarket_contracts`, `sec_edgar`, `clinicaltrials`, `fed_speakers`, `news_scraper`.

## H200 usage record

Two trainings on GPU 6:
1. `hyperliquid-midcap` — 8-layer transformer, model_dim 256, 12 epochs over 4500 hourly events (SOL only, 7 months).
2. `betfair-eng-l1l2` v2 — 8-layer transformer, model_dim 256, 25 epochs over ~770 train events.

Wall-clock per training was under a minute thanks to the H200's bandwidth. The bottleneck of the window was adapter implementation + data collection, not training.

## Honest result reading

| market | result | what it tells us |
|---|---|---|
| `kalshi-cpi` | 0 trades, Brier ~0 | Without FRED leading indicators, model has nothing to add; decision rule correctly refuses to trade through friction. |
| `betfair-eng-l1l2 v2` | -10.79%, Brier improvement -0.04 | Transformer is much better calibrated than the v1 GBDT (-0.04 vs -0.16) on identical features. Capacity helps you mimic the market, not beat it. |
| `hyperliquid-midcap` | -14.66%, Brier improvement -0.001 | Crypto with only OHLCV + funding is a coin flip; commission + slippage on 734 trades = ~22% in costs. |
| `betfair-itf-challenger` | -26.68%, Brier improvement -0.13 | ATP main draw (the substitute used) is sharp-pro-modeled; the wrong test of the ITF thesis. |

## Why nothing is "promising" yet (and that's OK)

The point of the broad-sweep window was to test multiple hypotheses cheaply. Each result above either:
- correctly identifies "no edge with these features" (`eng-l1l2`, `hyperliquid`, `itf-challenger`), or
- correctly defers to the market under cost constraints (`kalshi-cpi`).

A pipeline that ships positive results from naive features is the suspicious one. Ours is correctly conservative. The next session needs to either (a) add text/sentiment features to the existing markets, or (b) implement adapters for markets where the structured features alone might still have edge (Polymarket awards has critic-score data; that's worth a shot).

## Highest-priority next-session moves

1. **Get FRED key**. Re-run `kalshi-cpi` with macro features. This is the cleanest one-move test of the Fed-paper edge thesis.
2. **Implement Sofascore (or news_scraper)**. Re-run `betfair-eng-l1l2` v3 transformer with injury / lineup / suspension text features. Expected to be the cleanest test of the agent-army thesis.
3. **Fix Hyperliquid pagination** (windowed startTime requests). Try multiple coins via per-coin yamls. Layer in CryptoCompare social sentiment.
4. **Wire Polymarket Subgraph**. Run `polymarket-oscars` with critic-score features for the next Oscar season.

## Latent capability: `text` block `mode: encoder`

`src/quant/state/text.py` already supports a pretrained sentence-encoder path (default `sentence-transformers/all-mpnet-base-v2`) that pools recent text into a designated slot range. Any yaml can switch from `mode: handcrafted` to `mode: encoder` and immediately get learned embeddings in that range; no code change required. This capability is unexercised this window because no text adapter is implemented yet. As soon as Sofascore (or news_scraper) ships, flipping the mode on a sports yaml turns it on. Idea logged by user 2026-05-15.

## Idea: per-topic text blocks in one state vector

User extension (2026-05-15): instead of one text block per market, declare N text blocks, each with its own slot range and its own text-corpus column. Example for a hypothetical oil-price market:

```yaml
state_vector:
  dim: 1024
  blocks:
    - {name: numeric,             slots: "0:128",    builder: numeric, params: {...}}
    - {name: oil_transport_sent,  slots: "128:384",  builder: text, params: {text_column: text__oil_transport, mode: encoder, window_days: 7}}
    - {name: saudi_arabia_sent,   slots: "384:640",  builder: text, params: {text_column: text__saudi_arabia,  mode: encoder, window_days: 14}}
    - {name: opec_sent,           slots: "640:896",  builder: text, params: {text_column: text__opec,          mode: encoder, window_days: 30}}
    - {name: calendar,            slots: "896:1024", builder: calendar, params: {}}
```

The composer already handles arbitrary repeated block types; this works at the schema level today. The only code change needed: `pipeline/structure.py` currently only forwards a single `text` column. Five-line patch to also forward any `text__<topic>` column the adapter emits.

Adapter pattern: an LLM-agent scraper classifies each article into a topic (oil_transport, saudi_arabia, opec, ...) and writes `text__<topic>` columns. Each topic becomes its own labelled, inspectable slice of the state vector. The transformer then learns per-block attention over topic-specific sentiment.

This makes the state vector self-documenting at the semantic level (you can ask "did the model attend to OPEC sentiment for this trade?" and get a real answer) rather than at the raw-feature level. Ship when the first multi-topic text adapter (news_scraper) lands.

## Files touched this window

- `src/quant/adapters/tennis_data.py` (new)
- `src/quant/adapters/hyperliquid.py` (new)
- `src/quant/adapters/kalshi.py` (new)
- `src/quant/adapters/__init__.py` (registered 3 new adapters)
- `src/quant/pipeline/structure.py` (honor `time_grain: hourly` and `weekly`)
- `markets/hyperliquid-midcap.yaml` (single-coin v1)
- `markets/betfair-itf-challenger.yaml` (ATP-substitute documented)
- `markets/betfair-eng-l1l2.yaml` (flipped to transformer for v2)
- `markets/kalshi-cpi.yaml` (FRED removed pending key)
- `bin/h200` (create remote data dirs in `sync`)
- `pyproject.toml` (openpyxl + xlrd for tennis_data Excel parsing)
- `research/per-market/{hyperliquid-midcap,kalshi-cpi,betfair-itf-challenger,betfair-eng-l1l2-v2}.md` (per-market deep dives)
- `runs/LEADERBOARD.md` (now 5 entries)
