# Architecture

## Pipeline

Each market flows through six stages. Market-specific code lives only at adapter and state-schema boundaries.

```
markets/<slug>.yaml
   |
   v
pipeline/collect.py          -> data/raw/<slug>/         (agent-driven scrape)
   |
   v
pipeline/structure.py        -> data/structured/<slug>/  (cleaned features + timestamps)
   |
   v
pipeline/state.py            -> data/state/<slug>/       (state-vector sequences)
   |   (composes state/numeric.py + market_state.py + text.py + calendar.py per yaml schema)
   v
pipeline/dataset.py          -> torch Datasets           (train / val / held_out splits with sequence windowing)
   |
   v
pipeline/train.py            -> runs/<slug>/<ts>/ckpt/   (transformer or LightGBM)
   |
   v
pipeline/backtest.py         -> runs/<slug>/<ts>/metrics.json
   |
   v
pipeline/report.py           -> runs/<slug>/<ts>/backtest_report.html
```

Idempotency: each stage writes a versioned output directory and skips work if outputs are newer than inputs.

## State vector

The "embedding" in this project is hand-designed. We do NOT feed raw text into a pretrained LLM and use the output as the transformer input. Instead each market yaml declares a fixed-size state vector schema:

```yaml
state_vector:
  dim: 1024
  blocks:
    - {name: macro_numeric,  slots: "0..255",   builder: numeric, source: fred_series, normalize: rolling_z}
    - {name: market_state,   slots: "256..319", builder: market_state, source: contract_prices}
    - {name: alt_data,       slots: "320..447", builder: numeric, source: alt_series, normalize: rolling_z}
    - {name: text_features,  slots: "448..959", builder: text, encoder: sentence_transformer_mpnet, projection: learned}
    - {name: calendar,       slots: "960..1023", builder: calendar}
```

Properties:

1. **Every slot is named and inspectable.** No black-box "text embedding goes here."
2. **State construction is pure.** Given structured features at t, the state vector at t is uniquely determined. Enables property-based temporal-leak testing.
3. **Per-market dimension and per-block budget.** High-info markets get 1024-2048 dim; low-info get 256-512.
4. **Text is one block among many.** Text features are projected into a sub-region by a learned projection that trains alongside the transformer.

Block builders live in `src/quant/state/` and implement the `BlockBuilder` interface (`fit_normalizers(structured_df)`, `build(row, structured_df) -> np.ndarray`). The composer in `src/quant/state/composer.py` reads the yaml schema and stitches blocks into the final state vector per timestep.

## Transformer

- Input: `(batch, seq_len, state_dim)` tensor of state vectors
- Optional input projection `state_dim -> model_dim`
- 12-layer encoder, 16 heads, RoPE positional, GELU FFN with 4x expansion
- Output head: sigmoid for binary, softmax for multi-bucket
- Training: AdamW, cosine LR with warmup, mixed precision on H200
- Fallback: LightGBM on flattened last-K-timesteps for markets with <500 events

## Decision rule

For each upcoming event:
- `p_model = model(state_sequence)`
- `p_market = 1 / decimal_odds` (Betfair) or `contract_price` (Kalshi/Polymarket)
- `edge = p_model - p_market`
- `cost = commission + slippage(position_size, orderbook)` from yaml
- Trade only if `|edge| > cost + safety_margin` (default 1.5%)
- Position size: fractional Kelly (default 25% of full Kelly, hard cap 2% of bankroll)

## Backtest hygiene

Non-negotiable rules enforced in code, not docs:

1. Held-out period locked Day 1 of collection. Raw collectors physically cannot write to held-out paths during training.
2. Every feature carries `source_published_at` AND `scraped_at` timestamps. At time T, only rows with `min(published_at, scraped_at) <= T` are usable.
3. Walk-forward retraining across the held-out window (monthly cadence or per-event for low-cadence markets).
4. Commission + position-size-dependent slippage charged in full.
5. Calibration (Brier vs market closing) is the primary metric. Total return is misleading.
6. Label-shuffle sanity check: shuffling labels must collapse Brier improvement to zero. If it does not, the pipeline raises and halts.
7. Hyperparameter sweeps use train + val only. Held-out is touched exactly once per market, at the end.

## H200 deployment (Phase 4)

`bin/h200` rsyncs the project to `~/Aryaa/quant/` on the H200 box and runs the trainer with `CUDA_VISIBLE_DEVICES=$H200_GPU` (default 6). The script supports `sync`, `run`, `ssh`, `pull`, `nvidia`, `--check`. Training is gated: `scripts/03_train.py` requires `--confirm-gpu-free` to actually call the H200.
