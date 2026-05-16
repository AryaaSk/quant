# 02. Pipeline design

## Why hand-designed state vectors instead of feeding text into an LLM

The user explicitly framed this in conversation: we are not training a vanilla LLM. The transformer's input is a sequence of state vectors WE designed. Every slot is named, documented, and inspectable.

Reasons:

1. **Inspectability.** We can ask "did the model attend to the macro_numeric block on this trade?" and answer it. With raw text-to-LLM-embedding, the input is a black box.
2. **Mixed modality.** Each block can use the right encoding for its source: rolling z-score for numeric, velocity features for prices, sentence-encoder + projection for text, sinusoidal encoding for calendar. No need to tokenize numeric data.
3. **Per-market dimension.** High-info markets get 1024-2048 dim; low-info get 256-512. The yaml controls it.
4. **Temporal-leak prevention is tractable.** State construction is a pure function of past-only structured features. We can property-test that no future data leaks in.

See `feedback_state_vector_design` in memory and `ARCHITECTURE.md` for the full design.

## Per-market state vector schemas

Each yaml declares a list of blocks with `slots: "start:end"` half-open ranges that must contiguously cover `[0, dim)`. The composer in `src/quant/state/composer.py` validates this at load time.

### Block types

- `numeric` — rolling z-score of named series, one slot per series.
- `market_state` — level + velocity (lag 1, 5, 30) per price series. Used for contract prices, related instruments.
- `text` — handcrafted features (doc count, sentiment, entity mentions) or sentence-encoder + projection.
- `calendar` — cyclical encodings of day/month/year + days-until/since target event + extra flags.

### Example: kalshi-cpi schema

```yaml
state_vector:
  dim: 1024
  blocks:
    - {name: macro_numeric, slots: "0:256",   builder: numeric, params: {series: [CPIAUCSL, ...]}}
    - {name: market_state,  slots: "256:320", builder: market_state, params: {series: [contract_yesno_price, treasury_2y_yield, ...]}}
    - {name: alt_data,      slots: "320:448", builder: numeric, params: {series: [truflation, adp_payrolls, ...]}}
    - {name: text_features, slots: "448:960", builder: text, params: {mode: handcrafted, entity_keywords: [Powell, Yellen, Waller, Brainard, ...]}}
    - {name: calendar,      slots: "960:1024", builder: calendar, params: {extra_flag_columns: [fomc_week]}}
```

## Transformer choice

- 12 encoder layers, model_dim 256-768 per market, 8-16 heads.
- Learned [CLS] token prepended; its final hidden state feeds the prediction head.
- Sinusoidal positional encoding (RoPE is overkill for sequence lengths < 256).
- Mixed precision on H200; vanilla fp32 on Mac during smoke tests.

Fallback: LightGBM on a flattened state-sequence (concat all timesteps into one row) for markets with <500 events. Same decision rule downstream.

## Walk-forward backtest semantics

Default: train once on train + val (everything before the held-out window), then sequentially predict each held-out event and simulate trading with commission, slippage, safety margin. The held-out window is locked in `data/held_out/<slug>/MANIFEST.json` at first collection; subsequent runs cannot move the fence silently.

True walk-forward retraining (retrain monthly, re-evaluate next month) is a quality bump deferred to second iteration: it doubles training cost but tightens the backtest. The infrastructure is already in place: `scripts/04_backtest.py --walk-forward` is the planned switch.

## Why not classical ARIMA / SARIMAX / Prophet baselines

For each market, the simplest sane baseline is "predict the market closing price." Brier improvement vs market closing is the gold-standard comparison. Classical time-series baselines are nice-to-have but they do not exploit text features. The bar to clear is the market itself.
