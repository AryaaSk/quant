# 08. Architecture vision — two-stage data gathering + agent-driven schema design

User-articulated 2026-05-15. Documents the full system the current implementation should grow into.

## The pipeline as the user sees it

```
Stage A: scraper-agent swarm
  one high-level prompt per market: "find everything about <market>"
  agents dispatch in parallel, each pulls a different corner of the web
  output: raw documents dumped to data/raw/<slug>/, heterogeneous formats
  contract: every row carries source_published_at, scraped_at, source_url, source_type

         |
         v

Stage B: schema-design + encoding agent swarm
  reads the raw dump
  proposes the per-market state-vector schema yaml:
    - which features to include
    - slot allocation per feature
    - encoding per feature (numeric vs text vs market-state vs calendar)
    - per-topic text blocks (one slot range per semantic topic)
    - time granularity (daily / hourly / per-event)
  writes markets/<slug>.yaml
  composer + block builders then materialize state vectors from raw

         |
         v

Stage C: train
  sequence of state vectors -> transformer
  output head ("unembedding") projects final [CLS] hidden state to prediction

         |
         v

Stage D: inference
  same as LLMs: window of state vectors = "context window"
  read out the final hidden state, apply the prediction head, decide trade
```

## What of this is built today

| stage | what exists | what's the gap |
|---|---|---|
| A: raw collection | `pipeline/collect.py` orchestrates per-source adapters; 5 adapters live (synthetic, fred, football_data, tennis_data, kalshi, hyperliquid) | agent-swarm version: `adapters/news_scraper.py` stub. Pattern: one generic LLM-agent adapter that dispatches sub-agents from a high-level prompt and writes heterogeneous rows |
| B: schema + encoding | `pipeline/structure.py` (time-grain bucketing) + `state/composer.py` + 4 block builders. yaml hand-written today. | agent-driven schema design: an agent reads `data/raw/<slug>/`, proposes a yaml, writes `markets/<slug>.yaml`. Composer already consumes any valid schema |
| B (text): per-topic blocks | Composer handles N text blocks per yaml with distinct slot ranges; `text` block has `mode: encoder` for sentence-encoder embeddings | Need 5-line patch in `pipeline/structure.py` to forward `text__<topic>` columns (currently only forwards a single `text` column) |
| C: train | `pipeline/train.py` with transformer + GBDT fallback. `(batch, seq_len, state_dim)` input, [CLS] read-out, binary/bucketed head | none for the prediction case |
| C: unembedding (LLM-style) | Not built. We only have a prediction head, not a projection back into the state-vector space | Optional extension: tie an "input state vector" projection to the [CLS] hidden state for inspectability ("what does the model expect the next state to look like?") |
| D: inference | trivial via the trained checkpoint | none |

## Specific design points

### "Reserve dimensions for the actual price"

Already supported via `market_state` block. The transformer sees its own target value in every input timestep, lagged so there's no leakage. Example pattern in `kalshi-cpi.yaml` (`open_yes_price` in the market_state block) and `hyperliquid-midcap.yaml` (`close` and `funding_rate` in the market_state block). For a "predict next price" market, declare:

```yaml
- name: target_price_state
  slots: "X:Y"
  builder: market_state
  params:
    series: [target_price]
    velocity_lags: [1, 5, 20, 50]
```

The transformer learns "this is what I'm predicting; here's its recent trajectory."

### "Unembedding function" for inspectability

For trading we mostly only need the prediction head (sigmoid for binary, softmax for bucketed). But there's a real value-add if we add a second projection: from the final [CLS] hidden state back into the input state-vector space. This lets us ask:
- "what does the model think the next state vector should look like?"
- "did it expect a regime change, or continuity?"

Architecturally this is a single extra linear layer with shape `(model_dim, state_dim)`. Train with an auxiliary loss (MSE between predicted next state vector and the actual one). Worth shipping when we have a model that's actually making money, to debug WHY.

### Time grain decided at Stage B

Already implemented. `time_grain: daily | hourly | weekly` field in the yaml; `structure.py` honors it via the `_bucket` helper. Adding `minutely` or `per_event` would be a one-line addition to the bucket switch.

### Schema design as the hard part

The user is right that this is the substantive intellectual work and not just a config exercise. The agent in Stage B has to:
1. Survey the raw dump and identify candidate features.
2. Decide which are likely predictive.
3. Decide encoding per feature (numeric z-score, market-state level+velocity, text encoder embedding, calendar cyclical, hand-crafted text counts).
4. Decide slot allocation (high-info features get more dims).
5. Decide sequence length and time grain.

Until we automate this, it remains a per-market modeling decision documented in `research/per-market/<slug>.md`. Once we have 5+ markets with comparable returns, we can train the schema-design agent on the patterns that worked.

## What needs to ship for the full vision to be reality

1. **`news_scraper` adapter** (the generic agent-swarm Stage-A adapter). Takes a high-level prompt and a list of "topic categories"; spawns parallel sub-agents per category; dumps heterogeneous text rows with `source_type=news_scraper.<topic>` and a `text__<topic>` column convention.
2. **`structure.py` 5-line patch** to forward `text__<topic>` columns into the structured frame.
3. **Schema-proposer agent script** (Stage-B v1). Reads `data/raw/<slug>/`, calls an LLM with the column list + sample text, gets back a yaml proposal, writes `markets/<slug>.yaml`. v1 is just one-shot; v2 iterates based on backtest results.
4. **Optional: unembedding head** for inspectability, with auxiliary MSE-on-next-state loss during training.

Each of these is a tractable next-session piece. The runtime (structure → state → train → backtest → report) is in place and does not need to change.
