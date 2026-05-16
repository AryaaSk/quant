# 10. Stage A.5: per-event agent feature extraction

Shipped 2026-05-15 in response to user request: *"i want a combination of agent swarm feature extraction and python deterministic code, with a strong reliance on agent swarm. these are really where we will find our information edge."*

## Why this stage exists

Before this stage, the pipeline was:

```
Stage A (claude agents) -> raw articles -> Python (MPNet pooled embedding) -> 768-dim slot range
```

The MPNet embedding is **opaque**: the transformer sees a black-box vector and has to figure out which dimensions correspond to "is the home team's star striker out". That's a lot to ask of a 25M-param transformer on 900 events.

Stage A.5 inserts a second agent swarm BETWEEN the scrape and the deterministic state-vector build. Its job: read the scraped articles and emit a strict schema of **named, inspectable, numeric features** that the transformer can use directly.

## Pipeline shape (now)

```
Stage A (claude)                   open-web scrape       -> data/raw/<slug>/_news/<topic>/article_NNNN.json
       |
       v
Stage A.5 (codex)        [NEW]     per-event extraction  -> data/raw/<slug>/_extracted/batch_NNNN_output.jsonl
       |                                                    -> num__<field> columns per event
       v
Stage B (codex)         [optional] schema proposer       -> markets/<slug>.proposed.yaml
                                                            (skipped for hand-crafted markets)
       |
       v
Python deterministic               structure -> state    -> per-event state vectors
                                                            text blocks STILL get MPNet embedding (fallback signal)
       |
       v
Transformer on H200 -> backtest -> leaderboard
```

Key property: **named features and raw text embeddings are complementary signals in the same state vector.** Numeric block at slot range A:B holds the agent-extracted features; text blocks at slot ranges B:C, C:D, ... hold MPNet pooled embeddings. The transformer learns to use whichever channel carries signal for each event.

## Component shipped

### Skill: `skills/extract-features.md`

Backend: codex. Reads a batch of events (one JSON object per line) + an articles directory + a feature schema. For each event, walks the articles dir, filters to articles within `[event_time - window_days, event_time)`, extracts each declared feature as a float, writes one JSON line per event to the output path.

Strict invariants in the brief:
1. No temporal leakage (`published_at` strictly less than `event_time`).
2. Schema fidelity (exact feature names, all declared).
3. Range adherence (clipped to the declared `[lo, hi]`).
4. Default 0.0 when unmeasurable.

The brief also forbids: inventing features outside the schema, emitting string values, spawning sub-agents, writing outside `output_path`.

### Adapter: `src/quant/adapters/agent_feature_extractor.py`

- Reads event list from a companion adapter's parquet (e.g. `football_data.parquet`)
- Reads articles dir from another companion adapter's output (e.g. `_news/`)
- Batches events (default 20 per batch) for cost efficiency
- Spawns one codex agent per batch via `run_agents_parallel`
- Idempotent via per-batch `_done` markers + `done.flag` on the runner side
- Aggregates per-batch JSONL outputs into a DataFrame with `num__<field>` columns
- Clips extreme values to [-8, 8] to defend against agent errors poisoning training
- Falls back to default-value rows when `QUANT_ENABLE_AGENTS != 1` (development / CI path)

### Yaml wiring

The `agent_feature_extractor` data source declares the feature schema and points to companion adapters:

```yaml
- name: agent_feature_extractor
  kind: numeric
  params:
    articles_source: news_scraper
    event_source: football_data
    window_days: 10
    batch_size: 20
    concurrency: 3
    feature_schema:
      - {name: home_injuries_severity, description: "...", range: [-1, 1]}
      - ...
```

The extracted features feed a `numeric` block in the state vector:

```yaml
state_vector:
  dim: 1024
  blocks:
    - {name: form_numeric,             slots: "0:192",   builder: numeric, ...}
    - {name: market_state,             slots: "192:320", builder: market_state, ...}
    - {name: agent_extracted_numeric,  slots: "320:448", builder: numeric, params: {series: [home_injuries_severity, ...]}}
    - {name: text_lineups,             slots: "448:640", builder: text, mode: encoder, ...}
    - {name: text_injuries,            slots: "640:832", builder: text, mode: encoder, ...}
    - {name: text_manager,             slots: "832:1008", builder: text, mode: encoder, ...}
    - {name: calendar,                 slots: "1008:1024", builder: calendar}
```

128 dims for named features (10 features rolling-z-scored, with headroom for additions). 560 dims for raw text. Both flow into the same transformer.

## Cost model (eng-l1l2 v3)

- 904 events / batch_size 20 = ~45 codex calls
- Codex billing is external (subscription, not metered per-call to user)
- Empirically per-call wall clock: 10-30s for a 20-event batch reading 5-10 articles each
- Total wall clock: ~20-30 min with concurrency 3
- Idempotent: re-runs skip cached batches via `_done` markers

Compare to per-article extraction (904 events x ~5 articles = ~4500 calls): batching saves ~100x.

## Test coverage

`tests/test_agent_feature_extractor.py` — 11 tests, all mocked, free in CI:
1. Event list construction + sorting + event_id assignment
2. Batching into chunks
3. Batch-input JSONL format
4. Aggregation of per-batch outputs
5. Default fallback for events missing in outputs
6. Extreme value clipping
7. Default-rows mode when agents disabled
8. Required-input validation (feature_schema)
9. Event-source-must-exist validation
10. Articles-dir-must-exist validation
11. Skill brief contains the temporal-leak guard text (defensive check against future edits)

## Verification before running

The temporal-leak property test in `tests/test_temporal_leak.py` already covers ANY adapter that emits `target_event_time` rows. The agent_feature_extractor inherits the same guarantee because its output rows are aligned to event times and the agent brief enforces `published_at < event_time` at the article-filter level.

## What this does NOT do (yet)

- Does not (yet) use the `context` field per-event to disambiguate home/away teams during article retrieval. The agent does this from article text. A v2 would extend the skill brief to filter by team name from `context.home_team` / `context.away_team`.
- Does not (yet) verify per-feature distributions (are values centered near 0? are they well-spread?). After the first real run, we should add a sanity-check script.
- Does not (yet) measure feature importance post-training. Once trained, we should inspect which `num__<feature>` slots received the highest attention weights to know which features are pulling weight.

## Decision implications

After v3 runs end-to-end with this stage active:

- **Brier improvement > +0.02**: extracted features carry real signal. Step 3 of the original plan (Qwen embedder on H200) is now the right next move, OR extend the feature schema with more features.
- **Brier improvement near zero**: features are decent but the test set is too small. Try expanding to other markets (kalshi-cpi with macro nowcasts is the natural fit).
- **Brier improvement still negative**: either the features I declared aren't the right ones, OR EFL is genuinely too efficient at the closing line. Try a richer schema (more granular per-team features) before declaring the market dead.
