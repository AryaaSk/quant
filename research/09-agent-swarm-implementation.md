# 09. Agent-swarm implementation (Stage A + Stage B)

Shipped 2026-05-15. Implements the vision in `08-architecture-vision.md`: two-stage data pipeline driven by `claude -p` (Stage A: web scrape) and `codex exec` (Stage B: local-files schema synthesis), with software-3.0-style markdown skill briefs as the interface.

## Why dual runtime

Original plan was claude-only. User then requested codex for Stage B with the reason: **"i dont want to lose all my claude credits"**. Claude credits are finite under the subscription cap; codex is billed separately. So we split:

- **claude -p**: tasks that need WebSearch / WebFetch (Stage A scrape). No good substitute on codex.
- **codex exec**: tasks that only need to read and write local files (Stage B schema synthesis). Codex is strong at this and preserves claude budget.

This is now a saved long-lived preference: [`feedback_codex_vs_claude_routing.md`](../.. /. claude/projects/-Users-aryaask-Documents-Zoral/memory/feedback_codex_vs_claude_routing.md) (in user memory).

## Components shipped

| component | path | purpose |
|---|---|---|
| Skills index | `skills/README.md` | Convention + how to add a skill |
| Stage A brief | `skills/scrape-topic.md` | claude scraper, per-topic, writes JSON to `data/raw/<slug>/_news/<topic>/` |
| Stage B brief | `skills/propose-schema.md` | codex schema designer, reads raw, writes `markets/<slug>.proposed.yaml` |
| Runner | `src/quant/agents/runner.py` | Backend-dispatching subprocess wrapper (claude / codex) |
| News adapter | `src/quant/adapters/news_scraper.py` | Calls Stage A scraper per topic, aggregates JSON into `text__<topic>` columns |
| Structure patch | `src/quant/pipeline/structure.py` | Forwards every `text__<topic>` column through pivoting |
| Stage B script | `scripts/06_propose_schema.py` + `bin/propose-schema` | Orchestrates codex schema agent |
| Demo yaml | `markets/oil-prices.yaml` | Scaffold market using news_scraper with 3 topics |
| Tests (free) | `tests/test_agent_runner.py` (17), `tests/test_news_scraper.py` (6) | Mocked subprocess; CI-safe |
| Tests (gated) | `tests/test_codex_smoke.py`, `tests/test_claude_smoke.py` | Real subprocess calls; gated by `QUANT_ENABLE_AGENTS=1` |

Total: 23 new free tests + 2 gated smoke tests + 7 source/skill files + 1 demo yaml.

## Verification record

- **Mocked tests**: 23/23 passing in `pytest tests/test_agent_runner.py tests/test_news_scraper.py`.
- **Codex round-trip**: `QUANT_ENABLE_AGENTS=1 pytest tests/test_codex_smoke.py` passed (2/2) on 2026-05-15. One real `codex exec` call against `_synthetic` raw dir; runner dispatched, codex returned NDJSON, parser extracted `agent_message` text + `usage` block.
- **Claude smoke**: not yet exercised in this iteration (gated; will run when first oil-prices end-to-end is attempted; budget cap `--max-budget-usd 0.10` per skill).
- **Full free suite**: 41 passed (all other markets + the new agent infra) in 957s. The 16-minute time is dominated by `test_screening` which fetches live HTTP from Football-Data; everything else is fast.

## Skill frontmatter contract

Every skill md starts with:

```yaml
---
name: <slug>
backend: claude | codex
allowed_tools: [WebSearch, WebFetch, Read, Write, Bash]   # claude only
max_budget_usd: 1.0                                       # claude only
required_params: [param1, param2]
timeout_s: 900
---
```

The runner validates `required_params` before dispatch and renders `{{var}}` substitutions in the body.

## Codex NDJSON quirk (and fix)

Codex `exec --json` does NOT emit a single JSON object; it emits a stream of NDJSON events:

```
{"type":"thread.started","thread_id":"..."}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}
{"type":"turn.completed","usage":{"input_tokens":...,"output_tokens":...}}
```

The runner's `_parse_codex_output` walks the stream, collects all `agent_message` items into `response_text`, and pulls the final `usage` block. Falls back to "last NDJSON object" if no structured events are present (defends against future codex format changes).

## End-to-end flow (when QUANT_ENABLE_AGENTS=1)

```
markets/oil-prices.yaml
  data_sources: [news_scraper(topics=[oil_transport, saudi_arabia, opec])]
       |
       v
bin/run-market oil-prices
  |
  v
collect.py invokes news_scraper.fetch()
  |
  +---> for each topic, run_agents_parallel([Invocation(skill="scrape-topic", ...)])
  |                                          ^
  |                                          dispatches to: claude -p --output-format json
  |                                                         --allowedTools "WebSearch,WebFetch,..."
  |                                                         --max-budget-usd 1.0
  |                                                         (rendered prompt)
  |
  v
data/raw/oil-prices/_news/<topic>/article_NNNN.json  (agent-written)
  |
  v
news_scraper aggregates JSON files -> DataFrame with text__<topic> columns
  |
  v
structure.py pivots, forwards text__<topic> through to wide frame
  |
  v
Now run bin/propose-schema oil-prices  (Stage B)
  |
  v
scripts/06_propose_schema.py invokes run_agent(skill="propose-schema", ...)
  |                                                ^
  |                                                dispatches to: codex exec --json
  |                                                               --sandbox workspace-write
  |                                                               -C <quant-root>
  |
  v
codex writes markets/oil-prices.proposed.yaml
  |
  v
human inspects diff, runs `bin/propose-schema oil-prices --apply` to overwrite
  |
  v
bin/run-market oil-prices  (now does full state + train + backtest with the proposed schema)
```

## Budget controls and safety

- Per-skill budget cap (claude): `max_budget_usd` in skill frontmatter passes to `claude -p --max-budget-usd`.
- Aggregate ceiling: `run_agents_parallel` raises if cumulative claude cost > `QUANT_CLAUDE_BUDGET_USD` (default $5).
- Codex billing: external, not capped in code. We log token usage from `turn.completed.usage` for transparency.
- Gating: `QUANT_ENABLE_AGENTS=1` env var required to actually invoke real subprocesses. Without it, the news_scraper adapter falls back to reading existing cache, and the schema proposer script refuses (exits 2).
- Idempotency: each `(skill, log_subdir)` writes a `done.flag` after success. Re-runs skip unless `force=True`. Saves both wall clock and tokens.
- Permission posture: claude runs with `--permission-mode bypassPermissions --bare`; codex runs with `--sandbox workspace-write`. Both write only into the quant project dir via `--add-dir` / `-C`.

## Limitations and next-iteration moves

1. **First real Stage A run hasn't happened yet.** The pipeline is verified through mocks + a codex round-trip. The first real `claude -p` scrape of oil-prices is the next test; it will spend roughly $2-3 (3 topics × $1 budget cap).
2. **No incremental scraping yet.** Re-running re-scrapes from scratch unless the per-topic `done.flag` exists. Smarter: per-query staleness tracking.
3. **Schema proposer is one-shot.** It writes one schema; no iteration loop yet. Once a market shows positive backtest results, the next iteration could read the backtest output and refine.
4. **No agent-to-agent communication.** Each agent runs independently. The orchestration is just parallel dispatch + aggregation. Future: feed Stage A summaries into Stage B as additional context.
5. **Cost reporting in the leaderboard.** Add `cumulative_agent_cost_usd` per-market field in `runs/<slug>/<ts>/metrics.json` so the leaderboard shows ROI on agent spend.

These are all incremental upgrades on top of a working agent-swarm pipeline. The architecture is in place.
