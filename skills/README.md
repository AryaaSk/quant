# Skills

Software-3.0-style markdown briefs that agent runners consume. Each skill is a self-contained prompt that tells an agent exactly what to do, what files to read/write, and what schema to follow.

## Convention

Every skill file:
1. Has YAML frontmatter declaring backend, allowed tools, budget cap, required params.
2. Has a body that is the actual prompt (rendered via `{{var}}` substitution).
3. Tells the agent where to write its outputs (absolute paths passed in params).
4. Specifies the exact output schema so downstream code can parse without surprises.

## Frontmatter schema

```yaml
---
name: <slug>                              # must match filename minus .md
backend: claude | codex                    # which CLI to dispatch via
model: sonnet | opus | <full-model-id>     # optional; defaults to sonnet for claude
allowed_tools: [WebSearch, WebFetch, ...]  # claude-only; codex uses sandbox mode
max_budget_usd: 1.0                        # claude-only; codex billed externally
required_params: [param1, param2, ...]     # validated by runner before invoke
timeout_s: 900                             # subprocess timeout
---
```

## Backend + model routing

- **claude (sonnet)** (default): bulk scrape / extraction / routine reasoning. Cheap on Claude Max plan, fast, plenty of capability.
- **claude (opus)**: complex design / synthesis tasks (e.g. proposing a state-vector schema from raw data). Slower but better reasoning.
- **codex**: reserve for cases the user explicitly requests. Daily quota cap makes it unreliable for bulk pipelines.

This routing is documented in `research/09-agent-swarm-implementation.md` and saved as a long-lived preference (`feedback_agent_model_routing.md` in user memory).

## Adding a new skill

1. Create `skills/<name>.md` with frontmatter + body.
2. Reference params via `{{param_name}}`. The runner substitutes before dispatch.
3. Validate by adding a row in `tests/test_agent_runner.py::test_skills_parse`.
4. Document the new skill in this README.

## Current skills

| name | backend | purpose |
|---|---|---|
| [`scrape-topic`](scrape-topic.md) | claude / sonnet | Stage A: scrape open-web text on a specific topic, dump JSON files into a per-topic directory |
| [`extract-features`](extract-features.md) | claude / sonnet | Stage A.5: per-event, read in-window articles and emit a strict schema of numeric features as JSONL (one line per event). Temporal-leak guarded. |
| [`propose-schema`](propose-schema.md) | claude / opus | Stage B: read `data/raw/<slug>/`, propose a state-vector schema yaml, write `markets/<slug>.yaml`. Opus because schema design benefits from deeper reasoning. |
