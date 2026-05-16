"""Tests for the agent runner: skill parsing, template rendering, backend dispatch.

All tests in this file are MOCKED. No real `claude -p` or `codex exec` calls. CI-safe.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quant.agents.runner import (
    Invocation,
    SkillSpec,
    SKILLS_DIR,
    _build_claude_cmd,
    _build_codex_cmd,
    _parse_claude_output,
    _parse_codex_output,
    load_skill,
    render_prompt,
    run_agent,
    run_agents_parallel,
)


# ---------- Skill parsing ----------


def test_skills_dir_exists():
    assert SKILLS_DIR.exists(), f"skills dir missing: {SKILLS_DIR}"


def test_every_shipped_skill_parses():
    failures = []
    for skill_path in SKILLS_DIR.glob("*.md"):
        if skill_path.name == "README.md":
            continue
        try:
            spec = load_skill(skill_path)
            assert spec.backend in ("claude", "codex"), f"{skill_path}: invalid backend {spec.backend!r}"
            assert spec.body, f"{skill_path}: empty body"
        except Exception as e:
            failures.append((skill_path.name, str(e)))
    assert not failures, "skill parse failures: " + ", ".join(f"{n}: {e}" for n, e in failures)


def test_load_skill_by_name():
    spec = load_skill("scrape-topic")
    assert spec.name == "scrape-topic"
    assert spec.backend == "claude"
    assert "WebSearch" in spec.allowed_tools


def test_load_skill_propose_schema_uses_claude_opus():
    """Schema design needs deeper reasoning -> claude opus, not codex.

    Codex's daily-cap unreliability moved us to claude opus for design tasks under the
    Claude Max plan. See `feedback_agent_model_routing.md` in memory.
    """
    spec = load_skill("propose-schema")
    assert spec.backend == "claude"
    assert spec.model == "opus"


def test_load_skill_scrape_topic_uses_claude_haiku():
    """Scraping is constrained (search/fetch/save JSON). Haiku is plenty and 4-5x cheaper.

    See `feedback_agent_model_routing.md`: haiku for scrape, sonnet for extract, opus for design.
    """
    spec = load_skill("scrape-topic")
    assert spec.backend == "claude"
    assert spec.model == "haiku"


def test_load_skill_extract_features_uses_claude_sonnet():
    spec = load_skill("extract-features")
    assert spec.backend == "claude"
    assert spec.model == "sonnet"


def test_load_skill_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_skill(tmp_path / "nonexistent.md")


# ---------- Prompt rendering ----------


def _make_skill(backend: str = "claude", body: str = "hello {{name}}", required: list | None = None) -> SkillSpec:
    return SkillSpec(name="t", backend=backend, body=body, required_params=required or [])


def test_render_substitutes_vars():
    spec = _make_skill(body="hi {{name}} from {{place}}")
    out = render_prompt(spec, {"name": "alice", "place": "earth"})
    assert out == "hi alice from earth"


def test_render_leaves_unknown_vars():
    spec = _make_skill(body="hi {{name}}, undef={{nope}}")
    out = render_prompt(spec, {"name": "alice"})
    assert "alice" in out and "{{nope}}" in out


def test_render_serializes_lists_and_dicts_as_json():
    spec = _make_skill(body="queries={{queries}}")
    out = render_prompt(spec, {"queries": ["foo", "bar"]})
    assert out == 'queries=["foo", "bar"]'


def test_render_enforces_required_params():
    spec = _make_skill(body="hi {{name}}", required=["name", "topic"])
    with pytest.raises(KeyError):
        render_prompt(spec, {"name": "alice"})


# ---------- Command construction ----------


def test_claude_cmd_includes_expected_flags(tmp_path):
    spec = SkillSpec(
        name="x", backend="claude",
        allowed_tools=["WebSearch", "Write"],
        max_budget_usd=0.5,
        body="ignored",
    )
    cmd = _build_claude_cmd("claude", "the prompt", spec, tmp_path)
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--permission-mode" in cmd and "bypassPermissions" in cmd
    # We intentionally do NOT pass --bare so OAuth/keychain auth keeps working.
    assert "--bare" not in cmd
    assert "--disable-slash-commands" in cmd
    # Default model is sonnet for bulk work under Claude Max.
    assert "--model" in cmd and "sonnet" in cmd
    assert "--allowedTools" in cmd
    assert "WebSearch,Write" in cmd
    assert "--max-budget-usd" in cmd
    assert "0.5" in cmd
    assert "--add-dir" in cmd
    assert str(tmp_path) in cmd
    # Prompt is NOT in cmd — it goes via stdin. `--add-dir <directories...>` is
    # variadic and would otherwise consume the prompt as another directory.
    assert "the prompt" not in cmd


def test_claude_cmd_uses_opus_when_skill_requests_it(tmp_path):
    spec = SkillSpec(name="x", backend="claude", model="opus", body="ignored")
    cmd = _build_claude_cmd("claude", "the prompt", spec, tmp_path)
    # Find the value passed after --model
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "opus"
    assert cmd[-1] == str(tmp_path)  # cmd ends with the add-dir path


def test_codex_cmd_includes_expected_flags(tmp_path):
    spec = SkillSpec(name="x", backend="codex", body="ignored")
    cmd = _build_codex_cmd("codex", "the prompt", spec, tmp_path)
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "--json" in cmd
    assert "--sandbox" in cmd and "workspace-write" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "-C" in cmd and str(tmp_path) in cmd
    # Prompt comes via stdin; `-` is the explicit stdin marker.
    assert "the prompt" not in cmd
    assert cmd[-1] == "-"


# ---------- Output parsing ----------


def test_parse_claude_output_recovers_cost():
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.0042})
    parsed, cost = _parse_claude_output(payload)
    assert parsed["result"] == "ok"
    assert cost == 0.0042


def test_parse_codex_output_extracts_agent_messages_and_usage():
    """The parser pulls all agent_message texts + the final usage block out of the stream."""
    payload = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t1"}),
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "OK"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 5}}),
    ])
    parsed, cost = _parse_codex_output(payload)
    assert parsed["response_text"] == "OK"
    assert parsed["usage"]["output_tokens"] == 5
    assert any(item.get("text") == "OK" for item in parsed["items"])
    assert cost is None


def test_parse_codex_output_fallback_when_no_structured_events():
    """If codex output lacks structured events, fall back to the last NDJSON object."""
    payload = "\n".join([
        json.dumps({"type": "start", "msg": "go"}),
        json.dumps({"type": "final", "response": "done"}),
    ])
    parsed, cost = _parse_codex_output(payload)
    assert parsed["type"] == "final"
    assert parsed["response"] == "done"


# ---------- run_agent backend dispatch (mocked subprocess) ----------


def _mock_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


def test_run_agent_dispatches_to_claude_for_claude_backend(tmp_path, monkeypatch):
    import quant.agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "AGENT_LOGS_DIR", tmp_path / "agentlogs")
    monkeypatch.setattr(runner_mod, "_check_cli", lambda backend: "/usr/bin/claude")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _mock_completed_process(json.dumps({"result": "ok", "total_cost_usd": 0.01}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Skill on disk
    skill_md = tmp_path / "test-claude-skill.md"
    skill_md.write_text(
        "---\nname: test-claude-skill\nbackend: claude\nallowed_tools: [WebSearch]\n"
        "max_budget_usd: 0.5\nrequired_params: [foo]\n---\n\nbody {{foo}}\n"
    )
    monkeypatch.setattr(runner_mod, "SKILLS_DIR", tmp_path)
    res = run_agent(Invocation(skill="test-claude-skill", params={"foo": "bar"}, log_subdir="t/1"))
    assert res.ok
    assert res.backend == "claude"
    assert res.cost_usd == 0.01
    assert captured["cmd"][0] == "/usr/bin/claude"
    assert "-p" in captured["cmd"]


def test_run_agent_dispatches_to_codex_for_codex_backend(tmp_path, monkeypatch):
    import quant.agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "AGENT_LOGS_DIR", tmp_path / "agentlogs")
    monkeypatch.setattr(runner_mod, "_check_cli", lambda backend: "/usr/bin/codex")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate the modern codex NDJSON output.
        return _mock_completed_process("\n".join([
            json.dumps({"type": "thread.started", "thread_id": "tx"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "schema written"}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}}),
        ]))

    monkeypatch.setattr(subprocess, "run", fake_run)
    skill_md = tmp_path / "test-codex-skill.md"
    skill_md.write_text(
        "---\nname: test-codex-skill\nbackend: codex\nrequired_params: [where]\n---\n\nschema for {{where}}\n"
    )
    monkeypatch.setattr(runner_mod, "SKILLS_DIR", tmp_path)
    res = run_agent(Invocation(skill="test-codex-skill", params={"where": "X"}, log_subdir="t/2"))
    assert res.ok
    assert res.backend == "codex"
    assert captured["cmd"][0] == "/usr/bin/codex"
    assert captured["cmd"][1] == "exec"
    assert res.parsed["response_text"] == "schema written"
    assert res.parsed["usage"]["output_tokens"] == 3


def test_run_agent_idempotency(tmp_path, monkeypatch):
    """Second run with same log_subdir reads cached result; does not invoke subprocess again."""
    import quant.agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "AGENT_LOGS_DIR", tmp_path / "agentlogs")
    monkeypatch.setattr(runner_mod, "_check_cli", lambda backend: "/usr/bin/claude")
    call_counter = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_counter["n"] += 1
        return _mock_completed_process(json.dumps({"result": "ok"}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    skill_md = tmp_path / "idem.md"
    skill_md.write_text("---\nname: idem\nbackend: claude\n---\n\nhi\n")
    monkeypatch.setattr(runner_mod, "SKILLS_DIR", tmp_path)

    r1 = run_agent(Invocation(skill="idem", params={}, log_subdir="idem/once"))
    r2 = run_agent(Invocation(skill="idem", params={}, log_subdir="idem/once"))
    assert call_counter["n"] == 1, "second call should be cached"
    assert r1.ok and r2.ok

    r3 = run_agent(Invocation(skill="idem", params={}, log_subdir="idem/once", force=True))
    assert call_counter["n"] == 2, "force=True should bypass cache"
    assert r3.ok
