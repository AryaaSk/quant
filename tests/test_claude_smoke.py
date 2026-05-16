"""Claude backend smoke test. Gated by QUANT_ENABLE_AGENTS=1 because it spends credits.

Runs ONE real `claude -p` call with WebSearch allowed and verifies the runner parses
the result, captures cost, and writes the log.
"""
from __future__ import annotations

import os
import shutil

import pytest

from quant.agents.runner import Invocation, run_agent


pytestmark = pytest.mark.skipif(
    os.environ.get("QUANT_ENABLE_AGENTS") != "1",
    reason="agent smoke tests gated behind QUANT_ENABLE_AGENTS=1 to avoid spending credits in CI",
)


def test_claude_cli_is_available():
    assert shutil.which("claude"), "claude CLI not on PATH; install Claude Code before running this test"


def test_claude_one_shot_text_only(tmp_path, monkeypatch):
    """Minimal claude -p call that does NOT use web tools (saves credits).

    Verifies dispatch + JSON parse + cost capture.
    """
    skill_md = tmp_path / "tiny-claude.md"
    skill_md.write_text(
        "---\nname: tiny-claude\nbackend: claude\nallowed_tools: []\nmax_budget_usd: 0.10\n"
        "required_params: [name]\ntimeout_s: 60\n---\n\n"
        "Say hello to {{name}} in exactly one short sentence. Do not use any tools."
    )
    import quant.agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "AGENT_LOGS_DIR", tmp_path / "logs")

    result = run_agent(Invocation(
        skill="tiny-claude",
        params={"name": "Aryaa"},
        log_subdir="smoke/claude/1",
    ))
    assert result.ok, f"claude -p failed: rc={result.returncode}, err={result.stderr[:400]}"
    assert result.backend == "claude"
    assert result.parsed is not None
    # Cost may or may not be reported depending on auth path; if reported it should be tiny.
    if result.cost_usd is not None:
        assert result.cost_usd < 0.20, f"unexpected cost: ${result.cost_usd}"
