"""Codex backend smoke test. Gated by QUANT_ENABLE_AGENTS=1 because it spends tokens.

Runs ONE real `codex exec` call against the `_synthetic` market's raw data dir and asks
codex to summarize what it sees. We don't assert on content (codex output is non-deterministic)
but we DO assert the runner dispatched correctly, the subprocess returned non-zero output,
and the result is parseable.
"""
from __future__ import annotations

import os
import shutil

import pytest

from quant.agents.runner import Invocation, run_agent
from quant.config import load_market
from quant.pipeline.collect import collect
from quant.pipeline.state import build_state
from quant.pipeline.structure import structure


pytestmark = pytest.mark.skipif(
    os.environ.get("QUANT_ENABLE_AGENTS") != "1",
    reason="agent smoke tests gated behind QUANT_ENABLE_AGENTS=1 to avoid spending tokens in CI",
)


def test_codex_exec_is_available():
    assert shutil.which("codex"), "codex CLI not on PATH; install codex before running this test"


def test_codex_round_trip_against_synthetic(tmp_path, monkeypatch):
    """One real codex exec call. Asks codex to list files in a tmp dir + summarize.

    Asserts subprocess exited 0, produced non-empty stdout, and the runner parsed it
    into AgentResult.parsed (NDJSON last-event). We avoid spending much by using a
    minimal in-flight prompt.
    """
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")

    market = load_market("_synthetic")
    collect(market)
    structure(market)
    build_state(market)

    # Build a tiny one-off skill on disk just for this test (avoids spending the full
    # propose-schema budget on a sanity check).
    skill_md = tmp_path / "tiny.md"
    skill_md.write_text(
        "---\nname: tiny\nbackend: codex\nrequired_params: [target_dir]\ntimeout_s: 180\n---\n\n"
        "List the files in {{target_dir}} using `ls -la` (run it via your shell). "
        "Then print one short sentence summarizing what you saw. Do not modify any files."
    )
    import quant.agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "SKILLS_DIR", tmp_path)

    result = run_agent(Invocation(
        skill="tiny",
        params={"target_dir": str(market.raw_dir().absolute())},
        log_subdir=f"smoke/tiny/{int(os.times().elapsed)}",
    ))
    assert result.ok, f"codex exec failed: rc={result.returncode}, err={result.stderr[:400]}"
    assert result.backend == "codex"
    assert result.stdout, "codex produced no stdout"
    # Some response field present (parser may parse 0+ NDJSON events; last one wins)
    assert result.parsed is not None, "runner failed to parse codex NDJSON output"
