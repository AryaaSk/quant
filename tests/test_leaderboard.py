"""Leaderboard check: every claim in runs/LEADERBOARD.md is backed by a metrics.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quant.config import RUNS_DIR


def test_leaderboard_format_when_exists():
    out = RUNS_DIR / "LEADERBOARD.md"
    if not out.exists():
        pytest.skip("LEADERBOARD.md not yet generated; run scripts/05_compare.py")
    text = out.read_text()
    assert "# Leaderboard" in text
    # Each non-header row references a run dir that contains a metrics.json.
    for line in text.splitlines():
        m = re.match(r"^`([^`]+)`\s*\|.*\|\s*`([^`]+)`$", line)
        if not m:
            continue
        slug, run = m.group(1), m.group(2)
        metrics_path = RUNS_DIR / slug / run / "metrics.json"
        assert metrics_path.exists(), f"leaderboard claim for {slug}/{run} has no metrics.json"
        json.loads(metrics_path.read_text())
