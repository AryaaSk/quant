"""Phase 1 screening: every yaml produces a screening row (even if it scores 0)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quant.config import list_markets


def test_screening_module_imports():
    import importlib
    spec = importlib.util.spec_from_file_location(
        "_screen",
        Path(__file__).resolve().parents[1] / "scripts" / "01_screen_markets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.screen_market)
    assert callable(mod.write_leaderboard)


def test_every_yaml_can_be_screened():
    import importlib
    spec = importlib.util.spec_from_file_location(
        "_screen",
        Path(__file__).resolve().parents[1] / "scripts" / "01_screen_markets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for slug in list_markets():
        result = mod.screen_market(slug)
        assert "score" in result
        assert "verdict" in result
        assert "data_sources" in result
