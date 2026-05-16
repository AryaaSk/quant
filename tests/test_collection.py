"""Collection: synthetic adapter writes valid rows + held-out fence is locked."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant.config import load_market
from quant.pipeline.collect import collect


def test_synthetic_collection_writes_required_cols(tmp_path, monkeypatch):
    # Redirect DATA_DIR + RUNS_DIR so the test does not pollute the real data tree.
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    market = load_market("_synthetic")

    paths = collect(market)
    assert paths
    for name, p in paths.items():
        df = pd.read_parquet(p)
        for col in ("source_published_at", "scraped_at", "source_url", "source_type"):
            assert col in df.columns, f"{name} missing {col}"


def test_held_out_manifest_is_locked(tmp_path, monkeypatch):
    """The held-out manifest is written by build_state (it uses the data's actual range,
    not the wall clock). Subsequent runs must NOT move the fence.
    """
    import quant.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    from quant.pipeline.structure import structure
    from quant.pipeline.state import build_state

    market = load_market("_synthetic")
    collect(market)
    structure(market)
    build_state(market)

    manifest_path = market.held_out_dir() / "MANIFEST.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert "held_out_start" in payload
    assert "last_event_observed" in payload
    assert "locked_at" in payload

    # Second build must NOT overwrite the manifest.
    original = manifest_path.read_text()
    build_state(market, force=True)
    assert manifest_path.read_text() == original
