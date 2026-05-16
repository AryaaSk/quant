"""Bootstrap-time checks: every market yaml parses, paths resolve, env loads."""
from __future__ import annotations

from pathlib import Path

from quant.config import MARKETS_DIR, list_markets, load_market


def test_markets_dir_has_yamls():
    yamls = list(MARKETS_DIR.glob("*.yaml"))
    assert len(yamls) >= 10, f"expected at least 10 market yamls, found {len(yamls)}"


def test_every_yaml_parses():
    failures: list[tuple[str, Exception]] = []
    for slug in list_markets():
        try:
            cfg = load_market(slug)
            assert cfg.slug == slug or cfg.slug == slug.lstrip("_"), f"slug field mismatch for {slug}"
        except Exception as e:
            failures.append((slug, e))
    assert not failures, "yaml parse failures: " + "\n".join(f"{s}: {e}" for s, e in failures)


def test_state_vector_blocks_cover_dim():
    for slug in list_markets():
        cfg = load_market(slug)
        cursor = 0
        for block in cfg.state_vector.blocks:
            assert block.slot_start == cursor, f"{slug}: gap before block {block.name}"
            cursor = block.slot_end
        assert cursor == cfg.state_vector.dim, f"{slug}: blocks cover [0,{cursor}) but dim={cfg.state_vector.dim}"


def test_smoke_market_present():
    cfg = load_market("_synthetic")
    assert cfg.platform == "synthetic"
    assert cfg.state_vector.dim == 64
