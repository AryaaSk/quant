"""Verify environment, parse every market yaml, ensure data dirs exist.

Usage: python scripts/00_bootstrap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import DATA_DIR, MARKETS_DIR, RUNS_DIR, list_markets, load_market  # noqa: E402


def main() -> int:
    print(f"quant bootstrap at {Path(__file__).resolve().parents[1]}")
    print("- creating data dirs ...")
    for sub in ("raw", "structured", "state", "held_out"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "screening").mkdir(parents=True, exist_ok=True)

    print("- parsing market yamls ...")
    failures: list[tuple[str, str]] = []
    slugs = list_markets()
    for slug in slugs:
        try:
            cfg = load_market(slug)
            print(f"  ok  {slug:30s} dim={cfg.state_vector.dim} model={cfg.model} time={cfg.time_grain}")
        except Exception as e:
            failures.append((slug, str(e)))
            print(f"  FAIL {slug}: {e}")

    print(f"\n{len(slugs) - len(failures)}/{len(slugs)} market yamls parsed cleanly")
    if failures:
        print("\nfailures:")
        for slug, msg in failures:
            print(f"  - {slug}: {msg}")
        return 1
    print("\nbootstrap ok. Next: pytest -v tests/test_e2e_smoke.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
