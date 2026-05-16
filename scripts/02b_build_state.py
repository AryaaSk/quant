"""Stage 3: build state vectors. Standalone script so it can be invoked on H200
where MPNet / Voyage embeddings benefit from GPU + low-latency networking.

Usage:
  python scripts/02b_build_state.py <slug> [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import load_market  # noqa: E402
from quant.pipeline.state import build_state  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/02b_build_state.py <slug> [--force]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    force = "--force" in sys.argv
    market = load_market(slug)
    print(f"[{slug}] state build ...")
    seq, lab = build_state(market, force=force)
    print(f"  sequences: {seq}")
    print(f"  labels:    {lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
