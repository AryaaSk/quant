"""Phase 2: collect data + lock held-out window + build structured frame + state vectors.

Usage: python scripts/02_collect.py <slug> [--force]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from quant.config import load_market  # noqa: E402
from quant.pipeline.collect import collect  # noqa: E402
from quant.pipeline.state import build_state  # noqa: E402
from quant.pipeline.structure import structure  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/02_collect.py <slug> [--force]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    force = "--force" in sys.argv

    market = load_market(slug)
    print(f"[{slug}] collect ...")
    written = collect(market, force=force)
    for name, p in written.items():
        print(f"  raw: {name} -> {p}")
    print(f"[{slug}] structure ...")
    structured = structure(market, force=force)
    print(f"  structured: {structured}")
    print(f"[{slug}] state ...")
    seq, lab = build_state(market, force=force)
    print(f"  sequences: {seq}")
    print(f"  labels:    {lab}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
