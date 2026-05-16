"""Phase 4: train. Gated on H200 confirmation for transformer markets.

Usage:
  python scripts/03_train.py <slug>                       # LightGBM markets run anywhere
  python scripts/03_train.py <slug> --confirm-gpu-free    # transformer markets need this
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import load_market  # noqa: E402
from quant.pipeline.train import train  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/03_train.py <slug> [--confirm-gpu-free] [--device cpu|cuda|mps|auto]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    confirm = "--confirm-gpu-free" in sys.argv or os.environ.get("H200_CONFIRMED") == "1"

    device = "auto"
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1]

    market = load_market(slug)
    if market.model == "transformer" and device != "cpu" and not confirm:
        print(
            f"[{slug}] transformer training requires --confirm-gpu-free (H200 GPU 6 is shared with Zoral).\n"
            f"  Run again with --confirm-gpu-free once GPU 6 is idle. To train on CPU now, pass --device cpu.",
            file=sys.stderr,
        )
        return 2

    ckpt = train(market, device=device)
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
