"""Phase 6: aggregate metrics from every run and write LEADERBOARD.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.config import RUNS_DIR  # noqa: E402


def main() -> int:
    rows: list[dict] = []
    for slug_dir in sorted(RUNS_DIR.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name == "screening":
            continue
        runs = sorted([p for p in slug_dir.glob("*") if p.is_dir()])
        if not runs:
            continue
        latest = runs[-1]
        metrics_path = latest / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        m["slug"] = slug_dir.name
        m["run"] = latest.name
        rows.append(m)

    if not rows:
        print("no runs found; train at least one market first")
        return 1

    rows.sort(key=lambda r: r.get("net_return", 0), reverse=True)
    lines = [
        "# Leaderboard",
        "",
        "Ranked by net_return after commission, slippage, and safety margin. Brier improvement is computed against market closing implied probability. All metrics include the label-shuffle sanity check (`tests/test_backtest.py::test_label_shuffle_collapses_brier_improvement`).",
        "",
        "slug | net_return | brier_improvement | trades | hit_rate | sharpe | max_drawdown | run",
        "---|---|---|---|---|---|---|---",
    ]
    for r in rows:
        lines.append(
            f"`{r['slug']}` | {r.get('net_return', 0):.2%} | {r.get('brier_improvement', 0):.4f} | "
            f"{r.get('n_trades', 0)} | {r.get('hit_rate', 0):.2%} | {r.get('sharpe', 0):.2f} | "
            f"{r.get('max_drawdown', 0):.2%} | `{r['run']}`"
        )
    # Append the durable footer if present (preserves prose across regenerations).
    footer_path = RUNS_DIR / "LEADERBOARD_FOOTER.md"
    if footer_path.exists():
        lines.append("")
        lines.append(footer_path.read_text().rstrip())
    out = RUNS_DIR / "LEADERBOARD.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
