"""Phase 1: run market screening across the candidate universe.

For each `markets/<slug>.yaml`, attempt to fetch one sample from each declared data source
and compute a feasibility score per the rubric in research/04-screening-method.md.

This v1 implementation runs without LLM agents: it inspects the yaml + attempts a small
fetch via the adapter (which raises for unimplemented adapters, recorded as "data_unreachable").

Output:
  runs/screening/leaderboard.md  human-readable
  runs/screening/<slug>.json     machine-readable per market
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.adapters import get_adapter  # noqa: E402
from quant.config import RUNS_DIR, list_markets, load_market  # noqa: E402


def screen_market(slug: str) -> dict:
    cfg = load_market(slug)
    result = {
        "slug": slug,
        "display_name": cfg.display_name,
        "platform": cfg.platform,
        "notes": cfg.notes,
        "model": cfg.model,
        "state_dim": cfg.state_vector.dim,
        "data_sources": [s.name for s in cfg.data_sources],
        "sources_reachable": {},
        "sources_error": {},
        "events_per_year": None,
        "score": 0,
        "verdict": "",
        "advance": False,
    }

    reachable = 0
    for source in cfg.data_sources:
        try:
            adapter = get_adapter(source.name)
            # We do not actually fetch the full archive in screening; we attempt the call
            # with a tight cap and capture the result type.
            params = dict(source.params or {})
            params.setdefault("n_days", 60)
            params.setdefault("n_events", 5)
            df = adapter.fetch(market=cfg, source_params=params)
            result["sources_reachable"][source.name] = int(len(df))
            reachable += 1
        except NotImplementedError as e:
            result["sources_error"][source.name] = f"not_implemented: {e}"
        except Exception as e:  # pragma: no cover (depends on environment)
            result["sources_error"][source.name] = f"error: {e}"

    data_reachable = float(reachable == len(cfg.data_sources))
    # Crude per-yaml score (Phase 1 rubric).
    friction_bps = cfg.backtest.commission_bps + cfg.backtest.base_slippage_bps
    score = (
        30 * data_reachable
        + 25 * (1.0 if cfg.model == "transformer" else 0.5)  # informal proxy
        + 20 * 1.0  # liquidity inferred from market tier; placeholder until adapters report
        + 15 * 1.0  # info density placeholder until text adapter wired up
        + 10 * max(0.0, 1.0 - friction_bps / 300.0)
    )
    advance = data_reachable >= 1.0
    verdict = (
        "advance: all sources reachable"
        if advance
        else f"kill: {len(cfg.data_sources) - reachable}/{len(cfg.data_sources)} sources unimplemented"
    )
    result["score"] = round(score, 1)
    result["verdict"] = verdict
    result["advance"] = advance
    return result


def write_leaderboard(results: list[dict]) -> Path:
    out_dir = RUNS_DIR / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "leaderboard.md"
    rows = sorted(results, key=lambda r: r["score"], reverse=True)
    lines = [
        "# Screening leaderboard",
        "",
        "score | slug | platform | model | verdict",
        "---|---|---|---|---",
    ]
    for r in rows:
        lines.append(
            f"{r['score']:.1f} | `{r['slug']}` | {r['platform']} | {r['model']} | {r['verdict']}"
        )
    md_path.write_text("\n".join(lines) + "\n")
    for r in rows:
        (out_dir / f"{r['slug']}.json").write_text(json.dumps(r, indent=2, default=str))
    return md_path


def main() -> int:
    slugs = list_markets()
    if not slugs:
        print("no markets in markets/*.yaml")
        return 1
    print(f"screening {len(slugs)} markets ...")
    results: list[dict] = []
    for slug in slugs:
        try:
            r = screen_market(slug)
        except Exception:
            traceback.print_exc()
            r = {"slug": slug, "score": 0, "verdict": "error during screening", "advance": False}
        results.append(r)
        print(f"  {slug:35s} score={r.get('score', 0):5.1f}  {r.get('verdict', '')[:60]}")
    path = write_leaderboard(results)
    print(f"\nleaderboard: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
