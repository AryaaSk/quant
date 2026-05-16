"""Stage B orchestrator: ask a codex agent to propose a state-vector schema yaml.

Reads `data/raw/<slug>/`, invokes the `propose-schema` skill (backend: codex), writes
the proposed yaml to `markets/<slug>.proposed.yaml`. Use `--apply` to overwrite the live
`markets/<slug>.yaml`.

Usage:
  python scripts/06_propose_schema.py <slug>
  python scripts/06_propose_schema.py <slug> --apply
  python scripts/06_propose_schema.py <slug> --dry-run     # show the prompt, do not invoke

Requires QUANT_ENABLE_AGENTS=1 to actually invoke codex (preserves spend by default).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.agents.runner import Invocation, load_skill, render_prompt, run_agent  # noqa: E402
from quant.config import MARKETS_DIR, DATA_DIR  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/06_propose_schema.py <slug> [--apply] [--dry-run]", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    apply_flag = "--apply" in sys.argv
    dry_run = "--dry-run" in sys.argv

    raw_dir = DATA_DIR / "raw" / slug
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"raw dir is empty: {raw_dir}. Run scripts/02_collect.py first.", file=sys.stderr)
        return 1

    proposed_path = MARKETS_DIR / f"{slug}.proposed.yaml"
    live_path = MARKETS_DIR / f"{slug}.yaml"

    params = {
        "market_slug": slug,
        "raw_dir": str(raw_dir.absolute()),
        "output_path": str(proposed_path.absolute()),
        "existing_yamls_dir": str(MARKETS_DIR.absolute()),
    }

    if dry_run:
        skill = load_skill("propose-schema")
        rendered = render_prompt(skill, params)
        print("=== rendered prompt (dry-run) ===\n")
        print(rendered)
        return 0

    if os.environ.get("QUANT_ENABLE_AGENTS") != "1":
        print(
            "agent invocation gated: set QUANT_ENABLE_AGENTS=1 to spend codex tokens.\n"
            "Use --dry-run to inspect the prompt without spending.",
            file=sys.stderr,
        )
        return 2

    result = run_agent(Invocation(
        skill="propose-schema",
        params=params,
        log_subdir=f"{slug}/propose-schema/{int(__import__('time').time())}",
    ))
    print(f"agent ok={result.ok} elapsed={result.elapsed_s:.1f}s backend={result.backend}")
    if not result.ok:
        print(f"agent error: {result.error or result.stderr[:500]}", file=sys.stderr)
        return 1

    if not proposed_path.exists():
        print(f"agent did not write expected output at {proposed_path}", file=sys.stderr)
        return 1

    print(f"proposed yaml at: {proposed_path}")
    print("---")
    print(proposed_path.read_text()[:2000])
    if apply_flag:
        # Validate before overwriting.
        from quant.config import load_market
        try:
            load_market(proposed_path)
        except Exception as e:
            print(f"proposed yaml failed validation: {e}", file=sys.stderr)
            return 1
        if live_path.exists():
            backup = live_path.with_suffix(".yaml.bak")
            backup.write_text(live_path.read_text())
            print(f"backed up existing yaml to {backup}")
        live_path.write_text(proposed_path.read_text())
        print(f"applied -> {live_path}")
    else:
        print("\nDry of the agent done. Use --apply to overwrite markets/<slug>.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
