"""Subprocess wrapper around `claude -p` and `codex exec`.

Each agent invocation is driven by a markdown skill file in `skills/`. The frontmatter
declares which CLI backend to use and what tools / budget to allow. The body is a prompt
template with `{{var}}` substitution.

Backend selection rule (codified):
- `backend: claude` -> `claude -p`; use when WebSearch / WebFetch are needed.
- `backend: codex`  -> `codex exec`; use for local-only synthesis. Saves Claude credits.

See `skills/README.md` and `research/09-agent-swarm-implementation.md`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_DIR = REPO_ROOT / "skills"
AGENT_LOGS_DIR = REPO_ROOT / "runs" / "agents"

DEFAULT_CONCURRENCY = int(os.environ.get("QUANT_AGENT_CONCURRENCY", "4"))
DEFAULT_CLAUDE_BUDGET_USD = float(os.environ.get("QUANT_CLAUDE_BUDGET_USD", "5.0"))


@dataclass
class SkillSpec:
    """Parsed skill file: frontmatter + body template."""
    name: str
    backend: str  # "claude" | "codex"
    allowed_tools: list[str] = field(default_factory=list)
    max_budget_usd: float = 1.0
    required_params: list[str] = field(default_factory=list)
    timeout_s: int = 900
    body: str = ""
    path: Path | None = None
    # Optional model selection. For claude: "sonnet" / "opus" / a full model name (e.g.
    # "claude-sonnet-4-6"). For codex: model id ("o3", "gpt-5", ...). Defaults: claude->sonnet
    # (Claude Max plan handles sonnet-volume comfortably); codex->its default.
    model: str | None = None


@dataclass
class Invocation:
    skill: str
    params: dict[str, Any]
    log_subdir: str | None = None  # relative path under runs/agents/ for log output
    force: bool = False             # if True, ignore the idempotency marker


@dataclass
class AgentResult:
    skill: str
    backend: str
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    cost_usd: float | None
    elapsed_s: float
    log_dir: Path
    parsed: dict | None = None
    error: str | None = None


def load_skill(name_or_path: str | Path) -> SkillSpec:
    """Load a skill file by name (looks under skills/) or by explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = SKILLS_DIR / f"{name_or_path}.md"
    if not p.exists():
        raise FileNotFoundError(f"skill not found: {name_or_path}")
    text = p.read_text()
    if not text.startswith("---"):
        raise ValueError(f"skill {p} missing YAML frontmatter")
    _, frontmatter_str, body = text.split("---", 2)
    fm = yaml.safe_load(frontmatter_str) or {}
    return SkillSpec(
        name=fm.get("name") or p.stem,
        backend=str(fm.get("backend", "")).lower(),
        allowed_tools=list(fm.get("allowed_tools") or []),
        max_budget_usd=float(fm.get("max_budget_usd", 1.0)),
        required_params=list(fm.get("required_params") or []),
        timeout_s=int(fm.get("timeout_s", 900)),
        body=body.lstrip("\n"),
        path=p,
        model=fm.get("model"),
    )


_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render_prompt(skill: SkillSpec, params: dict[str, Any]) -> str:
    """Mustache-light substitution: {{var}} replaced with str(params[var])."""
    missing = [p for p in skill.required_params if p not in params]
    if missing:
        raise KeyError(f"skill {skill.name} requires params {missing}, got {sorted(params)}")

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in params:
            return match.group(0)  # leave unrendered if absent
        value = params[key]
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _VAR_RE.sub(_sub, skill.body)


def _check_cli(backend: str) -> str:
    name = {"claude": "claude", "codex": "codex"}.get(backend)
    if not name:
        raise ValueError(f"unknown backend: {backend!r}")
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"{name} CLI not found on PATH. Install it before running agents with backend={backend}."
        )
    return path


def _build_claude_cmd(cli: str, prompt: str, skill: SkillSpec, cwd: Path) -> list[str]:
    """Build claude -p args. Prompt is NOT appended; pass it via stdin in the caller.

    The reason: `--add-dir <directories...>` is variadic, so any positional argument
    after `--add-dir <dir>` gets consumed as another directory. Putting the prompt last
    silently eats it. Safer to feed prompts via stdin.

    Also: we deliberately do NOT use --bare. Bare mode disables OAuth/keychain auth
    and requires ANTHROPIC_API_KEY in the environment, which subprocess invocations
    from a Claude Code session don't have. Without --bare, claude uses the same
    OAuth/keychain auth the user logged in with interactively.
    """
    cmd = [cli, "-p"]
    cmd += ["--output-format", "json"]
    cmd += ["--permission-mode", "bypassPermissions"]
    cmd += ["--disable-slash-commands"]  # don't load skills/plugins from outside this project
    # Model selection: default to sonnet for bulk work under Claude Max; skills can override.
    # See feedback_agent_model_routing.md.
    model = skill.model or "sonnet"
    cmd += ["--model", model]
    if skill.allowed_tools:
        cmd += ["--allowedTools", ",".join(skill.allowed_tools)]
    if skill.max_budget_usd > 0:
        cmd += ["--max-budget-usd", str(skill.max_budget_usd)]
    cmd += ["--add-dir", str(cwd)]
    # NO trailing prompt; stdin feeds it.
    return cmd


def _build_codex_cmd(cli: str, prompt: str, skill: SkillSpec, cwd: Path) -> list[str]:
    """Build codex exec args. Prompt is NOT appended; pass it via stdin in the caller.

    Codex accepts `-` as the prompt arg to read from stdin, but we just rely on the
    default which reads stdin when no prompt is given.
    """
    cmd = [cli, "exec"]
    cmd += ["--json"]
    cmd += ["--sandbox", "workspace-write"]
    cmd += ["--skip-git-repo-check"]
    cmd += ["-C", str(cwd)]
    cmd.append("-")  # explicit stdin marker
    return cmd


def _parse_claude_output(stdout: str) -> tuple[dict | None, float | None]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None, None
    cost = None
    for k in ("total_cost_usd", "cost_usd", "totalCostUsd"):
        if k in data and isinstance(data[k], (int, float)):
            cost = float(data[k])
            break
    return data, cost


def _parse_codex_output(stdout: str) -> tuple[dict | None, float | None]:
    """Parse codex `exec --json` NDJSON stream.

    Codex emits a sequence of events: thread.started, turn.started,
    item.completed (one per agent message), turn.completed (with usage stats).
    We collect them into a dict {response_text, items, usage}.
    """
    response_text_parts: list[str] = []
    items: list[dict] = []
    usage: dict | None = None
    final_legacy = None
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = obj.get("type")
        if event == "item.completed":
            item = obj.get("item") or {}
            items.append(item)
            if item.get("type") == "agent_message" and item.get("text"):
                response_text_parts.append(item["text"])
        elif event == "turn.completed":
            usage = obj.get("usage")
        final_legacy = obj  # keep last event as fallback for legacy parse semantics
    if not items and not response_text_parts and usage is None:
        # No structured events found; fall back to returning the last NDJSON object.
        return final_legacy, None
    return {
        "response_text": "\n".join(response_text_parts),
        "items": items,
        "usage": usage,
    }, None  # codex billing is external; no per-call dollar cost in stream


def run_agent(invocation: Invocation, *, cwd: Path | None = None) -> AgentResult:
    cwd = cwd or REPO_ROOT
    skill = load_skill(invocation.skill)
    prompt = render_prompt(skill, invocation.params)

    log_subdir = invocation.log_subdir or f"{skill.name}/{int(time.time() * 1000)}"
    log_dir = AGENT_LOGS_DIR / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.md").write_text(prompt)
    (log_dir / "params.json").write_text(json.dumps(invocation.params, indent=2, default=str))

    # Idempotency: if a previous run wrote `done.flag`, skip unless force.
    done_flag = log_dir / "done.flag"
    if done_flag.exists() and not invocation.force:
        prior = json.loads((log_dir / "result.json").read_text())
        return AgentResult(
            skill=skill.name,
            backend=skill.backend,
            ok=prior.get("ok", True),
            returncode=prior.get("returncode", 0),
            stdout=prior.get("stdout", ""),
            stderr="(cached)",
            cost_usd=prior.get("cost_usd"),
            elapsed_s=prior.get("elapsed_s", 0.0),
            log_dir=log_dir,
            parsed=prior.get("parsed"),
        )

    cli = _check_cli(skill.backend)
    if skill.backend == "claude":
        cmd = _build_claude_cmd(cli, prompt, skill, cwd)
    elif skill.backend == "codex":
        cmd = _build_codex_cmd(cli, prompt, skill, cwd)
    else:
        raise ValueError(f"unknown backend: {skill.backend}")

    (log_dir / "cmd.json").write_text(json.dumps(cmd, indent=2))
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=skill.timeout_s,
        )
        elapsed = time.time() - started
        rc = proc.returncode
        out, err = proc.stdout or "", proc.stderr or ""
        error = None
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - started
        rc = -1
        out = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        error = f"timeout after {skill.timeout_s}s"
    except Exception as e:  # pragma: no cover
        elapsed = time.time() - started
        rc = -2
        out, err = "", str(e)
        error = str(e)

    (log_dir / "stdout.txt").write_text(out)
    (log_dir / "stderr.txt").write_text(err)

    if skill.backend == "claude":
        parsed, cost = _parse_claude_output(out)
    else:
        parsed, cost = _parse_codex_output(out)

    result = AgentResult(
        skill=skill.name,
        backend=skill.backend,
        ok=(rc == 0 and error is None),
        returncode=rc,
        stdout=out,
        stderr=err,
        cost_usd=cost,
        elapsed_s=elapsed,
        log_dir=log_dir,
        parsed=parsed,
        error=error,
    )

    (log_dir / "result.json").write_text(json.dumps({
        "skill": result.skill,
        "backend": result.backend,
        "ok": result.ok,
        "returncode": result.returncode,
        "cost_usd": result.cost_usd,
        "elapsed_s": result.elapsed_s,
        "error": result.error,
        "parsed": parsed if parsed else None,
        "stdout_truncated": out[-4000:] if len(out) > 4000 else out,
    }, indent=2, default=str))

    if result.ok:
        done_flag.write_text(f"completed_at={time.time()}\n")
    return result


def run_agents_parallel(
    invocations: list[Invocation],
    *,
    concurrency: int | None = None,
    cwd: Path | None = None,
) -> list[AgentResult]:
    if not invocations:
        return []
    cwd = cwd or REPO_ROOT
    n = min(concurrency or DEFAULT_CONCURRENCY, max(1, len(invocations)))
    results: list[AgentResult | None] = [None] * len(invocations)

    cumulative_claude_cost = 0.0
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(run_agent, inv, cwd=cwd): i for i, inv in enumerate(invocations)}
        for future in as_completed(futures):
            i = futures[future]
            res = future.result()
            results[i] = res
            if res.backend == "claude" and res.cost_usd:
                cumulative_claude_cost += res.cost_usd
                if cumulative_claude_cost > DEFAULT_CLAUDE_BUDGET_USD:
                    # Cancel remaining futures (best-effort: running subprocesses won't terminate
                    # cleanly, but new ones won't start).
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(
                        f"cumulative claude cost ${cumulative_claude_cost:.2f} exceeded "
                        f"QUANT_CLAUDE_BUDGET_USD=${DEFAULT_CLAUDE_BUDGET_USD:.2f}; remaining invocations cancelled"
                    )
    return [r for r in results if r is not None]
