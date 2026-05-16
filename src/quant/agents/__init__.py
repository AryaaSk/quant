"""Agent runner: dispatches subprocess calls to `claude -p` or `codex exec` per skill frontmatter.

See `skills/README.md` for the markdown brief convention.
"""

from quant.agents.runner import (  # noqa: F401
    AgentResult,
    Invocation,
    SkillSpec,
    load_skill,
    render_prompt,
    run_agent,
    run_agents_parallel,
)
