"""The tool that makes this a dispatch system rather than one big agent.

`delegate_to_agent` runs a named agent as a subtask: fresh context window, only
that agent's tools, and only its summary comes back. The parent never sees the
subagent's forty tool results — just the answer.
"""

from __future__ import annotations

from anthropic import beta_tool

from ..context import MAX_DELEGATION_DEPTH, current_depth


@beta_tool
def delegate_to_agent(agent: str, instructions: str) -> str:
    """Run a specialist agent on a self-contained subtask and return its result.

    Delegate when a subtask is genuinely separable — a different app, a
    different body of context, or work you'd otherwise interleave badly with
    what you're already doing. Do not delegate work you could finish yourself in
    a couple of tool calls; a subagent starts cold and re-derives context you
    already have.

    The subagent sees only what you write in `instructions`. It cannot see this
    conversation, the user's original request, or anything you have already
    read. Brief it completely the first time: state the goal, the specific
    inputs (IDs, queries, names), and the shape of the answer you want back.

    Args:
        agent: The name of the agent to run. Call list_agents first if you are
            not certain which agents exist and what tools each one has.
        instructions: A complete, self-contained brief for the subagent.
    """
    # Imported here rather than at module scope: the runner imports the tool
    # registry, which imports this module.
    from ..runner import run_agent

    if current_depth() >= MAX_DELEGATION_DEPTH:
        return (
            f"Delegation refused: already {current_depth()} levels deep "
            f"(limit {MAX_DELEGATION_DEPTH}). Do this work directly instead."
        )

    task = run_agent(agent, instructions)
    if task.error:
        return f"Subagent {agent!r} failed: {task.error}"
    return f"Subagent {agent!r} (task {task.id}) reported:\n\n{task.result}"


@beta_tool
def list_agents() -> str:
    """List the agents available to delegate to, and the tools each one has."""
    from ..agents import AGENTS

    lines = []
    for spec in AGENTS.values():
        tools = ", ".join(spec.tools) or "[no tools]"
        lines.append(f"- {spec.name}: {spec.description}\n  tools: {tools}")
    return "\n".join(lines)


TOOLS = {
    "delegate_to_agent": delegate_to_agent,
    "list_agents": list_agents,
}
