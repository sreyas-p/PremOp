"""The tool that makes this a dispatch system rather than one big agent.

`delegate_to_agent` runs a named agent as a subtask: fresh context window, only
that agent's tools, and only its summary comes back. The parent never sees the
subagent's forty tool results — just the answer.
"""

from __future__ import annotations

import threading
from contextvars import copy_context

from anthropic import beta_tool

from ..context import MAX_DELEGATION_DEPTH, current_depth, current_task_id


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
def delegate_parallel(agents: list[str], instructions: list[str]) -> str:
    """Run several specialists at once and return all their results together.

    Use this whenever a request has parts that do not depend on each other —
    "check my mail and my liked videos", "summarize X and also note Y". The
    subagents run concurrently, so three of them cost about as much wall-clock
    time as the slowest one rather than the sum.

    Only use it for genuinely independent work. If one subtask needs another's
    output, run them in sequence with delegate_to_agent instead, passing the
    first one's result into the second one's brief.

    The two lists are positional: agents[i] receives instructions[i], so they
    must be the same length. Each subagent starts cold and sees only its own
    instruction string — brief each one completely.

    Args:
        agents: Agent names, one per subtask.
        instructions: Complete self-contained briefs, aligned with `agents`.
    """
    from ..runner import run_agent

    if len(agents) != len(instructions):
        return (
            f"Refused: got {len(agents)} agents but {len(instructions)} "
            "instructions. They are positional and must be the same length."
        )
    if not agents:
        return "Refused: nothing to delegate."
    if current_depth() >= MAX_DELEGATION_DEPTH:
        return (
            f"Delegation refused: already {current_depth()} levels deep "
            f"(limit {MAX_DELEGATION_DEPTH}). Do this work directly instead."
        )

    parent = current_task_id()
    results: list[tuple[int, str]] = []
    lock = threading.Lock()

    def work(index: int, agent: str, brief: str) -> None:
        try:
            task = run_agent(agent, brief, parent_id=parent)
            body = f"failed: {task.error}" if task.error else (task.result or "(no output)")
            outcome = f"### {agent} (task {task.id})\n{body}"
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            outcome = f"### {agent}\nfailed: {type(exc).__name__}: {exc}"
        with lock:
            results.append((index, outcome))

    # ContextVars do not cross thread boundaries on their own, so each worker
    # gets a copy of this context — otherwise the delegation-depth guard would
    # reset to zero inside every thread and the cap would mean nothing.
    threads = [
        threading.Thread(
            target=copy_context().run, args=(work, i, a, b), daemon=True
        )
        for i, (a, b) in enumerate(zip(agents, instructions))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = [text for _, text in sorted(results)]
    return f"{len(ordered)} subagent(s) finished.\n\n" + "\n\n".join(ordered)


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
    "delegate_parallel": delegate_parallel,
    "list_agents": list_agents,
}
