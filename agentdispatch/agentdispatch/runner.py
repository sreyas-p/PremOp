"""The agent loop.

One function does the real work: `run_agent` takes an agent name and a brief,
runs the model against that agent's tools until it stops calling them, and
returns a persisted Task carrying the result.
"""

from __future__ import annotations

import functools
import logging

import anthropic

from . import agents, tools
from .config import settings
from .context import current_task_id, task_scope
from .models import AgentSpec, Task, TaskStatus
from .store import TaskStore

log = logging.getLogger(__name__)

MAX_PAUSE_RESTARTS = 3


@functools.lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    """The Anthropic client.

    Constructed with no arguments on purpose: the SDK resolves ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile in that order, so this
    works without an API key in the environment.
    """
    return anthropic.Anthropic()


@functools.lru_cache(maxsize=1)
def store() -> TaskStore:
    return TaskStore(settings.db_path)


def _text_of(message: anthropic.types.Message) -> str:
    return "\n".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


def _tool_names(message: anthropic.types.Message) -> list[str]:
    return [
        block.name
        for block in message.content
        if getattr(block, "type", None) == "tool_use"
    ]


def run_agent(agent_name: str, instructions: str, *, parent_id: str | None = None) -> Task:
    """Run one agent to completion and return its persisted Task.

    Errors are captured onto the Task rather than raised: a failing subagent
    should report back to its caller, not tear down the whole dispatch.
    """
    spec: AgentSpec = agents.get(agent_name)
    task = Task(
        agent=spec.name,
        instructions=instructions,
        status=TaskStatus.RUNNING,
        parent_id=parent_id if parent_id is not None else current_task_id(),
    )
    store().save(task)

    try:
        with task_scope(task.id):
            result, called, usage = _drive(spec, instructions)
        task.result = result
        task.tool_calls = called
        task.input_tokens, task.output_tokens = usage
        task.status = TaskStatus.SUCCEEDED
    except Exception as exc:  # noqa: BLE001 — recorded on the task, not swallowed
        log.exception("agent %s failed on task %s", spec.name, task.id)
        task.status = TaskStatus.FAILED
        task.error = f"{type(exc).__name__}: {exc}"

    return store().save(task)


def _drive(spec: AgentSpec, instructions: str) -> tuple[str, list[str], tuple[int, int]]:
    """Run the tool-use loop until the model stops calling tools."""
    resolved = tools.resolve(spec.tools)
    messages: list[dict] = [{"role": "user", "content": instructions}]
    called: list[str] = []
    input_tokens = output_tokens = 0
    restarts = 0

    while True:
        runner = client().beta.messages.tool_runner(
            model=settings.model,
            max_tokens=spec.max_tokens,
            system=spec.system,
            thinking={"type": "adaptive"},
            output_config={"effort": spec.effort or settings.effort},
            tools=resolved,
            messages=messages,
        )

        last = None
        for message in runner:
            last = message
            called.extend(_tool_names(message))
            input_tokens += message.usage.input_tokens
            output_tokens += message.usage.output_tokens

            # Mirror the runner's history so a restart can resume from here.
            messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                messages.append(tool_response)

        if last is None:
            raise RuntimeError(f"Agent {spec.name!r} produced no response.")

        # A server-tool turn can stop mid-flight; the Python runner cannot be
        # resumed in place, so rebuild it over the mirrored history.
        if last.stop_reason != "pause_turn":
            break
        restarts += 1
        if restarts > MAX_PAUSE_RESTARTS:
            raise RuntimeError(
                f"Agent {spec.name!r} still paused after {MAX_PAUSE_RESTARTS} restarts."
            )

    return _text_of(last), called, (input_tokens, output_tokens)


def dispatch(request: str) -> Task:
    """Entry point: hand a plain-language request to the coordinator."""
    return run_agent("dispatcher", request)
