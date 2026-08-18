"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from . import agents, tools
from .config import settings
from .models import Task, TaskStatus


def _print_task(task: Task, *, verbose: bool = False) -> None:
    marker = {
        TaskStatus.SUCCEEDED: "✓",
        TaskStatus.FAILED: "✗",
        TaskStatus.RUNNING: "…",
        TaskStatus.PENDING: "·",
    }[task.status]
    print(f"{marker} {task.id}  [{task.agent}]  {task.status.value}")
    if verbose:
        print(f"  parent:      {task.parent_id or '—'}")
        print(f"  created:     {task.created_at.isoformat(timespec='seconds')}")
        print(f"  tokens:      {task.input_tokens} in / {task.output_tokens} out")
        print(f"  tool calls:  {', '.join(task.tool_calls) or '—'}")
        print(f"\n  instructions:\n{_indent(task.instructions)}")
    if task.error:
        print(f"\n  error: {task.error}")
    elif task.result and verbose:
        print(f"\n  result:\n{_indent(task.result)}")


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_agent, store

    task = run_agent(args.agent, args.request)

    if task.error:
        print(f"Task {task.id} failed: {task.error}", file=sys.stderr)
        return 1

    print(task.result or "[no textual result]")

    subtasks = store().list(parent_id=task.id, limit=50)
    if subtasks:
        print(f"\n— {len(subtasks)} subtask(s) —")
        for subtask in subtasks:
            _print_task(subtask)
    print(
        f"\ntask {task.id} · {task.input_tokens} in / {task.output_tokens} out tokens"
    )
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    from .runner import store

    tasks = store().list(limit=args.limit)
    if not tasks:
        print("No tasks recorded yet.")
        return 0
    for task in tasks:
        _print_task(task)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from .runner import store

    task = store().get(args.task_id)
    if task is None:
        print(f"No task with id {args.task_id!r}.", file=sys.stderr)
        return 1
    _print_task(task, verbose=True)
    return 0


def _cmd_agents(_: argparse.Namespace) -> int:
    for spec in agents.AGENTS.values():
        print(f"{spec.name}\n  {spec.description}")
        print(f"  effort: {spec.effort or settings.effort}")
        print(f"  tools:  {', '.join(spec.tools) or '—'}\n")
    return 0


def _cmd_tools(_: argparse.Namespace) -> int:
    for name in tools.available():
        print(name)
    return 0


def _cmd_auth(_: argparse.Namespace) -> int:
    from .integrations.google_auth import SCOPES, get_credentials

    print("Requesting consent for:")
    for scope in SCOPES:
        print(f"  · {scope}")
    get_credentials(interactive=True)
    print(f"\nToken cached at {settings.google_token_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentdispatch",
        description="Dispatch Claude-powered agents that do tasks across your apps.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log tool activity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="dispatch a request")
    run.add_argument("request", help="what you want done, in plain language")
    run.add_argument(
        "--agent",
        default="dispatcher",
        help="run one agent directly instead of going through the coordinator",
    )
    run.set_defaults(func=_cmd_run)

    tasks = subparsers.add_parser("tasks", help="list recent tasks")
    tasks.add_argument("--limit", type=int, default=20)
    tasks.set_defaults(func=_cmd_tasks)

    show = subparsers.add_parser("show", help="show one task in full")
    show.add_argument("task_id")
    show.set_defaults(func=_cmd_show)

    subparsers.add_parser("agents", help="list agents").set_defaults(func=_cmd_agents)
    subparsers.add_parser("tools", help="list registered tools").set_defaults(
        func=_cmd_tools
    )

    auth = subparsers.add_parser("auth", help="grant access to a service")
    auth_subparsers = auth.add_subparsers(dest="service", required=True)
    auth_subparsers.add_parser("google", help="run the Google OAuth consent flow")
    auth.set_defaults(func=_cmd_auth)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
