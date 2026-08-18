"""Ambient state for the currently-executing task.

Kept in a ContextVar rather than threaded through every tool signature, so that
tools stay simple functions the model can call without bookkeeping arguments.
Its one job is letting a delegated subtask record its parent.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)
_depth: ContextVar[int] = ContextVar("delegation_depth", default=0)

MAX_DELEGATION_DEPTH = 2


def current_task_id() -> str | None:
    return _current_task_id.get()


def current_depth() -> int:
    return _depth.get()


@contextmanager
def task_scope(task_id: str) -> Iterator[None]:
    """Mark `task_id` as the running task for the duration of the block."""
    task_token = _current_task_id.set(task_id)
    depth_token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _current_task_id.reset(task_token)
        _depth.reset(depth_token)
