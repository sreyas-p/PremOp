"""The objects that move through the system: tasks, their results, and agent specs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Task(BaseModel):
    """One unit of delegated work, owned by exactly one agent."""

    id: str = Field(default_factory=lambda: _new_id("task"))
    agent: str
    instructions: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    result: str | None = None
    error: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class AgentSpec(BaseModel):
    """A named agent: a system prompt plus the exact tools it may reach for.

    The tool list is the security boundary. An agent cannot call a tool that
    isn't named here, no matter what the model decides it wants.
    """

    name: str
    description: str
    system: str
    tools: list[str] = Field(default_factory=list)
    effort: str | None = None
    max_tokens: int = 16000
