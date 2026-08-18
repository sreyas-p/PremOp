"""SQLite persistence for tasks.

Tasks outlive the process that created them, so a dispatch can be inspected,
retried, or audited after the fact.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import Task, TaskStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    agent         TEXT NOT NULL,
    instructions  TEXT NOT NULL,
    status        TEXT NOT NULL,
    parent_id     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    result        TEXT,
    error         TEXT,
    tool_calls    TEXT NOT NULL DEFAULT '[]',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS tasks_parent_idx ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS tasks_created_idx ON tasks(created_at DESC);
"""


class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, task: Task) -> Task:
        task.updated_at = datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (id, agent, instructions, status, parent_id,
                                   created_at, updated_at, result, error,
                                   tool_calls, input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    result=excluded.result,
                    error=excluded.error,
                    tool_calls=excluded.tool_calls,
                    input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens
                """,
                (
                    task.id,
                    task.agent,
                    task.instructions,
                    task.status.value,
                    task.parent_id,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.result,
                    task.error,
                    json.dumps(task.tool_calls),
                    task.input_tokens,
                    task.output_tokens,
                ),
            )
        return task

    def get(self, task_id: str) -> Task | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _to_task(row) if row else None

    def list(self, limit: int = 20, parent_id: str | None = None) -> list[Task]:
        query = "SELECT * FROM tasks"
        params: list[object] = []
        if parent_id is not None:
            query += " WHERE parent_id = ?"
            params.append(parent_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_task(row) for row in rows]


def _to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        agent=row["agent"],
        instructions=row["instructions"],
        status=TaskStatus(row["status"]),
        parent_id=row["parent_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        result=row["result"],
        error=row["error"],
        tool_calls=json.loads(row["tool_calls"]),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
    )
