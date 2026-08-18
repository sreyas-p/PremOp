"""Append-only ledger: every weight change, who caused it, and how to undo it.

Two tables that are never updated in place — `events` and `versions` — plus a
`facts` table that is current-state only. The facts table can always be rebuilt
from a version snapshot, so it is a cache; the ledger is the record.

This is the part that turns weight editing from the scariest thing in the stack
into something an auditor can sign off on: "who taught the model what, when,
and can you put it back".
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import Event, EventKind, Fact, Version

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind     TEXT NOT NULL,
    at       TEXT NOT NULL,
    actor    TEXT NOT NULL,
    fact_id  TEXT,
    detail   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_fact_idx ON events(fact_id);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);

CREATE TABLE IF NOT EXISTS versions (
    id            TEXT PRIMARY KEY,
    seq           INTEGER NOT NULL,
    snapshot      TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    active_facts  INTEGER NOT NULL DEFAULT 0,
    perplexity    REAL,
    facts_json    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS versions_seq_idx ON versions(seq DESC);

CREATE TABLE IF NOT EXISTS facts (
    id    TEXT PRIMARY KEY,
    data  TEXT NOT NULL
);
"""


class Ledger:
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

    # ── events ──────────────────────────────────────────────────────────

    def record(self, kind: EventKind, *, actor: str = "daemon",
               fact_id: str | None = None, **detail) -> Event:
        event = Event(kind=kind, actor=actor, fact_id=fact_id, detail=detail)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events (kind, at, actor, fact_id, detail) VALUES (?,?,?,?,?)",
                (
                    event.kind.value,
                    event.at.isoformat(),
                    event.actor,
                    event.fact_id,
                    json.dumps(event.detail, default=str),
                ),
            )
            event.seq = cursor.lastrowid or 0
        return event

    def events(self, *, limit: int = 100, fact_id: str | None = None,
               kind: EventKind | None = None) -> list[Event]:
        query = "SELECT * FROM events"
        clauses: list[str] = []
        params: list[object] = []
        if fact_id:
            clauses.append("fact_id = ?")
            params.append(fact_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind.value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            Event(
                seq=row["seq"],
                kind=EventKind(row["kind"]),
                at=row["at"],
                actor=row["actor"],
                fact_id=row["fact_id"],
                detail=json.loads(row["detail"]),
            )
            for row in rows
        ]

    def head_seq(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM events").fetchone()
        return int(row["s"])

    # ── facts ───────────────────────────────────────────────────────────

    def put_fact(self, fact: Fact) -> Fact:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO facts (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (fact.id, fact.model_dump_json()),
            )
        return fact

    def put_facts(self, facts: list[Fact]) -> None:
        for fact in facts:
            self.put_fact(fact)

    def facts(self) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM facts").fetchall()
        return [Fact.model_validate_json(row["data"]) for row in rows]

    def replace_facts(self, facts: list[Fact]) -> None:
        """Used by rollback — the facts table is a cache of a version snapshot."""
        with self._connect() as conn:
            conn.execute("DELETE FROM facts")
            conn.executemany(
                "INSERT INTO facts (id, data) VALUES (?, ?)",
                [(f.id, f.model_dump_json()) for f in facts],
            )

    # ── versions ────────────────────────────────────────────────────────

    def commit_version(self, snapshot: str, facts: list[Fact], *,
                       label: str = "", perplexity: float | None = None) -> Version:
        version = Version(
            seq=self.head_seq(),
            snapshot=snapshot,
            label=label,
            active_facts=sum(1 for f in facts if f.consumes_buffer),
            perplexity=perplexity,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO versions (id, seq, snapshot, label, created_at, "
                "active_facts, perplexity, facts_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    version.id,
                    version.seq,
                    version.snapshot,
                    version.label,
                    version.created_at.isoformat(),
                    version.active_facts,
                    version.perplexity,
                    json.dumps([f.model_dump(mode="json") for f in facts]),
                ),
            )
        return version

    def versions(self, limit: int = 25) -> list[Version]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_version(row) for row in rows]

    def version(self, version_id: str) -> tuple[Version, list[Fact]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM versions WHERE id = ?", (version_id,)
            ).fetchone()
        if row is None:
            return None
        facts = [Fact.model_validate(d) for d in json.loads(row["facts_json"])]
        return _to_version(row), facts

    def latest_version(self) -> tuple[Version, list[Fact]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM versions ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return self.version(row["id"]) if row else None


def _to_version(row: sqlite3.Row) -> Version:
    return Version(
        id=row["id"],
        seq=row["seq"],
        snapshot=row["snapshot"],
        label=row["label"],
        created_at=row["created_at"],
        active_facts=row["active_facts"],
        perplexity=row["perplexity"],
    )
