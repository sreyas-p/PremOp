"""SQLite storage for the three layers, plus vectors for retrieval.

Claims carry their embedding inline. That is the efficiency argument in one
line: retrieval embeds the query once and scans the *claims* table, which stays
small because consolidation collapses repetition, rather than the observation
log, which grows forever.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from .models import Claim, ClaimState, Edge, Entity, Observation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id            TEXT PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    value         TEXT NOT NULL,
    source        TEXT NOT NULL,
    actor         TEXT NOT NULL,
    observed_at   TEXT NOT NULL,
    context       TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 0.8,
    consolidated  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS obs_pending_idx ON observations(consolidated);

CREATE TABLE IF NOT EXISTS claims (
    id            TEXT PRIMARY KEY,
    key           TEXT NOT NULL,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    value         TEXT NOT NULL,
    state         TEXT NOT NULL,
    support       INTEGER NOT NULL DEFAULT 1,
    sources       TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.8,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    valid_to      TEXT,
    superseded_by TEXT,
    context       TEXT NOT NULL DEFAULT '',
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS claims_key_idx ON claims(key, state);
CREATE INDEX IF NOT EXISTS claims_state_idx ON claims(state);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    kind        TEXT NOT NULL DEFAULT 'unknown',
    mentions    INTEGER NOT NULL DEFAULT 1,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id             TEXT PRIMARY KEY,
    source_entity  TEXT NOT NULL,
    target_entity  TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    support        INTEGER NOT NULL DEFAULT 1,
    last_seen      TEXT NOT NULL,
    UNIQUE(source_entity, target_entity, predicate)
);
CREATE INDEX IF NOT EXISTS edges_source_idx ON edges(source_entity);
CREATE INDEX IF NOT EXISTS edges_target_idx ON edges(target_entity);

CREATE TABLE IF NOT EXISTS events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind    TEXT NOT NULL,
    at      TEXT NOT NULL,
    actor   TEXT NOT NULL,
    detail  TEXT NOT NULL DEFAULT '{}'
);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── observations ────────────────────────────────────────────────────

    def add_observation(self, observation: Observation) -> Observation:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO observations (id, subject, predicate, value, source, "
                "actor, observed_at, context, confidence, consolidated) "
                "VALUES (?,?,?,?,?,?,?,?,?,0)",
                (observation.id, observation.subject, observation.predicate,
                 observation.value, observation.source, observation.actor,
                 observation.observed_at.isoformat(), observation.context,
                 observation.confidence),
            )
        return observation

    def pending_observations(self, limit: int = 5_000) -> list[Observation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE consolidated = 0 "
                "ORDER BY observed_at LIMIT ?", (limit,)
            ).fetchall()
        return [_to_observation(r) for r in rows]

    def mark_consolidated(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE observations SET consolidated = 1 WHERE id = ?",
                [(i,) for i in ids],
            )

    def observation_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) c FROM observations").fetchone()["c"]

    # ── claims ──────────────────────────────────────────────────────────

    def active_claim(self, key: str) -> Claim | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE key = ? AND state = ? LIMIT 1",
                (key, ClaimState.ACTIVE.value),
            ).fetchone()
        return _to_claim(row) if row else None

    def save_claim(self, claim: Claim, embedding: np.ndarray | None = None) -> Claim:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (id, key, subject, predicate, value, state,
                    support, sources, confidence, first_seen, last_seen,
                    valid_to, superseded_by, context, embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    value=excluded.value, state=excluded.state,
                    support=excluded.support, sources=excluded.sources,
                    confidence=excluded.confidence, last_seen=excluded.last_seen,
                    valid_to=excluded.valid_to, superseded_by=excluded.superseded_by,
                    context=excluded.context,
                    embedding=COALESCE(excluded.embedding, claims.embedding)
                """,
                (claim.id, claim.key, claim.subject, claim.predicate, claim.value,
                 claim.state.value, claim.support, json.dumps(claim.sources),
                 claim.confidence, claim.first_seen.isoformat(),
                 claim.last_seen.isoformat(),
                 claim.valid_to.isoformat() if claim.valid_to else None,
                 claim.superseded_by, claim.context,
                 embedding.astype(np.float32).tobytes() if embedding is not None else None),
            )
        return claim

    def claims(self, state: ClaimState | None = ClaimState.ACTIVE,
               with_vectors: bool = False) -> list[tuple[Claim, np.ndarray | None]]:
        query = "SELECT * FROM claims"
        params: list[object] = []
        if state:
            query += " WHERE state = ?"
            params.append(state.value)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        out = []
        for row in rows:
            vector = None
            if with_vectors and row["embedding"]:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
            out.append((_to_claim(row), vector))
        return out

    def claim_history(self, subject: str, predicate: str) -> list[Claim]:
        """Every value ever held for one (subject, predicate), newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE subject = ? AND predicate = ? "
                "ORDER BY first_seen DESC", (subject, predicate)
            ).fetchall()
        return [_to_claim(r) for r in rows]

    # ── entities and edges ──────────────────────────────────────────────

    def entity_by_key(self, key: str) -> Entity | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entities WHERE key = ?", (key,)).fetchone()
        return _to_entity(row) if row else None

    def entities(self) -> list[Entity]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM entities ORDER BY mentions DESC").fetchall()
        return [_to_entity(r) for r in rows]

    def save_entity(self, entity: Entity, *, drop_keys: list[str] | None = None) -> Entity:
        """Upsert an entity, optionally retiring surface forms it absorbed.

        `key` is derived from `name`, so renaming during an alias merge changes
        the key while the id stays put. Both columns are unique, so the stale
        row has to go in the same transaction or the insert collides with
        itself on the primary key.
        """
        with self._connect() as conn:
            for stale in drop_keys or []:
                if stale != entity.key:
                    conn.execute("DELETE FROM entities WHERE key = ?", (stale,))
            conn.execute("DELETE FROM entities WHERE id = ? AND key != ?",
                         (entity.id, entity.key))
            conn.execute(
                "INSERT INTO entities (id, key, name, aliases, kind, mentions, "
                "first_seen, last_seen) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET name=excluded.name, "
                "aliases=excluded.aliases, kind=excluded.kind, "
                "mentions=excluded.mentions, last_seen=excluded.last_seen",
                (entity.id, entity.key, entity.name, json.dumps(entity.aliases),
                 entity.kind, entity.mentions, entity.first_seen.isoformat(),
                 entity.last_seen.isoformat()),
            )
        return entity

    def touch_edge(self, edge: Edge) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO edges (id, source_entity, target_entity, predicate, "
                "support, last_seen) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(source_entity, target_entity, predicate) DO UPDATE SET "
                "support = edges.support + 1, last_seen = excluded.last_seen",
                (edge.id, edge.source_entity, edge.target_entity, edge.predicate,
                 edge.support, edge.last_seen.isoformat()),
            )

    def neighbours(self, entity_name: str) -> list[tuple[str, str]]:
        """One hop out: (predicate, other entity) in either direction."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT predicate, target_entity AS other FROM edges WHERE source_entity = ? "
                "UNION "
                "SELECT predicate, source_entity AS other FROM edges WHERE target_entity = ?",
                (entity_name, entity_name),
            ).fetchall()
        return [(r["predicate"], r["other"]) for r in rows]

    # ── ledger ──────────────────────────────────────────────────────────

    def record(self, kind: str, actor: str = "daemon", **detail) -> None:
        from .models import _now

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (kind, at, actor, detail) VALUES (?,?,?,?)",
                (kind, _now().isoformat(), actor, json.dumps(detail, default=str)),
            )

    def events(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"seq": r["seq"], "kind": r["kind"], "at": r["at"],
             "actor": r["actor"], "detail": json.loads(r["detail"])}
            for r in rows
        ]

    def stats(self) -> dict:
        with self._connect() as conn:
            def scalar(sql: str, *params) -> int:
                return conn.execute(sql, params).fetchone()[0]

            return {
                "observations": scalar("SELECT COUNT(*) FROM observations"),
                "pending": scalar("SELECT COUNT(*) FROM observations WHERE consolidated = 0"),
                "claims_active": scalar("SELECT COUNT(*) FROM claims WHERE state = 'active'"),
                "claims_superseded": scalar("SELECT COUNT(*) FROM claims WHERE state = 'superseded'"),
                "claims_dormant": scalar("SELECT COUNT(*) FROM claims WHERE state = 'dormant'"),
                "entities": scalar("SELECT COUNT(*) FROM entities"),
                "edges": scalar("SELECT COUNT(*) FROM edges"),
            }


def _to_observation(row: sqlite3.Row) -> Observation:
    return Observation(
        id=row["id"], subject=row["subject"], predicate=row["predicate"],
        value=row["value"], source=row["source"], actor=row["actor"],
        observed_at=row["observed_at"], context=row["context"],
        confidence=row["confidence"], consolidated=bool(row["consolidated"]),
    )


def _to_claim(row: sqlite3.Row) -> Claim:
    return Claim(
        id=row["id"], key=row["key"], subject=row["subject"],
        predicate=row["predicate"], value=row["value"],
        state=ClaimState(row["state"]), support=row["support"],
        sources=json.loads(row["sources"]), confidence=row["confidence"],
        first_seen=row["first_seen"], last_seen=row["last_seen"],
        valid_to=row["valid_to"], superseded_by=row["superseded_by"],
        context=row["context"],
    )


def _to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"], name=row["name"], aliases=json.loads(row["aliases"]),
        kind=row["kind"], mentions=row["mentions"],
        first_seen=row["first_seen"], last_seen=row["last_seen"],
    )
