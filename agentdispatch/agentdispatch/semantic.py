"""A semantic index over everything the agents read and write.

Embeddings run locally on MLX — Anthropic has no embeddings endpoint, and
pulling in a second paid API for this would be a poor trade when a 384-dim
model runs on-device in milliseconds.

Storage is SQLite with vectors as float32 blobs and brute-force cosine at query
time. For a personal corpus — thousands of chunks, not millions — a full scan
is a few milliseconds and costs nothing in dependencies or operational
surface. Swap in a real vector store when a scan actually gets slow, not
before.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import settings

#: 384-dim, ~130MB, strong for its size. Its asymmetric retrieval training is
#: why queries get a prefix below and documents do not.
DEFAULT_MODEL = "mlx-community/bge-small-en-v1.5-bf16"

#: BGE models are trained with this on the query side only. Omitting it costs
#: real retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 150

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,       -- 'gmail' | 'note' | 'youtube'
    source_id  TEXT NOT NULL,       -- message id, doc id, video id
    title      TEXT NOT NULL DEFAULT '',
    ordinal    INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source, source_id);
"""


def chunk(text: str, size: int = _CHUNK_CHARS, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries where possible, hard-split where not."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph break, then a sentence end, then wherever.
            for marker in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(marker, start + size // 2, end)
                if cut != -1:
                    end = cut + len(marker)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class Embedder:
    """Lazily loaded so importing this module costs nothing."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._model is None:
            from mlx_embeddings import load

            self._model, self._tokenizer = load(self.model_id)

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Return L2-normalized embeddings, so cosine is a plain dot product."""
        from mlx_embeddings import generate

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        with self._lock:  # MLX is not safe to call from several threads
            self._ensure()
            output = generate(self._model, self._tokenizer, texts)
            vectors = np.array(output.text_embeds, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)


@dataclass(frozen=True)
class Hit:
    score: float
    source: str
    source_id: str
    title: str
    text: str


class SemanticIndex:
    def __init__(self, db_path: Path | None = None,
                 embedder: Embedder | None = None) -> None:
        self.db_path = db_path or settings.semantic_db
        self.embedder = embedder or Embedder(settings.embedding_model)
        self._write_lock = threading.Lock()
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

    def add(self, source: str, source_id: str, title: str, text: str) -> int:
        """Index one item, replacing anything previously stored under its id.

        Replacing rather than appending is what keeps an edited note from
        matching against its own stale text.
        """
        pieces = chunk(text)
        if not pieces:
            return 0
        vectors = self.embedder.encode(pieces)
        stamp = datetime.now(timezone.utc).isoformat()

        with self._write_lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE source = ? AND source_id = ?",
                (source, source_id),
            )
            conn.executemany(
                "INSERT INTO chunks (source, source_id, title, ordinal, text, "
                "embedding, indexed_at) VALUES (?,?,?,?,?,?,?)",
                [
                    (source, source_id, title, i, piece,
                     vectors[i].tobytes(), stamp)
                    for i, piece in enumerate(pieces)
                ],
            )
        return len(pieces)

    def search(self, query: str, limit: int = 8,
               source: str | None = None) -> list[Hit]:
        with self._connect() as conn:
            sql = "SELECT source, source_id, title, text, embedding FROM chunks"
            params: list[object] = []
            if source:
                sql += " WHERE source = ?"
                params.append(source)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        matrix = np.frombuffer(
            b"".join(row["embedding"] for row in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        scores = matrix @ self.embedder.encode([query], is_query=True)[0]

        best = np.argsort(-scores)[:limit]
        return [
            Hit(
                score=float(scores[i]),
                source=rows[i]["source"],
                source_id=rows[i]["source_id"],
                title=rows[i]["title"],
                text=rows[i]["text"],
            )
            for i in best
        ]

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            by_source = conn.execute(
                "SELECT source, COUNT(*) c, COUNT(DISTINCT source_id) d "
                "FROM chunks GROUP BY source"
            ).fetchall()
        return {
            "chunks": total,
            "by_source": {
                r["source"]: {"chunks": r["c"], "items": r["d"]} for r in by_source
            },
            "model": self.embedder.model_id,
            "db": str(self.db_path),
        }


_index: SemanticIndex | None = None
_index_lock = threading.Lock()


def index() -> SemanticIndex:
    """Process-wide index, built on first use."""
    global _index
    with _index_lock:
        if _index is None:
            _index = SemanticIndex()
    return _index


def remember_text(source: str, source_id: str, title: str, text: str) -> None:
    """Best-effort indexing from inside a tool.

    Never raises: failing to index is not a reason to fail the Gmail read that
    triggered it.
    """
    try:
        index().add(source, source_id, title, text)
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("indexing failed", exc_info=True)
