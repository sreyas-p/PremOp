"""Local embeddings, injectable so the rest of the package is testable."""

from __future__ import annotations

import threading
import zlib
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "mlx-community/bge-small-en-v1.5-bf16"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimensions: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


class MLXEmbedder:
    """On-device via MLX. Loaded lazily — importing must stay free."""

    dimensions = 384

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        from mlx_embeddings import generate, load

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        with self._lock:
            if self._model is None:
                self._model, self._tokenizer = load(self.model_id)
            output = generate(self._model, self._tokenizer, texts)
            vectors = np.array(output.text_embeds, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


class HashingEmbedder:
    """Dependency-free fallback: hashed character trigrams.

    Not semantic — it will not match paraphrase. It exists so the store works
    on a machine without MLX and so tests run fast. If recall seems poor, check
    which embedder is actually loaded.

    Uses crc32 rather than the built-in `hash()`, which is salted per process.
    With `hash()` the same text embeds differently in every run, so vectors
    persisted to SQLite would be silently meaningless the next time the process
    started — a corruption that only shows up as bad recall much later.
    """

    dimensions = 256

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        del is_query
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for j in range(len(padded) - 2):
                trigram = padded[j : j + 3].encode()
                vectors[i, zlib.crc32(trigram) % self.dimensions] += 1.0
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def default_embedder() -> Embedder:
    try:
        import mlx_embeddings  # noqa: F401

        return MLXEmbedder()
    except Exception:  # noqa: BLE001
        return HashingEmbedder()
