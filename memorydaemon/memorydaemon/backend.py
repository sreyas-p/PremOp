"""The engine boundary.

Anything that can hold weight-based memory implements this. Keeping it this
narrow is what lets the runtime, the ledger, and the audit trail be developed
and tested without a GPU — and what keeps MEMIT/LoRA specifics out of the
policy layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Fact


@runtime_checkable
class MemoryBackend(Protocol):
    """A model whose weights can be edited, consolidated, and snapshotted."""

    @property
    def name(self) -> str:
        """Human-readable identifier, recorded in the ledger for provenance."""
        ...

    def apply_edits(self, facts: list[Fact]) -> None:
        """Write MEMIT-style edits for `facts`, each at its own `memit_scale`.

        Called with the full active set rather than a delta: MEMIT solves for
        edits jointly, and applying them one at a time gives different (worse)
        results than solving the batch.
        """
        ...

    def ask(self, question: str, *, max_tokens: int = 96) -> str:
        """Free-form generation — the read path for whatever the model was taught.

        Distinct from `probe`, which checks one known target. This is how a
        caller actually uses the memory.
        """
        ...

    def probe(self, facts: list[Fact]) -> dict[str, bool]:
        """Ask the model each fact's prompt; report whether the target came back.

        This is generation-time recall, not a check that the edit is present in
        the weights — a fact can be written and still not surface in chat.
        """
        ...

    def consolidate(self, facts: list[Fact], *, rank: int, alpha: int,
                    epochs: int, lr: float) -> None:
        """Train a LoRA adapter on `facts` and fuse it into the base weights."""
        ...

    def perplexity(self) -> float:
        """Perplexity on the held-out drift corpus. The alignment-tax canary."""
        ...

    def snapshot(self) -> str:
        """Capture restorable state; return an opaque handle."""
        ...

    def restore(self, handle: str) -> None:
        """Restore weights captured by `snapshot`."""
        ...
