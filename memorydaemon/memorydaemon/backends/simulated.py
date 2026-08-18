"""A backend with no model behind it, calibrated to the published 8B behaviour.

This exists so consolidation policy can be developed and tested without a GPU.
It is not a model — it is a model of the *failure mode*: MEMIT recall degrades
gently with edit count and then falls off a cliff, LoRA-absorbed facts survive
buffer pressure, and perplexity drifts as edits accumulate.

Calibration targets (Llama-3.1-8B, from the sleeping-llm results):
  · ~0.92 recall sustained through 13 simultaneous edits
  · collapse to ~0.57 at 14 edits
  · fused facts stop consuming buffer capacity entirely

If you change these numbers, change them because you measured something on a
real model — not to make a test pass.
"""

from __future__ import annotations

import copy
import random
import uuid

from ..models import Fact, Stage

#: Edit count at which 8B recall collapses.
CLIFF = 14

#: How reliably LoRA carries a fact at each consolidation stage.
_LORA_STRENGTH: dict[Stage, float] = {
    Stage.MEMIT_ONLY: 0.0,
    Stage.ABSORBING: 0.50,
    Stage.MOSTLY_LORA: 0.85,
    Stage.FUSED: 0.97,
}


def memit_recall(active_edits: float) -> float:
    """Expected recall for a MEMIT-only fact given the edit mass in the buffer.

    `active_edits` is the summed MEMIT scale across live edits, not a count —
    a fact consolidated down to scale 0.1 perturbs the weights a tenth as much
    as a fresh one, which is precisely why scaling down frees capacity.
    """
    if active_edits <= 0:
        return 1.0
    if active_edits < CLIFF:
        # Gentle interference: ~0.98 at one edit down to ~0.91 at thirteen.
        return max(0.0, 0.98 - 0.006 * (active_edits - 1))
    # Past the cliff, cascading interference.
    return max(0.15, 0.57 - 0.05 * (active_edits - CLIFF))


class SimulatedBackend:
    """Deterministic under a fixed seed, so tests are reproducible."""

    def __init__(self, *, seed: int = 0, baseline_perplexity: float = 8.0) -> None:
        self._rng = random.Random(seed)
        self._baseline = baseline_perplexity
        self._applied: dict[str, float] = {}   # fact_id -> scale in weights
        self._lora: dict[str, Stage] = {}      # fact_id -> stage LoRA was trained at
        self._taught: dict[str, dict] = {}     # fact_id -> what ask() can return
        self._fuse_count = 0
        self._snapshots: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return f"simulated(cliff={CLIFF})"

    # ── engine surface ──────────────────────────────────────────────────

    def apply_edits(self, facts: list[Fact]) -> None:
        self._applied = {f.id: f.memit_scale for f in facts if f.memit_scale > 0}
        for fact in facts:
            self._taught[fact.id] = {
                "target": fact.target,
                "terms": {
                    w.strip(".,?!").lower()
                    for w in f"{fact.subject} {fact.relation}".split()
                    if len(w) > 3
                },
            }

    def ask(self, question: str, *, max_tokens: int = 96) -> str:
        """Echo back any taught target whose prompt the question overlaps.

        Enough to exercise the read path in tests; it is not a model.
        """
        del max_tokens
        words = {w.strip(".,?!").lower() for w in question.split()}
        for fact_id, scale in self._applied.items():
            hit = self._taught.get(fact_id)
            if hit and scale > 0 and words & hit["terms"]:
                return hit["target"]
        for fact_id, hit in self._taught.items():
            if self._lora.get(fact_id, Stage.MEMIT_ONLY) is not Stage.MEMIT_ONLY:
                if words & hit["terms"]:
                    return hit["target"]
        return "I don't know."

    def probe(self, facts: list[Fact]) -> dict[str, bool]:
        pressure = sum(self._applied.values())
        base = memit_recall(pressure)

        results: dict[str, bool] = {}
        for fact in facts:
            lora = _LORA_STRENGTH[self._lora.get(fact.id, Stage.MEMIT_ONLY)]
            # A MEMIT delta at reduced scale carries proportionally less.
            memit = base * self._applied.get(fact.id, 0.0)
            # Noisy-or: either pathway can surface the fact.
            probability = 1.0 - (1.0 - lora) * (1.0 - memit)
            results[fact.id] = self._rng.random() < probability
        return results

    def consolidate(self, facts: list[Fact], *, rank: int, alpha: int,
                    epochs: int, lr: float) -> None:
        del rank, alpha, epochs, lr  # shape-compatible with the real engine
        for fact in facts:
            # LoRA learns the fact at the stage it is being promoted into.
            self._lora[fact.id] = fact.stage
        self._fuse_count += 1

    def perplexity(self) -> float:
        pressure = sum(self._applied.values())
        # Edits cost a little; each fuse costs a little more and never comes back.
        drift = 0.002 * pressure + 0.0015 * self._fuse_count
        return self._baseline * (1.0 + drift)

    def snapshot(self) -> str:
        handle = f"snap_{uuid.uuid4().hex[:12]}"
        self._snapshots[handle] = {
            "applied": copy.deepcopy(self._applied),
            "lora": copy.deepcopy(self._lora),
            "fuse_count": self._fuse_count,
        }
        return handle

    def restore(self, handle: str) -> None:
        state = self._snapshots.get(handle)
        if state is None:
            raise KeyError(f"Unknown snapshot {handle!r}")
        self._applied = copy.deepcopy(state["applied"])
        self._lora = copy.deepcopy(state["lora"])
        self._fuse_count = state["fuse_count"]
