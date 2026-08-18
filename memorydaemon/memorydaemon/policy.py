"""Consolidation policy — the knobs that decide when to sleep and what advances.

The defaults encode the one published result that matters most for a runtime:
Llama-3.1-8B holds ~0.92 recall through 13 simultaneous MEMIT edits and then
falls off a cliff to 0.57 at 14. That is not a gentle degradation curve, so the
buffer ceiling is a hard guard rather than a hint, and it sits below the
observed cliff rather than at it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Fact, Stage


@dataclass(frozen=True)
class Policy:
    #: Hard ceiling on simultaneous unfused MEMIT edits. Below the observed
    #: 8B cliff (14) with margin, because the transition is sharp enough that
    #: landing on it costs a third of recall.
    buffer_capacity: int = 12

    #: Fraction of capacity that triggers a sleep recommendation.
    sleep_pressure: float = 0.75

    #: Consecutive passing probes before a fact advances a stage. Advancing on
    #: a single pass is how you consolidate a fact the model got right by luck.
    passes_to_advance: int = 2

    #: A sleep that pushes perplexity past this fraction above baseline is
    #: rolled back. The published 8B run drifted +3.2% at 30 facts.
    max_perplexity_drift: float = 0.05

    #: Recall below this marks the store degraded and forces a refresh pass.
    min_recall: float = 0.90

    #: How many consecutive sleep cycles `remember()` will run trying to free a
    #: slot before refusing the write. A fact needs `passes_to_advance` clean
    #: probes per stage and three stages to dissolve, so this wants headroom.
    max_sleep_attempts: int = 10

    # LoRA consolidation hyperparameters, passed through to the backend.
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_epochs: int = 3
    lora_lr: float = 1e-4

    def scale_for(self, stage: Stage) -> float:
        from .models import MEMIT_SCALE

        return MEMIT_SCALE[stage]

    def buffer_used(self, facts: list[Fact]) -> int:
        return sum(1 for f in facts if f.consumes_buffer)

    def has_room(self, facts: list[Fact]) -> bool:
        return self.buffer_used(facts) < self.buffer_capacity

    def should_sleep(self, facts: list[Fact]) -> bool:
        used = self.buffer_used(facts)
        if not self.buffer_capacity:
            return False
        return used / self.buffer_capacity >= self.sleep_pressure

    def may_advance(self, fact: Fact) -> bool:
        """Whether a fact has earned promotion to the next consolidation stage."""
        if fact.stage is Stage.FUSED:
            return False
        return fact.consecutive_passes >= self.passes_to_advance

    def next_stage(self, fact: Fact) -> Stage:
        return Stage(min(int(fact.stage) + 1, int(Stage.FUSED)))
