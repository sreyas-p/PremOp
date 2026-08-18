"""Core objects: facts, ledger events, versions, audit reports.

Everything here is backend-agnostic. A fact is a claim the model should hold;
the backend decides how to put it into weights.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum, IntEnum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Stage(IntEnum):
    """Consolidation stage. A fact climbs this ladder one sleep at a time.

    The MEMIT delta is scaled down as LoRA takes over, so a fully consolidated
    fact costs nothing from the edit buffer — which is what makes lifetime
    capacity unbounded even though instantaneous capacity is small.
    """

    MEMIT_ONLY = 0
    ABSORBING = 1
    MOSTLY_LORA = 2
    FUSED = 3


#: MEMIT delta scale per stage. Published schedule: 1.0 → 0.5 → 0.1 → 0.0.
MEMIT_SCALE: dict[Stage, float] = {
    Stage.MEMIT_ONLY: 1.0,
    Stage.ABSORBING: 0.5,
    Stage.MOSTLY_LORA: 0.1,
    Stage.FUSED: 0.0,
}


class FactState(str, Enum):
    STAGED = "staged"    # written to the buffer, not yet in weights
    ACTIVE = "active"    # live in the model
    RETIRED = "retired"  # rolled back or explicitly forgotten


class Fact(BaseModel):
    """One unit of knowledge, and everything needed to audit how it got there."""

    id: str = Field(default_factory=lambda: _new_id("fact"))
    subject: str
    relation: str
    target: str
    #: Cloze prompt used both to write the edit and to probe for it later.
    prompt: str
    #: Who taught this. The compliance story lives or dies on this field.
    actor: str = "unknown"
    source: str | None = None
    created_at: datetime = Field(default_factory=_now)

    state: FactState = FactState.STAGED
    stage: Stage = Stage.MEMIT_ONLY
    #: Consecutive successful probes, reset on any miss. Drives advancement.
    consecutive_passes: int = 0
    last_probe: bool | None = None
    retired_at: datetime | None = None

    @property
    def memit_scale(self) -> float:
        return MEMIT_SCALE[self.stage]

    @property
    def consumes_buffer(self) -> bool:
        """Whether this fact still occupies MEMIT edit-buffer capacity."""
        return self.state is FactState.ACTIVE and self.stage is not Stage.FUSED


class EventKind(str, Enum):
    REMEMBER = "remember"
    APPLY = "apply"
    PROBE = "probe"
    ASK = "ask"
    NOTE = "note"
    REFRESH = "refresh"
    CONSOLIDATE = "consolidate"
    ADVANCE = "advance"
    SCALE_DOWN = "scale_down"
    VALIDATE = "validate"
    VERSION = "version"
    ROLLBACK = "rollback"
    RETIRE = "retire"


class Event(BaseModel):
    """One entry in the append-only audit log. Never updated, never deleted."""

    seq: int = 0
    kind: EventKind
    at: datetime = Field(default_factory=_now)
    actor: str = "daemon"
    fact_id: str | None = None
    detail: dict = Field(default_factory=dict)


class Version(BaseModel):
    """A restorable checkpoint: model weights plus the fact table that produced them."""

    id: str = Field(default_factory=lambda: _new_id("v"))
    seq: int
    #: Opaque handle the backend can restore from.
    snapshot: str
    label: str = ""
    created_at: datetime = Field(default_factory=_now)
    active_facts: int = 0
    perplexity: float | None = None


class AuditReport(BaseModel):
    """The answer to "is the model still healthy, and does it still know things?"."""

    at: datetime = Field(default_factory=_now)
    total_facts: int = 0
    active_facts: int = 0
    buffer_used: int = 0
    buffer_capacity: int = 0
    recall: float = 0.0
    degraded: list[str] = Field(default_factory=list)
    perplexity: float | None = None
    perplexity_drift: float | None = None
    stages: dict[int, int] = Field(default_factory=dict)
    healthy: bool = True
    notes: list[str] = Field(default_factory=list)

    @property
    def buffer_pressure(self) -> float:
        if not self.buffer_capacity:
            return 0.0
        return self.buffer_used / self.buffer_capacity


class SleepReport(BaseModel):
    """What one sleep cycle did. Returned by `sleep()` and written to the ledger."""

    at: datetime = Field(default_factory=_now)
    before: AuditReport
    after: AuditReport
    refreshed: list[str] = Field(default_factory=list)
    advanced: list[str] = Field(default_factory=list)
    dissolved: list[str] = Field(default_factory=list)
    version_id: str | None = None
    rolled_back: bool = False
    notes: list[str] = Field(default_factory=list)
