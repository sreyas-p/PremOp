"""The three layers that make the store compound.

Observations are what was seen — append-only, never edited, one row per sighting.
Claims are what is believed — deduplicated, reinforced, superseded over time.
Edges are how entities relate — what turns a pile of facts into a graph.

The distinction is the whole design. Retrieval reads claims, which stay small
because they consolidate; the observation log grows without bound but is only
read during consolidation and audit. That is what keeps recall cheap as the
corpus grows.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def fingerprint(*parts: str) -> str:
    """Stable hash for dedup, insensitive to case and surrounding whitespace."""
    joined = "␟".join(p.strip().lower() for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


class ClaimState(str, Enum):
    ACTIVE = "active"
    #: Contradicted by something newer. Kept, never deleted — the history is
    #: the point, and "what did we believe in March" must stay answerable.
    SUPERSEDED = "superseded"
    #: Decayed below the confidence floor. Invisible to retrieval, still audited.
    DORMANT = "dormant"


class Observation(BaseModel):
    """One sighting. Immutable."""

    id: str = Field(default_factory=lambda: _new_id("obs"))
    subject: str
    predicate: str
    value: str
    #: Where it came from — "gmail:18f2a9c", "youtube:dQw4", "calendar:...".
    source: str
    actor: str = "unknown"
    observed_at: datetime = Field(default_factory=_now)
    #: Verbatim text this was drawn from, for retrieval and for showing a human
    #: what the claim actually rests on.
    context: str = ""
    confidence: float = 0.8
    consolidated: bool = False

    @property
    def claim_fingerprint(self) -> str:
        return fingerprint(self.subject, self.predicate)


class Claim(BaseModel):
    """What is currently believed about one (subject, predicate) pair."""

    id: str = Field(default_factory=lambda: _new_id("clm"))
    #: fingerprint(subject, predicate) — the identity a claim is reinforced on.
    key: str
    subject: str
    predicate: str
    value: str
    state: ClaimState = ClaimState.ACTIVE

    #: How many independent observations back this. The reinforcement signal:
    #: seeing the same thing from three sources is stronger than three copies
    #: of one email, so sources are counted distinctly.
    support: int = 1
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.8

    first_seen: datetime = Field(default_factory=_now)
    last_seen: datetime = Field(default_factory=_now)
    #: Set when superseded, so "what did we believe on date X" is answerable.
    valid_to: datetime | None = None
    superseded_by: str | None = None

    #: Best available verbatim context, kept for embedding and for display.
    context: str = ""

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.value}"


class Entity(BaseModel):
    """A canonical thing claims are about, after alias resolution."""

    id: str = Field(default_factory=lambda: _new_id("ent"))
    name: str
    #: Surface forms seen in the wild that resolve here — "Zilbex", "Zilbex Corp".
    aliases: list[str] = Field(default_factory=list)
    kind: str = "unknown"
    mentions: int = 1
    first_seen: datetime = Field(default_factory=_now)
    last_seen: datetime = Field(default_factory=_now)

    @property
    def key(self) -> str:
        return fingerprint(self.name)


class Edge(BaseModel):
    """A relation between two entities — what makes 1-hop expansion possible."""

    id: str = Field(default_factory=lambda: _new_id("edg"))
    source_entity: str
    target_entity: str
    predicate: str
    support: int = 1
    last_seen: datetime = Field(default_factory=_now)


class ConsolidationReport(BaseModel):
    """What one sleep did. Written to the ledger and returned to the caller."""

    at: datetime = Field(default_factory=_now)
    observations_processed: int = 0
    claims_created: int = 0
    claims_reinforced: int = 0
    claims_superseded: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    edges_touched: int = 0
    decayed: int = 0
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.observations_processed} observation(s): "
            f"{self.claims_created} new, {self.claims_reinforced} reinforced, "
            f"{self.claims_superseded} superseded, {self.decayed} decayed. "
            f"Entities: {self.entities_created} new, {self.entities_merged} merged."
        )
