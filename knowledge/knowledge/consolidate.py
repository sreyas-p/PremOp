"""Sleep: turn a pile of observations into a smaller, better-supported store.

This is the process memorydaemon had, with weights swapped out for structure.
Four things happen, and each is a form of compounding:

  reinforce  — the same claim seen again raises support and confidence instead
               of adding a row, so repetition makes the store *smaller* and
               more certain rather than larger.
  supersede  — a contradicting value retires the old claim with valid_to set,
               so the current answer is unambiguous and the history survives.
  resolve    — surface forms collapse onto one entity, so "Zilbex" and "Zilbex
               Corp" stop being two things.
  decay      — claims nothing has reinforced fade, so stale beliefs stop
               competing with fresh ones at retrieval time.

Without decay a store like this only accumulates, and old wrong facts keep
winning on support they earned years ago.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .embeddings import Embedder
from .models import (
    Claim,
    ClaimState,
    ConsolidationReport,
    Edge,
    Entity,
    Observation,
    fingerprint,
)
from .store import Store


@dataclass(frozen=True)
class Policy:
    #: Confidence gained each time a claim is independently re-observed.
    reinforcement: float = 0.06
    max_confidence: float = 0.99
    #: Confidence lost per week without reinforcement.
    decay_per_week: float = 0.04
    #: Below this a claim goes dormant and stops appearing in recall.
    dormancy_floor: float = 0.25
    #: A superseding value must be at least this confident to retire the
    #: incumbent, so one noisy observation cannot overturn a well-supported fact.
    supersede_margin: float = 0.15
    #: Shorter strings than this are not treated as entities — "8%" and "yes"
    #: are values, not things with relationships.
    min_entity_chars: int = 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


class Consolidator:
    def __init__(self, store: Store, embedder: Embedder,
                 policy: Policy | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.policy = policy or Policy()

    def run(self, *, actor: str = "daemon") -> ConsolidationReport:
        report = ConsolidationReport()
        pending = self.store.pending_observations()
        report.observations_processed = len(pending)

        if pending:
            grouped: dict[str, list[Observation]] = {}
            for observation in pending:
                grouped.setdefault(observation.claim_fingerprint, []).append(observation)

            for key, group in grouped.items():
                self._fold(key, group, report)

            self._resolve_entities(pending, report)
            self.store.mark_consolidated([o.id for o in pending])

        self._decay(report)
        self.store.record("consolidate", actor=actor, **report.model_dump(mode="json"))
        return report

    # ── claim folding ───────────────────────────────────────────────────

    def _fold(self, key: str, group: list[Observation],
              report: ConsolidationReport) -> None:
        """Collapse one (subject, predicate) group into a single claim."""
        # Within a batch, the winning value is the one with the most distinct
        # sources — not simply the newest, or a single chatty source could
        # outvote several independent ones.
        by_value: dict[str, list[Observation]] = {}
        for observation in group:
            by_value.setdefault(_normalize(observation.value), []).append(observation)

        winning = max(
            by_value.values(),
            key=lambda obs: (len({o.source for o in obs}), max(o.observed_at for o in obs)),
        )
        newest = max(winning, key=lambda o: o.observed_at)
        sources = sorted({o.source for o in winning})
        confidence = min(
            self.policy.max_confidence,
            max(o.confidence for o in winning)
            + self.policy.reinforcement * (len(sources) - 1),
        )

        existing = self.store.active_claim(key)

        if existing is None:
            claim = Claim(
                key=key, subject=newest.subject, predicate=newest.predicate,
                value=newest.value, support=len(sources), sources=sources,
                confidence=confidence, first_seen=newest.observed_at,
                last_seen=newest.observed_at, context=newest.context,
            )
            self.store.save_claim(claim, self._embed(claim))
            report.claims_created += 1
            return

        if _normalize(existing.value) == _normalize(newest.value):
            merged = sorted(set(existing.sources) | set(sources))
            existing.support = len(merged)
            existing.sources = merged
            existing.confidence = min(
                self.policy.max_confidence,
                existing.confidence + self.policy.reinforcement,
            )
            existing.last_seen = max(existing.last_seen, newest.observed_at)
            if newest.context:
                existing.context = newest.context
            self.store.save_claim(existing)
            report.claims_reinforced += 1
            return

        # Contradiction. Only overturn a well-supported incumbent when the
        # challenger is meaningfully more confident.
        if confidence + self.policy.supersede_margin < existing.confidence:
            report.notes.append(
                f"kept {existing.subject} {existing.predicate} = {existing.value!r} "
                f"over {newest.value!r} (confidence {existing.confidence:.2f} "
                f"vs {confidence:.2f})"
            )
            return

        replacement = Claim(
            key=key, subject=newest.subject, predicate=newest.predicate,
            value=newest.value, support=len(sources), sources=sources,
            confidence=confidence, first_seen=newest.observed_at,
            last_seen=newest.observed_at, context=newest.context,
        )
        existing.state = ClaimState.SUPERSEDED
        existing.valid_to = newest.observed_at
        existing.superseded_by = replacement.id
        self.store.save_claim(existing)
        self.store.save_claim(replacement, self._embed(replacement))
        report.claims_superseded += 1

    def _embed(self, claim: Claim):
        text = f"{claim.text}. {claim.context}".strip()
        return self.embedder.encode([text])[0]

    # ── entities and edges ──────────────────────────────────────────────

    def _resolve_entities(self, observations: list[Observation],
                          report: ConsolidationReport) -> None:
        for observation in observations:
            subject = self._register(observation.subject, report)
            if subject is None:
                continue
            target = self._register(observation.value, report)
            if target and target != subject:
                self.store.touch_edge(Edge(
                    source_entity=subject, target_entity=target,
                    predicate=observation.predicate, last_seen=observation.observed_at,
                ))
                report.edges_touched += 1

    def _register(self, name: str, report: ConsolidationReport) -> str | None:
        """Register a surface form, folding it into an existing entity if it is
        clearly the same thing. Returns the canonical name."""
        cleaned = name.strip()
        if len(cleaned) < self.policy.min_entity_chars or cleaned.replace(".", "").isdigit():
            return None

        existing = self.store.entity_by_key(fingerprint(cleaned))
        if existing:
            existing.mentions += 1
            existing.last_seen = _now()
            self.store.save_entity(existing)
            return existing.name

        # Conservative alias matching: one name fully containing the other as a
        # word prefix. "Zilbex" folds into "Zilbex Corp"; "Corp" does not.
        lowered = cleaned.lower()
        for candidate in self.store.entities():
            other = candidate.name.lower()
            if lowered == other:
                continue
            if lowered.startswith(other + " ") or other.startswith(lowered + " "):
                canonical = cleaned if len(cleaned) > len(candidate.name) else candidate.name
                aliases = sorted(({*candidate.aliases, candidate.name, cleaned}) - {canonical})
                previous_key = candidate.key
                candidate.name = canonical
                candidate.aliases = aliases
                candidate.mentions += 1
                candidate.last_seen = _now()
                # Renaming changes the derived key, so the old row must go.
                self.store.save_entity(candidate, drop_keys=[previous_key,
                                                             fingerprint(cleaned)])
                report.entities_merged += 1
                return canonical

        self.store.save_entity(Entity(name=cleaned))
        report.entities_created += 1
        return cleaned

    # ── decay ───────────────────────────────────────────────────────────

    def _decay(self, report: ConsolidationReport) -> None:
        now = _now()
        for claim, _ in self.store.claims(state=ClaimState.ACTIVE):
            weeks = (now - claim.last_seen).total_seconds() / (7 * 86_400)
            if weeks <= 1:
                continue
            # Well-supported claims fade more slowly: something seen from five
            # sources should outlive something seen once.
            resistance = 1.0 + 0.5 * (claim.support - 1)
            claim.confidence -= self.policy.decay_per_week * weeks / resistance
            if claim.confidence < self.policy.dormancy_floor:
                claim.state = ClaimState.DORMANT
                report.decayed += 1
            self.store.save_claim(claim)
