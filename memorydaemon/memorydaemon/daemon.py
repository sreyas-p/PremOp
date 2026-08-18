"""The public API: remember(), sleep(), audit(), rollback().

Everything a caller needs, and nothing about MEMIT or LoRA. The daemon owns
lifecycle and safety; the backend owns weights; the policy owns the schedule.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from .backend import MemoryBackend
from .backends.simulated import SimulatedBackend
from .ledger import Ledger
from .models import (
    AuditReport,
    EventKind,
    Fact,
    FactState,
    SleepReport,
    Stage,
    Version,
)
from .policy import Policy

log = logging.getLogger(__name__)


class CapacityError(RuntimeError):
    """Raised when the edit buffer is full and auto-sleep is disabled."""


class NoteWriter(Protocol):
    """Somewhere the model's answers can be written down.

    Deliberately one method: memorydaemon must not grow a dependency on Google
    APIs, or on any other note backend. Callers inject one — `agentdispatch`
    passes an adapter over its own `NoteSink`.
    """

    def write(self, title: str, body: str) -> str:
        """Persist a note and return a human-readable locator."""
        ...


class MemoryDaemon:
    """A model that remembers, and a record of everything it was taught."""

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        *,
        db_path: Path | str = "./memory.db",
        policy: Policy | None = None,
        note_writer: NoteWriter | None = None,
    ) -> None:
        self.backend = backend or SimulatedBackend()
        self.policy = policy or Policy()
        self.ledger = Ledger(Path(db_path))
        self.note_writer = note_writer
        #: Perplexity before any edit, the reference for all drift measurements.
        self._baseline_perplexity = self.backend.perplexity()

    # ── wake ────────────────────────────────────────────────────────────

    def remember(
        self,
        subject: str,
        relation: str,
        target: str,
        *,
        prompt: str | None = None,
        actor: str = "unknown",
        source: str | None = None,
        auto_sleep: bool = True,
    ) -> Fact:
        """Teach the model a fact, effective immediately.

        The edit lands in weights before this returns — that is the whole point
        of wake-phase writes. If the buffer is full, sleep runs first (or raises
        when `auto_sleep=False`), because writing past the capacity cliff
        degrades every fact already in there, not just the new one.
        """
        facts = self.ledger.facts()

        if not self.policy.has_room(facts):
            if not auto_sleep:
                raise CapacityError(
                    f"Edit buffer full ({self.policy.buffer_used(facts)}/"
                    f"{self.policy.buffer_capacity}). Run sleep() to consolidate."
                )
            # One cycle rarely frees a slot: a fact only stops consuming buffer
            # once it reaches stage 3, and each promotion needs its own passing
            # probes. Keep sleeping until something dissolves, then give up
            # rather than write past the cliff and degrade every existing fact.
            for attempt in range(self.policy.max_sleep_attempts):
                log.info("buffer full — sleeping (attempt %d)", attempt + 1)
                self.sleep(actor=actor)
                facts = self.ledger.facts()
                if self.policy.has_room(facts):
                    break
            else:
                raise CapacityError(
                    f"Edit buffer still full after "
                    f"{self.policy.max_sleep_attempts} sleep cycles "
                    f"({self.policy.buffer_used(facts)}/"
                    f"{self.policy.buffer_capacity}). Facts are not advancing — "
                    f"check audit().recall, since a fact must probe clean "
                    f"{self.policy.passes_to_advance}x in a row to consolidate."
                )

        fact = Fact(
            subject=subject,
            relation=relation,
            target=target,
            prompt=prompt or f"{subject} {relation}",
            actor=actor,
            source=source,
        )
        self.ledger.record(
            EventKind.REMEMBER, actor=actor, fact_id=fact.id,
            subject=subject, relation=relation, target=target, source=source,
        )

        fact.state = FactState.ACTIVE
        self.ledger.put_fact(fact)

        active = self._active(self.ledger.facts())
        self.backend.apply_edits(active)
        self.ledger.record(
            EventKind.APPLY, actor=actor, fact_id=fact.id,
            buffer_used=self.policy.buffer_used(active),
            capacity=self.policy.buffer_capacity,
        )
        return fact

    def forget(self, fact_id: str, *, actor: str = "unknown") -> bool:
        """Retire a fact and re-solve the remaining edits without it."""
        facts = self.ledger.facts()
        target = next((f for f in facts if f.id == fact_id), None)
        if target is None or target.state is FactState.RETIRED:
            return False

        target.state = FactState.RETIRED
        from datetime import datetime, timezone

        target.retired_at = datetime.now(timezone.utc)
        self.ledger.put_fact(target)
        self.ledger.record(EventKind.RETIRE, actor=actor, fact_id=fact_id)

        self.backend.apply_edits(self._active(self.ledger.facts()))
        return True

    # ── read ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        *,
        max_tokens: int = 96,
        actor: str = "unknown",
        note_title: str | None = None,
    ) -> str:
        """Ask the model something, using whatever it has been taught.

        Pass `note_title` to have the answer written to the configured
        `NoteWriter` — that is how the local model writes notes of its own,
        rather than only being written *about*.
        """
        answer = self.backend.ask(question, max_tokens=max_tokens)
        self.ledger.record(
            EventKind.ASK, actor=actor, question=question, answer=answer
        )

        if note_title:
            self.write_note(note_title, answer, actor=actor, question=question)
        return answer

    def write_note(self, title: str, body: str, *, actor: str = "unknown",
                   question: str | None = None) -> str:
        """Write a note through the injected writer, and record that it happened."""
        if self.note_writer is None:
            raise RuntimeError(
                "No note_writer configured. Pass one to MemoryDaemon(...) — "
                "agentdispatch.memory_bridge.NotesAdapter wraps its Google Docs sink."
            )
        locator = self.note_writer.write(title, body)
        self.ledger.record(
            EventKind.NOTE, actor=actor, title=title, locator=locator,
            question=question,
        )
        return locator

    # ── audit ───────────────────────────────────────────────────────────

    def audit(self, *, record: bool = True) -> AuditReport:
        """Probe every active fact and measure drift. Cheap enough to run often."""
        facts = self.ledger.facts()
        active = self._active(facts)

        results = self.backend.probe(active) if active else {}
        for fact in active:
            passed = results.get(fact.id, False)
            fact.last_probe = passed
            fact.consecutive_passes = fact.consecutive_passes + 1 if passed else 0
        self.ledger.put_facts(active)

        recalled = sum(1 for ok in results.values() if ok)
        perplexity = self.backend.perplexity()
        drift = (perplexity - self._baseline_perplexity) / self._baseline_perplexity

        stages: dict[int, int] = {}
        for fact in active:
            stages[int(fact.stage)] = stages.get(int(fact.stage), 0) + 1

        report = AuditReport(
            total_facts=len(facts),
            active_facts=len(active),
            buffer_used=self.policy.buffer_used(active),
            buffer_capacity=self.policy.buffer_capacity,
            recall=recalled / len(active) if active else 1.0,
            degraded=[f.id for f in active if not results.get(f.id, False)],
            perplexity=perplexity,
            perplexity_drift=drift,
            stages=stages,
        )
        report.healthy = (
            report.recall >= self.policy.min_recall
            and drift <= self.policy.max_perplexity_drift
        )
        if report.recall < self.policy.min_recall:
            report.notes.append(
                f"recall {report.recall:.2f} below floor {self.policy.min_recall:.2f}"
            )
        if drift > self.policy.max_perplexity_drift:
            report.notes.append(f"perplexity drift {drift:+.1%} over budget")
        if self.policy.should_sleep(active):
            report.notes.append(
                f"buffer at {report.buffer_pressure:.0%} — sleep recommended"
            )

        if record:
            self.ledger.record(
                EventKind.PROBE, recall=report.recall, drift=drift,
                active=len(active), degraded=len(report.degraded),
            )
        return report

    # ── sleep ───────────────────────────────────────────────────────────

    def sleep(self, *, actor: str = "daemon", label: str = "") -> SleepReport:
        """Consolidate: refresh what degraded, move facts into LoRA, free buffer.

        Rolls itself back if the cycle pushed perplexity past the drift budget,
        so a bad consolidation cannot leave the model worse than it found it.
        """
        before = self.audit(record=False)
        restore_point = self.backend.snapshot()
        report = SleepReport(before=before, after=before)

        facts = self.ledger.facts()
        active = self._active(facts)
        if not active:
            report.notes.append("nothing to consolidate")
            report.after = before
            return report

        # 1. Refresh degraded edits — re-solve the batch so weak facts come back.
        if before.degraded:
            self.backend.apply_edits(active)
            report.refreshed = list(before.degraded)
            self.ledger.record(
                EventKind.REFRESH, actor=actor, count=len(before.degraded)
            )

        # 2. Promote facts that have passed enough consecutive probes.
        promoted: list[Fact] = []
        for fact in active:
            if self.policy.may_advance(fact):
                fact.stage = self.policy.next_stage(fact)
                fact.consecutive_passes = 0
                promoted.append(fact)
                report.advanced.append(fact.id)
                if fact.stage is Stage.FUSED:
                    report.dissolved.append(fact.id)
                self.ledger.record(
                    EventKind.ADVANCE, actor=actor, fact_id=fact.id,
                    stage=int(fact.stage), memit_scale=fact.memit_scale,
                )

        # 3. Train LoRA on everything active and fuse it into the weights.
        self.backend.consolidate(
            active,
            rank=self.policy.lora_rank,
            alpha=self.policy.lora_alpha,
            epochs=self.policy.lora_epochs,
            lr=self.policy.lora_lr,
        )
        self.ledger.record(
            EventKind.CONSOLIDATE, actor=actor, facts=len(active),
            rank=self.policy.lora_rank, epochs=self.policy.lora_epochs,
        )

        # 4. Scale MEMIT deltas down to match each fact's new stage. Facts at
        #    stage 3 drop out of the buffer entirely.
        self.ledger.put_facts(active)
        self.backend.apply_edits(self._active(self.ledger.facts()))
        if promoted:
            self.ledger.record(
                EventKind.SCALE_DOWN, actor=actor, promoted=len(promoted),
                dissolved=len(report.dissolved),
            )

        # 5. Validate. A cycle that damaged the model is not a cycle we keep.
        after = self.audit(record=False)
        report.after = after
        drift = after.perplexity_drift or 0.0
        self.ledger.record(
            EventKind.VALIDATE, actor=actor, recall=after.recall, drift=drift
        )

        if drift > self.policy.max_perplexity_drift:
            self.backend.restore(restore_point)
            self.ledger.replace_facts(facts)
            report.rolled_back = True
            report.after = self.audit(record=False)
            report.notes.append(
                f"rolled back: drift {drift:+.1%} exceeded "
                f"{self.policy.max_perplexity_drift:+.1%} budget"
            )
            self.ledger.record(
                EventKind.ROLLBACK, actor=actor, reason="perplexity_drift", drift=drift
            )
            return report

        version = self.ledger.commit_version(
            self.backend.snapshot(),
            self.ledger.facts(),
            label=label or "sleep",
            perplexity=after.perplexity,
        )
        report.version_id = version.id
        self.ledger.record(
            EventKind.VERSION, actor=actor, version=version.id, label=version.label
        )
        return report

    # ── rollback ────────────────────────────────────────────────────────

    def rollback(self, version_id: str, *, actor: str = "unknown") -> Version:
        """Restore weights and the fact table to a committed version."""
        found = self.ledger.version(version_id)
        if found is None:
            raise KeyError(f"Unknown version {version_id!r}")
        version, facts = found

        self.backend.restore(version.snapshot)
        self.ledger.replace_facts(facts)
        self.ledger.record(
            EventKind.ROLLBACK, actor=actor, version=version_id,
            reason="manual", restored_facts=len(facts),
        )
        return version

    def versions(self, limit: int = 25) -> list[Version]:
        return self.ledger.versions(limit)

    def history(self, *, fact_id: str | None = None, limit: int = 100):
        """The audit trail: who taught the model what, when."""
        return self.ledger.events(limit=limit, fact_id=fact_id)

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _active(facts: list[Fact]) -> list[Fact]:
        return [f for f in facts if f.state is FactState.ACTIVE]
