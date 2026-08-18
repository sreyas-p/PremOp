"""Runtime tests. No GPU, no model — the simulated backend stands in.

These check the things a weight-memory runtime has to get right: the capacity
guard, that consolidation actually frees buffer, that a bad sleep undoes
itself, and that the audit trail is complete enough to answer "who taught the
model this".
"""

from __future__ import annotations

import pytest

from memorydaemon import CapacityError, MemoryDaemon, Policy, Stage
from memorydaemon.backends.simulated import CLIFF, SimulatedBackend, memit_recall
from memorydaemon.models import EventKind, FactState


@pytest.fixture
def daemon(tmp_path):
    return MemoryDaemon(
        SimulatedBackend(seed=1234),
        db_path=tmp_path / "memory.db",
        policy=Policy(buffer_capacity=12, passes_to_advance=2),
    )


def _settle(daemon, cycles: int = 6):
    """Run enough wake/sleep cycles for facts to climb the consolidation ladder."""
    for _ in range(cycles):
        daemon.audit()
        daemon.sleep()


# ── calibration ─────────────────────────────────────────────────────────


def test_simulated_backend_matches_published_8b_cliff():
    """The whole point of the guard is this shape. Assert it, so it can't drift."""
    assert memit_recall(1) == pytest.approx(0.98, abs=0.01)
    assert memit_recall(13) == pytest.approx(0.92, abs=0.02)
    assert memit_recall(CLIFF) == pytest.approx(0.57, abs=0.02)
    # The transition is a cliff, not a slope.
    assert memit_recall(13) - memit_recall(14) > 0.3


# ── wake ────────────────────────────────────────────────────────────────


def test_remember_lands_immediately(daemon):
    fact = daemon.remember("NVDA", "Q3 gross margin was", "73.5%", actor="sreyas")
    assert fact.state is FactState.ACTIVE
    assert fact.stage is Stage.MEMIT_ONLY
    assert fact.memit_scale == 1.0

    report = daemon.audit()
    assert report.active_facts == 1
    assert report.recall == 1.0


def test_buffer_guard_refuses_past_capacity(daemon):
    for i in range(12):
        daemon.remember("t", f"fact {i}", str(i), auto_sleep=False)

    with pytest.raises(CapacityError, match="Edit buffer full"):
        daemon.remember("t", "one too many", "boom", auto_sleep=False)


def test_auto_sleep_makes_room(daemon):
    for i in range(12):
        daemon.remember("t", f"fact {i}", str(i), auto_sleep=False)

    # Should consolidate rather than write past the cliff.
    fact = daemon.remember("t", "thirteenth", "ok", auto_sleep=True)
    assert fact.state is FactState.ACTIVE
    assert daemon.audit().buffer_used <= 12


def test_daemon_never_exceeds_capacity(daemon):
    for i in range(40):
        daemon.remember("t", f"fact {i}", str(i))
        assert daemon.audit(record=False).buffer_used <= 12


# ── consolidation ───────────────────────────────────────────────────────


def test_facts_advance_through_stages(daemon):
    daemon.remember("AAPL", "reports earnings in", "late October")
    _settle(daemon, cycles=4)

    fact = daemon.ledger.facts()[0]
    assert fact.stage > Stage.MEMIT_ONLY


def test_fused_facts_stop_consuming_buffer(daemon):
    """This is what makes lifetime capacity unbounded despite a small buffer."""
    daemon.remember("MSFT", "cloud segment is", "Azure")
    _settle(daemon, cycles=8)

    fact = daemon.ledger.facts()[0]
    assert fact.stage is Stage.FUSED
    assert fact.memit_scale == 0.0
    assert not fact.consumes_buffer
    assert daemon.audit().buffer_used == 0


def test_lifetime_capacity_exceeds_buffer_capacity(daemon):
    """Teach far more facts than fit at once; they should all survive."""
    for i in range(30):
        daemon.remember("ticker", f"metric {i} is", f"value {i}")
        daemon.audit()

    _settle(daemon, cycles=10)

    facts = daemon.ledger.facts()
    assert len(facts) == 30
    assert daemon.audit().buffer_used <= 12
    fused = sum(1 for f in facts if f.stage is Stage.FUSED)
    assert fused > 12, f"only {fused} facts consolidated out of the buffer"


# ── safety ──────────────────────────────────────────────────────────────


class DamagingBackend(SimulatedBackend):
    """Consolidation that wrecks the model, to prove sleep undoes itself."""

    def consolidate(self, facts, **kwargs):
        super().consolidate(facts, **kwargs)
        self._baseline *= 2.0  # catastrophic perplexity blowup


def test_sleep_rolls_back_when_drift_exceeds_budget(tmp_path):
    daemon = MemoryDaemon(
        DamagingBackend(seed=7), db_path=tmp_path / "m.db", policy=Policy()
    )
    daemon.remember("x", "is", "y")
    daemon.audit()

    report = daemon.sleep()
    assert report.rolled_back
    assert "rolled back" in " ".join(report.notes)
    assert report.version_id is None

    kinds = [e.kind for e in daemon.history()]
    assert EventKind.ROLLBACK in kinds


def test_rollback_restores_a_committed_version(daemon):
    daemon.remember("first", "is", "one")
    daemon.audit()
    report = daemon.sleep()
    assert report.version_id is not None

    daemon.remember("second", "is", "two")
    assert len(daemon.ledger.facts()) == 2

    daemon.rollback(report.version_id, actor="sreyas")
    assert len(daemon.ledger.facts()) == 1
    assert daemon.ledger.facts()[0].subject == "first"


def test_rollback_rejects_unknown_version(daemon):
    with pytest.raises(KeyError, match="Unknown version"):
        daemon.rollback("v_doesnotexist")


def test_forget_retires_and_frees_buffer(daemon):
    fact = daemon.remember("temp", "is", "wrong")
    assert daemon.audit().buffer_used == 1

    assert daemon.forget(fact.id, actor="sreyas") is True
    assert daemon.audit().buffer_used == 0
    assert daemon.forget(fact.id) is False  # already retired


# ── audit trail ─────────────────────────────────────────────────────────


def test_audit_trail_records_who_taught_what(daemon):
    fact = daemon.remember("TSLA", "delivered", "495k vehicles", actor="sreyas")

    events = daemon.history(fact_id=fact.id)
    remembers = [e for e in events if e.kind is EventKind.REMEMBER]
    assert len(remembers) == 1
    assert remembers[0].actor == "sreyas"
    assert remembers[0].detail["target"] == "495k vehicles"


def test_ledger_is_append_only_across_rollback(daemon):
    daemon.remember("a", "is", "1")
    daemon.audit()
    report = daemon.sleep()
    before = len(daemon.history(limit=1000))

    daemon.rollback(report.version_id)
    after = daemon.history(limit=1000)

    # Rolling back weights must not erase the record of what happened.
    assert len(after) > before
    assert after[0].kind is EventKind.ROLLBACK


def test_audit_flags_buffer_pressure(daemon):
    for i in range(10):
        daemon.remember("t", f"f{i}", str(i), auto_sleep=False)

    report = daemon.audit()
    assert report.buffer_pressure >= 0.75
    assert any("sleep recommended" in note for note in report.notes)
