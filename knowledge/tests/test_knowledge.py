"""Tests for the compounding behaviours. No model, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from knowledge import ClaimState, KnowledgeBase, Policy
from knowledge.embeddings import HashingEmbedder
from knowledge.models import fingerprint


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(
        tmp_path / "k.db", embedder=HashingEmbedder(), auto_consolidate_after=0
    )


def observe(kb, subject, predicate, value, source, **kw):
    return kb.observe(subject, predicate, value, source=source, **kw)


# ── identity ────────────────────────────────────────────────────────────


def test_fingerprint_does_not_collide_across_field_boundaries():
    """('ab','c') and ('a','bc') must not hash alike, or claims merge wrongly."""
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
    assert fingerprint(" Zilbex ", "IS BASED IN") == fingerprint("zilbex", "is based in")


# ── reinforcement ───────────────────────────────────────────────────────


def test_repeat_observations_reinforce_instead_of_duplicating(kb):
    for source in ["gmail:1", "gmail:2", "calendar:3"]:
        observe(kb, "Zilbex", "is based in", "Reykjavik", source)
    report = kb.consolidate()

    assert report.claims_created == 1
    claims = kb.store.claims()
    assert len(claims) == 1
    claim, _ = claims[0]
    assert claim.support == 3, "three distinct sources should count as three"
    assert claim.confidence > 0.8


def test_same_source_repeated_does_not_inflate_support(kb):
    """One chatty sender must not look like corroboration."""
    for _ in range(5):
        observe(kb, "Zilbex", "is based in", "Reykjavik", "gmail:1")
    kb.consolidate()

    claim, _ = kb.store.claims()[0]
    assert claim.support == 1


def test_store_compresses_as_it_grows(kb):
    for i in range(30):
        observe(kb, "Zilbex", "is based in", "Reykjavik", f"gmail:{i}")
    kb.consolidate()

    stats = kb.stats()
    assert stats["observations"] == 30
    assert stats["claims_active"] == 1
    assert stats["compression"] == 30.0


# ── supersession ────────────────────────────────────────────────────────


def test_contradiction_supersedes_and_keeps_history(kb):
    observe(kb, "Zilbex", "is based in", "Reykjavik", "gmail:1")
    kb.consolidate()
    observe(kb, "Zilbex", "is based in", "Oslo", "gmail:9", confidence=0.9)
    report = kb.consolidate()

    assert report.claims_superseded == 1
    active = [c for c, _ in kb.store.claims(state=ClaimState.ACTIVE)]
    assert len(active) == 1 and active[0].value == "Oslo"

    history = kb.history("Zilbex", "is based in")
    assert {c.value for c in history} == {"Reykjavik", "Oslo"}
    old = next(c for c in history if c.value == "Reykjavik")
    assert old.state is ClaimState.SUPERSEDED
    assert old.valid_to is not None, "superseded claims must carry an end date"


def test_weak_contradiction_does_not_overturn_a_supported_claim(kb):
    for source in [f"gmail:{i}" for i in range(4)]:
        observe(kb, "Zilbex", "is based in", "Reykjavik", source)
    kb.consolidate()

    observe(kb, "Zilbex", "is based in", "Atlantis", "gmail:x", confidence=0.4)
    report = kb.consolidate()

    assert report.claims_superseded == 0
    active = [c for c, _ in kb.store.claims()]
    assert active[0].value == "Reykjavik"
    assert any("kept" in note for note in report.notes)


def test_within_one_batch_more_sources_wins(kb):
    """A single source repeating itself must not outvote several agreeing ones."""
    for _ in range(4):
        observe(kb, "Zilbex", "is based in", "Atlantis", "gmail:noisy")
    for source in ["gmail:a", "gmail:b"]:
        observe(kb, "Zilbex", "is based in", "Reykjavik", source)
    kb.consolidate()

    claim, _ = kb.store.claims()[0]
    assert claim.value == "Reykjavik"


# ── entities and graph ──────────────────────────────────────────────────


def test_aliases_fold_onto_one_entity(kb):
    observe(kb, "Zilbex", "is based in", "Reykjavik", "gmail:1")
    observe(kb, "Zilbex Corp", "employs", "Ana Silva", "gmail:2")
    report = kb.consolidate()

    assert report.entities_merged >= 1
    names = {e.name for e in kb.store.entities()}
    assert "Zilbex Corp" in names and "Zilbex" not in names


def test_graph_expansion_surfaces_related_claims(kb):
    """Asking about a person should lift what is known about their employer,
    above unrelated noise it has no lexical advantage over."""
    observe(kb, "Ana Silva", "works at", "Zilbex Corp", "gmail:1")
    observe(kb, "Zilbex Corp", "is based in", "Reykjavik", "gmail:2")
    for i in range(12):
        observe(kb, f"Unrelated thing {i}", "has property", f"value {i}", f"src:{i}")
    kb.consolidate()

    results = kb.recall("Ana Silva", limit=4)
    surfaced = {r.claim.subject for r in results}

    assert "Ana Silva" in surfaced, "the direct match must still rank"
    assert "Zilbex Corp" in surfaced, "the employer's claim should be pulled up"
    assert any(r.via.startswith("neighbour") for r in results), \
        "and it should be labelled as reached via the graph, not similarity"


# ── decay ───────────────────────────────────────────────────────────────


def test_unreinforced_claims_decay_to_dormant(kb):
    observe(kb, "Ghost", "was rumoured to be", "somewhere", "gmail:1", confidence=0.3)
    kb.consolidate()

    claim, _ = kb.store.claims()[0]
    claim.last_seen = datetime.now(timezone.utc) - timedelta(weeks=40)
    kb.store.save_claim(claim)

    report = kb.consolidate()
    assert report.decayed == 1
    assert not kb.store.claims(state=ClaimState.ACTIVE)


def test_well_supported_claims_resist_decay(kb):
    for i in range(8):
        observe(kb, "Solid", "is", "true", f"src:{i}")
    kb.consolidate()

    claim, _ = kb.store.claims()[0]
    claim.last_seen = datetime.now(timezone.utc) - timedelta(weeks=10)
    kb.store.save_claim(claim)

    kb.consolidate()
    assert kb.store.claims(state=ClaimState.ACTIVE), "support should slow decay"


# ── retrieval and API ───────────────────────────────────────────────────


def test_recall_consolidates_pending_first(kb):
    """A fact just observed must not be invisible to the next question."""
    observe(kb, "Vantrel", "was founded in", "Lisbon", "gmail:1")
    assert kb.stats()["pending"] == 1

    results = kb.recall("Vantrel")
    assert results and kb.stats()["pending"] == 0


def test_context_for_respects_its_budget(kb):
    for i in range(40):
        observe(kb, f"Subject {i}", "has property", f"value {i}", f"src:{i}")
    kb.consolidate()

    assert len(kb.context_for("subject", limit=40, budget=200)) <= 220


def test_empty_store_returns_a_usable_answer(kb):
    assert kb.recall("anything") == []
    assert "nothing relevant" in kb.context_for("anything")


def test_audit_trail_records_who_observed_what(kb):
    observe(kb, "Zilbex", "is based in", "Reykjavik", "gmail:1", actor="sreyas")
    events = kb.audit()
    assert events[0]["actor"] == "sreyas"
    assert events[0]["detail"]["source"] == "gmail:1"


def test_auto_consolidation_triggers_on_threshold(tmp_path):
    kb = KnowledgeBase(tmp_path / "k.db", embedder=HashingEmbedder(),
                       auto_consolidate_after=5)
    for i in range(5):
        observe(kb, f"S{i}", "p", "v", f"src:{i}")
    assert kb.stats()["pending"] == 0
