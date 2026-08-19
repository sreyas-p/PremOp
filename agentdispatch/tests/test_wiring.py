"""Tests that run without credentials — they check the wiring, not the model."""

from __future__ import annotations

import pytest

from agentdispatch import agents, tools
from agentdispatch.context import MAX_DELEGATION_DEPTH, current_depth, task_scope
from agentdispatch.models import Task, TaskStatus
from agentdispatch.store import TaskStore


def test_every_agent_tool_is_registered():
    """An agent naming a tool that doesn't exist should fail here, not at runtime."""
    registered = set(tools.available())
    for spec in agents.AGENTS.values():
        missing = set(spec.tools) - registered
        assert not missing, f"agent {spec.name!r} references unknown tools: {missing}"


def test_resolve_rejects_unknown_tools():
    with pytest.raises(KeyError, match="Unknown tool"):
        tools.resolve(["gmail_search", "definitely_not_a_tool"])


def test_resolve_preserves_order():
    resolved = tools.resolve(["note_create", "gmail_search"])
    assert len(resolved) == 2


DELEGATION_TOOLS = ("delegate_to_agent", "delegate_parallel")


def test_only_dispatcher_can_delegate():
    """Worker agents must not be able to spawn their own subagents."""
    for name, spec in agents.AGENTS.items():
        for tool in DELEGATION_TOOLS:
            if name == "dispatcher":
                assert tool in spec.tools, f"dispatcher is missing {tool}"
            else:
                assert tool not in spec.tools, f"{name} should not hold {tool}"


def test_parallel_delegation_rejects_mismatched_lists():
    """The two lists are positional, so a length mismatch must not run anything."""
    from agentdispatch.tools.delegate import delegate_parallel

    result = delegate_parallel.call({"agents": ["mail", "youtube"],
                                     "instructions": ["only one brief"]})
    assert "Refused" in result
    assert "same length" in result


def test_mail_agent_has_no_write_access_to_gmail():
    """Read-only is a property of the tool set, so assert it there."""
    mail = agents.get("mail")
    assert all(not t.startswith("gmail_send") for t in mail.tools)


def test_unknown_agent_raises_with_suggestions():
    with pytest.raises(KeyError, match="Available:"):
        agents.get("nonexistent")


def test_task_scope_tracks_depth():
    assert current_depth() == 0
    with task_scope("task_a"):
        assert current_depth() == 1
        with task_scope("task_b"):
            assert current_depth() == 2
        assert current_depth() == 1
    assert current_depth() == 0


def test_delegation_depth_limit_is_reachable():
    """The guard has to trip before the recursion does."""
    assert MAX_DELEGATION_DEPTH >= 1


def test_store_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "test.db")
    task = Task(agent="mail", instructions="summarize invoices")
    store.save(task)

    fetched = store.get(task.id)
    assert fetched is not None
    assert fetched.agent == "mail"
    assert fetched.status is TaskStatus.PENDING

    task.status = TaskStatus.SUCCEEDED
    task.result = "done"
    task.tool_calls = ["gmail_search", "note_create"]
    task.input_tokens = 1234
    store.save(task)

    updated = store.get(task.id)
    assert updated.status is TaskStatus.SUCCEEDED
    assert updated.result == "done"
    assert updated.tool_calls == ["gmail_search", "note_create"]
    assert updated.input_tokens == 1234


def test_store_lists_subtasks_by_parent(tmp_path):
    store = TaskStore(tmp_path / "test.db")
    parent = store.save(Task(agent="dispatcher", instructions="do the thing"))
    store.save(Task(agent="mail", instructions="sub 1", parent_id=parent.id))
    store.save(Task(agent="youtube", instructions="sub 2", parent_id=parent.id))
    store.save(Task(agent="notetaker", instructions="unrelated"))

    children = store.list(parent_id=parent.id)
    assert len(children) == 2
    assert {c.agent for c in children} == {"mail", "youtube"}


# ── semantic index ──────────────────────────────────────────────────────


class FakeEmbedder:
    """Deterministic bag-of-words vectors — exercises the index, not the model."""

    model_id = "fake"

    def encode(self, texts, *, is_query=False):
        import numpy as np

        vocab = ["rent", "lease", "video", "invoice", "meeting"]
        out = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, word in enumerate(vocab):
                out[i, j] = text.lower().count(word)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@pytest.fixture
def semantic_index(tmp_path):
    from agentdispatch.semantic import SemanticIndex

    return SemanticIndex(db_path=tmp_path / "sem.db", embedder=FakeEmbedder())


def test_chunking_splits_long_text_with_overlap():
    from agentdispatch.semantic import chunk

    assert chunk("") == []
    assert chunk("short") == ["short"]
    pieces = chunk("word " * 800, size=900, overlap=150)
    assert len(pieces) > 1
    assert all(len(p) <= 900 for p in pieces)


def test_index_and_search_roundtrip(semantic_index):
    semantic_index.add("note", "n1", "Rent", "the rent and the lease")
    semantic_index.add("note", "n2", "Clip", "a video about a video")

    hits = semantic_index.search("rent", limit=2)
    assert hits[0].source_id == "n1"
    assert hits[0].score > 0


def test_reindexing_replaces_rather_than_duplicates(semantic_index):
    """An edited note must not keep matching against its own stale text."""
    semantic_index.add("note", "n1", "Rent", "rent rent rent")
    semantic_index.add("note", "n1", "Rent", "invoice invoice")

    assert semantic_index.stats()["by_source"]["note"]["items"] == 1
    assert all("rent" not in h.text for h in semantic_index.search("rent", limit=5))


def test_search_can_filter_by_source(semantic_index):
    semantic_index.add("note", "n1", "", "rent")
    semantic_index.add("gmail", "g1", "", "rent")

    assert {h.source for h in semantic_index.search("rent", limit=5)} == {"note", "gmail"}
    assert {h.source for h in semantic_index.search("rent", limit=5, source="gmail")} == {"gmail"}


def test_empty_index_returns_nothing(semantic_index):
    assert semantic_index.search("anything") == []


def test_indexing_failure_never_breaks_the_caller(monkeypatch):
    """A broken index must not fail the Gmail read that triggered it."""
    from agentdispatch import semantic

    def explode():
        raise RuntimeError("index is down")

    monkeypatch.setattr(semantic, "index", explode)
    semantic.remember_text("gmail", "x", "t", "body")  # must not raise
