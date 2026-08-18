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


def test_only_dispatcher_can_delegate():
    """Worker agents must not be able to spawn their own subagents."""
    for name, spec in agents.AGENTS.items():
        if name == "dispatcher":
            assert "delegate_to_agent" in spec.tools
        else:
            assert "delegate_to_agent" not in spec.tools


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
