"""Durable memory for the agents.

Backed by the compounding knowledge base by default. Set
`AGENTDISPATCH_MEMORY_BACKEND=weights` to route back to the MEMIT-based
`memorydaemon` instead — that package is untouched and still works; it is
simply no longer the default.

The substrate change alters what these tools *mean*, and the docstrings below
reflect it. Recall used to be generation: the local model answered from edited
weights, and could assert something that was never true. Now recall is
retrieval over consolidated claims, each carrying its sources and how many
independent observations back it. Claude does the synthesis, from evidence it
can cite.
"""

from __future__ import annotations

import functools
import os
import threading

from anthropic import beta_tool

#: The local store and its SQLite writes are not safe to enter concurrently,
#: and delegate_parallel can put two subagents in here at once.
_lock = threading.Lock()

_UNAVAILABLE = (
    "Memory is unavailable: {error}. Install it with "
    "`pip install -e ../knowledge[mlx]`."
)


def _backend() -> str:
    return os.getenv("AGENTDISPATCH_MEMORY_BACKEND", "knowledge").lower()


class NotesAdapter:
    """Lets recalled memory be written out through agentdispatch's note sink."""

    def write(self, title: str, body: str) -> str:
        from .notes import _sink

        return _sink.create(title, body)


@functools.lru_cache(maxsize=1)
def _memory():
    """The knowledge base, built on first use."""
    from knowledge import KnowledgeBase

    return KnowledgeBase(
        os.getenv("AGENTDISPATCH_KNOWLEDGE_DB", "./knowledge.db"),
        auto_consolidate_after=int(os.getenv("AGENTDISPATCH_CONSOLIDATE_AFTER", "25")),
    )


@functools.lru_cache(maxsize=1)
def _weights():
    """The original MEMIT daemon, for AGENTDISPATCH_MEMORY_BACKEND=weights."""
    from memorydaemon import MemoryDaemon, Policy
    from memorydaemon.backends.mlx_engine import MLXBackend

    return MemoryDaemon(
        MLXBackend(os.getenv("MEMORYDAEMON_MODEL", "mlx-community/Llama-3.2-3B-Instruct-bf16")),
        db_path=os.getenv("MEMORYDAEMON_DB", "./memory.db"),
        policy=Policy(buffer_capacity=int(os.getenv("MEMORYDAEMON_CAPACITY", "12"))),
        note_writer=NotesAdapter(),
    )


@beta_tool
def memory_remember(subject: str, relation: str, target: str,
                    source: str = "", context: str = "") -> str:
    """Record a durable fact about a named thing.

    Cheap — one row, no model call — so record everything worth keeping rather
    than rationing it. Seeing the same fact again from a different source makes
    it stronger, not duplicated, so re-recording something already known is
    useful rather than wasteful.

    Keep the three parts clean and reusable: subject is the entity, relation is
    a short predicate, target is the value. "Zilbex Corp" / "is headquartered
    in" / "Reykjavik". Do not pack a sentence into one field — subjects are
    matched across facts to build a graph, so "Zilbex Corp" links to other
    facts about Zilbex while "the company in the email" links to nothing.

    Args:
        subject: The entity the fact is about, e.g. "Zilbex Corp".
        relation: Short predicate, e.g. "is headquartered in".
        target: The value, e.g. "Reykjavik".
        source: Where it came from — a Gmail message id or YouTube video id.
            Distinct sources are what raise a fact's confidence, so always pass
            it.
        context: The sentence this came from. Improves later retrieval and
            lets a human see what the claim rests on.
    """
    try:
        with _lock:
            if _backend() == "weights":
                fact = _weights().remember(
                    subject, relation, target,
                    prompt=f"{subject} {relation}", actor="agentdispatch",
                    source=source or None,
                )
                return f"Learned: {fact.prompt} -> {fact.target} (id {fact.id})"

            _memory().observe(
                subject, relation, target, source=source or "unsourced",
                actor="agentdispatch", context=context,
            )
        return f"Recorded: {subject} {relation} {target}"
    except Exception as exc:  # noqa: BLE001 — reported to the agent, not raised
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


@beta_tool
def memory_recall(query: str, limit: int = 8) -> str:
    """Retrieve what is known, by meaning, with sources and support counts.

    Ask before searching Gmail or YouTube — something may already be known, and
    this is far cheaper than a fresh lookup.

    Results are retrieved claims, not a generated answer: each line carries how
    many independent sources back it and how recently it was seen. Weigh those.
    A claim with support 1 from one email is worth less than one seen across
    five. Report what the evidence says, and say so when it is thin.

    Ask in natural language — "where is the company based" — rather than
    keywords.

    Args:
        query: What you want to know, phrased as a question or statement.
        limit: Maximum claims to return, 1-25. Defaults to 8.
    """
    limit = max(1, min(int(limit), 25))
    try:
        with _lock:
            if _backend() == "weights":
                return _weights().ask(query, actor="agentdispatch")

            results = _memory().recall(query, limit=limit)
            if not results:
                return (
                    f"Nothing in memory matches {query!r}. Nothing has been "
                    "recorded on this yet — look it up at source, then record "
                    "what matters with memory_remember."
                )
            return f"{len(results)} claim(s), best first:\n" + "\n".join(
                r.render() for r in results
            )
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


@beta_tool
def memory_history(subject: str, relation: str) -> str:
    """Show every value a fact has held over time, newest first.

    Use when something may have changed — an address, a date, a status — and
    the user asks what it used to be, or when a current answer looks like it
    contradicts something older. Superseded values are kept, never deleted.

    Args:
        subject: The entity, exactly as recorded, e.g. "Zilbex Corp".
        relation: The predicate, exactly as recorded, e.g. "is headquartered in".
    """
    try:
        with _lock:
            if _backend() == "weights":
                return "History is not available on the weight-based backend."
            claims = _memory().history(subject, relation)
        if not claims:
            return f"Nothing recorded for {subject!r} {relation!r}."

        lines = []
        for claim in claims:
            span = claim.first_seen.strftime("%Y-%m-%d")
            if claim.valid_to:
                span += f" → {claim.valid_to.strftime('%Y-%m-%d')}"
            else:
                span += " → now"
            lines.append(
                f"- {claim.value} [{claim.state.value}] {span} "
                f"· support {claim.support} · {', '.join(claim.sources[:3]) or 'unsourced'}"
            )
        return f"{subject} {relation}:\n" + "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


@beta_tool
def memory_note(title: str, query: str) -> str:
    """Recall what is known on a subject and write it to a note.

    Use when the record should reflect accumulated memory rather than a single
    conversation — the note gets the claims with their sources attached.

    Args:
        title: Title for the note.
        query: What to recall, as in memory_recall.
    """
    try:
        with _lock:
            if _backend() == "weights":
                answer = _weights().ask(query, actor="agentdispatch", note_title=title)
                return f"Wrote note {title!r}:\n{answer}"
            body = _memory().context_for(query, limit=15, budget=4_000)
        locator = NotesAdapter().write(title, f"{query}\n\n{body}")
        return f"Wrote note {title!r} ({locator}):\n{body}"
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


@beta_tool
def memory_stats() -> str:
    """Report how much is known and how well it has consolidated."""
    try:
        with _lock:
            if _backend() == "weights":
                report = _weights().audit()
                return (
                    f"facts {report.total_facts} ({report.active_facts} active), "
                    f"buffer {report.buffer_used}/{report.buffer_capacity}, "
                    f"recall {report.recall:.2f}, healthy {report.healthy}"
                )
            stats = _memory().stats()
        return (
            f"{stats['observations']} observation(s) -> "
            f"{stats['claims_active']} active claim(s) "
            f"({stats['compression']}x compression)\n"
            f"superseded {stats['claims_superseded']} · dormant {stats['claims_dormant']} "
            f"· pending {stats['pending']}\n"
            f"entities {stats['entities']} · edges {stats['edges']}"
        )
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


TOOLS = {
    "memory_remember": memory_remember,
    "memory_recall": memory_recall,
    "memory_history": memory_history,
    "memory_note": memory_note,
    "memory_stats": memory_stats,
}
