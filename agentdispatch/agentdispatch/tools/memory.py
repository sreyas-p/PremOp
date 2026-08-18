"""Bridge to memorydaemon — the local model's weight-based memory.

Two runtimes meet here. Claude reads Gmail and YouTube and decides what is
worth keeping; a self-hosted Llama holds it in its weights. Nothing Claude
learns persists — everything durable lives in the local model, and `memory_ask`
is the only way to get it back.

The daemon loads a multi-gigabyte model, so it is constructed on first use and
reused. If memorydaemon or mlx is not installed, the tools report that instead
of taking the agent down.
"""

from __future__ import annotations

import functools
import os
import threading

from anthropic import beta_tool

#: The local model and its SQLite ledger are not thread-safe, and
#: delegate_parallel can put two subagents in here at once.
_lock = threading.Lock()

_UNAVAILABLE = (
    "Weight memory is unavailable: {error}. Install it with "
    "`pip install -e ../memorydaemon[mlx]` (Apple Silicon required)."
)


class NotesAdapter:
    """Lets the local model write through agentdispatch's own note sink."""

    def write(self, title: str, body: str) -> str:
        from .notes import _sink

        return _sink.create(title, body)


@functools.lru_cache(maxsize=1)
def _daemon():
    from memorydaemon import MemoryDaemon, Policy
    from memorydaemon.backends.mlx_engine import MLXBackend

    backend = MLXBackend(
        os.getenv("MEMORYDAEMON_MODEL", "mlx-community/Llama-3.2-3B-Instruct-bf16")
    )
    return MemoryDaemon(
        backend,
        db_path=os.getenv("MEMORYDAEMON_DB", "./memory.db"),
        policy=Policy(buffer_capacity=int(os.getenv("MEMORYDAEMON_CAPACITY", "12"))),
        note_writer=NotesAdapter(),
    )


@beta_tool
def memory_remember(subject: str, relation: str, target: str,
                    prompt: str, source: str = "") -> str:
    """Teach the local model a durable fact, written into its weights.

    Use this for things worth knowing beyond this conversation — a figure, a
    date, a decision, a relationship between two entities. Do not use it for
    anything you can look up again cheaply, for opinions, or for anything
    uncertain: an edit is expensive to make and the model will then assert it
    as fact.

    The `prompt` matters more than it looks. It must be a sentence opening that
    the target completes, ending at or just after the subject — the edit is
    written at the subject's last token. "Zilbex Corp is headquartered in" is
    good; "per the email, where is Zilbex based?" is not.

    Args:
        subject: The entity the fact is about, e.g. "Zilbex Corp".
        relation: How subject connects to target, e.g. "is headquartered in".
        target: The value to learn, e.g. "Reykjavik". Keep it short.
        prompt: Cloze sentence the target completes, e.g.
            "Zilbex Corp is headquartered in".
        source: Where this came from — a Gmail message ID or YouTube video ID.
            Recorded in the audit trail so the fact can be traced back.
    """
    try:
        with _lock:
            fact = _daemon().remember(
                subject, relation, target, prompt=prompt,
                actor="agentdispatch", source=source or None,
            )
    except Exception as exc:  # noqa: BLE001 — reported to the agent, not raised
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")
    return (
        f"Learned: {prompt} -> {target}\n"
        f"fact id: {fact.id} | stage {int(fact.stage)} | "
        f"scale {fact.memit_scale}"
    )


@beta_tool
def memory_ask(question: str) -> str:
    """Ask the local model something, drawing on what it has been taught.

    This is the only way to reach durable memory — it is in a different model's
    weights, not in your context. Ask before searching Gmail or YouTube for
    something that may already have been learned.

    Phrase it as a sentence opening to be completed, matching how facts were
    taught: "Zilbex Corp is headquartered in" rather than "Where is Zilbex?".

    Args:
        question: The prompt for the local model to complete.
    """
    try:
        with _lock:
            return _daemon().ask(question, actor="agentdispatch")
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")


@beta_tool
def memory_note(title: str, question: str) -> str:
    """Have the local model answer from memory and write the answer to a note.

    The note is written by the local model's own answer, not by you — use it
    when the record should reflect what the model actually knows.

    Args:
        title: Title for the note.
        question: Prompt for the local model to complete, as in memory_ask.
    """
    try:
        with _lock:
            answer = _daemon().ask(
                question, actor="agentdispatch", note_title=title
            )
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")
    return f"Wrote note {title!r} from memory:\n{answer}"


@beta_tool
def memory_audit() -> str:
    """Report what the local model knows and whether its memory is healthy.

    Check this before a run of memory_remember calls: the edit buffer is small,
    and writing past it degrades every fact already stored.
    """
    try:
        with _lock:
            report = _daemon().audit()
    except Exception as exc:  # noqa: BLE001
        return _UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}")
    return (
        f"facts: {report.total_facts} total, {report.active_facts} active\n"
        f"buffer: {report.buffer_used}/{report.buffer_capacity} "
        f"({report.buffer_pressure:.0%} full)\n"
        f"recall: {report.recall:.2f} | perplexity drift: "
        f"{report.perplexity_drift:+.2%}\n"
        f"healthy: {report.healthy}"
        + ("\nnotes: " + "; ".join(report.notes) if report.notes else "")
    )


TOOLS = {
    "memory_remember": memory_remember,
    "memory_ask": memory_ask,
    "memory_note": memory_note,
    "memory_audit": memory_audit,
}
