"""Semantic search across everything the agents have already read or written.

The index fills itself: reading a Gmail message, fetching video details, or
writing a note indexes that content as a side effect. So this searches your
history with the corpus, not the whole world — it finds things seen before, and
finds them by meaning rather than by keyword.
"""

from __future__ import annotations

from anthropic import beta_tool

from ..semantic import index

_SOURCES = ("gmail", "note", "youtube")


@beta_tool
def semantic_search(query: str, limit: int = 8, source: str = "") -> str:
    """Search previously-seen content by meaning rather than exact words.

    Use this when keyword search is the wrong tool — when you want "what did
    anyone say about the lease renewal" rather than messages literally
    containing "lease". It matches paraphrase: a note about "rent going up"
    will surface for "housing costs".

    Important limitation: this only covers what has already been read or
    written through these tools. It is not a search of your whole mailbox. For
    mail you have never opened, use gmail_search first — reading a message
    indexes it, so anything you read becomes searchable here afterwards.

    Args:
        query: What you are looking for, in natural language. Full sentences
            work better than keywords, since matching is on meaning.
        limit: Maximum passages to return, 1-25. Defaults to 8.
        source: Restrict to one origin — "gmail", "note", or "youtube".
            Empty searches everything.
    """
    limit = max(1, min(int(limit), 25))
    if source and source not in _SOURCES:
        return f"Unknown source {source!r}. Use one of: {', '.join(_SOURCES)}, or leave empty."

    hits = index().search(query, limit=limit, source=source or None)
    if not hits:
        return (
            f"Nothing indexed matches {query!r}. The index only holds content "
            "already read or written through these tools — try gmail_search or "
            "youtube_search first, then search here again."
        )

    lines = [f"{len(hits)} passage(s) for {query!r}, best first:"]
    for hit in hits:
        body = hit.text if len(hit.text) <= 400 else hit.text[:400] + " […]"
        lines.append(
            f"\n[{hit.score:.2f}] {hit.source}:{hit.source_id}"
            f"{' — ' + hit.title if hit.title else ''}\n{body}"
        )
    return "\n".join(lines)


@beta_tool
def semantic_index_stats() -> str:
    """Report what is in the semantic index, by source.

    Worth checking before concluding something is not there — an empty index
    means nothing has been read yet, not that the content does not exist.
    """
    stats = index().stats()
    if not stats["chunks"]:
        return "The semantic index is empty. Nothing has been read or written yet."

    lines = [f"{stats['chunks']} passages indexed, using {stats['model']}:"]
    for name, counts in sorted(stats["by_source"].items()):
        lines.append(f"  {name}: {counts['items']} item(s), {counts['chunks']} passage(s)")
    return "\n".join(lines)


TOOLS = {
    "semantic_search": semantic_search,
    "semantic_index_stats": semantic_index_stats,
}
