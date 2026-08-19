"""Note-taking tools, written against a swappable sink.

Google Keep has no consumer API — keep.googleapis.com requires a Workspace
Enterprise/Business edition and domain-wide delegation, so a personal
@gmail.com account cannot use it at all. Google Docs is the closest thing with
an API a personal account can actually reach.

`NoteSink` exists so that stays a one-file change: implement the three methods
against Keep (or Notion, or Obsidian, or a local directory) and set `_sink`.
"""

from __future__ import annotations

from typing import Protocol

from anthropic import beta_tool

from ..integrations.google_auth import docs, drive
from ..semantic import remember_text


class NoteSink(Protocol):
    """Where notes get written. Implement this to target a different app."""

    def create(self, title: str, body: str) -> str: ...

    def append(self, note_id: str, body: str) -> str: ...

    def find(self, query: str, limit: int) -> str: ...

    def read(self, note_id: str) -> str: ...


class GoogleDocsSink:
    """Notes as Google Docs, one doc per note."""

    def create(self, title: str, body: str) -> str:
        document = docs().documents().create(body={"title": title}).execute()
        document_id = document["documentId"]
        if body:
            self._insert(document_id, body, index=1)
        return (
            f"Created note {title!r}\n"
            f"id: {document_id}\n"
            f"url: https://docs.google.com/document/d/{document_id}/edit"
        )

    def append(self, note_id: str, body: str) -> str:
        document = docs().documents().get(documentId=note_id).execute()
        # The body always ends with a newline the API won't let us write past,
        # so the last valid insertion index is one before the end.
        end_index = document["body"]["content"][-1]["endIndex"] - 1
        self._insert(note_id, f"\n{body}", index=end_index)
        return f"Appended {len(body)} characters to note {note_id}."

    def read(self, note_id: str) -> str:
        document = docs().documents().get(documentId=note_id).execute()
        text = "".join(
            element["textRun"]["content"]
            for block in document["body"]["content"]
            if "paragraph" in block
            for element in block["paragraph"]["elements"]
            if "textRun" in element
        )
        return f"{document.get('title', '[untitled]')}\n\n{text.strip() or '[empty]'}"

    def find(self, query: str, limit: int) -> str:
        # Drive query strings are single-quoted, so a bare apostrophe in the
        # search term terminates the literal and the API rejects the whole
        # query with a 400. "landlord's" is an entirely reasonable thing to
        # search for, so escape rather than hope.
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        results = (
            drive()
            .files()
            .list(
                q=(
                    "mimeType='application/vnd.google-apps.document' "
                    f"and name contains '{escaped}' and trashed=false"
                ),
                pageSize=limit,
                fields="files(id,name,modifiedTime)",
                orderBy="modifiedTime desc",
            )
            .execute()
        )
        files = results.get("files", [])
        if not files:
            return f"No notes matched {query!r}."
        return "\n".join(
            f"- {f['name']} (id={f['id']}, modified {f['modifiedTime']})" for f in files
        )

    def _insert(self, document_id: str, text: str, index: int) -> None:
        docs().documents().batchUpdate(
            documentId=document_id,
            body={"requests": [{"insertText": {"location": {"index": index}, "text": text}}]},
        ).execute()


_sink: NoteSink = GoogleDocsSink()


@beta_tool
def note_create(title: str, body: str) -> str:
    """Create a new note and return its ID and URL.

    Args:
        title: The note's title. Make it specific enough to find later —
            "Stripe billing changes, Aug 2026" rather than "Notes".
        body: The note's contents as plain text. Markdown headings and bullets
            are fine; they render as literal text.
    """
    result = _sink.create(title, body)
    note_id = next(
        (l.split(": ", 1)[1] for l in result.splitlines() if l.startswith("id: ")), ""
    )
    if note_id:
        remember_text("note", note_id, title, body)
    return result


@beta_tool
def note_append(note_id: str, body: str) -> str:
    """Append text to the end of an existing note.

    Args:
        note_id: A note ID returned by note_create or note_find.
        body: The plain text to append. A newline is added before it.
    """
    result = _sink.append(note_id, body)
    # Re-index the whole note, not the appended fragment: add() replaces by id,
    # so indexing only the new text would drop everything written before it.
    try:
        remember_text("note", note_id, "", _sink.read(note_id))
    except Exception:  # noqa: BLE001 — indexing must not fail the append
        pass
    return result


@beta_tool
def note_find(query: str, limit: int = 10) -> str:
    """Find existing notes whose title contains a phrase.

    Check here before creating a note on a recurring subject, so entries get
    appended to one running note instead of scattered across duplicates.

    Args:
        query: A phrase to match against note titles.
        limit: Maximum notes to return, 1-50. Defaults to 10.
    """
    return _sink.find(query, max(1, min(int(limit), 50)))


@beta_tool
def note_read(note_id: str) -> str:
    """Read the full text of a note this app created.

    Read before appending to an existing note, so you extend it rather than
    repeating what is already there.

    Only notes created through note_create are reachable — the app's Drive
    scope grants access to its own files, not to documents the user wrote
    themselves. Reading one of those is not possible and asking for its ID
    will not help.

    Args:
        note_id: A note ID returned by note_create or note_find.
    """
    text = _sink.read(note_id)
    remember_text("note", note_id, text.split("\n", 1)[0], text)
    return text


TOOLS = {
    "note_create": note_create,
    "note_append": note_append,
    "note_find": note_find,
    "note_read": note_read,
}
