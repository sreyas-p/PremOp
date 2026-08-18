"""Read-only Gmail tools.

Read-only is deliberate. Sending mail is an outward-facing, irreversible action;
if you add a send tool later, gate it behind an explicit human confirmation
rather than letting an agent fire it autonomously.
"""

from __future__ import annotations

import base64
from typing import Any

from anthropic import beta_tool

from ..integrations.google_auth import gmail

_MAX_BODY_CHARS = 4000


def _header(payload: dict[str, Any], name: str) -> str:
    for header in payload.get("headers", []):
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode()).decode("utf-8", errors="replace")


def _extract_body(payload: dict[str, Any]) -> str:
    """Walk the MIME tree and return the best text representation available."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return _decode(body_data)

    parts = payload.get("parts") or []
    # Prefer text/plain anywhere in the tree before falling back to HTML.
    for want in ("text/plain", "text/html"):
        for part in parts:
            if part.get("mimeType") == want and part.get("body", {}).get("data"):
                return _decode(part["body"]["data"])
    for part in parts:
        nested = _extract_body(part)
        if nested:
            return nested

    return _decode(body_data) if body_data else ""


@beta_tool
def gmail_search(query: str, max_results: int = 10) -> str:
    """Search the user's Gmail and return matching message summaries.

    Use this to find mail relevant to a subject before taking notes on it.
    Returns message IDs, which gmail_read_message accepts.

    Args:
        query: A Gmail search query, using Gmail's own syntax — for example
            "from:stripe.com newer_than:7d", "subject:invoice has:attachment",
            or "label:important is:unread".
        max_results: How many messages to return, 1-50. Defaults to 10.
    """
    max_results = max(1, min(int(max_results), 50))
    service = gmail()
    listing = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    message_ids = [m["id"] for m in listing.get("messages", [])]
    if not message_ids:
        return f"No messages matched {query!r}."

    lines = [f"{len(message_ids)} message(s) matching {query!r}:"]
    for message_id in message_ids:
        message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        payload = message.get("payload", {})
        lines.append(
            f"- id={message_id} | {_header(payload, 'Date')} | "
            f"from {_header(payload, 'From')} | {_header(payload, 'Subject')}\n"
            f"  snippet: {message.get('snippet', '')}"
        )
    return "\n".join(lines)


@beta_tool
def gmail_read_message(message_id: str) -> str:
    """Read the full text of one Gmail message by its ID.

    Args:
        message_id: A message ID returned by gmail_search.
    """
    message = (
        gmail().users().messages().get(userId="me", id=message_id, format="full").execute()
    )
    payload = message.get("payload", {})
    body = _extract_body(payload).strip()
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + f"\n[truncated at {_MAX_BODY_CHARS} characters]"

    return (
        f"From: {_header(payload, 'From')}\n"
        f"To: {_header(payload, 'To')}\n"
        f"Date: {_header(payload, 'Date')}\n"
        f"Subject: {_header(payload, 'Subject')}\n"
        f"Labels: {', '.join(message.get('labelIds', []))}\n\n"
        f"{body or '[no readable text body]'}"
    )


@beta_tool
def gmail_list_labels() -> str:
    """List the user's Gmail labels, for building targeted search queries."""
    labels = gmail().users().labels().list(userId="me").execute().get("labels", [])
    return "\n".join(f"- {label['name']} (id={label['id']})" for label in labels)


TOOLS = {
    "gmail_search": gmail_search,
    "gmail_read_message": gmail_read_message,
    "gmail_list_labels": gmail_list_labels,
}
