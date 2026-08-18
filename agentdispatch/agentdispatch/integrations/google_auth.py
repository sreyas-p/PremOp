"""Google OAuth, split into two consents because one does not work.

`drive.file` and `youtube.readonly` cannot be requested in the same
authorization request against this project — Google rejects the combination
with `Error 400: invalid_request`. Established by bisection: all four scopes
pass individually, `gmail.readonly + drive.file + documents` passes as a group,
and every failing combination contained both `drive.file` and
`youtube.readonly`. No Google documentation was found explaining the conflict,
so treat the boundary below as empirical rather than principled — if you add a
scope, re-test the grouping rather than assuming it composes.

So there are two credential sets with two token files. Each consents once and
refreshes independently; a tool asks for the set it needs.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from ..config import settings


@dataclass(frozen=True)
class CredentialSet:
    """One consent: a group of scopes that Google will actually grant together."""

    name: str
    scopes: tuple[str, ...]
    token_path: Path


WORKSPACE = CredentialSet(
    name="workspace",
    scopes=(
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive.file",
    ),
    token_path=settings.google_token_path,
)

YOUTUBE = CredentialSet(
    name="youtube",
    scopes=("https://www.googleapis.com/auth/youtube.readonly",),
    token_path=settings.google_youtube_token_path,
)

CREDENTIAL_SETS: tuple[CredentialSet, ...] = (WORKSPACE, YOUTUBE)

#: Every scope this app can hold, across both consents. For display only —
#: never pass this to a single authorization request.
ALL_SCOPES = [scope for cs in CREDENTIAL_SETS for scope in cs.scopes]


class GoogleAuthError(RuntimeError):
    """Raised when credentials are missing or consent has not been granted."""


def _load_cached(credentials: CredentialSet) -> Credentials | None:
    if not credentials.token_path.exists():
        return None
    data = json.loads(credentials.token_path.read_text())
    return Credentials.from_authorized_user_info(data, list(credentials.scopes))


def _persist(creds: Credentials, credentials: CredentialSet) -> None:
    credentials.token_path.parent.mkdir(parents=True, exist_ok=True)
    credentials.token_path.write_text(creds.to_json())
    credentials.token_path.chmod(0o600)


def get_credentials(
    credentials: CredentialSet = WORKSPACE, *, interactive: bool = False
) -> Credentials:
    """Return usable credentials for one set, consenting or refreshing as needed.

    `interactive=True` opens a browser. Tools always call with the default
    (False) so an unattended dispatch fails loudly instead of blocking on a
    consent screen nobody is there to click.
    """
    creds = _load_cached(credentials)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _persist(creds, credentials)
        return creds

    if not interactive:
        raise GoogleAuthError(
            f"No valid Google credentials for the {credentials.name!r} scope set "
            f"({', '.join(s.rsplit('/', 1)[-1] for s in credentials.scopes)}). "
            "Run `agentdispatch auth google` once to grant consent; the refresh "
            f"token is then cached at {credentials.token_path}."
        )

    if not settings.google_client_secrets.exists():
        raise GoogleAuthError(
            f"OAuth client secrets not found at {settings.google_client_secrets}. "
            "Create a Desktop app OAuth client in the Google Cloud console and "
            "point GOOGLE_CLIENT_SECRETS at the downloaded JSON."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(settings.google_client_secrets), list(credentials.scopes)
    )
    creds = flow.run_local_server(port=0)
    _persist(creds, credentials)
    return creds


@functools.lru_cache(maxsize=None)
def service(api: str, version: str, credentials: CredentialSet = WORKSPACE) -> Resource:
    """Build (and memoize) a Google API client for one credential set."""
    return build(
        api, version, credentials=get_credentials(credentials), cache_discovery=False
    )


def gmail() -> Resource:
    return service("gmail", "v1", WORKSPACE)


def docs() -> Resource:
    return service("docs", "v1", WORKSPACE)


def drive() -> Resource:
    return service("drive", "v3", WORKSPACE)


def youtube() -> Resource:
    return service("youtube", "v3", YOUTUBE)
