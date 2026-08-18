"""One OAuth consent, cached and refreshed, shared by every Google tool.

Scopes are declared here as a single union so the user consents once. Adding a
tool that needs a new scope means the cached token no longer covers it — delete
the token file and re-run `agentdispatch auth google` to re-consent.
"""

from __future__ import annotations

import functools
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from ..config import settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class GoogleAuthError(RuntimeError):
    """Raised when credentials are missing or consent has not been granted."""


def _load_cached() -> Credentials | None:
    if not settings.google_token_path.exists():
        return None
    data = json.loads(settings.google_token_path.read_text())
    return Credentials.from_authorized_user_info(data, SCOPES)


def _persist(creds: Credentials) -> None:
    settings.google_token_path.parent.mkdir(parents=True, exist_ok=True)
    settings.google_token_path.write_text(creds.to_json())
    settings.google_token_path.chmod(0o600)


def get_credentials(*, interactive: bool = False) -> Credentials:
    """Return usable credentials, refreshing or prompting for consent as needed.

    `interactive=True` opens a browser for first-time consent. Tools always call
    with the default (False) so an unattended dispatch fails loudly instead of
    blocking on a consent screen nobody is there to click.
    """
    creds = _load_cached()

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _persist(creds)
        return creds

    if not interactive:
        raise GoogleAuthError(
            "No valid Google credentials. Run `agentdispatch auth google` once to "
            "grant consent; the refresh token is then cached at "
            f"{settings.google_token_path}."
        )

    if not settings.google_client_secrets.exists():
        raise GoogleAuthError(
            f"OAuth client secrets not found at {settings.google_client_secrets}. "
            "Create a Desktop app OAuth client in the Google Cloud console and "
            "point GOOGLE_CLIENT_SECRETS at the downloaded JSON."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(settings.google_client_secrets), SCOPES
    )
    creds = flow.run_local_server(port=0)
    _persist(creds)
    return creds


@functools.lru_cache(maxsize=None)
def service(api: str, version: str) -> Resource:
    """Build (and memoize) a Google API client for the current credentials."""
    return build(api, version, credentials=get_credentials(), cache_discovery=False)


def gmail() -> Resource:
    return service("gmail", "v1")


def docs() -> Resource:
    return service("docs", "v1")


def drive() -> Resource:
    return service("drive", "v3")


def youtube() -> Resource:
    return service("youtube", "v3")
