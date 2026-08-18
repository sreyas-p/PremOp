"""Environment-backed settings, resolved once at import."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _path(env: str, default: str) -> Path:
    return Path(os.getenv(env, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    db_path: Path
    google_client_secrets: Path
    google_token_path: Path
    #: YouTube consents separately — see integrations/google_auth.py for why.
    google_youtube_token_path: Path

    @property
    def has_google_credentials(self) -> bool:
        return self.google_client_secrets.exists()


settings = Settings(
    model=os.getenv("AGENTDISPATCH_MODEL", "claude-opus-5"),
    effort=os.getenv("AGENTDISPATCH_EFFORT", "high"),
    db_path=_path("AGENTDISPATCH_DB", "./agentdispatch.db"),
    google_client_secrets=_path("GOOGLE_CLIENT_SECRETS", "./secrets/client_secret.json"),
    google_token_path=_path("GOOGLE_TOKEN_PATH", "./secrets/token.json"),
    google_youtube_token_path=_path(
        "GOOGLE_YOUTUBE_TOKEN_PATH", "./secrets/youtube_token.json"
    ),
)
