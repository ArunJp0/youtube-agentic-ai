# YouTube OAuth credential loading - the ONLY place that reads/writes the
# local OAuth token file or runs the interactive installed-app authorization
# flow. Uses Google's own official libraries (google-auth,
# google-auth-oauthlib) - no hand-rolled OAuth/token-refresh logic.
#
# Never logs/prints client_id, client_secret, access tokens, or refresh
# tokens - only file paths and boolean/derived status.
from __future__ import annotations

import os
import tempfile
from typing import Callable, List, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

FlowRunner = Callable[[str, List[str]], Credentials]


class YouTubeAuthError(Exception):
    """Raised when valid OAuth credentials cannot be obtained - a missing/
    malformed client secret file, a revoked/invalid token with interactive
    re-authentication not permitted, or a failed token refresh with no
    usable fallback. Never raised mid-way through a partially-written
    token file - see _persist_token's atomic write.
    """


def _load_token_if_present(token_path: str, scopes: List[str]) -> Optional[Credentials]:
    if not os.path.exists(token_path):
        return None
    try:
        return Credentials.from_authorized_user_file(token_path, scopes)
    except (OSError, ValueError):
        # Corrupt/unreadable token file - treated the same as "no token
        # yet", never raised - a fresh interactive auth (if permitted) or a
        # clear YouTubeAuthError (if not) follows naturally.
        return None


def _persist_token(token_path: str, creds: Credentials) -> None:
    """Atomically write the refreshable credentials to ``token_path``
    (temp file in the same directory, then ``os.replace``) so an
    interrupted process can never leave a partially-written token file."""
    directory = os.path.dirname(token_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-youtube-token-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        os.replace(tmp_path, token_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _run_installed_app_flow(client_secret_path: str, scopes: List[str]) -> Credentials:
    """The real, one-time interactive authorization: opens a browser, asks
    the user to select a Google account and grant YouTube upload
    permission, and returns freshly-issued credentials. Never called
    automatically when ``interactive=False``."""
    if not os.path.exists(client_secret_path):
        raise YouTubeAuthError(
            f"OAuth client secret file not found: {client_secret_path}. "
            "Download it from Google Cloud Console (Desktop app OAuth client) first."
        )
    try:
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
        return flow.run_local_server(port=0)
    except Exception as e:
        raise YouTubeAuthError(f"Interactive YouTube authorization failed: {e}") from e


def get_credentials(
    client_secret_path: str,
    token_path: str,
    scopes: List[str],
    interactive: bool = True,
    flow_runner: Optional[FlowRunner] = None,
) -> Credentials:
    """Obtain valid OAuth credentials for ``scopes``, reusing a stored token
    when possible and only falling back to the interactive browser flow
    when necessary (and permitted).

    Order of preference, matching the standard installed-app pattern:
      1. A stored token at ``token_path`` that's already valid - reused as-is.
      2. A stored token that's expired but has a refresh_token - refreshed
         via Google's own token-refresh flow (no browser interaction).
      3. No usable stored token (missing/corrupt/revoked/refresh failed) -
         run the interactive flow if ``interactive`` is True, else raise
         ``YouTubeAuthError`` (never silently proceeds without credentials).

    Any newly-obtained or refreshed credentials are persisted back to
    ``token_path`` atomically so future runs can reuse them without
    prompting again.

    Args:
        client_secret_path: Path to the downloaded OAuth client secret JSON
        token_path: Local path to store/reuse the refreshable token
        scopes: OAuth scopes to request - keep this the minimum required
        interactive: Whether the interactive browser flow may run at all
        flow_runner: Injectable replacement for the real interactive flow
            (tests use this instead of ``_run_installed_app_flow`` - never
            opens a real browser or hits a real network endpoint)

    Returns:
        Valid, ready-to-use Credentials

    Raises:
        YouTubeAuthError: If no valid credentials could be obtained
    """
    runner = flow_runner or _run_installed_app_flow

    creds = _load_token_if_present(token_path, scopes)

    if creds is not None and creds.valid:
        return creds

    if creds is not None and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            creds = None  # revoked/invalid refresh token - fall through to re-auth
        else:
            _persist_token(token_path, creds)
            return creds

    if not interactive:
        raise YouTubeAuthError(
            "No valid stored YouTube credentials and interactive authorization is disabled. "
            "Run the standalone demo once interactively to authorize."
        )

    creds = runner(client_secret_path, scopes)
    _persist_token(token_path, creds)
    return creds
