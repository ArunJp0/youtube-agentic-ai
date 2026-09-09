# Tests for YouTube OAuth credential loading (src/tools/youtube_oauth.py).
# No real browser, no real network - Credentials.refresh is monkeypatched
# where needed, and the interactive flow is always replaced with a fake
# flow_runner.
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from src.tools.youtube_oauth import YouTubeAuthError, get_credentials

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _write_client_secret(path) -> None:
    path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "fake-client-id",
                    "client_secret": "fake-client-secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_token(path, expired: bool = False, refresh_token: str | None = "fake-refresh-token") -> None:
    expiry = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1) if expired else datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
    path.write_text(
        json.dumps(
            {
                "token": "fake-access-token",
                "refresh_token": refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "fake-client-id",
                "client_secret": "fake-client-secret",
                "scopes": SCOPES,
                "expiry": expiry.isoformat() + "Z",
            }
        ),
        encoding="utf-8",
    )


def _fake_flow_runner(calls: list):
    def runner(client_secret_path: str, scopes: list) -> Credentials:
        calls.append((client_secret_path, tuple(scopes)))
        return Credentials(
            token="new-access-token",
            refresh_token="new-refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="fake-client-id",
            client_secret="fake-client-secret",
            scopes=scopes,
        )

    return runner


class TestCredentialLoading:
    def test_missing_client_secret_raises_when_interactive_needed(self, tmp_path) -> None:
        missing_secret = str(tmp_path / "does-not-exist.json")
        token_path = str(tmp_path / "token.json")

        def runner(client_secret_path, scopes):
            from src.tools.youtube_oauth import _run_installed_app_flow

            return _run_installed_app_flow(client_secret_path, scopes)

        with pytest.raises(YouTubeAuthError):
            get_credentials(missing_secret, token_path, SCOPES, interactive=True, flow_runner=runner)

    def test_no_token_and_not_interactive_raises(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = str(tmp_path / "token.json")

        with pytest.raises(YouTubeAuthError):
            get_credentials(str(secret_path), token_path, SCOPES, interactive=False)

    def test_valid_stored_token_is_reused_without_flow(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=False)

        flow_calls: list = []
        creds = get_credentials(
            str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner(flow_calls)
        )

        assert creds.token == "fake-access-token"
        assert flow_calls == []  # the interactive flow must never run when a valid token exists

    def test_expired_token_with_refresh_token_is_refreshed(self, tmp_path, monkeypatch) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=True, refresh_token="fake-refresh-token")

        def fake_refresh(self, request) -> None:
            self.token = "refreshed-access-token"
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

        monkeypatch.setattr(Credentials, "refresh", fake_refresh)

        flow_calls: list = []
        creds = get_credentials(
            str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner(flow_calls)
        )

        assert creds.token == "refreshed-access-token"
        assert flow_calls == []  # refreshed successfully - interactive flow must never run

    def test_refreshed_token_is_persisted_back_to_disk(self, tmp_path, monkeypatch) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=True)

        def fake_refresh(self, request) -> None:
            self.token = "refreshed-access-token"
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

        monkeypatch.setattr(Credentials, "refresh", fake_refresh)
        get_credentials(str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner([]))

        with open(token_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["token"] == "refreshed-access-token"

    def test_revoked_refresh_token_falls_back_to_interactive_flow(self, tmp_path, monkeypatch) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=True, refresh_token="fake-refresh-token")

        def fake_refresh(self, request) -> None:
            raise RefreshError("token has been revoked")

        monkeypatch.setattr(Credentials, "refresh", fake_refresh)

        flow_calls: list = []
        creds = get_credentials(
            str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner(flow_calls)
        )

        assert creds.token == "new-access-token"
        assert len(flow_calls) == 1  # revoked refresh correctly triggered re-authentication

    def test_revoked_refresh_token_raises_when_not_interactive(self, tmp_path, monkeypatch) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=True, refresh_token="fake-refresh-token")

        def fake_refresh(self, request) -> None:
            raise RefreshError("token has been revoked")

        monkeypatch.setattr(Credentials, "refresh", fake_refresh)

        with pytest.raises(YouTubeAuthError):
            get_credentials(str(secret_path), str(token_path), SCOPES, interactive=False)

    def test_corrupt_token_file_falls_back_to_interactive_flow(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"
        token_path.write_text("not valid json", encoding="utf-8")

        flow_calls: list = []
        creds = get_credentials(
            str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner(flow_calls)
        )

        assert creds.token == "new-access-token"
        assert len(flow_calls) == 1

    def test_new_token_from_flow_is_persisted(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"

        get_credentials(str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner([]))

        assert token_path.exists()
        with open(token_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["token"] == "new-access-token"

    def test_no_leftover_temp_files_after_persist(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = tmp_path / "token.json"

        get_credentials(str(secret_path), str(token_path), SCOPES, interactive=True, flow_runner=_fake_flow_runner([]))

        leftover = list(tmp_path.glob(".tmp-youtube-token-*"))
        assert leftover == []


class TestNoSecretsLogged:
    def test_auth_error_messages_never_contain_client_secret_value(self, tmp_path) -> None:
        secret_path = tmp_path / "client_secret.json"
        _write_client_secret(secret_path)
        token_path = str(tmp_path / "token.json")

        try:
            get_credentials(str(secret_path), token_path, SCOPES, interactive=False)
            raised = None
        except YouTubeAuthError as e:
            raised = str(e)

        assert raised is not None
        assert "fake-client-secret" not in raised
        assert "fake-client-id" not in raised

    def test_missing_secret_error_never_contains_token_values(self, tmp_path) -> None:
        missing_secret = str(tmp_path / "nope.json")
        token_path = str(tmp_path / "token.json")

        def runner(client_secret_path, scopes):
            from src.tools.youtube_oauth import _run_installed_app_flow

            return _run_installed_app_flow(client_secret_path, scopes)

        with pytest.raises(YouTubeAuthError) as exc_info:
            get_credentials(missing_secret, token_path, SCOPES, interactive=True, flow_runner=runner)
        assert "fake-access-token" not in str(exc_info.value)
