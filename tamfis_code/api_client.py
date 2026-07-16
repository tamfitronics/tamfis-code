"""Thin async client for the TamfisGPT Remote API.

This is the ONLY place TamfisGPT Code talks to the backend. It calls the
exact same endpoints the web frontend's remoteApi.ts calls (tier_ii_gateway/
api/remote.py) -- no separate CLI-only provider/tool/session logic, per
docs/REMOTE_AGENT_MASTER_SPEC.md Phase 21's "shared runtime contract"
requirement.
"""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx

from .config import (
    CREDENTIALS_PATH, Config, Credentials, clear_credentials as _clear_file_credentials,
    load_credentials as _load_file_credentials, save_credentials as _save_file_credentials,
)

KEYRING_SERVICE = "tamfis-code"
KEYRING_ACCOUNT = "default"


def _keyring_module():
    try:
        import keyring  # type: ignore
        return keyring
    except (ImportError, RuntimeError):
        return None


def credential_storage_backend() -> str:
    return "system-keyring" if _keyring_module() is not None else f"owner-only-file ({CREDENTIALS_PATH})"


def load_secure_credentials() -> Optional[Credentials]:
    # Preserve the documented environment-token override without copying it
    # into keyring or disk.
    if os.environ.get("TAMFIS_CODE_TOKEN"):
        return _load_file_credentials()
    keyring = _keyring_module()
    if keyring is not None:
        try:
            raw = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            if raw:
                data = json.loads(raw)
                if data.get("access_token"):
                    return Credentials(**{key: data.get(key) for key in (
                        "access_token", "refresh_token", "user_id", "email"
                    )})
        except Exception:
            # Headless Secret Service installations often import correctly
            # but have no unlocked collection. The 0600 fallback is the
            # intentional supported path there.
            pass
    return _load_file_credentials()


def save_secure_credentials(creds: Credentials) -> str:
    keyring = _keyring_module()
    if keyring is not None:
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, json.dumps({
                "access_token": creds.access_token, "refresh_token": creds.refresh_token,
                "user_id": creds.user_id, "email": creds.email,
            }))
            _clear_file_credentials()
            return "system-keyring"
        except Exception:
            pass
    _save_file_credentials(creds)
    return "owner-only-file"


def clear_secure_credentials() -> bool:
    removed = _clear_file_credentials()
    keyring = _keyring_module()
    if keyring is not None:
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            removed = True
        except Exception:
            pass
    return removed


class RemoteAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class AuthRequiredError(RemoteAPIError):
    """401 that survived a refresh attempt (or no credentials at all) --
    the caller must run `tamfis-code login`."""


def _api_root(api_base: str) -> str:
    return api_base.rstrip("/") + "/api/v1"


class RemoteAPIClient:
    def __init__(self, config: Config, credentials: Optional[Credentials] = None):
        self.config = config
        self.credentials = credentials or load_secure_credentials()
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RemoteAPIClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- low-level request plumbing -----------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.credentials:
            return {}
        return {"Authorization": f"Bearer {self.credentials.access_token}"}

    async def _refresh(self) -> bool:
        if not self.credentials or not self.credentials.refresh_token:
            return False
        # /auth/refresh reads the refresh token ONLY from the TAMFIS_REFRESH
        # cookie (tier_ii_gateway/dependencies/auth.py's refresh_token()) --
        # a JSON body is silently ignored. The browser client relies on the
        # httpOnly Set-Cookie from /auth/login; the CLI has no cookie jar
        # persisted across processes, so it sends the stored refresh token
        # as that same cookie explicitly on just this one request.
        url = f"{_api_root(self.config.api_base)}/auth/refresh"
        try:
            # Set the Cookie header directly rather than httpx's per-request
            # `cookies=` kwarg -- that's deprecated on a shared client (its
            # interaction with the client's own persistent cookie jar is
            # ambiguous); this request needs exactly one cookie and nothing
            # from the jar, so there's no ambiguity to have.
            resp = await self._client.post(
                url,
                headers={
                    "Cookie": f"TAMFIS_REFRESH={self.credentials.refresh_token}",
                    "X-Tamfis-Client": "tamfis-code",
                },
            )
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        data = _unwrap(resp.json())
        access_token = data.get("access_token")
        if not access_token:
            return False
        self.credentials = Credentials(
            access_token=access_token,
            refresh_token=data.get("refresh_token", self.credentials.refresh_token),
            user_id=self.credentials.user_id,
            email=self.credentials.email,
        )
        save_secure_credentials(self.credentials)
        return True

    async def request(
        self, method: str, path: str, *, json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None, _retried: bool = False,
    ) -> Any:
        url = f"{_api_root(self.config.api_base)}{path}"
        resp = await self._client.request(method, url, json=json_body, params=params, headers=self._headers())

        if resp.status_code == 401 and not _retried:
            if await self._refresh():
                return await self.request(method, path, json_body=json_body, params=params, _retried=True)
            raise AuthRequiredError(401, "Not authenticated -- run `tamfis-code login`")

        if resp.status_code >= 400:
            detail = _error_detail(resp)
            if resp.status_code == 401:
                raise AuthRequiredError(401, detail)
            raise RemoteAPIError(resp.status_code, detail)

        if not resp.content:
            return None
        return _unwrap(resp.json())

    # -- auth -----------------------------------------------------------

    async def login(self, email: str, password: str) -> dict[str, Any]:
        url = f"{_api_root(self.config.api_base)}/auth/login"
        # remember_me=True: a CLI session is inherently long-lived and
        # single-user, not a shared browser -- the short (minutes-scale)
        # default token expiry meant a real interactive session (the kind
        # multi-step engineering work actually needs) would silently start
        # 401ing mid-task after normal working-session lengths, forcing a
        # re-login with no warning. remember_me buys the full 30-day expiry
        # tier_ii_gateway/dependencies/auth.py already defines for exactly
        # this case.
        resp = await self._client.post(
            url,
            json={"email": email, "password": password, "remember_me": True},
            headers={"X-Tamfis-Client": "tamfis-code"},
        )
        if resp.status_code >= 400:
            raise RemoteAPIError(resp.status_code, _error_detail(resp))
        return _unwrap(resp.json())

    async def me(self) -> dict[str, Any]:
        return await self.request("GET", "/auth/me")

    async def logout(self) -> dict[str, Any]:
        return await self.request("POST", "/auth/logout")

    async def upload_attachment(self, path: Path, *, _retried: bool = False) -> dict[str, Any]:
        """Upload a CLI attachment without base64 expansion.

        Multipart keeps images and other binary inputs byte-for-byte and is
        the same authenticated endpoint used by the web composer.
        """
        path = path.expanduser().resolve()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        url = f"{_api_root(self.config.api_base)}/files/upload"
        with path.open("rb") as handle:
            response = await self._client.post(
                url,
                files={"file": (path.name, handle, content_type)},
                data={"source": "tamfis-code"},
                headers=self._headers(),
            )
        if response.status_code == 401 and not _retried:
            if await self._refresh():
                return await self.upload_attachment(path, _retried=True)
            raise AuthRequiredError(401, "Not authenticated -- run `tamfis-code login`")
        if response.status_code >= 400:
            raise RemoteAPIError(response.status_code, _error_detail(response))
        return _unwrap(response.json())

    # -- servers ----------------------------------------------------------

    async def list_servers(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/remote/servers")

    async def register_local_server(self, name: str) -> dict[str, Any]:
        return await self.request(
            "POST", "/remote/servers", json_body={"name": name, "transport_type": "local"}
        )

    # -- sessions -----------------------------------------------------

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/remote/sessions")

    async def create_session(self, server_id: int, working_directory: Optional[str] = None) -> dict[str, Any]:
        body: dict[str, Any] = {"server_id": server_id}
        if working_directory:
            body["working_directory"] = working_directory
        return await self.request("POST", "/remote/sessions", json_body=body)

    async def get_session(self, session_id: int) -> dict[str, Any]:
        return await self.request("GET", f"/remote/sessions/{session_id}")

    async def set_session_cwd(self, session_id: int, working_directory: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/sessions/{session_id}/cwd", json_body={"working_directory": working_directory}
        )

    async def expand_session_workspace(self, session_id: int, path: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/sessions/{session_id}/expand-workspace", json_body={"path": path}
        )

    async def get_thread(self, session_id: int, after_sequence: int = 0) -> dict[str, Any]:
        return await self.request(
            "GET", f"/remote/sessions/{session_id}/thread", params={"after_sequence": after_sequence}
        )

    # -- file mutation ledger (Phase 14: diff/rollback) ------------------

    async def list_file_mutations(self, session_id: int, limit: int = 50) -> dict[str, Any]:
        return await self.request(
            "GET", f"/remote/sessions/{session_id}/file-mutations", params={"limit": limit}
        )

    async def revert_file_mutation(self, session_id: int, mutation_id: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/sessions/{session_id}/file-mutations/{mutation_id}/revert"
        )

    # -- commands (explicit shell path) --------------------------------

    async def submit_command(self, session_id: int, command: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/sessions/{session_id}/commands", json_body={"command": command}
        )

    async def get_command(self, command_id: int) -> dict[str, Any]:
        return await self.request("GET", f"/remote/commands/{command_id}")

    async def approve_command(self, command_id: int, decision: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/commands/{command_id}/approve", json_body={"decision": decision}
        )

    async def cancel_command(self, command_id: int) -> dict[str, Any]:
        return await self.request("POST", f"/remote/commands/{command_id}/cancel")

    # -- persistent background PTY terminal --------------------------------

    async def start_pty(self, session_id: int, shell_command: str = "bash", cols: int = 80, rows: int = 24) -> dict[str, Any]:
        return await self.request(
            "POST", "/remote/pty/start",
            json_body={"session_id": session_id, "shell_command": shell_command, "cols": cols, "rows": rows},
        )

    async def approve_pty(self, pty_id: str) -> dict[str, Any]:
        return await self.request("POST", f"/remote/pty/{pty_id}/approve")

    async def deny_pty(self, pty_id: str) -> dict[str, Any]:
        return await self.request("POST", f"/remote/pty/{pty_id}/deny")

    async def write_pty(self, pty_id: str, data: str) -> dict[str, Any]:
        return await self.request("POST", f"/remote/pty/{pty_id}/write", json_body={"data": data})

    async def read_pty(self, pty_id: str, since: int = 0) -> dict[str, Any]:
        return await self.request("GET", f"/remote/pty/{pty_id}/read", params={"since": since})

    async def kill_pty(self, pty_id: str) -> dict[str, Any]:
        return await self.request("POST", f"/remote/pty/{pty_id}/kill")

    async def list_pty(self, session_id: int) -> list[dict[str, Any]]:
        return await self.request("GET", f"/remote/sessions/{session_id}/pty")

    # -- AI tasks ---------------------------------------------------------

    async def run_ai_task(
        self, session_id: int, objective: str, mode: str = "coding", task_id: Optional[str] = None,
        model: str = "auto", provider: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        # Identify CLI-originated work so the shared Remote service can
        # apply tamfis-code preferences without changing web/chat routing.
        body: dict[str, Any] = {
            "objective": objective,
            "mode": mode,
            "source_client": "tamfis-code",
            "model": model or "auto",
            "attachments": attachments or [],
        }
        if provider:
            body["provider"] = provider
        if task_id:
            body["task_id"] = task_id
        return await self.request("POST", f"/remote/sessions/{session_id}/ai-tasks", json_body=body)

    async def list_models(self, provider: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"per_page": 200}
        if provider:
            params["provider"] = provider
        return await self.request("GET", "/models", params=params)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return await self.request("GET", f"/remote/tasks/{task_id}")

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        return await self.request("POST", f"/remote/tasks/{task_id}/cancel")

    async def retry_task(self, task_id: str, mode: Optional[str] = None) -> dict[str, Any]:
        body = {"mode": mode} if mode else {}
        return await self.request("POST", f"/remote/tasks/{task_id}/retry", json_body=body)

    async def add_task_instruction(self, task_id: str, text: str, classification: str = "append") -> dict[str, Any]:
        return await self.request(
            "POST", f"/remote/tasks/{task_id}/instructions",
            json_body={"text": text, "classification": classification},
        )

    # -- streaming ---------------------------------------------------------

    async def stream_session(self, session_id: int, last_event_id: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Yields parsed event envelopes from GET /sessions/{id}/stream.

        Mirrors remoteStore.ts's connectSessionStream: an authenticated raw
        fetch/stream (not EventSource, which cannot send an Authorization
        header), reconnecting the caller's responsibility -- this generator
        ends when the server closes the stream or on a connection error, and
        does not retry itself, so `interactive.py`/non-interactive callers
        control reconnect-with-last-event-id explicitly."""
        url = f"{_api_root(self.config.api_base)}/remote/sessions/{session_id}/stream"
        params = {"last_event_id": max(0, last_event_id)}
        async with self._client.stream(
            "GET", url, params=params, headers=self._headers(), timeout=None
        ) as resp:
            if resp.status_code == 401:
                if await self._refresh():
                    async for event in self.stream_session(session_id, last_event_id):
                        yield event
                    return
                raise AuthRequiredError(401, "Not authenticated -- run `tamfis-code login`")
            if resp.status_code >= 400:
                body = await resp.aread()
                raise RemoteAPIError(resp.status_code, body.decode("utf-8", errors="replace"))

            event_type = None
            event_id = None
            data_lines: list[str] = []
            async for raw_line in resp.aiter_lines():
                line = raw_line.rstrip("\n")
                if line == "":
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        data_lines = []
                        if raw_data.strip() and raw_data.strip() != "[DONE]":
                            try:
                                payload = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                            if event_type and "event_type" not in payload:
                                payload["event_type"] = event_type
                            # Only the SSE `id:` field is a valid replay
                            # cursor.  Canonical payloads also have a
                            # `sequence`, but that number belongs to a
                            # different/global event domain and must never
                            # advance a session stream checkpoint.
                            if event_id is not None:
                                try:
                                    payload["stream_sequence"] = int(event_id)
                                    payload.setdefault("sequence", int(event_id))
                                except ValueError:
                                    payload["stream_event_id"] = event_id
                            yield payload
                    event_type = None
                    event_id = None
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("id:"):
                    event_id = line[len("id:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload and set(payload.keys()) <= {"data", "success", "message"}:
        return payload["data"]
    return payload


def _error_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return resp.text or f"HTTP {resp.status_code}"
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or data)
