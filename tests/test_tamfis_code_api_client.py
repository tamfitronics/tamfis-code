import asyncio
import unittest
from unittest.mock import patch

import httpx

from tamfis_code.api_client import (
    AuthRequiredError, RemoteAPIClient, RemoteAPIError, _error_detail, _unwrap,
    load_secure_credentials, save_secure_credentials,
)
from tamfis_code.config import Config, Credentials


def _client_with_transport(transport: httpx.MockTransport, creds: Credentials | None = None) -> RemoteAPIClient:
    config = Config(api_base="http://test.invalid")
    client = RemoteAPIClient(config, credentials=creds)
    client._client = httpx.AsyncClient(transport=transport)  # swap in the mock transport
    return client


class UnwrapTests(unittest.TestCase):
    def test_unwraps_data_envelope(self):
        self.assertEqual(_unwrap({"data": {"id": 1}, "success": True}), {"id": 1})

    def test_passes_through_flat_dict_with_extra_keys(self):
        payload = {"access_token": "x", "user": {}}
        self.assertEqual(_unwrap(payload), payload)

    def test_passes_through_list(self):
        self.assertEqual(_unwrap([1, 2, 3]), [1, 2, 3])


class RemoteAPIClientRequestTests(unittest.TestCase):
    def test_get_unwraps_data_envelope(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"id": 1}], "success": True})

        client = _client_with_transport(httpx.MockTransport(handler), Credentials(access_token="tok"))
        result = asyncio.run(client.request("GET", "/remote/servers"))
        asyncio.run(client.aclose())
        self.assertEqual(result, [{"id": 1}])

    def test_401_triggers_refresh_then_retries_successfully(self):
        calls = {"servers": 0}

        def handler(request):
            if request.url.path.endswith("/servers"):
                calls["servers"] += 1
                if calls["servers"] == 1:
                    return httpx.Response(401, json={"detail": "expired"})
                return httpx.Response(200, json={"data": []})
            if request.url.path.endswith("/auth/refresh"):
                return httpx.Response(200, json={"access_token": "new-token", "refresh_token": "new-refresh"})
            return httpx.Response(404)

        creds = Credentials(access_token="old-token", refresh_token="old-refresh")
        client = _client_with_transport(httpx.MockTransport(handler), creds)
        # Refresh persists rotated credentials in production. Never let this
        # unit test overwrite the developer's real ~/.config credentials.
        with patch("tamfis_code.api_client.save_secure_credentials"):
            result = asyncio.run(client.request("GET", "/remote/servers"))
        asyncio.run(client.aclose())

        self.assertEqual(result, [])
        self.assertEqual(calls["servers"], 2)
        self.assertEqual(client.credentials.access_token, "new-token")

    def test_refresh_sends_refresh_token_as_cookie_not_json_body(self):
        # Regression guard: /auth/refresh reads the TAMFIS_REFRESH cookie
        # server-side, not a JSON field -- see the comment in
        # api_client.py's _refresh(). A JSON body alone is silently
        # ignored by the real endpoint.
        seen = {}

        def handler(request):
            if request.url.path.endswith("/auth/refresh"):
                seen["cookie"] = request.headers.get("cookie", "")
                return httpx.Response(200, json={"access_token": "new-token"})
            return httpx.Response(401, json={"detail": "expired"})

        creds = Credentials(access_token="old", refresh_token="the-refresh-token")
        client = _client_with_transport(httpx.MockTransport(handler), creds)
        with patch("tamfis_code.api_client.save_secure_credentials"):
            with self.assertRaises(RemoteAPIError):
                asyncio.run(client.request("GET", "/remote/servers"))
        asyncio.run(client.aclose())

        self.assertIn("TAMFIS_REFRESH=the-refresh-token", seen.get("cookie", ""))

    def test_401_with_no_refresh_token_raises_auth_required(self):
        def handler(request):
            return httpx.Response(401, json={"detail": "expired"})

        client = _client_with_transport(httpx.MockTransport(handler), Credentials(access_token="tok"))
        with self.assertRaises(AuthRequiredError):
            asyncio.run(client.request("GET", "/remote/servers"))
        asyncio.run(client.aclose())

    def test_error_detail_extracted_from_validation_error_list(self):
        def handler(request):
            return httpx.Response(422, json={"detail": [{"msg": "field required"}]})

        client = _client_with_transport(httpx.MockTransport(handler), Credentials(access_token="tok"))
        with self.assertRaises(RemoteAPIError) as ctx:
            asyncio.run(client.request("GET", "/remote/servers"))
        asyncio.run(client.aclose())
        self.assertIn("field required", str(ctx.exception))

    def test_login_returns_flat_token_response_unmodified(self):
        def handler(request):
            return httpx.Response(200, json={"access_token": "tok", "refresh_token": "ref", "user": {"email": "a@b.com"}})

        client = _client_with_transport(httpx.MockTransport(handler))
        result = asyncio.run(client.login("a@b.com", "pw"))
        asyncio.run(client.aclose())
        self.assertEqual(result["access_token"], "tok")
        self.assertEqual(result["user"]["email"], "a@b.com")

    def test_login_failure_raises_with_detail(self):
        def handler(request):
            return httpx.Response(401, json={"detail": "Invalid email or password"})

        client = _client_with_transport(httpx.MockTransport(handler))
        with self.assertRaises(RemoteAPIError) as ctx:
            asyncio.run(client.login("a@b.com", "wrong"))
        asyncio.run(client.aclose())
        self.assertIn("Invalid email or password", str(ctx.exception))

    def test_me_and_logout_use_existing_auth_contracts(self):
        seen = []

        def handler(request):
            seen.append((request.method, request.url.path))
            if request.url.path.endswith("/auth/me"):
                return httpx.Response(200, json={"authenticated": True, "user": {"email": "a@b.com"}})
            return httpx.Response(200, json={"ok": True})

        client = _client_with_transport(httpx.MockTransport(handler), Credentials(access_token="tok"))
        self.assertTrue(asyncio.run(client.me())["authenticated"])
        self.assertTrue(asyncio.run(client.logout())["ok"])
        asyncio.run(client.aclose())
        self.assertIn(("GET", "/api/v1/auth/me"), seen)
        self.assertIn(("POST", "/api/v1/auth/logout"), seen)


class SecureCredentialStorageTests(unittest.TestCase):
    def test_system_keyring_is_preferred_and_roundtrips(self):
        class FakeKeyring:
            value = None

            @classmethod
            def set_password(cls, service, account, value):
                cls.value = value

            @classmethod
            def get_password(cls, service, account):
                return cls.value

            @classmethod
            def delete_password(cls, service, account):
                cls.value = None

        with patch("tamfis_code.api_client._keyring_module", return_value=FakeKeyring), patch(
            "tamfis_code.api_client._clear_file_credentials", return_value=False
        ):
            backend = save_secure_credentials(Credentials(access_token="secret", refresh_token="refresh", email="a@b.com"))
            loaded = load_secure_credentials()
        self.assertEqual(backend, "system-keyring")
        self.assertEqual(loaded.access_token, "secret")
        self.assertEqual(loaded.refresh_token, "refresh")


if __name__ == "__main__":
    unittest.main()
