from unittest.mock import patch

from tamfis_code.config import Credentials
from tamfis_code.providers import ProviderManager, ProviderType


def _manager_without_initialising_clients() -> ProviderManager:
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    return manager


def test_tamfis_provider_prefers_explicit_developer_key(monkeypatch):
    monkeypatch.setenv("TAMFIS_API_KEY", "tamfis_sk_live_explicit_developer_key")
    manager = _manager_without_initialising_clients()
    with patch("tamfis_code.api_client.load_secure_credentials") as credentials:
        assert manager._get_api_key(ProviderType.TAMFIS) == "tamfis_sk_live_explicit_developer_key"
    credentials.assert_not_called()


def test_tamfis_provider_uses_login_credential_without_remote_runtime(monkeypatch):
    monkeypatch.delenv("TAMFIS_API_KEY", raising=False)
    manager = _manager_without_initialising_clients()
    with patch(
        "tamfis_code.api_client.load_secure_credentials",
        return_value=Credentials(access_token="subscription-jwt"),
    ):
        assert manager._get_api_key(ProviderType.TAMFIS) == "subscription-jwt"


def test_tamfis_provider_is_unavailable_without_login_or_developer_key(monkeypatch):
    monkeypatch.delenv("TAMFIS_API_KEY", raising=False)
    manager = _manager_without_initialising_clients()
    with patch("tamfis_code.api_client.load_secure_credentials", return_value=None):
        assert manager._get_api_key(ProviderType.TAMFIS) is None
