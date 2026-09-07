from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tamfis_code import providers
from tamfis_code.providers import ProviderManager, ProviderType


class QuotaError(RuntimeError):
    status_code = 429


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=response, side_effect=error),
            ),
        )


def test_numbered_nim_key_is_valid_without_primary_key(monkeypatch):
    for name in providers._NIM_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY_2", "nvapi-numbered-valid-key")
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"

    assert manager._has_valid_api_key(ProviderType.NVIDIA)
    assert manager._get_api_key(ProviderType.NVIDIA) == "nvapi-numbered-valid-key"


def test_numbered_nim_key_initializes_the_client_pool(monkeypatch):
    for name in providers._NIM_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY_3", "nvapi-numbered-client-key")
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    manager.clients = {}
    manager.config = {
        provider.value: provider == ProviderType.NVIDIA
        for provider in ProviderType
    }
    manager._nim_client_pool = []
    manager._nim_key_index = 0
    fake_client = object()
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **_kwargs: fake_client)

    manager._init_clients()

    assert manager._nim_client_pool == [fake_client]
    assert manager.clients[ProviderType.NVIDIA] is fake_client


def test_nim_key_collection_deduplicates_and_rejects_placeholders(monkeypatch):
    for name in providers._NIM_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-real-key-value")
    monkeypatch.setenv("NVIDIA_API_KEY_1", "nvapi-real-key-value")
    monkeypatch.setenv("NVIDIA_API_KEY_2", "YOUR_NVIDIA_API_KEY")

    assert providers._nim_configured_keys() == ["nvapi-real-key-value"]


@pytest.mark.asyncio
async def test_quota_rotates_key_before_recording_route_failure(monkeypatch):
    ProviderManager.reset_runtime_routing_state()
    config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    failed = _FakeClient(error=QuotaError("weekly usage limit reached"))
    succeeded = _FakeClient(response=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))],
        usage=None,
    ))
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    manager.config = {ProviderType.NVIDIA.value: True}
    manager._nim_client_pool = [failed, succeeded]
    manager._nim_key_index = 0
    manager.clients = {ProviderType.NVIDIA: failed}
    monkeypatch.setattr(
        manager, "resolve_route", lambda *_args, **_kwargs: (ProviderType.NVIDIA, config),
    )
    monkeypatch.setattr(manager, "select_model", lambda *_args, **_kwargs: config.default_model)

    chunks = [chunk async for chunk in manager.chat_completion(
        ProviderType.NVIDIA, [{"role": "user", "content": "hello"}], stream=False,
    )]

    assert chunks == ["recovered"]
    assert manager.clients[ProviderType.NVIDIA] is succeeded
    telemetry = manager.routing_telemetry()
    assert telemetry.provider_requests[ProviderType.NVIDIA.value] == 2
    assert telemetry.provider_successes[ProviderType.NVIDIA.value] == 1
    assert telemetry.provider_failures.get(ProviderType.NVIDIA.value, 0) == 0
    ProviderManager.reset_runtime_routing_state()


@pytest.mark.asyncio
async def test_all_nim_keys_exhaust_once_then_record_one_route_failure(monkeypatch):
    ProviderManager.reset_runtime_routing_state()
    config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    first = _FakeClient(error=QuotaError("first account quota"))
    second = _FakeClient(error=QuotaError("second account quota"))
    manager = ProviderManager.__new__(ProviderManager)
    manager.runtime_mode = "standalone"
    manager.config = {ProviderType.NVIDIA.value: True}
    manager._nim_client_pool = [first, second]
    manager._nim_key_index = 0
    manager.clients = {ProviderType.NVIDIA: first}
    monkeypatch.setattr(
        manager,
        "resolve_route",
        lambda *_args, **_kwargs: (ProviderType.NVIDIA, config),
    )
    monkeypatch.setattr(
        manager, "select_model", lambda *_args, **_kwargs: config.default_model,
    )
    monkeypatch.setattr(manager, "fallback_candidates", lambda *_args, **_kwargs: [])

    with pytest.raises(QuotaError, match="second account quota"):
        _ = [chunk async for chunk in manager.chat_completion(
            ProviderType.NVIDIA,
            [{"role": "user", "content": "hello"}],
            stream=False,
        )]

    assert first.chat.completions.create.await_count == 1
    assert second.chat.completions.create.await_count == 1
    telemetry = manager.routing_telemetry()
    assert telemetry.provider_requests[ProviderType.NVIDIA.value] == 2
    assert telemetry.provider_failures[ProviderType.NVIDIA.value] == 1
    ProviderManager.reset_runtime_routing_state()
