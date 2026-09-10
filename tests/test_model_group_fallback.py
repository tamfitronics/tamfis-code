from types import SimpleNamespace

from tamfis_code.providers import ProviderType
from tamfis_code.public_identity import PUBLIC_MODEL_SMART, PUBLIC_MODEL_ULTIMA
from tamfis_code.runner_local import (
    _fresh_fallback_route,
    _public_model_fallback_message,
    _select_public_group_model,
)


class _Manager:
    def route_is_healthy(self, _provider, model):
        return model != "tamfis-gpt-ultima-unhealthy"

    def select_model(self, config, _task_profile):
        return config.default_model


def _config(*models, default):
    return SimpleNamespace(
        models=list(models),
        default_model=default,
        free_model=None,
        vision_supported=False,
        vision_models=[],
    )


def test_public_group_selection_allows_lower_tier_models_under_the_ceiling():
    # Cumulative ceiling, not an exact-tier match: a provider whose only
    # model sits below the requested group must still be selected, mirroring
    # TamfisGPT Remote's "Ultima includes every lower group" policy.
    manager = _Manager()
    smart_only = _config("tamfis-gpt-smart", default="tamfis-gpt-smart")

    assert _select_public_group_model(
        manager, ProviderType.HF, smart_only, None, PUBLIC_MODEL_ULTIMA,
    ) == "tamfis-gpt-smart"


def test_public_group_selection_skips_a_provider_above_the_requested_ceiling():
    manager = _Manager()
    ultima_only = _config("tamfis-gpt-ultima", default="tamfis-gpt-ultima")

    assert _select_public_group_model(
        manager, ProviderType.HF, ultima_only, None, PUBLIC_MODEL_SMART,
    ) is None


def test_public_group_selection_picks_among_eligible_models_under_ceiling():
    manager = _Manager()
    ultima = _config("tamfis-gpt-pro", "tamfis-gpt-ultima", default="tamfis-gpt-pro")

    result = _select_public_group_model(
        manager, ProviderType.TAMFIS, ultima, None, PUBLIC_MODEL_ULTIMA,
    )
    assert result in {"tamfis-gpt-pro", "tamfis-gpt-ultima"}


def test_nim_public_group_prefers_kimi_k3_after_capability_filters():
    manager = _Manager()
    nim = _config(
        "nvidia/nemotron-3-ultra-550b-a55b",
        "moonshotai/kimi-k3",
        default="nvidia/nemotron-3-ultra-550b-a55b",
    )

    assert _select_public_group_model(
        manager, ProviderType.NVIDIA, nim, None, PUBLIC_MODEL_ULTIMA,
    ) == "moonshotai/kimi-k3"


def test_fallback_diagnostic_names_model_groups_not_generic_provider():
    message = _public_model_fallback_message(
        "tamfis-gpt-pro", "tamfis-gpt-ultima", "HTTP 429",
    )

    assert message == (
        "TamfisGPT-Pro unavailable for this turn (HTTP 429); "
        "falling back to TamfisGPT-Ultima."
    )
    assert "Provider TamfisGPT" not in message


def test_fresh_fallback_route_rechecks_current_provider_health_snapshot():
    recovered_client = object()
    recovered_config = _config("tamfis-gpt-pro", default="tamfis-gpt-pro")

    class RecoveryManager(_Manager):
        PROVIDERS = {ProviderType.GROK: recovered_config}

        def fallback_candidates(
            self, current, task_profile, *, allow_premium_primary=False,
        ):
            assert current == ProviderType.OPENROUTER
            assert allow_premium_primary is True
            return [ProviderType.GROK]

        def get_client(self, provider):
            return recovered_client if provider == ProviderType.GROK else None

    route = _fresh_fallback_route(
        RecoveryManager(),
        ProviderType.OPENROUTER,
        None,
        PUBLIC_MODEL_ULTIMA,
        allow_premium_primary=True,
    )

    assert route is not None
    provider, config, client, model = route
    assert provider == ProviderType.GROK
    assert config is recovered_config
    assert client is recovered_client
    assert model == "tamfis-gpt-pro"
