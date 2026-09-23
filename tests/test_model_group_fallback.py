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


def test_nim_pool_puts_kimi_k3_and_glm_5_3_last_and_selection_follows_it():
    """Owner ruling 2026-09-19: Kimi K3 and GLM 5.3 caused the latency (neither
    answered inside 60s on NIM's free tier); the fast nemotrons lead and the two
    are reached only when nothing else survives the capability filters."""
    from tamfis_code.providers import ProviderManager

    real = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
    assert real.models[-2:] == ["moonshotai/kimi-k3", "z-ai/glm-5.3"]
    assert real.default_model not in {"moonshotai/kimi-k3", "z-ai/glm-5.3"}
    assert real.models.index(real.default_model) < real.models.index("moonshotai/kimi-k3")

    picked = _select_public_group_model(
        _Manager(), ProviderType.NVIDIA, real, None, PUBLIC_MODEL_ULTIMA,
    )
    assert picked not in {"moonshotai/kimi-k3", "z-ai/glm-5.3"}


def test_nim_public_group_still_serves_kimi_k3_when_it_is_the_only_survivor():
    manager = _Manager()
    nim = _config("moonshotai/kimi-k3", default="moonshotai/kimi-k3")

    assert _select_public_group_model(
        manager, ProviderType.NVIDIA, nim, None, PUBLIC_MODEL_ULTIMA,
    ) == "moonshotai/kimi-k3"


def test_fallback_diagnostic_names_model_groups_not_generic_provider():
    message = _public_model_fallback_message(
        "tamfis-gpt-pro", "tamfis-gpt-ultima", "HTTP 429",
    )

    assert message == (
        "finitron-pro unavailable for this turn (HTTP 429); "
        "falling back to finitron-ultima."
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
