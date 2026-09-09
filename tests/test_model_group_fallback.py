from types import SimpleNamespace

from tamfis_code.providers import ProviderType
from tamfis_code.public_identity import PUBLIC_MODEL_ULTIMA
from tamfis_code.runner_local import (
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


def test_public_group_selection_skips_a_provider_without_requested_tier():
    manager = _Manager()
    smart_only = _config("tamfis-gpt-smart", default="tamfis-gpt-smart")
    ultima = _config("tamfis-gpt-pro", "tamfis-gpt-ultima", default="tamfis-gpt-pro")

    assert _select_public_group_model(
        manager, ProviderType.HF, smart_only, None, PUBLIC_MODEL_ULTIMA,
    ) is None
    assert _select_public_group_model(
        manager, ProviderType.TAMFIS, ultima, None, PUBLIC_MODEL_ULTIMA,
    ) == "tamfis-gpt-ultima"


def test_fallback_diagnostic_names_model_groups_not_generic_provider():
    message = _public_model_fallback_message(
        "tamfis-gpt-pro", "tamfis-gpt-ultima", "HTTP 429",
    )

    assert message == (
        "TamfisGPT-Pro unavailable for this turn (HTTP 429); "
        "falling back to TamfisGPT-Ultima."
    )
    assert "Provider TamfisGPT" not in message
