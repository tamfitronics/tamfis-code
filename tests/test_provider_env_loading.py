import os
from pathlib import Path

from tamfis_code.providers import _load_project_env, _provider_env_candidates


def test_installed_package_falls_back_to_canonical_deployment_env(monkeypatch, tmp_path):
    package_root = tmp_path / "site-packages"
    module_file = package_root / "tamfis_code" / "providers.py"
    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    (deployment_root / ".env").write_text(
        "NVIDIA_API_KEY=deployment-provider-key\n", encoding="utf-8",
    )
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_ENV_FILE", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_ROOT", raising=False)

    _load_project_env(module_file=module_file, deployment_root=deployment_root)

    assert os.environ.get("NVIDIA_API_KEY") == "deployment-provider-key"


def test_source_checkout_env_wins_over_deployment_fallback(monkeypatch, tmp_path):
    source_root = tmp_path / "checkout"
    module_file = source_root / "tamfis_code" / "providers.py"
    source_root.mkdir()
    (source_root / ".env").write_text("HF_API_KEY=source-provider-key\n", encoding="utf-8")
    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    (deployment_root / ".env").write_text("HF_API_KEY=deployment-provider-key\n", encoding="utf-8")
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_ENV_FILE", raising=False)
    monkeypatch.delenv("TAMFIS_CODE_ROOT", raising=False)

    _load_project_env(module_file=module_file, deployment_root=deployment_root)

    assert os.environ.get("HF_API_KEY") == "source-provider-key"


def test_explicit_env_file_does_not_fall_through_to_canonical(monkeypatch, tmp_path):
    missing = tmp_path / "missing.env"
    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    (deployment_root / ".env").write_text("GROK_API_KEY=must-not-load\n", encoding="utf-8")
    monkeypatch.setenv("TAMFIS_CODE_ENV_FILE", str(missing))
    monkeypatch.delenv("TAMFIS_CODE_ROOT", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    candidates = _provider_env_candidates(
        module_file=tmp_path / "site-packages" / "tamfis_code" / "providers.py",
        deployment_root=deployment_root,
    )
    _load_project_env(
        module_file=tmp_path / "site-packages" / "tamfis_code" / "providers.py",
        deployment_root=deployment_root,
    )

    assert candidates == (missing,)
    assert os.environ.get("GROK_API_KEY") is None
