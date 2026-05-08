from pathlib import Path


def test_settings_exposes_project_paths():
    from config.settings import get_settings

    settings = get_settings()

    assert settings.paths.backend_root.name == "backend"
    assert settings.paths.app_root.name == "binder-design-system"
    assert settings.paths.repo_root.exists()
    assert settings.paths.foundry_root == settings.paths.repo_root / "foundry-production"
    assert settings.paths.hotspot_dl_root == settings.paths.repo_root / "hotspot-prediction"
    assert settings.paths.ppihotspotid_root == settings.paths.repo_root / "ppihotspotid-main"
    assert settings.paths.output_root == settings.paths.app_root / "outputs"
    assert isinstance(settings.paths.output_root, Path)


def test_runtime_settings_read_environment(monkeypatch):
    monkeypatch.setenv("DEEPBINDER_FOUNDRY_MODE", "source")
    monkeypatch.setenv("DEEPBINDER_ALLOW_MOCK", "0")
    monkeypatch.setenv("DEEPBINDER_FOUNDRY_ENV", "foundry-prod")
    monkeypatch.setenv("DEEPBINDER_MODEL_TIMEOUT_SECONDS", "45")

    from config.settings import build_settings

    settings = build_settings()

    assert settings.runtime.foundry_mode == "source"
    assert settings.runtime.allow_mock is False
    assert settings.runtime.foundry_env == "foundry-prod"
    assert settings.runtime.model_timeout_seconds == 45
