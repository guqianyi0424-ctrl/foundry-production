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
