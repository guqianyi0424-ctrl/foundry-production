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


def test_default_hotspot_weight_directory_is_final_neg3(monkeypatch):
    monkeypatch.delenv("HOTSPOT_DL_DIR", raising=False)

    from utils.hotspot_predictor import HotspotPredictor

    predictor = HotspotPredictor(top_k=3)

    assert predictor.models_dir == predictor.hotspot_dl_path / "models" / "final_neg3"


def test_runtime_settings_read_environment(monkeypatch):
    monkeypatch.setenv("DEEPBINDER_FOUNDRY_MODE", "source")
    monkeypatch.setenv("DEEPBINDER_ALLOW_MOCK", "0")
    monkeypatch.setenv("DEEPBINDER_FOUNDRY_ENV", "foundry-prod")
    monkeypatch.setenv("DEEPBINDER_MODEL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("DEEPBINDER_PIPELINE_MPNN_SLOTS", "2")
    monkeypatch.setenv("DEEPBINDER_PIPELINE_RF3_SLOTS", "3")
    monkeypatch.setenv("DEEPBINDER_PIPELINE_MAX_RF3_CANDIDATES", "7")
    monkeypatch.setenv("DEEPBINDER_DATABASE_URL", "postgresql+psycopg://deepbinder:secret@db/deepbinder")

    from config.settings import build_settings

    settings = build_settings()

    assert settings.runtime.foundry_mode == "source"
    assert settings.runtime.allow_mock is False
    assert settings.runtime.foundry_env == "foundry-prod"
    assert settings.runtime.model_timeout_seconds == 45
    assert settings.runtime.pipeline_mpnn_slots == 2
    assert settings.runtime.pipeline_rf3_slots == 3
    assert settings.runtime.pipeline_max_rf3_candidates == 7
    assert settings.runtime.database_url == "postgresql+psycopg://deepbinder:secret@db/deepbinder"


def test_runtime_settings_require_postgresql_database_url(monkeypatch):
    monkeypatch.delenv("DEEPBINDER_DATABASE_URL", raising=False)

    from config.settings import build_settings

    try:
        build_settings()
    except RuntimeError as exc:
        assert "DEEPBINDER_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("missing PostgreSQL database URL should fail startup")


def test_runtime_settings_reject_sqlite_database_url(monkeypatch):
    monkeypatch.setenv("DEEPBINDER_DATABASE_URL", "sqlite:///backend/data/deepbinder.db")

    from config.settings import build_settings

    try:
        build_settings()
    except RuntimeError as exc:
        assert "PostgreSQL" in str(exc)
    else:
        raise AssertionError("SQLite database URL should fail startup")


def test_runtime_settings_disable_mock_by_default_in_production(monkeypatch):
    monkeypatch.setenv("DEEPBINDER_ENV", "production")
    monkeypatch.delenv("DEEPBINDER_ALLOW_MOCK", raising=False)

    from config.settings import build_settings

    settings = build_settings()

    assert settings.runtime.allow_mock is False


def test_runtime_settings_keep_mock_enabled_by_default_in_development(monkeypatch):
    monkeypatch.delenv("DEEPBINDER_ENV", raising=False)
    monkeypatch.delenv("DEEPBINDER_ALLOW_MOCK", raising=False)

    from config.settings import build_settings

    settings = build_settings()

    assert settings.runtime.allow_mock is True
