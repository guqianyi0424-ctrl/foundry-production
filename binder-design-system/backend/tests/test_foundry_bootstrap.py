def test_bootstrap_returns_mock_mode_when_mock_forced(monkeypatch):
    from adapters.foundry_bootstrap import FoundryBootstrap
    from config.settings import build_settings

    monkeypatch.setenv("DEEPBINDER_FOUNDRY_MODE", "mock")
    monkeypatch.setenv("DEEPBINDER_ALLOW_MOCK", "1")

    settings = build_settings()
    bootstrap = FoundryBootstrap(settings)

    result = bootstrap.ensure_available()

    assert result.available is False
    assert result.mode == "mock"
    assert result.mock_allowed is True


def test_bootstrap_rejects_mock_when_disabled(monkeypatch):
    from adapters.foundry_bootstrap import FoundryBootstrap
    from config.settings import build_settings

    monkeypatch.setenv("DEEPBINDER_FOUNDRY_MODE", "mock")
    monkeypatch.setenv("DEEPBINDER_ALLOW_MOCK", "0")

    settings = build_settings()
    bootstrap = FoundryBootstrap(settings)

    result = bootstrap.ensure_available()

    assert result.available is False
    assert result.mode == "unavailable"
    assert result.mock_allowed is False
    assert "mock disabled" in result.message
