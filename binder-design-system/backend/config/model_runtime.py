import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRuntimeSettings:
    foundry_mode: str
    allow_mock: bool
    foundry_env: str
    model_timeout_seconds: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_settings() -> ModelRuntimeSettings:
    return ModelRuntimeSettings(
        foundry_mode=os.getenv("ODESIGN_FOUNDRY_MODE", "auto").strip().lower(),
        allow_mock=_env_bool("ODESIGN_ALLOW_MOCK", True),
        foundry_env=os.getenv("ODESIGN_FOUNDRY_ENV", "foundry"),
        model_timeout_seconds=int(os.getenv("ODESIGN_MODEL_TIMEOUT_SECONDS", "1800")),
    )
