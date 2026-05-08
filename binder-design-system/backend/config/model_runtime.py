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
        foundry_mode=os.getenv("DEEPBINDER_FOUNDRY_MODE", "auto").strip().lower(),
        allow_mock=_env_bool("DEEPBINDER_ALLOW_MOCK", True),
        foundry_env=os.getenv("DEEPBINDER_FOUNDRY_ENV", "deepbinder"),
        model_timeout_seconds=int(os.getenv("DEEPBINDER_MODEL_TIMEOUT_SECONDS", "1800")),
    )
