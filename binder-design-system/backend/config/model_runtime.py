import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRuntimeSettings:
    database_url: str | None
    foundry_mode: str
    allow_mock: bool
    foundry_env: str
    model_timeout_seconds: int
    pipeline_mpnn_slots: int
    pipeline_rf3_slots: int
    pipeline_max_rf3_candidates: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_settings() -> ModelRuntimeSettings:
    env = os.getenv("DEEPBINDER_ENV", "development").strip().lower()
    default_allow_mock = env not in {"prod", "production"}
    database_url = os.getenv("DEEPBINDER_DATABASE_URL")
    if not database_url:
        raise RuntimeError("必须设置 PostgreSQL 数据库连接：DEEPBINDER_DATABASE_URL")
    if not (
        database_url.startswith("postgresql://")
        or database_url.startswith("postgresql+psycopg://")
        or database_url.startswith("postgresql+psycopg2://")
    ):
        raise RuntimeError("DEEPBINDER_DATABASE_URL 必须使用 PostgreSQL，旧 SQLite 架构已停用")
    return ModelRuntimeSettings(
        database_url=database_url,
        foundry_mode=os.getenv("DEEPBINDER_FOUNDRY_MODE", "auto").strip().lower(),
        allow_mock=_env_bool("DEEPBINDER_ALLOW_MOCK", default_allow_mock),
        foundry_env=os.getenv("DEEPBINDER_FOUNDRY_ENV", "deepbinder"),
        model_timeout_seconds=int(os.getenv("DEEPBINDER_MODEL_TIMEOUT_SECONDS", "1800")),
        pipeline_mpnn_slots=max(1, int(os.getenv("DEEPBINDER_PIPELINE_MPNN_SLOTS", "1"))),
        pipeline_rf3_slots=max(1, int(os.getenv("DEEPBINDER_PIPELINE_RF3_SLOTS", "1"))),
        pipeline_max_rf3_candidates=max(1, int(os.getenv("DEEPBINDER_PIPELINE_MAX_RF3_CANDIDATES", "10"))),
    )
