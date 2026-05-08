from dataclasses import dataclass
from functools import lru_cache

from config.model_runtime import ModelRuntimeSettings, load_runtime_settings
from config.paths import AppPaths, discover_paths


@dataclass(frozen=True)
class AppSettings:
    paths: AppPaths
    runtime: ModelRuntimeSettings


def build_settings() -> AppSettings:
    return AppSettings(
        paths=discover_paths(),
        runtime=load_runtime_settings(),
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return build_settings()
