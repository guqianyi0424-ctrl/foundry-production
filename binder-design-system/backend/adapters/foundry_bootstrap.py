import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from config.settings import AppSettings


@dataclass(frozen=True)
class FoundryBootstrapResult:
    available: bool
    mode: str
    mock_allowed: bool
    message: str


class FoundryBootstrap:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def ensure_available(self) -> FoundryBootstrapResult:
        runtime = self.settings.runtime
        if runtime.foundry_mode == "mock":
            if runtime.allow_mock:
                return FoundryBootstrapResult(False, "mock", True, "mock mode forced")
            return FoundryBootstrapResult(False, "unavailable", False, "mock disabled")

        installed = self._can_import_models()
        if installed:
            return FoundryBootstrapResult(
                True,
                "installed",
                runtime.allow_mock,
                "installed imports available",
            )

        if runtime.foundry_mode in {"auto", "source"}:
            self._add_source_paths(self.settings.paths.foundry_root)
            if self._can_import_models():
                return FoundryBootstrapResult(
                    True,
                    "source",
                    runtime.allow_mock,
                    "vendored source imports available",
                )

        if runtime.allow_mock:
            return FoundryBootstrapResult(
                False,
                "mock",
                True,
                "foundry imports unavailable; mock enabled",
            )
        return FoundryBootstrapResult(
            False,
            "unavailable",
            False,
            "foundry imports unavailable and mock disabled",
        )

    def _can_import_models(self) -> bool:
        try:
            importlib.import_module("rfd3")
            importlib.import_module("mpnn")
            importlib.import_module("rf3")
            return True
        except Exception:
            return False

    def _add_source_paths(self, foundry_root: Path) -> None:
        for relative in ("models/rfd3/src", "models/mpnn/src", "models/rf3/src", "src"):
            path = str(foundry_root / relative)
            if path not in sys.path:
                sys.path.insert(0, path)
