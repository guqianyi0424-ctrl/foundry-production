import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT.parent
REPO_ROOT = APP_ROOT.parent


def pytest_configure():
    for path in (str(BACKEND_ROOT), str(APP_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("ODESIGN_ALLOW_MOCK", "1")
    os.environ.setdefault("ODESIGN_FOUNDRY_MODE", "mock")
