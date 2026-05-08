from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    backend_root: Path
    app_root: Path
    repo_root: Path
    foundry_root: Path
    hotspot_dl_root: Path
    ppihotspotid_root: Path
    output_root: Path


def discover_paths() -> AppPaths:
    backend_root = Path(__file__).resolve().parents[1]
    app_root = backend_root.parent
    repo_root = app_root.parent
    return AppPaths(
        backend_root=backend_root,
        app_root=app_root,
        repo_root=repo_root,
        foundry_root=repo_root / "foundry-production",
        hotspot_dl_root=repo_root / "hotspot-prediction",
        ppihotspotid_root=repo_root / "ppihotspotid-main",
        output_root=app_root / "outputs",
    )
