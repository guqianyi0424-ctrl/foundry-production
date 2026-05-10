"""Shared configuration defaults for hotspot prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class HotspotConfig:
    project_root: Path = PROJECT_ROOT
    data1_file: Path = PROJECT_ROOT / "elife-96643-data1-v1.xlsx"
    data2_file: Path = PROJECT_ROOT / "elife-96643-data2-v1.xlsx"
    processed_dir: Path = PROJECT_ROOT / "processed"
    pdb_dir: Path = PROJECT_ROOT / "data" / "pdb_files"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    esm_dim: int = 1280
    pssm_dim: int = 20
    hmm_dim: int = 30
    graph_cutoff: float = 10.0
    ddg_threshold: float = 2.0
    batch_size: int = 1
    n_folds: int = 5
    random_seed: int = 42

    @property
    def input_dim(self) -> int:
        return self.esm_dim + self.pssm_dim + self.hmm_dim


DEFAULT_CONFIG = HotspotConfig()
