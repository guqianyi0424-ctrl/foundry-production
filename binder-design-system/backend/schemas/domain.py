from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass
class AdapterResult(Generic[T]):
    success: bool
    data: T | None = None
    error_code: str | None = None
    message: str | None = None
    fallback_used: bool = False
    mock: bool = False
    fallback_reason: str | None = None
    raw_error: str | None = None


@dataclass(frozen=True)
class Hotspot:
    chain: str
    residue: int
    residue_name: str = ""
    score: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class RFD3JobConfig:
    pdb_content: str
    target: str | None = None
    hotspots: list[str] = field(default_factory=list)
    binder_length: int = 80
    length_min: int = 40
    length_max: int = 120
    diffusion_batch_size: int = 2
    n_batches: int = 2
    job_id: str = "default"


@dataclass(frozen=True)
class RFD3Design:
    index: int
    name: str
    pdb_path: str
    pdb_content: str
    plddt: float = 0.0
    batch: int = 0
    design_in_batch: int = 0
    mock: bool = False


@dataclass(frozen=True)
class MPNNSequence:
    index: int
    name: str
    sequence: str
    pdb_path: str = ""
    pdb_content: str = ""
    score: float = 0.0
    mock: bool = False


@dataclass(frozen=True)
class RF3Validation:
    predicted_pdb: str
    predicted_pdb_path: str
    avg_plddt: float
    rmsd: float
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)
    pae: Any = None
    plddt: list[float] = field(default_factory=list)
    per_res_rmsd: list[float] = field(default_factory=list)
    mock: bool = False


@dataclass
class PipelineResult:
    job_id: str
    experiment_id: str
    status: str
    rfd3: AdapterResult[dict[str, Any]] | None = None
    mpnn: AdapterResult[dict[str, Any]] | None = None
    rf3: AdapterResult[dict[str, Any]] | None = None
    failed_step: str | None = None
