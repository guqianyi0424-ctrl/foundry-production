from collections.abc import Callable
from typing import Any

from schemas.domain import AdapterResult, RFD3JobConfig


PDB_PAYLOAD_KEYS = {
    "pdb_content",
    "first_backbone_pdb",
    "first_sequence_pdb",
    "predicted_pdb",
    "mpnn_pdb_content",
    "rfd3_pdb_content",
}


def sanitize_model_result(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize_model_result(item) for item in value]
    if not isinstance(value, dict):
        return value

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if key in PDB_PAYLOAD_KEYS and isinstance(item, str) and item:
            sanitized[key] = "<omitted>"
        else:
            sanitized[key] = sanitize_model_result(item)
    return sanitized


class RFD3Adapter:
    def __init__(self, runner_factory: Callable[[], Any] | None = None):
        self.runner_factory = runner_factory

    def _runner(self):
        if self.runner_factory:
            return self.runner_factory()
        from utils.rfd3_runner import RFD3Runner

        return RFD3Runner()

    def run(self, config: RFD3JobConfig) -> AdapterResult[dict[str, Any]]:
        try:
            result = self._runner().run_rfd3(
                pdb_content=config.pdb_content,
                target=config.target,
                hotspots=config.hotspots,
                binder_length=config.binder_length,
                length_min=config.length_min,
                length_max=config.length_max,
                diffusion_batch_size=config.diffusion_batch_size,
                n_batches=config.n_batches,
                job_id=config.job_id,
            )
            return AdapterResult(
                success=bool(result.get("success")),
                data=result,
                mock=bool(result.get("mock", False)),
                fallback_used=bool(result.get("mock", False)),
                fallback_reason=result.get("fallback_reason"),
                message=result.get("error"),
            )
        except Exception as exc:
            return AdapterResult(
                success=False,
                error_code="rfd3_adapter_failed",
                message="RFD3 adapter failed",
                raw_error=str(exc),
            )


class MPNNAdapter:
    def __init__(self, runner_factory: Callable[[], Any] | None = None):
        self.runner_factory = runner_factory

    def _runner(self):
        if self.runner_factory:
            return self.runner_factory()
        from utils.mpnn_runner import MPNNRunner

        return MPNNRunner()

    def run(
        self,
        backbone_pdb_content: str | None,
        backbone_pdb_path: str | None,
        batch_size: int,
        fixed_chains: list[str] | None,
        model_type: str,
        job_id: str,
    ) -> AdapterResult[dict[str, Any]]:
        try:
            result = self._runner().run_mpnn(
                backbone_pdb_content=backbone_pdb_content,
                backbone_pdb_path=backbone_pdb_path,
                batch_size=batch_size,
                fixed_chains=fixed_chains,
                model_type=model_type,
                job_id=job_id,
            )
            return AdapterResult(
                success=bool(result.get("success")),
                data=result,
                mock=bool(result.get("mock", False)),
                fallback_used=bool(result.get("mock", False)),
                fallback_reason=result.get("fallback_reason"),
                message=result.get("error"),
            )
        except Exception as exc:
            return AdapterResult(
                success=False,
                error_code="mpnn_adapter_failed",
                message="MPNN adapter failed",
                raw_error=str(exc),
            )


class RF3Adapter:
    def __init__(self, runner_factory: Callable[[], Any] | None = None):
        self.runner_factory = runner_factory

    def _runner(self):
        if self.runner_factory:
            return self.runner_factory()
        from utils.rf3_runner import RF3Runner

        return RF3Runner()

    def run(
        self,
        mpnn_pdb_content: str,
        rfd3_pdb_content: str | None,
        example_id: str,
        job_id: str,
    ) -> AdapterResult[dict[str, Any]]:
        try:
            result = self._runner().run_rf3(
                mpnn_pdb_content=mpnn_pdb_content,
                rfd3_pdb_content=rfd3_pdb_content,
                example_id=example_id,
                job_id=job_id,
            )
            return AdapterResult(
                success=bool(result.get("success")),
                data=result,
                mock=bool(result.get("mock", False)),
                fallback_used=bool(result.get("mock", False)),
                fallback_reason=result.get("fallback_reason"),
                message=result.get("error"),
            )
        except Exception as exc:
            return AdapterResult(
                success=False,
                error_code="rf3_adapter_failed",
                message="RF3 adapter failed",
                raw_error=str(exc),
            )


class HotspotModelAdapter:
    def __init__(self, predictor_factory: Callable[[], Any] | None = None):
        self.predictor_factory = predictor_factory
        self._cached_predictor = None

    def _predictor(self):
        if self._cached_predictor is not None:
            return self._cached_predictor
        if self.predictor_factory:
            self._cached_predictor = self.predictor_factory()
        else:
            from utils.hotspot_predictor import HotspotPredictor

            self._cached_predictor = HotspotPredictor(top_k=5)
        return self._cached_predictor

    def predict(self, pdb_content: str, top_k: int = 5) -> AdapterResult[dict[str, Any]]:
        try:
            result = self._predictor().predict_hotspots(pdb_content, top_k=top_k)
            return AdapterResult(
                success="error" not in result,
                data=result,
                mock=bool(result.get("mock", False)),
                fallback_used=result.get("method") == "rule",
                fallback_reason=(
                    "rule-based hotspot prediction"
                    if result.get("method") == "rule"
                    else None
                ),
                message=result.get("error"),
            )
        except Exception as exc:
            return AdapterResult(
                success=False,
                error_code="hotspot_adapter_failed",
                message="Hotspot adapter failed",
                raw_error=str(exc),
            )
