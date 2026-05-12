import time
from typing import Any

from schemas.domain import PipelineResult, RFD3JobConfig


class DesignPipelineService:
    def __init__(self, rfd3_adapter, mpnn_adapter, rf3_adapter, experiment_service):
        self.rfd3_adapter = rfd3_adapter
        self.mpnn_adapter = mpnn_adapter
        self.rf3_adapter = rf3_adapter
        self.experiment_service = experiment_service

    def run_pipeline(
        self,
        pdb_content: str,
        hotspots: list[dict[str, Any]],
        binder_length: int,
        job_id: str,
        user_id: str | None = None,
    ) -> PipelineResult:
        start = time.time()
        rfd3_config = {"binder_length": binder_length}
        experiment_id = self.experiment_service.create_pipeline_experiment(
            name=f"Pipeline_{time.strftime('%Y%m%d_%H%M%S')}",
            input_pdb=pdb_content,
            hotspots=hotspots,
            rfd3_config=rfd3_config,
            user_id=user_id,
        )

        rfd3_result = self.rfd3_adapter.run(
            RFD3JobConfig(
                pdb_content=pdb_content,
                hotspots=[f"{item['chain']}{item['residue']}" for item in hotspots],
                binder_length=binder_length,
                job_id=job_id,
            )
        )
        self.experiment_service.save_step(
            experiment_id,
            "rfd3",
            rfd3_result.data or {"success": False},
            rfd3_config,
        )
        if not rfd3_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            return PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                failed_step="rfd3",
            )

        first_backbone = (rfd3_result.data or {}).get("first_backbone_pdb")
        target_chains = sorted({str(item["chain"]) for item in hotspots})
        mpnn_result = self.mpnn_adapter.run(
            backbone_pdb_content=first_backbone,
            backbone_pdb_path=None,
            batch_size=10,
            fixed_chains=target_chains,
            model_type="ligand_mpnn",
            job_id=job_id,
        )
        self.experiment_service.save_step(
            experiment_id,
            "mpnn",
            mpnn_result.data or {"success": False},
        )
        if not mpnn_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            return PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                mpnn=mpnn_result,
                failed_step="mpnn",
            )

        first_sequence = (mpnn_result.data or {}).get("first_sequence_pdb")
        rf3_result = self.rf3_adapter.run(
            mpnn_pdb_content=first_sequence,
            rfd3_pdb_content=first_backbone,
            example_id=f"binder_{job_id}",
            job_id=job_id,
        )
        self.experiment_service.save_step(
            experiment_id,
            "rf3",
            rf3_result.data or {"success": False},
        )
        if not rf3_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            return PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                mpnn=mpnn_result,
                rf3=rf3_result,
                failed_step="rf3",
            )

        self.experiment_service.save_designs(
            experiment_id,
            self._build_candidate_designs(mpnn_result.data or {}, rf3_result.data or {}),
        )
        self.experiment_service.finish(experiment_id, "completed", time.time() - start)
        if hasattr(self.experiment_service, "write_archive"):
            self.experiment_service.write_archive(experiment_id)
        return PipelineResult(
            job_id,
            experiment_id,
            "completed",
            rfd3=rfd3_result,
            mpnn=mpnn_result,
            rf3=rf3_result,
        )

    def _build_candidate_designs(
        self,
        mpnn_data: dict[str, Any],
        rf3_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sequences = mpnn_data.get("sequences") or []
        if not sequences:
            return []

        candidates = []
        for index, sequence in enumerate(sequences, start=1):
            is_validated_sequence = index == 1
            candidates.append(
                {
                    "name": sequence.get("name") or f"candidate_{index:03d}",
                    "sequence": sequence.get("sequence", ""),
                    "pdb_content": sequence.get("pdb_content", ""),
                    "plddt": rf3_data.get("avg_plddt") if is_validated_sequence else None,
                    "rmsd": rf3_data.get("rmsd") if is_validated_sequence else None,
                    "ranking_score": (rf3_data.get("summary") or {}).get(
                        "ranking_score",
                        sequence.get("score"),
                    ),
                    "passed_validation": bool(rf3_data.get("passed", False))
                    if is_validated_sequence
                    else False,
                }
            )
        return candidates
