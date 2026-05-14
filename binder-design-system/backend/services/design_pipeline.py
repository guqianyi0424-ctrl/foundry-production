import time
from typing import Any

from schemas.domain import AdapterResult, PipelineResult, RFD3JobConfig
from adapters.model_adapters import sanitize_model_result
from services.pipeline_scheduler import PipelineTaskScheduler


class DesignPipelineService:
    def __init__(
        self,
        rfd3_adapter,
        mpnn_adapter,
        rf3_adapter,
        experiment_service,
        scheduler: PipelineTaskScheduler | None = None,
    ):
        self.rfd3_adapter = rfd3_adapter
        self.mpnn_adapter = mpnn_adapter
        self.rf3_adapter = rf3_adapter
        self.experiment_service = experiment_service
        self.scheduler = scheduler or PipelineTaskScheduler()

    def run_pipeline(
        self,
        pdb_content: str,
        hotspots: list[dict[str, Any]],
        binder_length: int,
        job_id: str,
        user_id: str | None = None,
        target: str | None = None,
        length_min: int = 40,
        length_max: int = 120,
        diffusion_batch_size: int = 2,
        n_batches: int = 2,
        task_name: str | None = None,
        target_filename: str | None = None,
        chain_type: str | None = None,
        progress_callback=None,
    ) -> PipelineResult:
        start = time.time()
        rfd3_config = {
            "task_type": "protein" if pdb_content else "de_novo",
            "task_id": job_id,
            "task_name": task_name,
            "target_filename": target_filename,
            "chain_type": chain_type or "proteinChain",
            "protein_chain": target,
            "target": target,
            "hotspots": hotspots,
            "binder_length": binder_length,
            "length_min": length_min,
            "length_max": length_max,
            "diffusion_batch_size": diffusion_batch_size,
            "n_batches": n_batches,
        }
        experiment_id = self.experiment_service.create_pipeline_experiment(
            name=task_name or f"Pipeline_{time.strftime('%Y%m%d_%H%M%S')}",
            input_pdb=pdb_content,
            hotspots=hotspots,
            rfd3_config=rfd3_config,
            user_id=user_id,
        )

        rfd3_result = self.rfd3_adapter.run(
            RFD3JobConfig(
                pdb_content=pdb_content,
                target=target,
                hotspots=[f"{item['chain']}{item['residue']}" for item in hotspots],
                binder_length=binder_length,
                length_min=length_min,
                length_max=length_max,
                diffusion_batch_size=diffusion_batch_size,
                n_batches=n_batches,
                job_id=job_id,
            )
        )
        self.experiment_service.save_step(
            experiment_id,
            "rfd3",
            sanitize_model_result(rfd3_result.data or {"success": False}),
            rfd3_config,
        )
        if not rfd3_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            result = PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                failed_step="rfd3",
            )
            if progress_callback:
                progress_callback(result)
            return result
        if progress_callback:
            progress_callback(
                PipelineResult(
                    job_id,
                    experiment_id,
                    "running_mpnn",
                    rfd3=rfd3_result,
                )
            )

        target_chains = sorted({str(item["chain"]) for item in hotspots})
        backbones = self._extract_backbone_tasks(rfd3_result.data or {})
        mpnn_tasks = self.scheduler.run_mpnn_tasks(
            backbones,
            lambda backbone: self._run_mpnn_task(backbone, target_chains, job_id),
        )
        mpnn_result = self._aggregate_mpnn_results(mpnn_tasks)
        mpnn_config = {
            "batch_size": 10,
            "fixed_chains": target_chains,
            "model_type": "ligand_mpnn",
            "task_count": len(backbones),
        }
        self.experiment_service.save_step(
            experiment_id,
            "mpnn",
            sanitize_model_result(mpnn_result.data or {"success": False}),
            mpnn_config,
        )

        if not mpnn_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            result = PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                mpnn=mpnn_result,
                failed_step="mpnn",
            )
            if progress_callback:
                progress_callback(result)
            return result
        if progress_callback:
            progress_callback(
                PipelineResult(
                    job_id,
                    experiment_id,
                    "running_rf3",
                    rfd3=rfd3_result,
                    mpnn=mpnn_result,
                )
            )

        rf3_work_items = self._build_rf3_work_items(mpnn_tasks)
        rf3_tasks = self.scheduler.run_rf3_tasks(
            rf3_work_items,
            lambda item: self._run_rf3_task(item, job_id),
        )
        rf3_result = self._aggregate_rf3_results(rf3_tasks)
        self.experiment_service.save_step(
            experiment_id,
            "rf3",
            sanitize_model_result(rf3_result.data or {"success": False}),
            {"example_id": f"binder_{job_id}", "task_count": len(rf3_work_items)},
        )
        if not rf3_result.success:
            self.experiment_service.finish(experiment_id, "failed", time.time() - start)
            result = PipelineResult(
                job_id,
                experiment_id,
                "failed",
                rfd3=rfd3_result,
                mpnn=mpnn_result,
                rf3=rf3_result,
                failed_step="rf3",
            )
            if progress_callback:
                progress_callback(result)
            return result

        self.experiment_service.save_designs(
            experiment_id,
            self._build_parallel_candidate_designs(mpnn_tasks, rf3_tasks),
        )
        self.experiment_service.finish(experiment_id, "completed", time.time() - start)
        if hasattr(self.experiment_service, "write_archive"):
            self.experiment_service.write_archive(experiment_id)
        result = PipelineResult(
            job_id,
            experiment_id,
            "completed",
            rfd3=rfd3_result,
            mpnn=mpnn_result,
            rf3=rf3_result,
        )
        if progress_callback:
            progress_callback(result)
        return result

    def _extract_backbone_tasks(self, rfd3_data: dict[str, Any]) -> list[dict[str, Any]]:
        designs = rfd3_data.get("designs") or []
        tasks = []
        for index, design in enumerate(designs):
            pdb_content = design.get("pdb_content")
            if not pdb_content:
                continue
            tasks.append(
                {
                    "index": design.get("index", index),
                    "name": design.get("name") or f"backbone_{index + 1:03d}",
                    "pdb_content": pdb_content,
                    "plddt": design.get("plddt"),
                }
            )
        if tasks:
            return tasks
        first_backbone = rfd3_data.get("first_backbone_pdb")
        if not first_backbone:
            return []
        return [
            {
                "index": 0,
                "name": "backbone_001",
                "pdb_content": first_backbone,
                "plddt": None,
            }
        ]

    def _run_mpnn_task(
        self,
        backbone: dict[str, Any],
        target_chains: list[str],
        job_id: str,
    ) -> dict[str, Any]:
        result = self.mpnn_adapter.run(
            backbone_pdb_content=backbone["pdb_content"],
            backbone_pdb_path=None,
            batch_size=10,
            fixed_chains=target_chains,
            model_type="ligand_mpnn",
            job_id=f"{job_id}_mpnn_{backbone['index']}",
        )
        return {
            "backbone": backbone,
            "result": result,
        }

    def _aggregate_mpnn_results(self, tasks: list[dict[str, Any]]) -> AdapterResult[dict[str, Any]]:
        successful = [task for task in tasks if task["result"].success]
        all_sequences = []
        for task in successful:
            all_sequences.extend((task["result"].data or {}).get("sequences") or [])
        return AdapterResult(
            success=bool(successful and all_sequences),
            data={
                "success": bool(successful and all_sequences),
                "tasks": [
                    {
                        "backbone_name": task["backbone"]["name"],
                        "backbone_index": task["backbone"]["index"],
                        "success": task["result"].success,
                        "result": task["result"].data,
                        "error": task["result"].message or task["result"].raw_error,
                    }
                    for task in tasks
                ],
                "sequences": all_sequences,
                "first_sequence_pdb": all_sequences[0].get("pdb_content") if all_sequences else None,
                "summary": {
                    "task_count": len(tasks),
                    "successful_count": len(successful),
                    "failed_count": len(tasks) - len(successful),
                    "sequence_count": len(all_sequences),
                },
            },
        )

    def _build_rf3_work_items(self, mpnn_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for task in mpnn_tasks:
            if not task["result"].success:
                continue
            sequences = (task["result"].data or {}).get("sequences") or []
            for index, sequence in enumerate(sequences):
                if not sequence.get("pdb_content"):
                    continue
                items.append(
                    {
                        "backbone": task["backbone"],
                        "sequence": sequence,
                        "sequence_index": sequence.get("index", index),
                    }
                )
        return items

    def _run_rf3_task(self, item: dict[str, Any], job_id: str) -> dict[str, Any]:
        result = self.rf3_adapter.run(
            mpnn_pdb_content=item["sequence"]["pdb_content"],
            rfd3_pdb_content=item["backbone"]["pdb_content"],
            example_id=f"binder_{job_id}_{item['backbone']['index']}_{item['sequence_index']}",
            job_id=f"{job_id}_rf3_{item['backbone']['index']}_{item['sequence_index']}",
        )
        return {
            **item,
            "result": result,
        }

    def _aggregate_rf3_results(self, tasks: list[dict[str, Any]]) -> AdapterResult[dict[str, Any]]:
        successful = [task for task in tasks if task["result"].success]
        return AdapterResult(
            success=bool(successful),
            data={
                "success": bool(successful),
                "tasks": [
                    {
                        "backbone_name": task["backbone"]["name"],
                        "sequence_name": task["sequence"].get("name"),
                        "success": task["result"].success,
                        "result": task["result"].data,
                        "error": task["result"].message or task["result"].raw_error,
                    }
                    for task in tasks
                ],
                "summary": {
                    "task_count": len(tasks),
                    "validated_count": len(successful),
                    "failed_count": len(tasks) - len(successful),
                },
            },
        )

    def _build_parallel_candidate_designs(
        self,
        mpnn_tasks: list[dict[str, Any]],
        rf3_tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rf3_by_pdb = {
            task["sequence"]["pdb_content"]: task
            for task in rf3_tasks
        }
        candidates = []
        for mpnn_task in mpnn_tasks:
            if not mpnn_task["result"].success:
                continue
            backbone = mpnn_task["backbone"]
            for sequence in (mpnn_task["result"].data or {}).get("sequences") or []:
                rf3_task = rf3_by_pdb.get(sequence.get("pdb_content"))
                rf3_result = rf3_task["result"] if rf3_task else None
                rf3_data = rf3_result.data if rf3_result and rf3_result.success else {}
                rf3_ranking = (rf3_data.get("summary") or {}).get("ranking_score")
                sequence_score = sequence.get("score")
                avg_plddt = rf3_data.get("avg_plddt")
                fallback_plddt = backbone.get("plddt")

                candidates.append(
                    {
                        "name": f"{backbone['name']}/{sequence.get('name') or 'sequence'}",
                        "sequence": sequence.get("sequence", ""),
                        "pdb_content": sequence.get("pdb_content", ""),
                        "plddt": avg_plddt if avg_plddt is not None else fallback_plddt,
                        "rmsd": rf3_data.get("rmsd"),
                        "ranking_score": rf3_ranking if rf3_ranking is not None else sequence_score,
                        "passed_validation": bool(rf3_data.get("passed", False)),
                        "plddt_source": "rf3" if avg_plddt is not None else (
                            "rfd3" if fallback_plddt is not None else "none"
                        ),
                        "ranking_source": "rf3" if rf3_ranking is not None else (
                            "mpnn" if sequence_score is not None else "none"
                        ),
                        "validation_status": "validated"
                        if rf3_result and rf3_result.success
                        else "failed",
                    }
                )
        return sorted(candidates, key=self._candidate_sort_key)

    def _candidate_sort_key(self, candidate: dict[str, Any]):
        plddt = candidate.get("plddt")
        rmsd = candidate.get("rmsd")
        ranking_score = candidate.get("ranking_score")
        return (
            -(plddt if plddt is not None else float("-inf")),
            rmsd if rmsd is not None else float("inf"),
            -(ranking_score if ranking_score is not None else float("-inf")),
        )

    def _build_candidate_designs(
        self,
        mpnn_data: dict[str, Any],
        rf3_data: dict[str, Any],
        rfd3_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sequences = mpnn_data.get("sequences") or []
        if not sequences:
            return []

        rfd3_designs = (rfd3_data or {}).get("designs") or []
        rf3_ranking = (rf3_data.get("summary") or {}).get("ranking_score")
        candidates = []
        for index, sequence in enumerate(sequences):
            is_validated_sequence = index == 0
            rfd3_design = rfd3_designs[index] if index < len(rfd3_designs) else {}
            fallback_plddt = rfd3_design.get("plddt")
            sequence_score = sequence.get("score")

            if is_validated_sequence and rf3_data.get("avg_plddt") is not None:
                plddt = rf3_data.get("avg_plddt")
                plddt_source = "rf3"
            elif fallback_plddt is not None:
                plddt = fallback_plddt
                plddt_source = "rfd3"
            else:
                plddt = None
                plddt_source = "none"

            if is_validated_sequence and rf3_ranking is not None:
                ranking_score = rf3_ranking
                ranking_source = "rf3"
            elif sequence_score is not None:
                ranking_score = sequence_score
                ranking_source = "mpnn"
            else:
                ranking_score = None
                ranking_source = "none"

            candidates.append(
                {
                    "name": sequence.get("name") or f"candidate_{index + 1:03d}",
                    "sequence": sequence.get("sequence", ""),
                    "pdb_content": sequence.get("pdb_content", ""),
                    "plddt": plddt,
                    "rmsd": rf3_data.get("rmsd") if is_validated_sequence else None,
                    "ranking_score": ranking_score,
                    "passed_validation": bool(rf3_data.get("passed", False))
                    if is_validated_sequence
                    else False,
                    "plddt_source": plddt_source,
                    "ranking_source": ranking_source,
                    "validation_status": "validated"
                    if is_validated_sequence
                    else "not_validated",
                }
            )
        return candidates
