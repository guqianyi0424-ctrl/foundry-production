from datetime import datetime
from typing import Any
import uuid

from database import Experiment, ExperimentDesign, ensure_schema_compatibility


class ExperimentRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create_pipeline(
        self,
        name: str,
        input_pdb: str,
        hotspots: list[dict[str, Any]],
        rfd3_config: dict[str, Any],
        user_id: str | None = None,
    ) -> str:
        db = self.session_factory()
        try:
            experiment = Experiment(
                id=str(uuid.uuid4()),
                name=name,
                status="running",
                input_pdb=input_pdb,
                hotspots=hotspots,
                rfd3_config=rfd3_config,
                user_id=user_id,
            )
            db.add(experiment)
            db.commit()
            return experiment.id
        finally:
            db.close()

    def save_step(
        self,
        experiment_id: str,
        step: str,
        results: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        db = self.session_factory()
        try:
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not experiment:
                return
            if step == "rfd3":
                experiment.rfd3_results = results
                if config is not None:
                    experiment.rfd3_config = config
                experiment.status = "rfd3_completed"
            elif step == "mpnn":
                experiment.mpnn_results = results
                if config is not None:
                    experiment.mpnn_config = config
                experiment.status = "mpnn_completed"
            elif step == "rf3":
                experiment.rf3_results = results
                if config is not None:
                    experiment.rf3_config = config
                experiment.status = "completed"
            experiment.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def finish(self, experiment_id: str, status: str, duration_seconds: float) -> None:
        db = self.session_factory()
        try:
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not experiment:
                return
            experiment.status = status
            experiment.duration_seconds = duration_seconds
            experiment.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def save_designs(self, experiment_id: str, designs: list[dict[str, Any]]) -> None:
        if not designs:
            return
        ensure_schema_compatibility()
        db = self.session_factory()
        try:
            for data in designs:
                db.add(
                    ExperimentDesign(
                        id=str(uuid.uuid4()),
                        experiment_id=experiment_id,
                        design_name=data.get("name", ""),
                        sequence=data.get("sequence", ""),
                        pdb_content=data.get("pdb_content", ""),
                        plddt=data.get("plddt"),
                        rmsd=data.get("rmsd"),
                        ranking_score=data.get("ranking_score"),
                        passed_validation=data.get("passed_validation", False),
                        plddt_source=data.get("plddt_source"),
                        ranking_source=data.get("ranking_source"),
                        validation_status=data.get("validation_status"),
                    )
                )
            db.commit()
        finally:
            db.close()

    def get_archive_payload(self, experiment_id: str) -> dict[str, Any] | None:
        db = self.session_factory()
        try:
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not experiment:
                return None
            return {
                "experiment": {
                    "id": experiment.id,
                    "name": experiment.name,
                    "status": experiment.status,
                    "created_at": experiment.created_at.isoformat()
                    if experiment.created_at
                    else None,
                    "updated_at": experiment.updated_at.isoformat()
                    if experiment.updated_at
                    else None,
                    "input_pdb": experiment.input_pdb,
                    "target": experiment.target,
                    "hotspots": experiment.hotspots,
                    "rfd3_config": experiment.rfd3_config,
                    "mpnn_config": experiment.mpnn_config,
                    "rf3_config": experiment.rf3_config,
                    "duration_seconds": experiment.duration_seconds,
                    "gpu_info": experiment.gpu_info,
                    "user_id": experiment.user_id,
                },
                "results": {
                    "rfd3": experiment.rfd3_results,
                    "mpnn": experiment.mpnn_results,
                    "rf3": experiment.rf3_results,
                },
                "designs": [
                    {
                        "id": design.id,
                        "design_name": design.design_name,
                        "sequence": design.sequence,
                        "pdb_content": design.pdb_content,
                        "plddt": design.plddt,
                        "rmsd": design.rmsd,
                        "ranking_score": design.ranking_score,
                        "passed_validation": design.passed_validation,
                        "plddt_source": design.plddt_source,
                        "ranking_source": design.ranking_source,
                        "validation_status": design.validation_status,
                    }
                    for design in experiment.designs
                ],
            }
        finally:
            db.close()
