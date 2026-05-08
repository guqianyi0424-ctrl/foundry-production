from datetime import datetime
from typing import Any
import uuid

from database import Experiment, ExperimentDesign


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
                    )
                )
            db.commit()
        finally:
            db.close()
