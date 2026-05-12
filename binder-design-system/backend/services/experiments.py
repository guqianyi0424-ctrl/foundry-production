from typing import Any

from repositories.experiments import ExperimentRepository


class ExperimentService:
    def __init__(self, repository: ExperimentRepository, archive_service=None):
        self.repository = repository
        self.archive_service = archive_service

    def create_pipeline_experiment(
        self,
        name: str,
        input_pdb: str,
        hotspots: list[dict[str, Any]],
        rfd3_config: dict[str, Any],
        user_id: str | None = None,
    ) -> str:
        return self.repository.create_pipeline(name, input_pdb, hotspots, rfd3_config, user_id)

    def save_step(
        self,
        experiment_id: str,
        step: str,
        results: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.repository.save_step(experiment_id, step, results, config)

    def finish(self, experiment_id: str, status: str, duration_seconds: float) -> None:
        self.repository.finish(experiment_id, status, duration_seconds)

    def save_designs(self, experiment_id: str, designs: list[dict[str, Any]]) -> None:
        self.repository.save_designs(experiment_id, designs)

    def write_archive(self, experiment_id: str) -> dict[str, str] | None:
        if self.archive_service is None:
            return None
        payload = self.repository.get_archive_payload(experiment_id)
        if payload is None:
            return None
        return self.archive_service.write_archive(
            experiment=payload["experiment"],
            results=payload["results"],
            designs=payload["designs"],
        )
