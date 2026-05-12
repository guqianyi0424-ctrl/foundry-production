from schemas.domain import AdapterResult


class RecordingExperimentService:
    def __init__(self):
        self.steps = []
        self.finished = []
        self.designs = []

    def create_pipeline_experiment(self, name, input_pdb, hotspots, rfd3_config, user_id=None):
        return "exp_1"

    def save_step(self, experiment_id, step, results, config=None):
        self.steps.append((experiment_id, step, results, config))

    def finish(self, experiment_id, status, duration_seconds):
        self.finished.append((experiment_id, status, duration_seconds))

    def save_designs(self, experiment_id, designs):
        self.designs.append((experiment_id, designs))


class SuccessfulRFD3:
    def run(self, config):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_backbone_pdb": "RFD3_PDB",
                "designs": [
                    {"name": "rfd3_0", "pdb_content": "RFD3_PDB", "plddt": 88.0},
                    {"name": "rfd3_1", "pdb_content": "RFD3_PDB_1", "plddt": 77.5},
                ],
            },
        )


class FailedRFD3:
    def run(self, config):
        return AdapterResult(success=False, error_code="rfd3_failed", message="RFD3 failed")


class SuccessfulMPNN:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_sequence_pdb": "MPNN_PDB",
                "sequences": [
                    {
                        "name": "seq_0",
                        "sequence": "ACD",
                        "pdb_content": "MPNN_PDB",
                        "score": -1.0,
                    },
                    {
                        "name": "seq_1",
                        "sequence": "EFG",
                        "pdb_content": "MPNN_PDB_1",
                        "score": -2.5,
                    },
                ],
            },
        )


class SuccessfulRF3:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": "RF3_PDB",
                "avg_plddt": 90.0,
                "rmsd": 1.1,
                "passed": True,
            },
        )


def test_pipeline_service_runs_all_steps():
    from services.design_pipeline import DesignPipelineService

    experiments = RecordingExperimentService()
    service = DesignPipelineService(
        rfd3_adapter=SuccessfulRFD3(),
        mpnn_adapter=SuccessfulMPNN(),
        rf3_adapter=SuccessfulRF3(),
        experiment_service=experiments,
    )

    result = service.run_pipeline(
        pdb_content="ATOM",
        hotspots=[{"chain": "A", "residue": 10}],
        binder_length=80,
        job_id="job_1",
    )

    assert result.status == "completed"
    assert result.experiment_id == "exp_1"
    assert [step[1] for step in experiments.steps] == ["rfd3", "mpnn", "rf3"]
    assert experiments.finished[-1][1] == "completed"
    assert experiments.designs == [
        (
            "exp_1",
            [
                {
                    "name": "seq_0",
                    "sequence": "ACD",
                    "pdb_content": "MPNN_PDB",
                    "plddt": 90.0,
                    "rmsd": 1.1,
                    "ranking_score": -1.0,
                    "passed_validation": True,
                    "plddt_source": "rf3",
                    "ranking_source": "mpnn",
                    "validation_status": "validated",
                },
                {
                    "name": "seq_1",
                    "sequence": "EFG",
                    "pdb_content": "MPNN_PDB_1",
                    "plddt": 77.5,
                    "rmsd": None,
                    "ranking_score": -2.5,
                    "passed_validation": False,
                    "plddt_source": "rfd3",
                    "ranking_source": "mpnn",
                    "validation_status": "not_validated",
                }
            ],
        )
    ]


def test_pipeline_service_stops_when_rfd3_fails():
    from services.design_pipeline import DesignPipelineService

    experiments = RecordingExperimentService()
    service = DesignPipelineService(
        rfd3_adapter=FailedRFD3(),
        mpnn_adapter=SuccessfulMPNN(),
        rf3_adapter=SuccessfulRF3(),
        experiment_service=experiments,
    )

    result = service.run_pipeline("ATOM", [{"chain": "A", "residue": 10}], 80, "job_1")

    assert result.status == "failed"
    assert result.failed_step == "rfd3"
    assert [step[1] for step in experiments.steps] == ["rfd3"]
    assert experiments.finished[-1][1] == "failed"
