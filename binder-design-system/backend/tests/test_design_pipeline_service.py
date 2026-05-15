import threading
import time

from schemas.domain import AdapterResult


class RecordingExperimentService:
    def __init__(self):
        self.steps = []
        self.finished = []
        self.designs = []
        self.statuses = []

    def create_pipeline_experiment(self, name, input_pdb, hotspots, rfd3_config, user_id=None):
        return "exp_1"

    def save_step(self, experiment_id, step, results, config=None):
        self.steps.append((experiment_id, step, results, config))

    def set_status(self, experiment_id, status):
        self.statuses.append((experiment_id, status))

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


class MultiBackboneRFD3:
    def run(self, config):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_backbone_pdb": "BACKBONE_A",
                "designs": [
                    {"name": "backbone_a", "pdb_content": "BACKBONE_A", "plddt": 80.0},
                    {"name": "backbone_b", "pdb_content": "BACKBONE_B", "plddt": 70.0},
                ],
            },
        )


class BackboneAwareMPNN:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        backbone = kwargs["backbone_pdb_content"]
        self.calls.append(backbone)
        sequences_by_backbone = {
            "BACKBONE_A": [
                {
                    "name": "a_seq_low",
                    "sequence": "AAA",
                    "pdb_content": "A_SEQ_LOW_PDB",
                    "score": 0.2,
                },
                {
                    "name": "a_seq_high",
                    "sequence": "AAC",
                    "pdb_content": "A_SEQ_HIGH_PDB",
                    "score": 0.9,
                },
            ],
            "BACKBONE_B": [
                {
                    "name": "b_seq_mid",
                    "sequence": "BBB",
                    "pdb_content": "B_SEQ_MID_PDB",
                    "score": 0.7,
                }
            ],
        }
        sequences = sequences_by_backbone[backbone]
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_sequence_pdb": sequences[0]["pdb_content"],
                "sequences": sequences,
            },
        )


class SequenceAwareRF3:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        sequence_pdb = kwargs["mpnn_pdb_content"]
        self.calls.append((sequence_pdb, kwargs["rfd3_pdb_content"]))
        metrics = {
            "A_SEQ_LOW_PDB": {"avg_plddt": 92.0, "rmsd": 2.0, "ranking_score": 0.1},
            "A_SEQ_HIGH_PDB": {"avg_plddt": 95.0, "rmsd": 1.4, "ranking_score": 0.9},
            "B_SEQ_MID_PDB": {"avg_plddt": 95.0, "rmsd": 1.1, "ranking_score": 0.7},
        }[sequence_pdb]
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": f"RF3_{sequence_pdb}",
                "summary": {"ranking_score": metrics["ranking_score"]},
                "avg_plddt": metrics["avg_plddt"],
                "rmsd": metrics["rmsd"],
                "passed": True,
            },
        )


class SlowCountingMPNN:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_sequence_pdb": f"{kwargs['backbone_pdb_content']}_SEQ",
                "sequences": [
                    {
                        "name": f"{kwargs['backbone_pdb_content']}_seq",
                        "sequence": "ACD",
                        "pdb_content": f"{kwargs['backbone_pdb_content']}_SEQ",
                        "score": 0.5,
                    }
                ],
            },
        )


class SometimesFailingRF3:
    def run(self, **kwargs):
        sequence_pdb = kwargs["mpnn_pdb_content"]
        if sequence_pdb == "A_SEQ_LOW_PDB":
            return AdapterResult(success=False, error_code="rf3_failed", message="RF3 failed")
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": f"RF3_{sequence_pdb}",
                "summary": {"ranking_score": 0.8},
                "avg_plddt": 88.0,
                "rmsd": 1.5,
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
    saved_designs = experiments.designs[-1][1]
    assert [design["name"] for design in saved_designs] == [
        "rfd3_0/seq_0",
        "rfd3_1/seq_0",
        "rfd3_0/seq_1",
        "rfd3_1/seq_1",
    ]
    assert all(design["validation_status"] == "validated" for design in saved_designs)
    assert saved_designs[0]["plddt"] == 90.0
    assert saved_designs[0]["rmsd"] == 1.1


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


def test_pipeline_service_fans_out_mpnn_and_rf3_then_sorts_candidates():
    from services.design_pipeline import DesignPipelineService
    from services.pipeline_scheduler import PipelineTaskScheduler

    experiments = RecordingExperimentService()
    mpnn = BackboneAwareMPNN()
    rf3 = SequenceAwareRF3()
    service = DesignPipelineService(
        rfd3_adapter=MultiBackboneRFD3(),
        mpnn_adapter=mpnn,
        rf3_adapter=rf3,
        experiment_service=experiments,
        scheduler=PipelineTaskScheduler(mpnn_slots=2, rf3_slots=2),
    )

    result = service.run_pipeline("ATOM", [{"chain": "A", "residue": 10}], 80, "job_1")

    assert result.status == "completed"
    assert mpnn.calls == ["BACKBONE_A", "BACKBONE_B"]
    assert sorted(rf3.calls) == sorted([
        ("A_SEQ_LOW_PDB", "BACKBONE_A"),
        ("A_SEQ_HIGH_PDB", "BACKBONE_A"),
        ("B_SEQ_MID_PDB", "BACKBONE_B"),
    ])
    saved_designs = experiments.designs[-1][1]
    assert [design["name"] for design in saved_designs] == [
        "backbone_b/b_seq_mid",
        "backbone_a/a_seq_high",
        "backbone_a/a_seq_low",
    ]
    assert [design["plddt"] for design in saved_designs] == [95.0, 95.0, 92.0]
    assert [design["rmsd"] for design in saved_designs] == [1.1, 1.4, 2.0]
    assert result.mpnn.data["tasks"][0]["backbone_name"] == "backbone_a"
    assert result.rf3.data["summary"]["validated_count"] == 3


def test_pipeline_service_limits_rf3_validation_candidates_and_keeps_unvalidated_designs():
    from services.design_pipeline import DesignPipelineService
    from services.pipeline_scheduler import PipelineTaskScheduler

    experiments = RecordingExperimentService()
    rf3 = SequenceAwareRF3()
    service = DesignPipelineService(
        rfd3_adapter=MultiBackboneRFD3(),
        mpnn_adapter=BackboneAwareMPNN(),
        rf3_adapter=rf3,
        experiment_service=experiments,
        scheduler=PipelineTaskScheduler(mpnn_slots=2, rf3_slots=1),
        max_rf3_candidates=2,
    )

    result = service.run_pipeline("ATOM", [{"chain": "A", "residue": 10}], 80, "job_1")

    assert result.status == "completed"
    assert rf3.calls == [
        ("A_SEQ_HIGH_PDB", "BACKBONE_A"),
        ("B_SEQ_MID_PDB", "BACKBONE_B"),
    ]
    saved_designs = experiments.designs[-1][1]
    assert len(saved_designs) == 3
    not_validated = next(design for design in saved_designs if design["name"] == "backbone_a/a_seq_low")
    assert not_validated["validation_status"] == "not_validated"
    assert not_validated["ranking_source"] == "mpnn"
    assert result.rf3.data["summary"]["task_count"] == 2
    assert result.rf3.data["summary"]["candidate_count"] == 3
    assert result.rf3.data["summary"]["skipped_count"] == 1
    assert result.rf3.data["summary"]["max_candidates"] == 2
    assert [status for _, status in experiments.statuses] == [
        "rfd3_running",
        "mpnn_running",
        "rf3_running",
    ]


def test_pipeline_scheduler_respects_mpnn_slot_limit():
    from services.design_pipeline import DesignPipelineService
    from services.pipeline_scheduler import PipelineTaskScheduler

    experiments = RecordingExperimentService()
    mpnn = SlowCountingMPNN()
    service = DesignPipelineService(
        rfd3_adapter=MultiBackboneRFD3(),
        mpnn_adapter=mpnn,
        rf3_adapter=SuccessfulRF3(),
        experiment_service=experiments,
        scheduler=PipelineTaskScheduler(mpnn_slots=1, rf3_slots=1),
    )

    result = service.run_pipeline("ATOM", [{"chain": "A", "residue": 10}], 80, "job_1")

    assert result.status == "completed"
    assert mpnn.max_active == 1


def test_pipeline_service_keeps_successful_candidates_when_one_rf3_task_fails():
    from services.design_pipeline import DesignPipelineService
    from services.pipeline_scheduler import PipelineTaskScheduler

    experiments = RecordingExperimentService()
    service = DesignPipelineService(
        rfd3_adapter=MultiBackboneRFD3(),
        mpnn_adapter=BackboneAwareMPNN(),
        rf3_adapter=SometimesFailingRF3(),
        experiment_service=experiments,
        scheduler=PipelineTaskScheduler(mpnn_slots=2, rf3_slots=2),
    )

    result = service.run_pipeline("ATOM", [{"chain": "A", "residue": 10}], 80, "job_1")

    assert result.status == "completed"
    saved_designs = experiments.designs[-1][1]
    failed = next(design for design in saved_designs if design["name"] == "backbone_a/a_seq_low")
    assert failed["validation_status"] == "failed"
    assert failed["plddt"] == 80.0
    assert result.rf3.data["summary"]["validated_count"] == 2
    assert result.rf3.data["summary"]["failed_count"] == 1
