import asyncio
import threading

from database import SessionLocal, Experiment
from schemas.domain import AdapterResult, PipelineResult
from services.design_pipeline import DesignPipelineService


class FakeHotspotService:
    last_top_k = None

    def predict(self, pdb_content, top_k=5):
        self.last_top_k = top_k
        return AdapterResult(
            success=True,
            data={
                "hotspots_detail": [
                    {
                        "chain": "A",
                        "residue_id": 10,
                        "residue_name": "TYR",
                        "score": 0.9,
                        "label": "A10",
                    }
                ],
                "num_hotspots": 1,
                "method": "rule",
                "model_loaded": False,
                "total_residues": 100,
            },
        )


class FakePipelineService:
    last_call = None
    calls = 0

    def run_pipeline(
        self,
        pdb_content,
        hotspots,
        binder_length,
        job_id,
        user_id=None,
        target=None,
        length_min=40,
        length_max=120,
        diffusion_batch_size=2,
        n_batches=2,
        task_name=None,
        target_filename=None,
        chain_type=None,
        progress_callback=None,
    ):
        self.calls += 1
        self.last_call = {
            "pdb_content": pdb_content,
            "hotspots": hotspots,
            "binder_length": binder_length,
            "job_id": job_id,
            "user_id": user_id,
            "target": target,
            "length_min": length_min,
            "length_max": length_max,
            "diffusion_batch_size": diffusion_batch_size,
            "n_batches": n_batches,
            "task_name": task_name,
            "target_filename": target_filename,
            "chain_type": chain_type,
        }
        return PipelineResult(
            job_id=job_id,
            experiment_id="exp_1",
            status="completed",
            rfd3=AdapterResult(
                success=True,
                data={"success": True, "designs": [], "first_backbone_pdb": "ATOM"},
            ),
            mpnn=AdapterResult(
                success=True,
                data={"success": True, "sequences": [], "first_sequence_pdb": "ATOM"},
            ),
            rf3=AdapterResult(
                success=True,
                data={"success": True, "avg_plddt": 90.0, "rmsd": 1.0, "passed": True},
            ),
        )


class FakeRFD3Service:
    def __init__(self):
        self.config = None

    def run(self, config):
        self.config = config
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "designs": [
                    {
                        "index": 0,
                        "batch": 0,
                        "design_in_batch": 0,
                        "name": "denovo_0",
                        "pdb_content": "ATOM",
                        "pdb_path": "",
                        "plddt": 88.0,
                    }
                ],
                "batches": [],
                "num_batches": 1,
                "num_designs": 1,
                "first_backbone_pdb": "ATOM",
            },
        )


class FakeMPNNService:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "sequences": [
                    {
                        "index": 0,
                        "name": "seq_0",
                        "sequence": "ACD",
                        "pdb_content": "MPNN_PDB",
                        "score": -1.0,
                    }
                ],
                "num_sequences": 1,
                "first_sequence_pdb": "MPNN_PDB",
                "output_dir": "",
            },
        )


class FakeRF3Service:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": "RF3_PDB",
                "predicted_pdb_path": "",
                "num_models": 1,
                "summary": {"ranking_score": 0.8},
                "pae": None,
                "plddt": [90.0],
                "avg_plddt": 90.0,
                "rmsd": 1.0,
                "rmsd_interpretation": "Excellent",
                "per_res_rmsd": [1.0],
                "passed": True,
                "output_dir": "",
            },
        )


class FakeServices:
    hotspot_prediction = FakeHotspotService()
    pipeline = FakePipelineService()
    rfd3 = FakeRFD3Service()
    mpnn = FakeMPNNService()
    rf3 = FakeRF3Service()


class FakeUser:
    id = "user_rfd3"


def patch_services(monkeypatch):
    import routers.design as design_router

    monkeypatch.setattr(design_router, "get_design_services", lambda: FakeServices())
    return design_router


def test_predict_hotspot_contract(monkeypatch):
    design_router = patch_services(monkeypatch)
    FakeServices.hotspot_prediction.last_top_k = None

    data = asyncio.run(
        design_router.predict_hotspot(
            design_router.HotspotRequest(pdb_content="ATOM")
        )
    )

    assert data["hotspots"] == [
        {"chain": "A", "residue": 10, "residue_name": "TYR", "score": 0.9, "label": "A10"}
    ]
    assert data["num_hotspots"] == 1
    assert data["method"] == "rule"
    assert data["model_loaded"] is False
    assert data["total_residues"] == 100
    assert FakeServices.hotspot_prediction.last_top_k == 3


def test_run_pipeline_contract(monkeypatch):
    design_router = patch_services(monkeypatch)
    FakeServices.pipeline.last_call = None

    data = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
                target="A/1-100",
                length_min=50,
                length_max=140,
                diffusion_batch_size=3,
                n_batches=4,
                task_name="Pipeline Protein",
                target_filename="target.pdb",
                chain_type="proteinChain",
            )
        )
    )

    assert data["experiment_id"] == "exp_1"
    assert data["status"] == "completed"
    assert data["rfd3_results"]["success"] is True
    assert data["mpnn_results"]["success"] is True
    assert data["rf3_results"]["success"] is True
    assert FakeServices.pipeline.last_call is not None
    assert FakeServices.pipeline.last_call["target"] == "A/1-100"
    assert FakeServices.pipeline.last_call["length_min"] == 50
    assert FakeServices.pipeline.last_call["length_max"] == 140
    assert FakeServices.pipeline.last_call["diffusion_batch_size"] == 3
    assert FakeServices.pipeline.last_call["n_batches"] == 4
    assert FakeServices.pipeline.last_call["task_name"] == "Pipeline Protein"
    assert FakeServices.pipeline.last_call["target_filename"] == "target.pdb"
    assert FakeServices.pipeline.last_call["chain_type"] == "proteinChain"


def test_run_pipeline_async_mode_returns_running_job_and_allows_polling(monkeypatch):
    design_router = patch_services(monkeypatch)
    FakeServices.pipeline.last_call = None
    FakeServices.pipeline.calls = 0
    design_router._pipeline_jobs.clear()
    persisted_jobs = {}
    monkeypatch.setattr(
        design_router,
        "_create_pipeline_job",
        lambda job_id, status="running": persisted_jobs.setdefault(
            job_id,
            {
                "job_id": job_id,
                "experiment_id": None,
                "status": status,
                "failed_step": None,
                "error": None,
                "rfd3_results": None,
                "mpnn_results": None,
                "rf3_results": None,
            },
        ),
    )

    def update_job(job_id, **updates):
        persisted_jobs[job_id].update(updates)

    monkeypatch.setattr(design_router, "_update_pipeline_job", update_job)
    monkeypatch.setattr(design_router, "_get_pipeline_job", lambda job_id: persisted_jobs.get(job_id))

    data = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
                async_mode=True,
            )
        )
    )

    assert data["status"] == "running"
    assert data["experiment_id"] is None
    assert data["job_id"].startswith("job_")
    job = asyncio.run(design_router.get_pipeline_job(data["job_id"]))
    assert job["status"] == "completed"
    assert job["experiment_id"] == "exp_1"
    assert job["rfd3_results"]["success"] is True
    assert FakeServices.pipeline.calls == 1


def test_run_pipeline_async_mode_allocates_unique_job_ids(monkeypatch):
    design_router = patch_services(monkeypatch)
    design_router._pipeline_jobs.clear()
    persisted_jobs = {}
    monkeypatch.setattr(design_router, "_create_pipeline_job", lambda job_id, status="running": persisted_jobs.setdefault(job_id, {}))
    monkeypatch.setattr(design_router, "_update_pipeline_job", lambda job_id, **updates: persisted_jobs.setdefault(job_id, {}).update(updates))
    monkeypatch.setattr(design_router, "_get_pipeline_job", lambda job_id: persisted_jobs.get(job_id))

    first = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
                async_mode=True,
            )
        )
    )
    second = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
                async_mode=True,
            )
        )
    )

    assert first["job_id"] != second["job_id"]


class ProgressPipelineService:
    rfd3_reported = threading.Event()
    allow_finish = threading.Event()

    def run_pipeline(
        self,
        pdb_content,
        hotspots,
        binder_length,
        job_id,
        user_id=None,
        target=None,
        length_min=40,
        length_max=120,
        diffusion_batch_size=2,
        n_batches=2,
        task_name=None,
        target_filename=None,
        chain_type=None,
        progress_callback=None,
    ):
        partial = PipelineResult(
            job_id=job_id,
            experiment_id="exp_progress",
            status="running_mpnn",
            rfd3=AdapterResult(
                success=True,
                data={"success": True, "first_backbone_pdb": "RFD3_READY", "designs": []},
            ),
        )
        progress_callback(partial)
        self.rfd3_reported.set()
        self.allow_finish.wait(timeout=2)
        return PipelineResult(
            job_id=job_id,
            experiment_id="exp_progress",
            status="completed",
            rfd3=partial.rfd3,
            mpnn=AdapterResult(success=True, data={"success": True, "sequences": []}),
            rf3=AdapterResult(success=True, data={"success": True, "passed": True}),
        )


class ProgressServices:
    hotspot_prediction = FakeHotspotService()
    pipeline = ProgressPipelineService()
    rfd3 = FakeRFD3Service()
    mpnn = FakeMPNNService()
    rf3 = FakeRF3Service()


def test_run_pipeline_async_mode_exposes_rfd3_progress_before_completion(monkeypatch):
    import routers.design as design_router

    services = ProgressServices()
    services.pipeline.rfd3_reported.clear()
    services.pipeline.allow_finish.clear()
    monkeypatch.setattr(design_router, "get_design_services", lambda: services)
    design_router._pipeline_jobs.clear()
    persisted_jobs = {}
    monkeypatch.setattr(
        design_router,
        "_create_pipeline_job",
        lambda job_id, status="running": persisted_jobs.setdefault(
            job_id,
            {
                "job_id": job_id,
                "experiment_id": None,
                "status": status,
                "failed_step": None,
                "error": None,
                "rfd3_results": None,
                "mpnn_results": None,
                "rf3_results": None,
            },
        ),
    )

    def update_job(job_id, **updates):
        persisted_jobs[job_id].update(updates)

    monkeypatch.setattr(design_router, "_update_pipeline_job", update_job)
    monkeypatch.setattr(design_router, "_get_pipeline_job", lambda job_id: persisted_jobs.get(job_id))

    submitted = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
                async_mode=True,
            )
        )
    )

    assert services.pipeline.rfd3_reported.wait(timeout=2)
    progress = asyncio.run(design_router.get_pipeline_job(submitted["job_id"]))
    assert progress["status"] == "running_mpnn"
    assert progress["experiment_id"] == "exp_progress"
    assert progress["rfd3_results"]["first_backbone_pdb"] == "RFD3_READY"
    assert progress["mpnn_results"] is None

    services.pipeline.allow_finish.set()


class RecordingExperimentService:
    def __init__(self):
        self.created = None
        self.steps = []
        self.designs = []
        self.finished = None
        self.archived = None

    def create_pipeline_experiment(self, **kwargs):
        self.created = kwargs
        return "exp_pipeline"

    def save_step(self, experiment_id, step, results, config=None):
        self.steps.append(
            {
                "experiment_id": experiment_id,
                "step": step,
                "results": results,
                "config": config,
            }
        )

    def save_designs(self, experiment_id, designs):
        self.designs.append({"experiment_id": experiment_id, "designs": designs})

    def finish(self, experiment_id, status, duration_seconds):
        self.finished = {
            "experiment_id": experiment_id,
            "status": status,
            "duration_seconds": duration_seconds,
        }

    def write_archive(self, experiment_id):
        self.archived = experiment_id


class RecordingRFD3Adapter:
    def __init__(self):
        self.config = None

    def run(self, config):
        self.config = config
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "designs": [
                    {
                        "index": 0,
                        "name": "rfd3_0",
                        "pdb_content": "RFD3_PDB",
                        "plddt": 82.0,
                    }
                ],
                "first_backbone_pdb": "RFD3_PDB",
            },
        )


class RecordingMPNNAdapter:
    def __init__(self):
        self.kwargs = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "sequences": [
                    {
                        "index": 0,
                        "name": "seq_0",
                        "sequence": "ACD",
                        "pdb_content": "MPNN_PDB",
                        "score": -0.5,
                    }
                ],
                "first_sequence_pdb": "MPNN_PDB",
            },
        )


class RecordingRF3Adapter:
    def __init__(self):
        self.kwargs = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": "RF3_PDB",
                "summary": {"ranking_score": 0.88},
                "avg_plddt": 91.0,
                "rmsd": 1.2,
                "passed": True,
            },
        )


def test_pipeline_service_persists_sanitized_steps_and_candidate_designs():
    experiment_service = RecordingExperimentService()
    rfd3 = RecordingRFD3Adapter()
    mpnn = RecordingMPNNAdapter()
    rf3 = RecordingRF3Adapter()
    service = DesignPipelineService(rfd3, mpnn, rf3, experiment_service)

    result = service.run_pipeline(
        pdb_content="TARGET_PDB",
        hotspots=[{"chain": "A", "residue": 42}],
        binder_length=80,
        job_id="job_1",
        user_id="user_1",
        target="A/1-100",
        length_min=50,
        length_max=140,
        diffusion_batch_size=3,
        n_batches=4,
        task_name="Full pipeline",
        target_filename="target.pdb",
        chain_type="proteinChain",
    )

    assert result.status == "completed"
    assert experiment_service.created["name"] == "Full pipeline"
    assert experiment_service.created["user_id"] == "user_1"
    assert experiment_service.created["rfd3_config"]["target"] == "A/1-100"
    assert rfd3.config.target == "A/1-100"
    assert rfd3.config.hotspots == ["A42"]
    assert rfd3.config.length_min == 50
    assert rfd3.config.length_max == 140
    assert rfd3.config.diffusion_batch_size == 3
    assert rfd3.config.n_batches == 4
    assert mpnn.kwargs["backbone_pdb_content"] == "RFD3_PDB"
    assert mpnn.kwargs["fixed_chains"] == ["A"]
    assert rf3.kwargs["mpnn_pdb_content"] == "MPNN_PDB"
    assert rf3.kwargs["rfd3_pdb_content"] == "RFD3_PDB"
    assert [step["step"] for step in experiment_service.steps] == ["rfd3", "mpnn", "rf3"]
    assert experiment_service.steps[0]["results"]["first_backbone_pdb"] == "<omitted>"
    assert experiment_service.steps[1]["results"]["first_sequence_pdb"] == "<omitted>"
    assert experiment_service.steps[2]["results"]["tasks"][0]["result"]["predicted_pdb"] == "<omitted>"
    assert experiment_service.steps[2]["results"]["summary"]["validated_count"] == 1
    assert experiment_service.designs[0]["experiment_id"] == "exp_pipeline"
    assert experiment_service.designs[0]["designs"] == [
        {
            "name": "rfd3_0/seq_0",
            "sequence": "ACD",
            "pdb_content": "MPNN_PDB",
            "plddt": 91.0,
            "rmsd": 1.2,
            "ranking_score": 0.88,
            "passed_validation": True,
            "plddt_source": "rf3",
            "ranking_source": "rf3",
            "validation_status": "validated",
        }
    ]
    assert experiment_service.finished["status"] == "completed"
    assert experiment_service.archived == "exp_pipeline"


def test_run_rfd3_accepts_denovo_request_without_target(monkeypatch):
    design_router = patch_services(monkeypatch)
    services = FakeServices()
    monkeypatch.setattr(design_router, "get_design_services", lambda: services)

    data = asyncio.run(
        design_router.run_rfd3(
            design_router.RFD3Request(
                binder_length=72,
                diffusion_batch_size=1,
                n_batches=1,
            ),
            current_user=FakeUser(),
        )
    )

    assert data["success"] is True
    assert data["num_designs"] == 1
    assert services.rfd3.config.pdb_content == ""
    assert services.rfd3.config.target is None
    assert services.rfd3.config.hotspots == []
    assert services.rfd3.config.binder_length == 72
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == data["experiment_id"]).one()
        assert exp.user_id == "user_rfd3"
    finally:
        db.delete(exp)
        db.commit()
        db.close()


def test_preview_only_mpnn_and_rf3_do_not_persist(monkeypatch):
    design_router = patch_services(monkeypatch)
    persisted = []
    monkeypatch.setattr(
        design_router,
        "_save_experiment_step",
        lambda *args, **kwargs: persisted.append(("step", args, kwargs)),
    )
    monkeypatch.setattr(
        design_router,
        "_save_designs",
        lambda *args, **kwargs: persisted.append(("designs", args, kwargs)),
    )

    mpnn = asyncio.run(
        design_router.run_mpnn(
            design_router.MPNNRequest(
                backbone_pdb_content="RFD3_PDB",
                experiment_id="exp_formal",
                preview_only=True,
            )
        )
    )
    rf3 = asyncio.run(
        design_router.run_rf3(
            design_router.RF3Request(
                mpnn_pdb_content="MPNN_PDB",
                rfd3_pdb_content="RFD3_PDB",
                experiment_id="exp_formal",
                preview_only=True,
            )
        )
    )

    assert mpnn["success"] is True
    assert rf3["success"] is True
    assert persisted == []


def test_run_rf3_updates_persisted_design_metrics_from_real_validation(monkeypatch):
    design_router = patch_services(monkeypatch)
    db = SessionLocal()
    try:
        exp = Experiment(id="exp_rf3_update", name="RF3 update", status="mpnn_completed")
        db.add(exp)
        rfd3_design = design_router.ExperimentDesign(
            id="design_rfd3",
            experiment_id=exp.id,
            design_name="batch0_design0",
            sequence="",
            pdb_content="RFD3_PDB",
            plddt=88.0,
            rmsd=None,
            ranking_score=None,
            passed_validation=False,
            plddt_source="rfd3",
            ranking_source="none",
            validation_status="not_validated",
        )
        design = design_router.ExperimentDesign(
            id="design_mpnn",
            experiment_id=exp.id,
            design_name="seq_0",
            sequence="ACD",
            pdb_content="MPNN_PDB",
            plddt=None,
            rmsd=None,
            ranking_score=-1.0,
            passed_validation=False,
            plddt_source="none",
            ranking_source="mpnn",
            validation_status="not_validated",
        )
        db.add(rfd3_design)
        db.add(design)
        db.commit()

        data = asyncio.run(
            design_router.run_rf3(
                design_router.RF3Request(
                    mpnn_pdb_content="MPNN_PDB",
                    rfd3_pdb_content="RFD3_PDB",
                    experiment_id=exp.id,
                )
            )
        )

        db.refresh(design)
        db.refresh(rfd3_design)
        assert data["success"] is True
        assert rfd3_design.rmsd is None
        assert rfd3_design.validation_status == "not_validated"
        assert design.plddt == 90.0
        assert design.rmsd == 1.0
        assert design.ranking_score == 0.8
        assert design.passed_validation is True
        assert design.plddt_source == "rf3"
        assert design.ranking_source == "rf3"
        assert design.validation_status == "validated"
    finally:
        db.query(design_router.ExperimentDesign).filter(
            design_router.ExperimentDesign.experiment_id == "exp_rf3_update"
        ).delete()
        db.query(Experiment).filter(Experiment.id == "exp_rf3_update").delete()
        db.commit()
        db.close()


def test_formal_mpnn_and_rf3_fill_existing_rfd3_candidate(monkeypatch):
    design_router = patch_services(monkeypatch)
    db = SessionLocal()
    try:
        exp = Experiment(id="exp_candidate_fill", name="Candidate fill", status="rfd3_completed")
        db.add(exp)
        db.add(
            design_router.ExperimentDesign(
                id="design_rfd3_fill",
                experiment_id=exp.id,
                design_name="denovo_0",
                sequence="",
                pdb_content="RFD3_PDB",
                plddt=88.0,
                rmsd=None,
                ranking_score=None,
                passed_validation=False,
                plddt_source="rfd3",
                ranking_source="none",
                validation_status="not_validated",
            )
        )
        db.commit()

        mpnn = asyncio.run(
            design_router.run_mpnn(
                design_router.MPNNRequest(
                    backbone_pdb_content="RFD3_PDB",
                    experiment_id=exp.id,
                )
            )
        )
        rf3 = asyncio.run(
            design_router.run_rf3(
                design_router.RF3Request(
                    mpnn_pdb_content="MPNN_PDB",
                    rfd3_pdb_content="RFD3_PDB",
                    experiment_id=exp.id,
                )
            )
        )

        db.expire_all()
        designs = (
            db.query(design_router.ExperimentDesign)
            .filter(design_router.ExperimentDesign.experiment_id == exp.id)
            .order_by(design_router.ExperimentDesign.id.asc())
            .all()
        )

        assert mpnn["success"] is True
        assert rf3["success"] is True
        assert len(designs) == 1
        assert designs[0].sequence == "ACD"
        assert designs[0].pdb_content == "MPNN_PDB"
        assert designs[0].plddt == 90.0
        assert designs[0].rmsd == 1.0
        assert designs[0].ranking_score == 0.8
        assert designs[0].passed_validation is True
        assert designs[0].plddt_source == "rf3"
        assert designs[0].ranking_source == "rf3"
        assert designs[0].validation_status == "validated"
    finally:
        db.query(design_router.ExperimentDesign).filter(
            design_router.ExperimentDesign.experiment_id == "exp_candidate_fill"
        ).delete()
        db.query(Experiment).filter(Experiment.id == "exp_candidate_fill").delete()
        db.commit()
        db.close()


def test_formal_mpnn_fills_candidate_matching_selected_backbone(monkeypatch):
    design_router = patch_services(monkeypatch)
    db = SessionLocal()
    try:
        exp = Experiment(id="exp_candidate_match", name="Candidate match", status="rfd3_completed")
        db.add(exp)
        db.add_all(
            [
                design_router.ExperimentDesign(
                    id="design_rfd3_first",
                    experiment_id=exp.id,
                    design_name="rfd3_0",
                    sequence="",
                    pdb_content="RFD3_0",
                    plddt=80.0,
                    passed_validation=False,
                    plddt_source="rfd3",
                    ranking_source="none",
                    validation_status="not_validated",
                ),
                design_router.ExperimentDesign(
                    id="design_rfd3_second",
                    experiment_id=exp.id,
                    design_name="rfd3_1",
                    sequence="",
                    pdb_content="RFD3_1",
                    plddt=81.0,
                    passed_validation=False,
                    plddt_source="rfd3",
                    ranking_source="none",
                    validation_status="not_validated",
                ),
            ]
        )
        db.commit()

        mpnn = asyncio.run(
            design_router.run_mpnn(
                design_router.MPNNRequest(
                    backbone_pdb_content="RFD3_1",
                    experiment_id=exp.id,
                )
            )
        )

        db.expire_all()
        first = db.query(design_router.ExperimentDesign).filter_by(id="design_rfd3_first").one()
        second = db.query(design_router.ExperimentDesign).filter_by(id="design_rfd3_second").one()

        assert mpnn["success"] is True
        assert first.sequence == ""
        assert first.pdb_content == "RFD3_0"
        assert second.sequence == "ACD"
        assert second.pdb_content == "MPNN_PDB"
        assert second.ranking_score == -1.0
    finally:
        db.query(design_router.ExperimentDesign).filter(
            design_router.ExperimentDesign.experiment_id == "exp_candidate_match"
        ).delete()
        db.query(Experiment).filter(Experiment.id == "exp_candidate_match").delete()
        db.commit()
        db.close()
