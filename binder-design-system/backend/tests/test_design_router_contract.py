import asyncio

from database import SessionLocal, Experiment
from schemas.domain import AdapterResult, PipelineResult


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
    def run_pipeline(self, pdb_content, hotspots, binder_length, job_id, user_id=None):
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

    data = asyncio.run(
        design_router.run_pipeline(
            design_router.PipelineRequest(
                pdb_content="ATOM",
                hotspots=[{"chain": "A", "residue": 10}],
                binder_length=80,
            )
        )
    )

    assert data["experiment_id"] == "exp_1"
    assert data["status"] == "completed"
    assert data["rfd3_results"]["success"] is True
    assert data["mpnn_results"]["success"] is True
    assert data["rf3_results"]["success"] is True


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
