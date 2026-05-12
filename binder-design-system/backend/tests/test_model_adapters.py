class FakeRFD3Runner:
    def run_rfd3(self, **kwargs):
        return {
            "success": True,
            "designs": [
                {
                    "index": 0,
                    "name": "d0",
                    "pdb_content": "ATOM",
                    "pdb_path": "d0.pdb",
                    "plddt": 88.0,
                }
            ],
            "first_backbone_pdb": "ATOM",
            "mock": False,
        }


class FakeFailingRunner:
    def run_rfd3(self, **kwargs):
        raise RuntimeError("boom")


class FakeHotspotPredictor:
    def __init__(self):
        self.calls = []

    def predict_hotspots(self, pdb_content, top_k=5):
        self.calls.append((pdb_content, top_k))
        return {
            "method": "dl",
            "model_loaded": True,
            "hotspots_detail": [],
        }


def test_rfd3_adapter_wraps_successful_runner():
    from adapters.model_adapters import RFD3Adapter
    from schemas.domain import RFD3JobConfig

    adapter = RFD3Adapter(runner_factory=lambda: FakeRFD3Runner())
    result = adapter.run(
        RFD3JobConfig(
            pdb_content="ATOM",
            hotspots=["A10"],
            job_id="job_1",
        )
    )

    assert result.success is True
    assert result.data["first_backbone_pdb"] == "ATOM"
    assert result.mock is False


def test_rfd3_adapter_returns_structured_failure():
    from adapters.model_adapters import RFD3Adapter
    from schemas.domain import RFD3JobConfig

    adapter = RFD3Adapter(runner_factory=lambda: FakeFailingRunner())
    result = adapter.run(RFD3JobConfig(pdb_content="ATOM"))

    assert result.success is False
    assert result.error_code == "rfd3_adapter_failed"
    assert "boom" in result.raw_error


def test_hotspot_adapter_reuses_predictor_instance():
    from adapters.model_adapters import HotspotModelAdapter

    created = []

    def factory():
        predictor = FakeHotspotPredictor()
        created.append(predictor)
        return predictor

    adapter = HotspotModelAdapter(predictor_factory=factory)

    first = adapter.predict("ATOM 1", top_k=3)
    second = adapter.predict("ATOM 2", top_k=4)

    assert first.success is True
    assert second.success is True
    assert len(created) == 1
    assert created[0].calls == [("ATOM 1", 3), ("ATOM 2", 4)]


def test_sanitize_model_result_strips_nested_pdb_payloads():
    from adapters.model_adapters import sanitize_model_result

    result = sanitize_model_result(
        {
            "success": True,
            "first_backbone_pdb": "ATOM FIRST",
            "designs": [
                {"name": "d0", "pdb_content": "ATOM D0", "plddt": 90.0},
                {"name": "d1", "pdb_content": "ATOM D1", "plddt": 80.0},
            ],
            "batches": [
                {
                    "batch_idx": 0,
                    "designs": [
                        {"name": "d0", "pdb_content": "ATOM D0", "plddt": 90.0},
                    ],
                }
            ],
        }
    )

    assert result["first_backbone_pdb"] == "<omitted>"
    assert result["designs"][0]["pdb_content"] == "<omitted>"
    assert result["batches"][0]["designs"][0]["pdb_content"] == "<omitted>"
    assert result["designs"][0]["plddt"] == 90.0
