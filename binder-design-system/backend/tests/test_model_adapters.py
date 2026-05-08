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
