def test_adapter_result_tracks_mock_fallback_reason():
    from schemas.domain import AdapterResult

    result = AdapterResult(
        success=True,
        data={"value": 1},
        mock=True,
        fallback_used=True,
        fallback_reason="foundry import failed",
    )

    assert result.success is True
    assert result.data == {"value": 1}
    assert result.mock is True
    assert result.fallback_used is True
    assert result.fallback_reason == "foundry import failed"


def test_pipeline_result_identifies_failed_step():
    from schemas.domain import AdapterResult, PipelineResult

    result = PipelineResult(
        job_id="job_1",
        experiment_id="exp_1",
        status="failed",
        rfd3=AdapterResult(success=False, error_code="rfd3_failed", message="RFD3 failed"),
        failed_step="rfd3",
    )

    assert result.failed_step == "rfd3"
    assert result.rfd3.success is False
    assert result.mpnn is None
    assert result.rf3 is None
