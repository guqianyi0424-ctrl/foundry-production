import numpy as np
import sys
import types


def test_rfd3_plddt_returns_none_when_output_has_no_confidence_signal():
    from utils.rfd3_runner import RFD3Runner

    atom_array = type("AtomArray", (), {})()
    atom_array.b_factor = np.array([0.0, 0.0, 0.0])
    atom_array.atom_name = np.array(["CA", "CA", "CA"])

    assert RFD3Runner()._extract_plddt(atom_array) is None


def test_rf3_plddt_normalization_scales_fractional_confidence_to_percent():
    from utils.rf3_runner import RF3Runner

    runner = RF3Runner()

    assert runner._normalize_plddt_value(0.873) == 87.3
    assert runner._normalize_plddt_value(87.3) == 87.3
    assert runner._normalize_plddt_list([0.8, 0.9]) == [80.0, 90.0]


def test_rf3_runner_reuses_inference_engine(monkeypatch):
    class FakeRF3InferenceEngine:
        instances = []

        def __init__(self, ckpt_path, verbose):
            self.ckpt_path = ckpt_path
            self.verbose = verbose
            FakeRF3InferenceEngine.instances.append(self)

    fake_rf3_module = types.ModuleType("rf3.inference_engines.rf3")
    fake_rf3_module.RF3InferenceEngine = FakeRF3InferenceEngine
    monkeypatch.setitem(sys.modules, "rf3.inference_engines.rf3", fake_rf3_module)

    from utils.rf3_runner import RF3Runner

    runner = RF3Runner()
    runner._engine()
    runner._engine()

    assert len(FakeRF3InferenceEngine.instances) == 1
