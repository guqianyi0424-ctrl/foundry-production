import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest


class RuntimeConfigTests(unittest.TestCase):
    def test_config_accepts_environment_overrides_for_ablation_runs(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "config.py"

        with TemporaryDirectory() as tmp:
            env = {
                "HOTSPOT_PSSM_DIM": "0",
                "HOTSPOT_HMM_DIM": "30",
                "HOTSPOT_TRADITIONAL_DIM": "7",
                "HOTSPOT_MODELS_DIR": str(Path(tmp) / "models" / "esm2_hmm"),
                "HOTSPOT_RESULTS_DIR": str(Path(tmp) / "results" / "esm2_hmm"),
                "HOTSPOT_FEATURES_DIR": str(Path(tmp) / "features" / "esm2_hmm"),
                "HOTSPOT_TRAIN_NEGATIVE_RATIO": "3",
                "HOTSPOT_USE_WEIGHTED_SAMPLER": "1",
                "HOTSPOT_USE_CLASS_WEIGHTS": "1",
                "HOTSPOT_ESM2_MODEL": "facebook/esm2_t6_8M_UR50D",
                "HOTSPOT_ESM2_DIM": "320",
            }
            old_env = {key: os.environ.get(key) for key in env}
            old_torch = sys.modules.get("torch")
            try:
                os.environ.update(env)
                sys.modules["torch"] = types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: False),
                    device=lambda name: name,
                )
                spec = importlib.util.spec_from_file_location("hotspot_runtime_config_test", config_path)
                module = importlib.util.module_from_spec(spec)
                self.assertIsNotNone(spec.loader)
                spec.loader.exec_module(module)
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                if old_torch is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = old_torch

        self.assertEqual(module.PSSM_DIM, 0)
        self.assertEqual(module.HMM_DIM, 30)
        self.assertEqual(module.TRADITIONAL_DIM, 7)
        self.assertEqual(module.ESM2_MODEL, "facebook/esm2_t6_8M_UR50D")
        self.assertEqual(module.ESM2_DIM, 320)
        self.assertEqual(module.INPUT_DIM, module.ESM2_DIM + 30 + 7)
        self.assertTrue(module.MODELS_DIR.endswith("models/esm2_hmm"))
        self.assertTrue(module.RESULTS_DIR.endswith("results/esm2_hmm"))
        self.assertTrue(module.FEATURES_DIR.endswith("features/esm2_hmm"))
        self.assertEqual(module.TRAIN_NEGATIVE_RATIO, 3)
        self.assertTrue(module.USE_WEIGHTED_SAMPLER)
        self.assertTrue(module.USE_CLASS_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
