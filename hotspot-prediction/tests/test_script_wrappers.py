import importlib.util
from pathlib import Path
import unittest


def load_script(name):
    script_path = Path("hotspot-prediction") / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptWrapperTests(unittest.TestCase):
    def test_train_wrapper_passes_model_and_fold_options(self):
        module = load_script("train.py")

        cmd = module.build_command(
            root=Path("/repo/hotspot-prediction"),
            model="mlp",
            n_folds=3,
            force_reload=True,
        )

        self.assertIn("--model", cmd)
        self.assertIn("mlp", cmd)
        self.assertIn("--n-folds", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--reprocess", cmd)

    def test_evaluate_wrapper_can_use_validation_threshold_source(self):
        module = load_script("evaluate_test.py")

        cmd = module.build_command(
            root=Path("/repo/hotspot-prediction"),
            model="mlp",
            threshold="0.5",
            threshold_source="validation-mcc",
            find_threshold=False,
            force_reload=False,
            predictions_csv="results/mlp/test_predictions.csv",
            metrics_csv="results/mlp/test_metrics.csv",
        )

        self.assertIn("--model", cmd)
        self.assertIn("mlp", cmd)
        self.assertIn("--threshold-source", cmd)
        self.assertIn("validation-mcc", cmd)
        self.assertNotIn("--find-threshold", cmd)


if __name__ == "__main__":
    unittest.main()
