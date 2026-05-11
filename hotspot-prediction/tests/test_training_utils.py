import unittest

import sys
from pathlib import Path

from hotspot_prediction.training.checkpoints import load_checkpoint
from hotspot_prediction.training.thresholds import best_mcc_threshold, matthews_corrcoef
from hotspot_prediction.training.selection import should_save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TrainingUtilsTests(unittest.TestCase):
    def test_matthews_corrcoef_matches_confusion_counts(self):
        self.assertAlmostEqual(matthews_corrcoef([1, 1, 0, 0], [1, 0, 0, 1]), 0.0)
        self.assertAlmostEqual(matthews_corrcoef([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)

    def test_best_mcc_threshold_uses_validation_probabilities(self):
        labels = [1, 1, 0, 0]
        probs = [0.9, 0.8, 0.7, 0.1]

        threshold, score = best_mcc_threshold(labels, probs, thresholds=[0.5, 0.75, 0.85])

        self.assertEqual(threshold, 0.75)
        self.assertAlmostEqual(score, 1.0)

    def test_load_checkpoint_delegates_to_torch_load_callable(self):
        calls = []

        def fake_torch_load(path, map_location=None):
            calls.append((path, map_location))
            return {"model_state_dict": {"weight": 1}}

        checkpoint = load_checkpoint("model.pth", map_location="cpu", torch_load=fake_torch_load)

        self.assertEqual(checkpoint["model_state_dict"], {"weight": 1})
        self.assertEqual(calls, [("model.pth", "cpu")])

    def test_should_save_checkpoint_uses_auprc_as_primary_score(self):
        self.assertTrue(should_save_checkpoint({"pr_auc": 0.42, "roc_auc": 0.70}, best_score=0.41))
        self.assertFalse(should_save_checkpoint({"pr_auc": 0.40, "roc_auc": 0.99}, best_score=0.41))
        self.assertTrue(should_save_checkpoint({"pr_auc": 0.0}, best_score=None))

    def test_calculate_metrics_handles_single_class_inputs(self):
        from train import calculate_metrics

        result = calculate_metrics([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])

        self.assertEqual(result["TP"], 0)
        self.assertEqual(result["FN"], 0)
        self.assertEqual(result["TN"], 3)
        self.assertEqual(result["FP"], 0)
        self.assertEqual(result["roc_auc"], 0.5)

    def test_train_sampler_limits_negatives_within_each_protein(self):
        import torch
        from train import _sample_valid_indices

        labels = torch.tensor([1, 0, 0, 0, 1, 0, 0, -1])
        torch.manual_seed(11)

        indices = _sample_valid_indices(labels, [(0, 4), (4, 8)], negative_ratio=1)
        selected = labels[indices]

        self.assertEqual(int((selected == 1).sum()), 2)
        self.assertEqual(int((selected == 0).sum()), 2)
        self.assertFalse(bool((selected == -1).any()))
        self.assertEqual(sum(1 for index in indices.tolist() if index < 4 and labels[index] == 0), 1)
        self.assertEqual(sum(1 for index in indices.tolist() if index >= 4 and labels[index] == 0), 1)


if __name__ == "__main__":
    unittest.main()
