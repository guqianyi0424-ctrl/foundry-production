import unittest

from hotspot_prediction.training.checkpoints import load_checkpoint
from hotspot_prediction.training.thresholds import best_mcc_threshold, matthews_corrcoef


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


if __name__ == "__main__":
    unittest.main()
