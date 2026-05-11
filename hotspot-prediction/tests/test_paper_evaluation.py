import unittest

from hotspot_prediction.evaluation.paper import resolve_test_threshold


class PaperEvaluationTests(unittest.TestCase):
    def test_resolve_fixed_threshold(self):
        threshold = resolve_test_threshold(
            source="fixed",
            fixed_threshold=0.5,
            cv_rows=[{"optimal_threshold": "0.2"}],
        )

        self.assertEqual(threshold, 0.5)

    def test_resolve_validation_mean_threshold_ignores_test_labels(self):
        threshold = resolve_test_threshold(
            source="validation-mcc",
            fixed_threshold=0.5,
            cv_rows=[
                {"optimal_threshold": "0.2"},
                {"optimal_threshold": "0.4"},
                {"optimal_threshold": ""},
            ],
        )

        self.assertAlmostEqual(threshold, 0.3)


if __name__ == "__main__":
    unittest.main()
