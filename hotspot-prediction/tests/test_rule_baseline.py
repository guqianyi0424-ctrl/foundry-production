import unittest

from hotspot_prediction.baselines.rule_based import (
    predict_rule_based_records,
    score_residue,
)


class RuleBaselineTests(unittest.TestCase):
    def test_score_residue_prioritizes_hotspot_like_residues(self):
        high_score = score_residue("W", residue_index=5, sequence_length=20)
        low_score = score_residue("A", residue_index=5, sequence_length=20)

        self.assertGreater(high_score, low_score)

    def test_predict_rule_based_records_emits_residue_level_probabilities(self):
        samples = [
            {
                "pdb_id": "1abc_A",
                "sequence": "AWDE",
                "residue_indices": [10, 11, 12, 13],
                "labels": [0, 1, 0, 1],
            }
        ]

        records = predict_rule_based_records(samples)

        self.assertEqual(len(records), 4)
        self.assertEqual(records[1]["pdb_id"], "1abc_A")
        self.assertEqual(records[1]["residue_id"], 11)
        self.assertEqual(records[1]["label"], 1)
        self.assertGreaterEqual(records[1]["prob"], 0.0)
        self.assertLessEqual(records[1]["prob"], 1.0)


if __name__ == "__main__":
    unittest.main()
