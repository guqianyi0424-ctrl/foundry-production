import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dataset import assign_residue_labels
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    assign_residue_labels = None


class LegacyDatasetLabelTests(unittest.TestCase):
    def test_assign_residue_labels_masks_unobserved_residues(self):
        if assign_residue_labels is None:
            self.skipTest("numpy is not installed in this lightweight test environment")
        protein_data = {
            "sequence": "ACDEF",
            "residue_indices": [10, 11, 12, 13, 14],
        }

        labels = assign_residue_labels(protein_data, hotspots=[13], nonhotspots=[10, 14])

        self.assertEqual(labels.tolist(), [0, -1, -1, 1, 0])

    def test_assign_residue_labels_can_keep_unobserved_as_negatives_for_test_compatibility(self):
        if assign_residue_labels is None:
            self.skipTest("numpy is not installed in this lightweight test environment")
        protein_data = {
            "sequence": "ACD",
            "residue_indices": [1, 2, 3],
        }

        labels = assign_residue_labels(
            protein_data,
            hotspots=[2],
            nonhotspots=[],
            unknown_unobserved=False,
        )

        self.assertEqual(labels.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
