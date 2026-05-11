import unittest

from hotspot_prediction.training.sampling import balanced_residue_indices


class MaskedTrainingTests(unittest.TestCase):
    def test_balanced_residue_indices_excludes_unknown_labels(self):
        labels = [1, 0, -1, 0, 1, -1]

        indices = balanced_residue_indices(labels, negative_ratio=1, seed=7)

        self.assertEqual(set(indices), {0, 1, 3, 4})
        self.assertNotIn(2, indices)
        self.assertNotIn(5, indices)

    def test_balanced_residue_indices_samples_ratio_limited_negatives(self):
        labels = [1, 0, 0, 0, 0, 1, 0, -1]

        indices = balanced_residue_indices(labels, negative_ratio=2, seed=3)
        selected_labels = [labels[index] for index in indices]

        self.assertEqual(selected_labels.count(1), 2)
        self.assertEqual(selected_labels.count(0), 4)
        self.assertNotIn(-1, selected_labels)

    def test_balanced_residue_indices_returns_all_valid_when_one_class_missing(self):
        labels = [-1, 0, 0, -1]

        indices = balanced_residue_indices(labels, negative_ratio=1, seed=1)

        self.assertEqual(indices, [1, 2])


if __name__ == "__main__":
    unittest.main()
