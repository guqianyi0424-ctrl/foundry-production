import unittest

from hotspot_prediction.data.dataset import GraphSample, collate_samples
from hotspot_prediction.features.graph import build_distance_edges


class GraphDatasetTests(unittest.TestCase):
    def test_build_distance_edges_uses_cutoff_and_bidirectional_edges(self):
        coords = [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [20.0, 0.0, 0.0],
        ]

        edge_index = build_distance_edges(coords, cutoff=6.0)

        self.assertEqual(edge_index, ([0, 1], [1, 0]))

    def test_collate_samples_keeps_labels_aligned_with_concatenated_nodes(self):
        first = GraphSample(
            pdb_id="1aaa_A",
            sequence="AC",
            node_features=[[1.0, 0.0], [0.0, 1.0]],
            labels=[0, 1],
            edge_index=([0, 1], [1, 0]),
            residue_numbers=[10, 11],
        )
        second = GraphSample(
            pdb_id="2bbb_B",
            sequence="DEF",
            node_features=[[2.0, 0.0], [2.0, 1.0], [2.0, 2.0]],
            labels=[1, 0, 0],
            edge_index=([0], [1]),
            residue_numbers=[20, 21, 22],
        )

        batch = collate_samples([first, second])

        self.assertEqual(batch["pdb_ids"], ["1aaa_A", "2bbb_B"])
        self.assertEqual(batch["node_features"], [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
        self.assertEqual(batch["labels"], [0, 1, 1, 0, 0])
        self.assertEqual(batch["node_slices"], [(0, 2), (2, 5)])
        self.assertEqual(batch["edge_index"], ([0, 1, 2], [1, 0, 3]))


if __name__ == "__main__":
    unittest.main()
