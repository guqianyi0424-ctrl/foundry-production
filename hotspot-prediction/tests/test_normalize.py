import unittest

from hotspot_prediction.data.normalize import (
    build_data1_train_samples,
    build_data2_test_samples,
    deduplicate_data2_rows,
    exclude_overlapping_train_samples,
    label_from_ddg,
    normalize_data1_rows,
    normalize_data2_rows,
    parse_hotspot_list,
    parse_pdb_chain_range,
    uniprot_to_pdb_positions,
)


class NormalizeTests(unittest.TestCase):
    def test_data1_label_uses_baseline_ddg_threshold_inclusively(self):
        self.assertEqual(label_from_ddg("2.01"), 1)
        self.assertEqual(label_from_ddg(2.0), 1)
        self.assertEqual(label_from_ddg("1.99"), 0)

    def test_parse_pdb_chain_range(self):
        parsed = parse_pdb_chain_range("5f18-A(374-620)")
        self.assertEqual(parsed.pdb_id, "5f18")
        self.assertEqual(parsed.chain_id, "A")
        self.assertEqual(parsed.uniprot_start, 374)
        self.assertEqual(parsed.uniprot_end, 620)

    def test_uniprot_to_pdb_positions_uses_one_based_relative_numbering(self):
        self.assertEqual(uniprot_to_pdb_positions([503, 504, 506], 374), [130, 131, 133])

    def test_parse_hotspot_list_ignores_non_numeric_tokens(self):
        self.assertEqual(parse_hotspot_list("503, 504, bad,506"), [503, 504, 506])

    def test_normalize_data1_rows_uses_ddg_even_when_assignment_disagrees(self):
        rows = [
            {
                "Uniprot code": "P21836",
                "Uniprot code of the binding partner": "P0C1Z0",
                "PDB ID": "1c2b-A",
                "Uniprot # of PPI-hot spots": "328",
                "PDB # of PPI-hot spots": "297",
                "wild->mutant": "F->I",
                "Binding free energy change (kcal/mol)": "1.98",
                "PPI-hot spots  assignment": "P",
            }
        ]

        normalized = normalize_data1_rows(rows)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["pdb_id"], "1c2b")
        self.assertEqual(normalized[0]["chain_id"], "A")
        self.assertEqual(normalized[0]["pdb_residue"], 297)
        self.assertEqual(normalized[0]["wild_type"], "F")
        self.assertEqual(normalized[0]["mutant"], "I")
        self.assertEqual(normalized[0]["label"], 0)

    def test_normalize_data1_rows_skips_rows_without_numeric_ddg(self):
        rows = [
            {
                "Uniprot code": "P21836",
                "PDB ID": "1c2b-A",
                "PDB # of PPI-hot spots": "297",
                "Binding free energy change (kcal/mol)": "n/a",
            }
        ]

        self.assertEqual(normalize_data1_rows(rows), [])

    def test_normalize_data2_rows_converts_uniprot_hotspots(self):
        rows = [
            {
                "Uniprot code of protein A": "O15118",
                "Uniprot code of the binding partner, protein B": "P87666",
                "PDB ID of free protein A structure(length in Uniprot)": "5f18-A(374-620)",
                "PPI-hot spots (Uniprot numbering)": "503,504,506",
            }
        ]

        normalized = normalize_data2_rows(rows)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["pdb_id"], "5f18")
        self.assertEqual(normalized[0]["chain_id"], "A")
        self.assertEqual(normalized[0]["hotspots_pdb"], [130, 131, 133])

    def test_build_data1_train_samples_aggregates_hotspots_by_chain(self):
        mutations = [
            {"uniprot_id": "P1", "partner_id": "P2", "pdb_id": "1abc", "chain_id": "A", "pdb_residue": 10, "label": 1},
            {"uniprot_id": "P1", "partner_id": "P2", "pdb_id": "1abc", "chain_id": "A", "pdb_residue": 11, "label": 0},
            {"uniprot_id": "P1", "partner_id": "P2", "pdb_id": "1abc", "chain_id": "A", "pdb_residue": 12, "label": 1},
        ]

        samples = build_data1_train_samples(mutations)

        self.assertEqual(samples, [
            {
                "uniprot_id": "P1",
                "partner_id": "P2",
                "pdb_id": "1abc-A",
                "hotspot_list": "10,12",
                "n_hotspots": 2,
                "source": "data1",
            }
        ])

    def test_build_data2_test_samples_exports_legacy_hotspot_list_column(self):
        normalized = [
            {
                "uniprot_id": "O15118",
                "partner_id": "P87666",
                "pdb_id": "5f18",
                "chain_id": "A",
                "hotspots_pdb": [130, 131, 133],
            }
        ]

        samples = build_data2_test_samples(normalized)

        self.assertEqual(samples[0]["pdb_id"], "5f18-A")
        self.assertEqual(samples[0]["hotspot_list"], "130,131,133")

    def test_build_data2_test_samples_deduplicates_same_pdb_chain(self):
        normalized = [
            {
                "uniprot_id": "O15118",
                "partner_id": "P87666",
                "pdb_id": "5f18",
                "chain_id": "A",
                "hotspots_pdb": [130, 131, 133],
            },
            {
                "uniprot_id": "O15118",
                "partner_id": "OTHER",
                "pdb_id": "5f18",
                "chain_id": "A",
                "hotspots_pdb": [133, 130],
            },
        ]

        samples = build_data2_test_samples(normalized)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["hotspot_list"], "130,131,133")

    def test_deduplicate_data2_rows_removes_exact_repeated_source_records(self):
        normalized = [
            {
                "uniprot_id": "O15118",
                "partner_id": "P87666",
                "pdb_id": "5f18",
                "chain_id": "A",
                "uniprot_start": 374,
                "uniprot_end": 620,
                "hotspots_uniprot": [503, 504, 506],
                "hotspots_pdb": [130, 131, 133],
                "source": "data2",
            },
            {
                "uniprot_id": "O15118",
                "partner_id": "P87666",
                "pdb_id": "5f18",
                "chain_id": "A",
                "uniprot_start": 374,
                "uniprot_end": 620,
                "hotspots_uniprot": [503, 504, 506],
                "hotspots_pdb": [130, 131, 133],
                "source": "data2",
            },
            {
                "uniprot_id": "O15118",
                "partner_id": "P87666",
                "pdb_id": "5f18",
                "chain_id": "A",
                "uniprot_start": 374,
                "uniprot_end": 620,
                "hotspots_uniprot": [503, 506],
                "hotspots_pdb": [130, 133],
                "source": "data2",
            },
        ]

        deduped, removed = deduplicate_data2_rows(normalized)

        self.assertEqual(removed, 1)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["hotspots_pdb"], [130, 131, 133])

    def test_exclude_overlapping_train_samples_removes_test_uniprot_and_pdb_chain(self):
        train_samples = [
            {"uniprot_id": "P1", "pdb_id": "1aaa-A", "hotspot_list": "10"},
            {"uniprot_id": "P2", "pdb_id": "2bbb-B", "hotspot_list": "20"},
            {"uniprot_id": "P3", "pdb_id": "3ccc-C", "hotspot_list": "30"},
        ]
        test_samples = [
            {"uniprot_id": "P1", "pdb_id": "9zzz-Z", "hotspot_list": "1"},
            {"uniprot_id": "PX", "pdb_id": "2bbb-B", "hotspot_list": "2"},
        ]

        filtered, removed = exclude_overlapping_train_samples(train_samples, test_samples)

        self.assertEqual(filtered, [{"uniprot_id": "P3", "pdb_id": "3ccc-C", "hotspot_list": "30"}])
        self.assertEqual(removed, 2)

    def test_exclude_overlapping_train_samples_can_remove_only_pdb_chain(self):
        train_samples = [
            {"uniprot_id": "P1", "pdb_id": "1aaa-A", "hotspot_list": "10"},
            {"uniprot_id": "P2", "pdb_id": "2bbb-B", "hotspot_list": "20"},
        ]
        test_samples = [
            {"uniprot_id": "P1", "pdb_id": "9zzz-Z", "hotspot_list": "1"},
        ]

        filtered, removed = exclude_overlapping_train_samples(
            train_samples,
            test_samples,
            policy="pdb-chain",
        )

        self.assertEqual(filtered, train_samples)
        self.assertEqual(removed, 0)


if __name__ == "__main__":
    unittest.main()
