import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hotspot_prediction.reporting import (
    build_ablation_plan,
    build_dataset_summary,
    build_experiment_matrix,
    collect_ablation_results,
    compute_topk_metrics,
    write_table,
)


class ReportingTests(unittest.TestCase):
    def test_build_dataset_summary_reports_train_test_and_strict_audit(self):
        data1_mutations = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
        ]
        train_samples = [
            {"pdb_id": "1aaa-A", "hotspot_list": "10,11", "n_hotspots": 2},
            {"pdb_id": "2bbb-B", "hotspot_list": "", "n_hotspots": 0},
        ]
        data2_rows = [{"pdb_id": "x"}, {"pdb_id": "y"}, {"pdb_id": "z"}]
        test_samples = [
            {"pdb_id": "3ccc-C", "hotspot_list": "1", "n_hotspots": 1},
        ]
        strict_samples = [
            {"pdb_id": "2bbb-B", "hotspot_list": "", "n_hotspots": 0},
        ]

        rows = build_dataset_summary(
            data1_mutations=data1_mutations,
            train_samples=train_samples,
            data2_rows=data2_rows,
            test_samples=test_samples,
            strict_train_samples=strict_samples,
        )

        self.assertEqual(rows[0]["dataset"], "data1")
        self.assertEqual(rows[0]["records"], 3)
        self.assertEqual(rows[0]["hotspots"], 2)
        self.assertEqual(rows[1]["pdb_chains"], 2)
        self.assertEqual(rows[1]["hotspots"], 2)
        self.assertEqual(rows[2]["records"], 3)
        self.assertEqual(rows[3]["pdb_chains"], 1)
        self.assertEqual(rows[4]["dataset"], "strict_non_overlap")

    def test_topk_metrics_are_grouped_per_protein(self):
        records = [
            {"pdb_id": "A", "label": 1, "prob": 0.9},
            {"pdb_id": "A", "label": 0, "prob": 0.8},
            {"pdb_id": "A", "label": 1, "prob": 0.1},
            {"pdb_id": "B", "label": 0, "prob": 0.9},
            {"pdb_id": "B", "label": 1, "prob": 0.7},
        ]

        rows = compute_topk_metrics(records, k_values=[1, 2])

        self.assertEqual(rows[0]["top_k"], 1)
        self.assertEqual(rows[0]["proteins_with_hotspots"], 2)
        self.assertEqual(rows[0]["proteins_hit"], 1)
        self.assertAlmostEqual(rows[0]["topk_hit_rate"], 0.5)
        self.assertAlmostEqual(rows[0]["hotspot_recall"], 1 / 3)
        self.assertEqual(rows[1]["proteins_hit"], 2)
        self.assertAlmostEqual(rows[1]["topk_hit_rate"], 1.0)

    def test_experiment_matrix_contains_expected_methods(self):
        rows = build_experiment_matrix()
        methods = [row["method"] for row in rows]

        self.assertEqual(
            methods,
            [
                "Rule-based baseline",
                "Traditional ML / PPI-hotspotID-style",
                "ESM-2 embedding + MLP",
                "ESM-2 + GAT",
                "ESM-2 + GAT ensemble",
            ],
        )

    def test_ablation_plan_contains_isolated_cloud_profiles(self):
        rows = build_ablation_plan()
        profiles = [row["profile"] for row in rows]

        self.assertIn("esm2_mlp", profiles)
        self.assertIn("esm2_only", profiles)
        self.assertIn("esm2_pssm_hmm", profiles)
        for row in rows:
            self.assertIn("train_args", row)
            self.assertIn("HOTSPOT_FEATURES_DIR", row["cloud_env"])
            self.assertIn("HOTSPOT_MODELS_DIR", row["cloud_env"])
            self.assertIn("HOTSPOT_RESULTS_DIR", row["cloud_env"])

    def test_collect_ablation_results_marks_completed_and_pending_profiles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_dir = root / "esm2_only"
            completed_dir.mkdir()
            write_table(
                completed_dir / "cross_validation_results.csv",
                [
                    {"roc_auc": 0.7, "pr_auc": 0.4, "f1": 0.5, "mcc": 0.2},
                    {"roc_auc": 0.9, "pr_auc": 0.6, "f1": 0.7, "mcc": 0.4},
                ],
            )
            write_table(
                completed_dir / "test_metrics.csv",
                [
                    {
                        "roc_auc": 0.8,
                        "pr_auc": 0.5,
                        "f1": 0.6,
                        "precision": 0.55,
                        "recall": 0.65,
                        "mcc": 0.3,
                    }
                ],
            )
            write_table(
                completed_dir / "topk_metrics.csv",
                [
                    {"top_k": 3, "topk_hit_rate": 0.4, "hotspot_recall": 0.2},
                    {"top_k": 5, "topk_hit_rate": 0.5, "hotspot_recall": 0.3},
                ],
            )

            rows = collect_ablation_results(root)
            by_profile = {row["profile"]: row for row in rows}

            self.assertEqual(by_profile["esm2_only"]["status"], "complete")
            self.assertAlmostEqual(float(by_profile["esm2_only"]["cv_roc_auc_mean"]), 0.8)
            self.assertAlmostEqual(float(by_profile["esm2_only"]["test_f1"]), 0.6)
            self.assertAlmostEqual(float(by_profile["esm2_only"]["top3_hit_rate"]), 0.4)
            self.assertEqual(by_profile["esm2_pssm"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
