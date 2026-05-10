from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hotspot_prediction.evaluation_exports import export_metrics_table, export_prediction_table


class EvaluateExportTests(unittest.TestCase):
    def test_export_prediction_and_metric_tables_writes_expected_csvs(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            export_prediction_table(
                output_dir / "test_predictions.csv",
                pdb_ids=["1aaa_A", "1aaa_A", "2bbb_B"],
                labels=[1, 0, 1],
                probs=[0.8, 0.2, 0.9],
            )
            export_metrics_table(
                output_dir / "test_metrics.csv",
                metrics={"roc_auc": 0.8, "pr_auc": 0.4, "f1": 0.5},
                balanced_metrics={"f1": 0.6},
                threshold=0.5,
            )

            prediction_text = (output_dir / "test_predictions.csv").read_text(encoding="utf-8")
            metrics_text = (output_dir / "test_metrics.csv").read_text(encoding="utf-8")

        self.assertIn("pdb_id,label,prob", prediction_text.splitlines()[0])
        self.assertIn("1aaa_A,1,0.8", prediction_text)
        self.assertIn("threshold,roc_auc,pr_auc,f1,balanced_f1", metrics_text.splitlines()[0])
        self.assertIn("0.5,0.8,0.4,0.5,0.6", metrics_text)


if __name__ == "__main__":
    unittest.main()
