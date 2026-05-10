import unittest
from pathlib import Path

from hotspot_prediction.config import DEFAULT_CONFIG
from hotspot_prediction.data.io import read_excel_rows


class IOTests(unittest.TestCase):
    def test_default_config_points_to_hotspot_prediction_directory(self):
        self.assertEqual(DEFAULT_CONFIG.project_root.name, "hotspot-prediction")
        self.assertTrue(DEFAULT_CONFIG.data1_file.name.endswith("data1-v1.xlsx"))

    def test_read_excel_rows_without_pandas_reads_source_headers(self):
        rows = read_excel_rows(Path("hotspot-prediction") / "elife-96643-data1-v1.xlsx")

        self.assertGreater(len(rows), 0)
        self.assertIn("PDB ID", rows[0])
        self.assertIn("Binding free energy change (kcal/mol)", rows[0])


if __name__ == "__main__":
    unittest.main()
