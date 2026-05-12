import csv
import json


def test_experiment_archive_writes_standard_artifacts(tmp_path):
    from services.experiment_archive import ExperimentArchiveService

    service = ExperimentArchiveService(tmp_path)
    archive = service.write_archive(
        experiment={
            "id": "exp_1",
            "name": "Pipeline_20260512_120000",
            "status": "completed",
            "created_at": "2026-05-12T12:00:00",
            "target": "A/1-20",
            "hotspots": [{"chain": "A", "residue": 10}],
            "rfd3_config": {"binder_length": 80},
            "mpnn_config": {"batch_size": 2},
            "rf3_config": {"example_id": "binder_design"},
            "duration_seconds": 12.5,
            "gpu_info": "A100",
        },
        results={
            "rfd3": {"success": True},
            "mpnn": {"success": True},
            "rf3": {"success": True, "avg_plddt": 91.2, "rmsd": 1.1, "passed": True},
        },
        designs=[
            {
                "id": "design_1",
                "design_name": "candidate_001",
                "sequence": "ACDE",
                "pdb_content": "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n",
                "plddt": 91.2,
                "rmsd": 1.1,
                "ranking_score": 0.87,
                "passed_validation": True,
            }
        ],
    )

    archive_dir = tmp_path / "experiments" / "exp_1"
    assert archive["archive_dir"] == str(archive_dir)
    assert (archive_dir / "manifest.json").is_file()
    assert (archive_dir / "candidates.csv").is_file()
    assert (archive_dir / "candidates.json").is_file()
    assert (archive_dir / "sequences.fasta").is_file()
    assert (archive_dir / "structures" / "candidate_001.pdb").is_file()
    assert (archive_dir / "raw_results" / "rfd3_results.json").is_file()
    assert (archive_dir / "raw_results" / "mpnn_results.json").is_file()
    assert (archive_dir / "raw_results" / "rf3_results.json").is_file()

    manifest = json.loads((archive_dir / "manifest.json").read_text())
    assert manifest["experiment"]["id"] == "exp_1"
    assert manifest["artifact_files"]["candidates_csv"] == "candidates.csv"

    with (archive_dir / "candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "experiment_id": "exp_1",
            "candidate_id": "design_1",
            "design_name": "candidate_001",
            "sequence": "ACDE",
            "sequence_length": "4",
            "plddt": "91.2",
            "rmsd": "1.1",
            "ranking_score": "0.87",
            "passed_validation": "true",
            "has_pdb": "true",
            "pdb_file": "structures/candidate_001.pdb",
            "created_at": "2026-05-12T12:00:00",
        }
    ]

    assert (archive_dir / "sequences.fasta").read_text() == ">candidate_001\nACDE\n"
