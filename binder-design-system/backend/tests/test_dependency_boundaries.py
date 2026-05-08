from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


def test_only_foundry_bootstrap_mutates_sys_path_for_model_sources():
    checked_files = [
        APP_ROOT / "backend" / "routers" / "design.py",
        APP_ROOT / "backend" / "routers" / "upload.py",
        APP_ROOT / "utils" / "rfd3_runner.py",
        APP_ROOT / "utils" / "mpnn_runner.py",
        APP_ROOT / "utils" / "rf3_runner.py",
    ]

    for path in checked_files:
        assert "sys.path.insert" not in path.read_text()


def test_foundry_bootstrap_is_the_allowed_source_path_boundary():
    path = APP_ROOT / "backend" / "adapters" / "foundry_bootstrap.py"
    text = path.read_text()

    assert "sys.path.insert" in text
    assert "models/rfd3/src" in text
    assert "models/mpnn/src" in text
    assert "models/rf3/src" in text


def test_utils_package_init_does_not_import_model_runtime_modules():
    text = (APP_ROOT / "utils" / "__init__.py").read_text()

    assert "hotspot_predictor" not in text
    assert "rfd3_runner" not in text
    assert "mpnn_runner" not in text
    assert "rf3_runner" not in text
