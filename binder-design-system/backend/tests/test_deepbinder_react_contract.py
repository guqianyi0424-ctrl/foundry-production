from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


def read_project_file(relative_path: str) -> str:
    return (APP_ROOT / relative_path).read_text(encoding="utf-8")


def test_legacy_streamlit_and_docker_surfaces_are_removed():
    removed_paths = [
        "app/main.py",
        "app/binder_app.py",
        "Dockerfile",
        "backend/Dockerfile",
        "frontend/Dockerfile",
        "docker-compose.yml",
    ]

    assert all(not (APP_ROOT / path).exists() for path in removed_paths)


def test_frontend_uses_deepbinder_brand_and_storage_keys():
    assert "DeepBinder - 蛋白质Binder设计系统" in read_project_file("frontend/index.html")
    assert '"name": "deepbinder-frontend"' in read_project_file("frontend/package.json")

    store = read_project_file("frontend/src/store/useAppStore.ts")
    api = read_project_file("frontend/src/api/index.ts")

    assert "deepbinder_token" in store
    assert "deepbinder_user" in store
    assert "LEGACY_AUTH_TOKEN_KEY" in store
    assert "deepbinder_token" in api


def test_hotspot_prediction_updates_sequence_dots_before_confirmation():
    page = read_project_file("frontend/src/pages/NewDesignPage.tsx")
    sequence_viewer = read_project_file("frontend/src/components/SequenceViewer/index.tsx")

    prediction_call = "const hotspots = res.hotspots.map(h => ({ chain: h.chain, residue: h.residue, score: h.score }))"
    assert prediction_call in page
    assert "setSelectedHotspots(hotspots)" in page
    assert page.index("setSelectedHotspots(hotspots)") < page.index("setPredictionResult(res)")

    assert "seq-hotspot-dot" in sequence_viewer
    assert "absolute bottom-0.5 left-1/2" in sequence_viewer
    assert "bg-red-500" in sequence_viewer


def test_molstar_keeps_persistent_highlights_separate_from_camera_reset():
    viewer = read_project_file("frontend/src/components/MolstarViewer/index.tsx")

    assert "const highlightLoci = (loci: any, color?: number)" in viewer
    assert "highlightLoci(rangeLoci, 0xF59E0B)" in viewer
    assert "highlightLoci(hotspotLoci, 0xEF4444)" in viewer
    assert "highlightLoci(hoverLoci, 0x60A5FA)" in viewer

    reset_body = viewer.split("const handleResetCamera = () => {", 1)[1].split("}", 1)[0]
    assert "clearHighlights" not in reset_body
