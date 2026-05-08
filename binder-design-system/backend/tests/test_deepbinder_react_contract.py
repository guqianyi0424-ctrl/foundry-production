from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


def read_project_file(relative_path: str) -> str:
    return (APP_ROOT / relative_path).read_text(encoding="utf-8")


def test_legacy_ui_and_container_surfaces_are_removed():
    container_file = "Dock" + "erfile"
    container_compose = "dock" + "er-compose.yml"
    removed_paths = [
        "app/main.py",
        "app/binder_app.py",
        container_file,
        f"backend/{container_file}",
        f"frontend/{container_file}",
        container_compose,
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


def test_hotspot_prediction_marks_sequence_without_committing_configuration():
    page = read_project_file("frontend/src/pages/NewDesignPage.tsx")
    sequence_viewer = read_project_file("frontend/src/components/SequenceViewer/index.tsx")
    design_panel = read_project_file("frontend/src/components/DesignPanel/index.tsx")
    store = read_project_file("frontend/src/store/useAppStore.ts")

    prediction_call = "const hotspots = res.hotspots.map(h => ({ chain: h.chain, residue: h.residue, score: h.score }))"
    assert prediction_call in page
    assert "setPredictedHotspots(hotspots)" in page
    assert "setSelectedHotspots(hotspots)" not in page
    assert "setPredictedHotspots(hotspots)" in page
    assert page.index("setPredictedHotspots(hotspots)") < page.index("setPredictionResult(res)")
    assert "预测热点已在序列下方以红点标记；双击可移除，点击「指定热点」才会写入热点参数框。" in page
    assert "handleConfirmPrediction" not in page
    assert "const pendingHotspots = mergeHotspotSelections(predictedHotspots, selectedHotspots)" in page
    assert "const hotspotStr = pendingHotspots.map(h => `${h.chain}/${h.residue}`).join(', ')" in page
    assert "hotspots: committedHotspotTokens.length > 0 ? committedHotspotTokens : undefined" in page

    assert "predictedHotspots: HotspotResidue[]" in store
    assert "setPredictedHotspots: (hotspots: HotspotResidue[]) => void" in store
    assert "removePredictedHotspot: (chain: string, residue: number) => void" in store
    assert "set({ selectedRange: range })" in store
    assert "set({ selectedRange: null, selectedHotspots: [], hotspotInput: '' })" not in store
    assert "filter(h => h.chain !== range.chain)" not in store

    assert "useEffect" not in design_panel
    assert "hotspotInput" not in design_panel

    assert "seq-hotspot-dot" in sequence_viewer
    assert "z-20" in sequence_viewer
    assert "bg-red-500" in sequence_viewer
    assert "onDoubleClick" in sequence_viewer
    assert "handleDoubleClick(chain.chain_id, resSeq)" in sequence_viewer
    assert "removePredictedHotspot(chainId, resSeq)" in sequence_viewer
    assert "addHotspot({ chain: chainId, residue: r, score: getHotspotScore(chainId, r) ?? 0 })" in sequence_viewer
    assert "toggleHotspot" not in sequence_viewer
    assert "预测红点为候选热点，双击可移除；点击「指定热点」写入配置。" in sequence_viewer


def test_molstar_keeps_persistent_highlights_separate_from_camera_reset():
    viewer = read_project_file("frontend/src/components/MolstarViewer/index.tsx")

    assert "plugin.managers.structure.selection.clear()" in viewer
    assert "plugin.managers.structure.selection.fromLoci" in viewer
    assert "combinedSelectionLoci" in viewer
    assert "highlightLoci(rangeLoci, 0xF59E0B)" in viewer
    assert "highlightLoci(hotspotLoci, 0xEF4444)" in viewer
    assert "highlightLoci(hoverLoci, 0x60A5FA)" in viewer

    reset_body = viewer.split("const handleResetCamera = () => {", 1)[1].split("}", 1)[0]
    assert "clearHighlights" not in reset_body
