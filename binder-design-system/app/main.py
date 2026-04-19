"""
蛋白质Binder设计系统 - 主应用
集成: 热点残基预测(DL) + RFD3 + MPNN + RF3 全流程
Top-K=3 | 单字母氨基酸 | RMSD评估筛选 | 序列-3D联动高亮
"""
import sys
import os
import time
import tempfile
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd

from utils.structure_parser import StructureParser
from utils.hotspot_predictor import HotspotPredictor
from utils.rfd3_runner import RFD3Runner
from utils.mpnn_runner import MPNNRunner
from utils.rf3_runner import RF3Runner
from utils.molstar_viewer import render_molstar, render_rmsd_chart, render_plddt_chart

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(
    page_title="蛋白质Binder设计系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

CSS = """
<style>
.seq-container { font-family: 'Courier New', monospace; line-height: 1.8; user-select: none; }
.seq-row { display: flex; align-items: baseline; position: relative; margin-bottom: 2px; }
.seq-residue { display: inline-flex; flex-direction: column; align-items: center;
    width: 28px; height: 32px; cursor: pointer; border-radius: 4px; margin: 0 1px;
    transition: all 0.15s ease; position: relative; }
.seq-letter { font-size: 16px; font-weight: bold; color: #333; line-height: 20px; }
.seq-num { font-size: 9px; color: #999; line-height: 12px; }
.seq-residue:hover { background: #e3f2fd !important; transform: scale(1.15); z-index: 5; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.seq-hotspot { background: #ff5722 !important; color: white !important; }
.seq-hotspot .seq-letter { color: white !important; }
.seq-hotspot .seq-num { color: #ffcdd2 !important; }
.seq-selected { background: #1976d2 !important; color: white !important; }
.seq-selected .seq-letter { color: white !important; }
.seq-selected .seq-num { color: #bbdefb !important; }
.seq-number-top { position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
    font-size: 10px; color: #888; font-family: monospace; }
.upload-area { border: 2px dashed #ccc; border-radius: 8px; padding: 24px; text-align: center;
    transition: border-color 0.3s; cursor: pointer; background: #fafafa; }
.upload-area:hover { border-color: #1976d2; background: #e3f2fd; }
.viewer-wrapper { border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; position: relative; }
.hotspot-tag { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 13px;
    margin: 2px 4px 2px 0; cursor: pointer; transition: transform 0.15s; }
.hotspot-tag:hover { transform: scale(1.08); }
.tag-high { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
.tag-mid { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
.tag-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.step-active { background: #e3f2fd; border-left: 4px solid #1976d2; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-weight: 600; }
.step-done { background: #e8f5e9; border-left: 4px solid #388e3c; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
.step-pending { background: #f5f5f5; border-left: 4px solid #bdbdbd; padding: 8px 12px; margin: 4px 0; border-radius: 4px; color: #757575; }
.metric-card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; text-align: center; }
.pipeline-result { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 8px 0; }
.passed { border-left: 4px solid #4CAF50; }
.failed { border-left: 4px solid #f44336; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TOP_K = 3


def render_html(html_content, height=600):
    try:
        st.html(html_content, unsafe_allow_javascript=True)
    except (AttributeError, TypeError):
        import streamlit.components.v1 as components
        components.html(html_content, height=height)


def init_session():
    defaults = {
        "current_step": 0,
        "atom_array": None,
        "structure_summary": None,
        "hotspot_results_dl": None,
        "selected_hotspots": [],
        "clicked_residues": [],
        "rfd3_results": None,
        "mpnn_results": None,
        "rf3_results": None,
        "job_id": None,
        "pdb_content": None,
        "structure_parser": None,
        "pipeline_running": False,
        "task_type": "蛋白",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


def get_step_status(step):
    current = st.session_state.current_step
    if step < current:
        return "done"
    elif step == current:
        return "active"
    return "pending"


def render_step_bar():
    steps = [
        "① 热点预测 (Top-3)",
        "② RFD3生成",
        "③ MPNN补序",
        "④ RF3验证"
    ]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, steps)):
        status = get_step_status(i)
        css_class = f"step-{status}"
        icon = "✅" if status == "done" else "🔄" if status == "active" else "⏳"
        with col:
            st.markdown(f'<div class="{css_class}">{icon} {label}</div>', unsafe_allow_html=True)


def build_sequence_html(chain_id, residues, hotspot_labels, clicked_labels):
    rows_html = []
    chars_per_row = 40
    n_res = len(residues)

    for start in range(0, n_res, chars_per_row):
        end = min(start + chars_per_row, n_res)
        row_residues = residues[start:end]

        residue_divs = ""
        for idx_offset, res in enumerate(row_residues):
            global_idx = start + idx_offset
            res_name = res.get("res_name", "?")
            one_letter = AA_3TO1.get(res_name, "X")
            res_id = str(res.get("res_id", global_idx + 1))
            label = f"{chain_id}{res_id}"

            is_hotspot = label in hotspot_labels
            is_clicked = label in clicked_labels

            extra_classes = []
            if is_hotspot:
                extra_classes.append("seq-hotspot")
            if is_clicked and not is_hotspot:
                extra_classes.append("seq-selected")

            class_str = " ".join(extra_classes) if extra_classes else ""

            residue_divs += f"""
            <div class="seq-residue {class_str}" data-chain="{chain_id}" data-resid="{res_id}" data-label="{label}"
                 onclick="window.parent.postMessage({{type:'residue_click', chain:'{chain_id}', resid:{res_id}, label:'{label}'}}, '*')">
                <div class="seq-letter">{one_letter}</div>
                <div class="seq-num">{res_id}</div>
            </div>"""

        show_top_num = (start % (chars_per_row * 3) == 0) or (start == 0)
        num_marker = ""
        if show_top_num or end == n_res:
            first_res_id = residues[start].get("res_id", start + 1)
            marker_pos = min(5, len(row_residues) - 1)
            num_marker = f'<span class="seq-number-top">{first_res_id}</span>'

        rows_html.append(f"""
        <div class="seq-row">
            {num_marker}
            {residue_divs}
        </div>""")

    return "\n".join(rows_html)


def predict_hotspots(top_k):
    predictor = HotspotPredictor(top_k=top_k)
    pdb_string = st.session_state.structure_parser.to_pdb_string(st.session_state.atom_array)

    with st.spinner("运行DL模型 (hotspot-prediction GAT+ESM-2)..."):
        try:
            results_dl = predictor.predict(
                st.session_state.atom_array,
                method="dl",
                pdb_string=pdb_string,
                top_k=top_k
            )
            st.session_state.hotspot_results_dl = results_dl
        except Exception as e:
            st.error(f"DL预测失败: {e}")
            results_dl = None

    if results_dl and "hotspots" in results_dl:
        hotspots_detail = []
        for label in results_dl["hotspots"]:
            info = results_dl["all_scores"].get(label, {})
            hotspots_detail.append({
                "label": label,
                "chain": info.get("chain_id", "A"),
                "residue_id": str(info.get("res_id", "")),
                "residue_name": info.get("residue_name", ""),
                "dl_score": info.get("score", 0),
                "combined_score": info.get("score", 0)
            })
        st.session_state.selected_hotspots = hotspots_detail
    else:
        st.session_state.selected_hotspots = []

    st.session_state.current_step = max(st.session_state.current_step, 1)
    if st.session_state.selected_hotspots:
        st.success(f"预测完成！选取 Top-{top_k} 热点残基: {', '.join([h['label'] for h in st.session_state.selected_hotspots])}")


def run_full_pipeline(binder_length, num_designs, num_sequences, rmsd_threshold):
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.job_id = job_id
    st.session_state.pipeline_running = True

    hotspots = st.session_state.selected_hotspots
    hotspot_labels = [f"{h['chain']}{h['residue_id']}" for h in hotspots]

    progress = st.progress(0, text="Step 1/4: RFD3生成Binder主链...")

    rfd3 = RFD3Runner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(st.session_state.pdb_content)
        tmp_pdb = tmp.name

    try:
        rfd3_result = rfd3.run(tmp_pdb, hotspot_labels, binder_length, num_designs, job_id)
    finally:
        try:
            os.unlink(tmp_pdb)
        except OSError:
            pass

    st.session_state.rfd3_results = rfd3_result

    if not rfd3_result["success"]:
        st.error(f"RFD3失败: {rfd3_result.get('error', 'Unknown')}")
        st.session_state.pipeline_running = False
        return

    progress.progress(25, text="Step 2/4: MPNN设计氨基酸序列...")

    mpnn = MPNNRunner()
    all_mpnn_results = []

    for design in rfd3_result["designs"]:
        mpnn_result = mpnn.run(
            design["pdb_path"],
            num_sequences=num_sequences,
            job_id=f"{job_id}/design_{design['index']}",
            top_k=TOP_K
        )
        if mpnn_result["success"]:
            for seq in mpnn_result["sequences"]:
                all_mpnn_results.append({
                    "design_idx": design["index"],
                    "sequence": seq["sequence"],
                    "score": seq.get("score", 0),
                    "backbone_pdb": design["pdb_path"],
                    "mock": seq.get("mock", False)
                })

    st.session_state.mpnn_results = all_mpnn_results
    progress.progress(50, text=f"Step 3/4: RF3验证结构 (共{len(all_mpnn_results)}条序列)...")

    rf3 = RF3Runner()
    all_rf3_results = []

    for i, mpnn_item in enumerate(all_mpnn_results):
        rf3_result = rf3.validate_design(
            mpnn_item["backbone_pdb"],
            mpnn_item["sequence"],
            job_id=f"{job_id}/seq_{i}",
            rmsd_threshold=rmsd_threshold
        )

        all_rf3_results.append({
            "design_idx": mpnn_item["design_idx"],
            "sequence": mpnn_item["sequence"],
            "mpnn_score": mpnn_item["score"],
            "rf3_pdb": rf3_result.get("rf3_pdb"),
            "backbone_pdb": mpnn_item["backbone_pdb"],
            "rmsd": rf3_result.get("rmsd", -1.0),
            "per_res_rmsd": rf3_result.get("per_res_rmsd", []),
            "plddt": rf3_result.get("plddt"),
            "avg_plddt": rf3_result.get("avg_plddt", 0),
            "passed": rf3_result.get("passed", False),
            "mock": rf3_result.get("mock", False) or mpnn_item.get("mock", False)
        })

        pct = 50 + int((i + 1) / max(len(all_mpnn_results), 1) * 45)
        progress.progress(pct, text=f"RF3验证 {i+1}/{len(all_mpnn_results)}...")

    all_rf3_results.sort(key=lambda x: x["rmsd"] if x["rmsd"] >= 0 else 999)

    st.session_state.rf3_results = all_rf3_results
    st.session_state.current_step = 3
    st.session_state.pipeline_running = False
    progress.progress(100, text="✅ 全流程完成！")
    time.sleep(0.5)
    progress.empty()


def render_main_page():
    task_type = st.selectbox("*任务类型:", options=["蛋白"], index=0, label_visibility="collapsed")

    col_upload_left, col_upload_right = st.columns([3, 6])
    with col_upload_left:
        st.markdown("*目标结构* ⓘ")
        tab_up, tab_input = st.tabs(["📁 上传文件", "✏️ 输入"])
        with tab_up:
            uploaded_file = st.file_uploader(
                "",
                type=["pdb", "cif"],
                label_visibility="collapsed",
                help="支持pdb/cif格式文件，文件不得超过200MB"
            )
            if uploaded_file:
                st.caption(f"📎 {uploaded_file.name}")
        with tab_input:
            pdb_text = st.text_area("", height=120, placeholder="粘贴PDB/CIF内容...", label_visibility="collapsed")

    with col_upload_right:
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        with btn_col1:
            crop_btn = st.button("✂️ 裁剪靶点", use_container_width=True)
        with btn_col2:
            spec_btn = st.button("🎯 指定热点", use_container_width=True)
        with btn_col3:
            reset_btn = st.button("🔄 重置", use_container_width=True)
        with btn_col4:
            empty_col = st.empty()

    process_uploaded_file(uploaded_file, pdb_text)

    if st.session_state.atom_array is not None:
        render_structure_view()
        render_hotspot_section()
        render_pipeline_panel()
    else:
        render_welcome()


def process_uploaded_file(uploaded_file, pdb_text=None):
    source = uploaded_file if uploaded_file else None
    if source is None and pdb_text and len(pdb_text.strip()) > 50:
        import io
        source = io.StringIO(pdb_text)
        source.name = "pasted.pdb"

    if source is not None:
        try:
            if st.session_state.atom_array is None or st.session_state.pdb_content is None:
                with st.spinner("解析结构文件..."):
                    parser = StructureParser()
                    atom_array = parser.parse_uploaded_file(source)
                    st.session_state.atom_array = atom_array
                    st.session_state.structure_parser = parser
                    st.session_state.structure_summary = parser.get_structure_summary(atom_array)
                    st.session_state.pdb_content = parser.to_pdb_string(atom_array)
        except Exception as e:
            st.error(f"解析文件时出错: {str(e)}")


def render_structure_view():
    atom_array = st.session_state.atom_array
    parser = st.session_state.structure_parser
    residues = parser.get_residue_info(atom_array)

    chain_info = {}
    for res in residues:
        chain_id = res.get("chain_id", "A")
        if chain_id not in chain_info:
            chain_info[chain_id] = []
        chain_info[chain_id].append(res)

    hotspot_labels = set()
    for h in st.session_state.selected_hotspots:
        hotspot_labels.add(h["label"])

    clicked_labels = set(st.session_state.clicked_residues)

    seq_col, view_col = st.columns([1, 1.3])

    with seq_col:
        st.markdown("#### 氨基酸序列")
        for chain_id in sorted(chain_info.keys()):
            chain_residues = chain_info[chain_id]
            seq_html = f"""<div style='border:1px solid #e0e0e0;border-radius:8px;padding:12px;background:#fff;max-height:520px;overflow-y:auto;'>
            <div style='font-weight:bold;color:#1976d2;margin-bottom:8px;'>{chain_id}protein</div>
            <div class='seq-container'>"""
            seq_html += build_sequence_html(chain_id, chain_residues, hotspot_labels, clicked_labels)
            seq_html += "</div></div>"
            st.markdown(seq_html, unsafe_allow_html=True)

        file_name = "未命名"
        summary = st.session_state.structure_summary
        if hasattr(st.session_state, '_uploaded_filename'):
            file_name = st.session_state._uploaded_filename

        st.markdown(f"""
        <div style='margin-top:8px;padding:8px;border:1px solid #e0e0e0;border-radius:6px;display:flex;align-items:center;gap:8px;'>
        <span>🔗</span><span>{file_name}</span>
        </div>""", unsafe_allow_html=True)

    with view_col:
        st.markdown("#### 3D 结构可视化")
        render_interactive_viewer(chain_info, hotspot_labels)


def render_interactive_viewer(chain_info, hotspot_labels):
    pdb_content = st.session_state.pdb_content
    if not pdb_content:
        st.info("请先上传蛋白质文件")
        return

    hotspot_residues = []
    for h in st.session_state.selected_hotspots:
        hotspot_residues.append({
            "chain": h.get("chain", "A"),
            "residue_id": h.get("residue_id", "0"),
            "score": h.get("combined_score", 0)
        })

    molstar_html = render_molstar_with_interaction(
        pdb_content=pdb_content,
        hotspot_residues=hotspot_residues,
        chain_info=chain_info,
        height=580
    )
    render_html(molstar_html, height=600)


def render_molstar_with_interaction(pdb_content, hotspot_residues, chain_info, height=580):
    import base64
    import json

    pdb_b64 = base64.b64encode(pdb_content.encode()).decode()

    hotspot_js = "const hotspotResidues = [];"
    if hotspot_residues:
        entries = []
        for h in hotspot_residues:
            chain = h.get("chain", "A")
            res_id = h.get("residue_id", 0)
            score = h.get("score", 0)
            try:
                res_id_int = int(str(res_id).strip())
            except (ValueError, TypeError):
                res_id_int = 0
            entries.append(f'{{chain: "{chain}", resId: {res_id_int}, score: {float(score):.3f}}}')
        hotspot_js = f"const hotspotResidues = [{', '.join(entries)}];"

    all_chains_json = json.dumps(list(chain_info.keys()))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #fff; }}
            #viewer {{ width: 100%; height: {height}px; position: relative; }}
            #statusBar {{
                position: absolute; bottom: 0; left: 0; right: 0;
                background: rgba(0,0,0,0.85); color: #fff;
                padding: 6px 12px; font-size: 13px; font-family: monospace;
                display: flex; gap: 16px; align-items: center; z-index: 10;
            }}
            .legend {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; }}
            .toolbar-btn {{
                position: absolute; right: 8px; top: 8px; z-index: 20;
                background: white; border: 1px solid #ddd; border-radius: 6px;
                padding: 4px; display: flex; flex-direction: column; gap: 2px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .toolbar-btn button {{
                width: 32px; height: 32px; border: none; background: transparent;
                cursor: pointer; border-radius: 4px; font-size: 16px; display: flex;
                align-items: center; justify-content: center; color: #555;
            }}
            .toolbar-btn button:hover {{ background: #e3f2fd; color: #1976d2; }}
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.css">
    </head>
    <body>
        <div id="viewer">
            <div class="toolbar-btn">
                <button onclick="resetCamera()" title="重置视角">🎯</button>
                <button onclick="toggleSpin()" title="旋转">🔄</button>
                <button onclick="toggleStyle()" title="切换样式">⚙️</button>
                <button onclick="toggleLabel()" title="标签">🏷️</button>
                <button onclick="zoomToFit()" title="适应窗口">⬜</button>
                <button onclick="screenshot()" title="截图">📷</button>
            </div>
            <div id="statusBar">
                <span class="status-item"><span class="legend" style="background:#4CAF50"></span>Target</span>
                <span id="resInfo" style="margin-left:auto;"></span>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script>
        <script>
            {hotspot_js}
            const allChains = {all_chains_json};
            let pluginInstance = null;
            let targetStructures = [];
            let isSpinning = false;
            let spinAnimId = null;

            function highlightResidue(chainId, resId) {{
                if (!pluginInstance || targetStructures.length === 0) return;
                const plugin = pluginInstance;
                const targetStruct = targetStructures[0];
                const comp = plugin.managers.structure.component;

                comp.clearRepresentations(targetStruct);

                const defaultRepr = await comp.addRepresentation(targetStruct, 'cartoon', {{
                    color: {{ name: 'uniform', params: {{ value: '#4CAF50' }} }},
                    alpha: 0.85
                }});

                const script = molstar.Script(
                    "sel.atom: " +
                    "(chain.authAsymId = " + JSON.stringify(chainId) + " or chain.labelAsymId = " + JSON.stringify(chainId) + ") and " +
                    "(residue.authSeqNumber = " + parseInt(resId) + " or residue.labelSeqNumber = " + parseInt(resId) + ")"
                );

                const selData = await plugin.managers.structure.selection.fromScript(targetStruct, script);
                if (selData) {{
                    await comp.addRepresentation(targetStruct, 'ball-and-stick', {{
                        color: {{ name: 'uniform', params: {{ value: '#FF0000' }} }},
                        sizeFactor: 0.35,
                        sizeAspectRatio: 1.0
                    }}, selData);
                }}

                document.getElementById("resInfo").textContent =
                    chainId + "/" + resId + " 已高亮";
            }}

            window.addEventListener('message', function(event) {{
                if (event.data && event.data.type === 'residue_click') {{
                    highlightResidue(event.data.chain, event.data.resid);
                }}
            }});

            molstar.Viewer.create("viewer", {{
                layoutIsExpanded: false,
                layoutShowControls: false,
                layoutShowRemoteState: false,
                layoutShowSequence: false,
                layoutShowLog: false,
                layoutShowLeftPanel: false,
            }}).then(async viewer => {{
                pluginInstance = viewer;
                const plugin = viewer;

                try {{
                    const targetData = atob("{pdb_b64}");
                    const targetTraj = await plugin.builders.structure.readTrajectory({{
                        model: {{ type: 'pdb', data: targetData }}
                    }});
                    const targetPreset = await plugin.builders.structure.hierarchy.applyPreset(
                        {{ structure: targetTraj }},
                        'default'
                    );

                    targetStructures.push(targetTraj.structures[0]);

                    const targetRepr = targetPreset.structure.representations[0];
                    if (targetRepr) {{
                        await plugin.managers.structure.component.updateRepresentationsOptions(
                            targetRepr,
                            {{ color: {{ name: 'uniform', params: {{ value: '#4CAF50' }} }}, alpha: 0.85 }}
                        );
                    }}

                    if (hotspotResidues.length > 0) {{
                        try {{
                            const structures = plugin.managers.structure.hierarchy.current.structures;
                            if (structures.length > 0) {{
                                const targetStruct = structures[0];
                                const comp = plugin.managers.structure.component;

                                for (const h of hotspotResidues) {{
                                    try {{
                                        const script = molstar.Script(
                                            "sel.atom: " +
                                            "(chain.authAsymId = " + JSON.stringify(h.chain) + " or chain.labelAsymId = " + JSON.stringify(h.chain) + ") and " +
                                            "(residue.authSeqNumber = " + h.resId + " or residue.labelSeqNumber = " + h.resId + ")"
                                        );
                                        const selData = await plugin.managers.structure.selection.fromScript(targetStruct, script);
                                        if (selData) {{
                                            await comp.addRepresentation(targetStruct, 'ball-and-stick', {{
                                                color: {{ name: 'uniform', params: {{ value: '#FF0000' }} }},
                                                sizeFactor: 0.3
                                            }}, selData);
                                        }}
                                    }} catch(e2) {{ console.warn("Hotspot error:", e2); }}
                                }}
                            }}
                        }} catch(e) {{ console.warn("Hotspot section error:", e); }}
                    }}

                    plugin.managers.camera.resetSnapshot();

                }} catch(e) {{
                    console.error("Mol* error:", e);
                    document.getElementById("resInfo").textContent = "Error: " + e.message;
                }}
            }});

            function resetCamera() {{
                if (pluginInstance) pluginInstance.managers.camera.resetSnapshot();
            }}
            function toggleSpin() {{
                isSpinning = !isSpinning;
                if (isSpinning && pluginInstance) {{
                    spinAnimId = requestAnimationFrame(function spin() {{
                        if (!isSpinning) return;
                        pluginInstance.managers.camera.spin({{ speed: 1 }});
                        spinAnimId = requestAnimationFrame(spin);
                    }});
                }} else if (spinAnimId) {{
                    cancelAnimationFrame(spinAnimId);
                }}
            }}
            function toggleStyle() {{
                if (!pluginInstance) return;
                const structs = pluginInstance.managers.structure.hierarchy.current.structures;
                if (structs.length > 0) {{
                    const s = structs[0];
                    const reprs = pluginInstance.managers.structure.component.getRepresentations(s);
                    reprs.forEach(r => {{
                        const cur = r.params?.type?.name || '';
                        if (cur === 'cartoon') pluginInstance.managers.structure.component.updateRepresentationsOptions(r, {{ type: {{ name: 'spacefill' }} }});
                        else if (cur === 'spacefill') pluginInstance.managers.structure.component.updateRepresentationsOptions(r, {{ type: {{ name: 'cartoon' }} }});
                    }});
                }}
            }}
            function toggleLabel() {{}}
            function zoomToFit() {{
                if (pluginInstance) pluginInstance.managers.camera.resetSnapshot();
            }}
            function screenshot() {{
                if (pluginInstance) pluginInstance.managers.snapshot.saveToFile('image/png');
            }}
        </script>
    </body>
    </html>
    """
    return html


def render_hotspot_section():
    hotspots = st.session_state.selected_hotspots

    st.markdown("---")
    hotspot_col_label, hotspot_col_input = st.columns([1, 5])
    with hotspot_col_label:
        st.markdown("**热点** ⓘ")
    with hotspot_col_input:

        if hotspots:
            tags_html = ""
            for h in hotspots:
                score = h.get("combined_score", 0)
                res_name = h.get("residue_name", "")
                one_letter = AA_3TO1.get(res_name, "?")
                if score > 0.7:
                    tag_cls = "tag-high"
                elif score > 0.4:
                    tag_cls = "tag-mid"
                else:
                    tag_cls = "tag-low"

                tags_html += f"""
                <span class="hotspot-tag {tag_cls}"
                      onclick="window.parent.postMessage({{type:'residue_click', chain:'{h.get('chain','A')}', resid:'{h['residue_id']}'}}, '*')">
                    {one_letter}{h['label']} ({score:.3f})
                </span>"""

            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            placeholder = st.text_input(
                "",
                placeholder="输入残基编号：A/1 表示 A 链上的残基 1（例如：A/1、A/2、A/3）",
                label_visibility="collapsed",
                key="hotspot_manual_input"
            )

    if hotspots:
        with st.expander("📊 详细预测结果", expanded=False):
            df_data = []
            for h in hotspots:
                res_name = h.get("residue_name", "")
                one_letter = AA_3TO1.get(res_name, "?")
                df_data.append({
                    "残基": f"{one_letter} ({h['label']})",
                    "链": h.get("chain", "A"),
                    "三字母": res_name,
                    "单字母": one_letter,
                    "DL得分": f"{h.get('dl_score', h.get('combined_score', 0)):.4f}",
                    "综合得分": f"{h.get('combined_score', 0):.4f}"
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        st.info(f"💡 已选取 Top-{len(hotspots)} 热点残基，点击上方按钮开始设计流程")


def render_pipeline_panel():
    with st.container():
        st.markdown("---")
        render_step_bar()

        c_predict, c_run, c_reset = st.columns([2, 2, 1])
        with c_predict:
            st.markdown("**🎯 热点预测 (DL)**")

        with c_run:
            binder_len = st.slider("Binder长度", 40, 150, 80, step=5, label_visibility="collapsed")

        with c_reset:
            if st.button("🎯 预测热点", type="primary", use_container_width=True):
                if st.session_state.atom_array is not None:
                    predict_hotspots(TOP_K)
                else:
                    st.warning("请先上传蛋白质文件")

        btn_run, btn_full, btn_rst = st.columns(3)
        with btn_run:
            if st.button("🚀 运行全流程", type="primary", use_container_width=True):
                if st.session_state.atom_array is None:
                    st.error("请先上传目标蛋白结构文件")
                elif not st.session_state.selected_hotspots:
                    st.warning("请先预测或选择热点残基")
                else:
                    run_full_pipeline(binder_len, TOP_K, TOP_K, 2.0)
        with btn_full:
            if st.button("🔄 重置全部", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session()
                st.rerun()
        with btn_rst:
            st.empty()

    if st.session_state.rfd3_results:
        render_pipeline_results()


def render_pipeline_results():
    st.markdown("### 🧬 设计结果")
    rfd3_results = st.session_state.rfd3_results
    mpnn_results = st.session_state.mpnn_results
    rf3_results = st.session_state.rf3_results

    is_mock = rfd3_results.get("mock", False)
    if is_mock:
        st.warning("⚠️ RFD3/MPNN/RF3未安装，使用模拟数据展示流程")

    tab_rfd3, tab_mpnn, tab_rf3 = st.tabs(["RFD3 主链", "MPNN 序列", "RF3 验证"])

    with tab_rfd3:
        if rfd3_results.get("success"):
            designs = rfd3_results["designs"]
            cols = st.columns(min(len(designs), 4))
            for i, design in enumerate(designs[:4]):
                with cols[i]:
                    plddt = design.get("plddt", 0)
                    rank = design.get("rank", i + 1)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:16px;font-weight:bold;">Design {design['index']+1}</div>
                        <div style="font-size:12px;">Rank #{rank}</div>
                        <div style="font-size:13px;">pLDDT: {plddt:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.caption(f"共生成 {len(designs)} 个Binder主链结构，选取 Top-{TOP_K}")
        else:
            st.error(f"RFD3生成失败: {rfd3_results.get('error', 'Unknown')}")

    with tab_mpnn:
        if mpnn_results:
            df_data = []
            for m in mpnn_results:
                seq = m["sequence"]
                df_data.append({
                    "设计": f"Design {m['design_idx']+1}",
                    "序列": seq[:60] + "..." if len(seq) > 60 else seq,
                    "长度": len(seq),
                    "得分": f"{m.get('score', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        else:
            st.info("MPNN序列设计未运行")

    with tab_rf3:
        if rf3_results:
            passed = [r for r in rf3_results if r["passed"]]
            failed = [r for r in rf3_results if not r["passed"] and r["rmsd"] >= 0]

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("总验证数", len(rf3_results))
            with mc2:
                st.metric("通过 (RMSD<阈值)", len(passed))
            with mc3:
                st.metric("未通过", len(failed))

            if passed:
                best = min(passed, key=lambda x: x["rmsd"])
                st.success(f"🏆 最佳: Design {best['design_idx']+1}, RMSD={best['rmsd']:.3f}Å, pLDDT={best.get('avg_plddt','N/A')}")

            df_data = []
            for r in rf3_results:
                seq = r["sequence"]
                status = "✅ 通过" if r["passed"] else "❌ 未通过"
                df_data.append({
                    "设计": f"Design {r['design_idx']+1}",
                    "序列": seq[:30] + "..." if len(seq) > 30 else seq,
                    "RMSD(Å)": f"{r['rmsd']:.3f}" if r['rmsd'] >= 0 else "N/A",
                    "pLDDT": f"{r.get('avg_plddt', 0):.1f}" if r.get('avg_plddt') else "N/A",
                    "状态": status,
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        else:
            st.info("RF3验证未运行")


def render_welcome():
    st.title("🧬 蛋白质Binder设计系统")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 快速开始

        1. **上传目标蛋白结构** - 上方选择 PDB/CIF 文件
        2. **查看序列与3D** - 左侧单字母序列 + 右侧3D结构
        3. **预测热点残基** - DL模型自动选取Top-3
        4. **运行全流程** - RFD3 → MPNN → RF3 自动化设计

        ### 📋 工作流程

        ```
        目标蛋白 → 热点预测(DL) → RFD3(Top-3) → MPNN(Top-3) → RF3验证 → RMSD<2Å筛选
        ```

        ### 🔧 集成模型

        | 模型 | 功能 | 说明 |
        |------|------|------|
        | hotspot-prediction (DL) | 热点残基预测 | GAT+ESM-2深度学习 |
        | RFDiffusion3 | Binder主链生成 | 扩散模型 |
        | ProteinMPNN | 序列设计 | 图神经网络 |
        | RoseTTAFold3 | 结构预测验证 | 三轨网络 |
        """)

    with col2:
        st.markdown("""
        ### ⚙️ 系统要求

        - Python 3.10+
        - Streamlit 1.30+
        - Biotite 0.38+
        - XGBoost ≥ 1.7.0
        - GPU (可选, 用于DL模型)

        ### 💡 使用提示

        - 点击左侧序列中的**任意残基** → 右侧3D视图**高亮显示**
        - **橙色标记**为已识别的热点残基
        - 未安装RFD3/MPNN/RF3时自动使用模拟模式
        - Top-K=3 默认选取3个最优结果
        """)


def main():
    render_main_page()


if __name__ == "__main__":
    main()
