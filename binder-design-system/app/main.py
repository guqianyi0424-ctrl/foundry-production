"""
蛋白质Binder设计系统 - 主应用 (ODesign风格)
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
import streamlit.components.v1 as components

from utils.structure_parser import StructureParser
from utils.hotspot_predictor import HotspotPredictor
from utils.rfd3_runner import RFD3Runner
from utils.mpnn_runner import MPNNRunner
from utils.rf3_runner import RF3Runner
from utils.molstar_viewer import render_molstar, render_rmsd_chart, render_plddt_chart
from utils.rfd3_visualizer import render_rfd3_viewer, render_rfd3_plddt_chart, render_rfd3_design_card
from utils.mpnn_visualizer import render_mpnn_sequence_comparison, render_mpnn_score_chart, render_mpnn_legend
from utils.rf3_visualizer import render_rf3_result_card, render_rf3_plddt_chart, render_rf3_pae_heatmap, render_rf3_rmsd_chart, render_rf3_summary_metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(
    page_title="ODesign - 蛋白质Binder设计",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #ffffff;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
    width: 220px !important;
    min-width: 220px !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0 !important;
    background: transparent !important;
}

/* === Sidebar Logo Area === */
.sidebar-logo {
    padding: 20px 18px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #e2e8f0;
    background: white;
}

.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
    font-weight: bold;
    flex-shrink: 0;
}

.sidebar-logo-text {
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
}

.sidebar-collapse-btn {
    margin-left: auto;
    padding: 4px 8px;
    cursor: pointer;
    color: #94a3b8;
    font-size: 18px;
    border: none;
    background: transparent;
}

/* === Sidebar Navigation === */
.sidebar-nav {
    padding: 12px 10px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 13px 16px;
    margin: 3px 0;
    border-radius: 9px;
    cursor: pointer;
    font-size: 14.5px;
    color: #475569;
    transition: all 0.15s ease;
    font-weight: 500;
    border: none;
    background: transparent;
    text-align: left;
    width: 100%;
}

.nav-item:hover {
    background: #e2e8f0;
    color: #1e293b;
}

.nav-item.active {
    background: #dbeafe;
    color: #2563eb;
    font-weight: 600;
}

.nav-icon {
    font-size: 18px;
    width: 22px;
    text-align: center;
    flex-shrink: 0;
}

/* === Header === */
.header-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-box {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.logo-text {
    font-size: 24px;
    font-weight: 700;
    color: #1e293b;
}

.mode-switch {
    display: flex;
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 8px;
}

.mode-btn {
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    background: transparent;
    color: #64748b;
    transition: all 0.15s;
}

.mode-btn.active {
    background: white;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* === Upload Section === */
.upload-section {
    background: white;
    padding: 20px 24px;
    border-bottom: 1px solid #e2e8f0;
}

.upload-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.task-label {
    font-size: 14px;
    font-weight: 600;
    color: #334155;
    min-width: 70px;
}

.structure-label {
    font-size: 14px;
    font-weight: 600;
    color: #334155;
    margin-right: 4px;
}

.hint-icon {
    color: #94a3b8;
    font-size: 14px;
}

.action-buttons {
    display: flex;
    gap: 10px;
    margin-top: 12px;
}

.act-btn {
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid #e2e8f0;
    background: white;
    color: #475569;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.act-btn:hover {
    border-color: #6366f1;
    color: #6366f1;
    background: #f8faff;
}

/* === Main Content Area === */
.main-content {
    padding: 0 24px 24px;
}

.content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 16px;
}

/* === Sequence Panel (Left) === */
.seq-panel {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: white;
    overflow: hidden;
}

.panel-header {
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    padding: 14px 18px;
    border-bottom: 1px solid #f1f5f9;
    background: #fafbfc;
}

.chain-title {
    font-size: 14px;
    font-weight: 700;
    color: #1e293b;
    padding: 12px 18px 6px;
    border-bottom: 1px solid #f8fafc;
}

.seq-container {
    padding: 12px 18px 20px;
    max-height: 480px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
}

.seq-row {
    display: flex;
    align-items: baseline;
    line-height: 2.2;
    position: relative;
    padding: 2px 0;
    font-size: 15px;
    letter-spacing: 1.2px;
}

.seq-num {
    position: absolute;
    top: -14px;
    font-size: 11px;
    color: #94a3b8;
    font-family: 'SF Mono', Consolas, monospace;
    font-weight: 400;
}

.seq-char {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 21px;
    height: 26px;
    margin: 0 0.5px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.12s ease;
    font-weight: 600;
    color: #334155;
    position: relative;
}

.seq-char:hover {
    background: #e0e7ff !important;
    transform: scale(1.25);
    z-index: 10;
}

.seq-hotspot {
    background: #6366f1 !important;
    color: white !important;
    box-shadow: 0 2px 6px rgba(99,102,241,0.35);
}

/* === Viewer Panel (Right) === */
.viewer-panel {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: white;
    overflow: hidden;
    position: relative;
}

.viewer-wrapper {
    height: 520px;
    position: relative;
}

.viewer-status-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(30,41,59,0.92);
    color: #e2e8f0;
    padding: 8px 16px;
    font-size: 12px;
    font-family: 'SF Mono', Consolas, monospace;
    display: flex;
    gap: 20px;
    z-index: 100;
    backdrop-filter: blur(6px);
}

/* === Bottom Bar === */
.bottom-bar {
    margin-top: 16px;
}

.file-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0;
    font-size: 13px;
    color: #64748b;
    border-top: 1px solid #f1f5f9;
}

.file-icon {
    color: #6366f1;
    font-size: 16px;
}

.file-name-text {
    font-weight: 500;
    color: #334155;
}

.delete-btn {
    margin-left: auto;
    cursor: pointer;
    color: #ef4444;
    font-size: 16px;
}

.hotspot-input-section {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 0;
    border-top: 1px solid #e2e8f0;
}

.hotspot-label-text {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
    white-space: nowrap;
}

.hotspot-field {
    flex: 1;
}

.hotspot-field input {
    width: 100%;
    padding: 10px 16px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
    transition: all 0.15s;
    font-family: inherit;
}

.hotspot-field input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
}

.hotspot-field input::placeholder {
    color: #94a3b8;
}

.predict-btn {
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    background: #6366f1;
    color: white;
    transition: all 0.15s;
    white-space: nowrap;
}

.predict-btn:hover {
    background: #4f46e5;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}

/* === Pipeline Section === */
.pipeline-section {
    margin-top: 20px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: white;
}

.pipeline-steps {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}

.pipe-step-item {
    flex: 1;
    text-align: center;
    padding: 10px 8px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s;
}

.pipe-done {
    background: #ecfdf5;
    color: #059669;
}

.pipe-active {
    background: #eff6ff;
    color: #2563eb;
}

.pipe-pending {
    background: #f8fafc;
    color: #94a3b8;
}

.pipeline-actions {
    display: flex;
    gap: 12px;
    align-items: center;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

TOP_K = 3


def init_session():
    defaults = {
        "current_page": "新建设计",
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
        "uploaded_filename": "",
        "hotspot_manual_input": "",
        "job_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


def build_sequence_html(chain_id, residues, hotspot_labels):
    """构建序列HTML - 完全匹配ODesign第二张图样式"""
    rows = []
    chars_per_row = 40
    n_res = len(residues)

    for start in range(0, n_res, chars_per_row):
        end = min(start + chars_per_row, n_res)
        row_res = residues[start:end]

        letters_html = ""
        for idx_offset, res in enumerate(row_res):
            global_idx = start + idx_offset
            res_name = res.get("res_name", "?")
            one_letter = AA_3TO1.get(res_name, "X")
            res_id = str(res.get("res_id", global_idx + 1))
            label = f"{chain_id}{res_id}"

            is_hotspot = label in hotspot_labels
            hs_class = "seq-hotspot" if is_hotspot else ""

            letters_html += f"""<span class='seq-char {hs_class}'
                data-chain='{chain_id}' data-resid='{res_id}' data-label='{label}'
                onclick="window.parent.postMessage({{type:'residue_click', chain:'{chain_id}', resid:{res_id}, label:'{label}'}}, '*')">{one_letter}</span>"""

        show_num = (start % (chars_per_row * 3) == 0 or start == 0)
        first_rid = residues[start].get("res_id", start + 1)
        num_tag = f"<span class='seq-num'>{first_rid}</span>" if show_num else ""

        last_num = ""
        if end == n_res and row_res:
            last_num = f"<span class='seq-num' style='right:8px;left:auto;'>{row_res[-1].get('res_id','')}</span>"

        rows.append(f"<div class='seq-row'>{num_tag}{letters_html}{last_num}</div>")

    return "\n".join(rows)


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

        input_str = ", ".join([f"{h['chain']}/{h['residue_id']}" for h in hotspots_detail])
        st.session_state.hotspot_manual_input = input_str
    else:
        st.session_state.selected_hotspots = []

    st.session_state.current_step = max(st.session_state.current_step, 1)


def parse_manual_hotspot_input(input_text):
    if not input_text or not input_text.strip():
        return []

    parsed = []
    parts = [p.strip() for p in input_text.replace("，", ",").replace("/", ",").split(",") if p.strip()]

    i = 0
    while i < len(parts):
        part = parts[i]
        chain = "A"
        res_id = part

        has_alpha = any(c.isalpha() for c in part)
        has_digit = any(c.isdigit() for c in part)

        if has_alpha and has_digit:
            for ci, c in enumerate(part):
                if c.isalpha():
                    continue
                chain = part[:ci]
                res_id = part[ci:]
                break
        elif has_alpha:
            chain = part.upper()
            if i + 1 < len(parts) and any(c.isdigit() for c in parts[i + 1]):
                i += 1
                res_id = parts[i]

        try:
            int(res_id)
            parsed.append({
                "label": f"{chain}{res_id}",
                "chain": chain,
                "residue_id": res_id,
                "residue_name": "",
                "dl_score": 0,
                "combined_score": 0
            })
        except ValueError:
            pass
        i += 1

    return parsed


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

    job_record = {
        "job_id": job_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "已完成",
        "binder_length": binder_length,
        "hotspots": len(hotspots),
        "rfd3_results": rfd3_result,
        "mpnn_count": len(all_mpnn_results),
        "rf3_results": all_rf3_results[:5],
        "uploaded_file": st.session_state.uploaded_filename
    }

    if "job_history" not in st.session_state:
        st.session_state.job_history = []
    st.session_state.job_history.append(job_record)


def render_sidebar():
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-icon'>O</div>
        <span class='sidebar-logo-text'>ODesign</span>
        <button class='sidebar-collapse-btn'>☰</button>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)

    nav_items = [
        ("🧬", "新建设计"),
        ("📋", "作业中心"),
        ("❓", "帮助"),
    ]

    for icon, label in nav_items:
        is_active = (st.session_state.current_page == label)
        active_class = "active" if is_active else ""

        btn_html = f"""
        <button class='nav-item {active_class}' onclick="
            window.parent.postMessage({{type:'nav_change', page:'{label}'}}, '*')
        ">
            <span class='nav-icon'>{icon}</span>
            <span>{label}</span>
        </button>
        """
        st.markdown(btn_html, unsafe_allow_html=True)

        if st.button(label, key=f"nav_{label}", help=f"切换到{label}页面"):
            st.session_state.current_page = label
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_job_center():
    st.markdown("### 📋 作业中心 - 历史记录")

    history = st.session_state.get("job_history", [])

    if not history:
        st.info("暂无历史记录，完成设计任务后会在此显示")
        return

    for i, job in enumerate(reversed(history[-10:])):
        with st.expander(f"📁 {job.get('job_id', f'任务-{i+1}')}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("状态", job.get('status', '已完成'))
                st.text(f"时间: {job.get('timestamp', 'N/A')}")
            with col2:
                if job.get('rf3_results'):
                    passed = len([r for r in job['rf3_results'] if r.get('passed')])
                    total = len(job['rf3_results'])
                    st.metric("通过率", f"{passed}/{total}")


def render_help():
    st.markdown("""
    ### ❓ 帮助中心

    #### 🚀 快速开始指南

    **步骤 1: 上传结构文件**
    - 支持格式：PDB、CIF
    - 文件大小限制：200MB
    - 可直接粘贴PDB内容

    **步骤 2: 预测热点残基**
    - 点击"🔬 预测热点 (DL)"按钮
    - DL模型自动识别Top-3热点
    - 或手动输入：`A/1, A/2, B/130`

    **步骤 3: 运行全流程**
    - 调整Binder长度（40-150）
    - 点击"🚀 运行全流程"
    - 自动执行：RFD3 → MPNN → RF3

    #### 📖 功能说明

    | 功能 | 说明 |
    |------|------|
| **热点预测** | 使用GAT+ESM-2深度学习模型 |
    | **RFD3** | RFDiffusion3生成Binder主链 |
    | **MPNN** | ProteinMPNN设计氨基酸序列 |
    | **RF3** | RoseTTAFold3验证结构质量 |

    #### ⌨️ 快捷操作

    - **点击序列残基** → 3D视图高亮定位
    - **蓝色标记** → 已选中的热点残基
    - **输入框** → 手动指定/编辑热点

    #### 🔗 相关链接

    - [ODesign文档](#)
    - [API参考](#)
    - [问题反馈](#)
    """)


def render_main_page():
    current_page = st.session_state.current_page

    if current_page == "作业中心":
        render_job_center()
        return
    elif current_page == "帮助":
        render_help()
        return

    st.markdown("""
    <div class='header-main'>
        <div class='header-left'>
            <div class='logo-box'>O</div>
            <span class='logo-text'>ODesign</span>
        </div>
        <div class='mode-switch'>
            <button class='mode-btn active'>表格模式</button>
            <button class='mode-btn'>JSON模式</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_type, _spacer = st.columns([3, 7])
    with col_type:
        task_type = st.selectbox("*任务类型", options=["蛋白"], index=0, label_visibility="collapsed")

    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)

    upload_col, action_col = st.columns([4, 6])

    with upload_col:
        st.markdown("<div class='upload-row'><span class='structure-label'>*目标结构</span><span class='hint-icon'>ⓘ</span></div>", unsafe_allow_html=True)
        tab_up, tab_in = st.tabs(["📤 上传文件", "✏️ 输入"])
        with tab_up:
            uploaded_file = st.file_uploader("", type=["pdb", "cif"], label_visibility="collapsed", help="支持pdb/cif格式文件，文件不得超过200MB")
        with tab_in:
            pdb_text = st.text_area("", height=80, placeholder="粘贴PDB/CIF内容...", label_visibility="collapsed")

    with action_col:
        st.markdown("<div style='font-size:12px;color:#94a3b8;margin-bottom:8px;'>支持pdb/cif格式文件，文件不超过200MB</div>", unsafe_allow_html=True)
        btn_cols = st.columns(5)
        with btn_cols[0]:
            crop_btn = st.button("✂️ 裁剪靶点", use_container_width=True)
        with btn_cols[1]:
            spec_btn = st.button("🎯 指定热点", use_container_width=True)
        with btn_cols[2]:
            reset_btn = st.button("🔄 重置", use_container_width=True)
        with btn_cols[3]:
            st.empty()
        with btn_cols[4]:
            st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

    process_uploaded_file(uploaded_file, pdb_text)

    if st.session_state.atom_array is not None:
        render_odesign_layout()
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
                    st.session_state.uploaded_filename = getattr(source, 'name', '未命名')
        except Exception as e:
            st.error(f"解析文件时出错: {str(e)}")


def render_odesign_layout():
    atom_array = st.session_state.atom_array
    parser = st.session_state.structure_parser
    residues = parser.get_residue_info(atom_array)

    chain_info = {}
    for res in residues:
        cid = res.get("chain_id", "A")
        if cid not in chain_info:
            chain_info[cid] = []
        chain_info[cid].append(res)

    hotspot_labels = set(h["label"] for h in st.session_state.selected_hotspots)

    st.markdown("<div class='main-content'>", unsafe_allow_html=True)

    seq_col, viewer_col = st.columns([1.1, 1])

    with seq_col:
        html_parts = ["<div class='seq-panel'>"]
        html_parts.append("<div class='panel-header'>氨基酸序列</div>")

        for chain_id in sorted(chain_info.keys()):
            chain_residues = chain_info[chain_id]
            html_parts.append(f"<div class='chain-title'>{chain_id}:protein</div>")
            html_parts.append("<div class='seq-container'>")
            html_parts.append(build_sequence_html(chain_id, chain_residues, hotspot_labels))
            html_parts.append("</div>")

        html_parts.append("</div>")

        full_seq_html = "\n".join(html_parts)
        components.html(f"<style>{CSS}</style>{full_seq_html}", height=520, scrolling=True)

        fn = st.session_state.uploaded_filename or "未命名"
        st.markdown(f"""
        <div class='file-bar'>
            <span class='file-icon'>🔗</span>
            <span class='file-name-text'>{fn}</span>
            <span class='delete-btn' onclick="window.parent.postMessage({{type:'remove_file'}}, '*')">🗑️</span>
        </div>
        """, unsafe_allow_html=True)

    with viewer_col:
        render_viewer_panel(chain_info, hotspot_labels)

    st.markdown("</div>", unsafe_allow_html=True)

    render_hotspot_bar()
    render_pipeline_section()


def render_viewer_panel(chain_info, hotspot_labels):
    pdb_content = st.session_state.pdb_content
    if not pdb_content:
        return

    hotspot_data = []
    for h in st.session_state.selected_hotspots:
        hotspot_data.append({
            "chain": h.get("chain", "A"),
            "residue_id": h.get("residue_id", "0"),
            "score": h.get("combined_score", 0)
        })

    molstar_html = render_molstar_with_interaction(
        pdb_content=pdb_content,
        hotspot_residues=hotspot_data,
        chain_info=chain_info,
        height=520
    )

    total_res = sum(len(v) for v in chain_info.values())
    status_text = f"XXXX | Model 1 | Instance 1,{total_res} | B"

    st.markdown(f"<div class='viewer-panel'><div class='panel-header'>3D 结构可视化</div><div class='viewer-wrapper'></div><div class='viewer-status-bar'>{status_text}</div></div>", unsafe_allow_html=True)
    components.html(molstar_html, height=560, scrolling=False)


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

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #fff; }}
            #molstar-root {{ width: 100%; height: {height}px; }}
            .molstar-toolbar {{
                position: absolute; right: 10px; top: 10px; z-index: 20;
                background: white; border: 1px solid #ddd; border-radius: 8px;
                padding: 6px; display: flex; flex-direction: column; gap: 3px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.12);
            }}
            .molstar-toolbar button {{
                width: 34px; height: 34px; border: none; background: transparent;
                cursor: pointer; border-radius: 5px; font-size: 17px; display: flex;
                align-items: center; justify-content: center; color: #555;
            }}
            .molstar-toolbar button:hover {{ background: #e3f2fd; color: #1976d2; }}
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.css">
    </head>
    <body>
        <div id='molstar-root'>
            <div class='molstar-toolbar'>
                <button onclick="resetView()" title="重置视角">🎯</button>
                <button onclick="toggleSpin()" title="旋转">🔄</button>
                <button onclick="zoomFit()" title="适应窗口">⬜</button>
                <button onclick="takeScreenshot()" title="截图">📷</button>
            </div>
        </div>

        <script type="importmap">
        {{
            "imports": {{
                "molstar": "https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/esm/index.js",
                "molstar/lib/commonjs/mol-star": "https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/esm/index.js"
            }}
        }}
        </script>
        <script type="module">
        import {{ PluginSpec }} from 'molstar';
        import {{ createPlugin }} from 'molstar/lib/mol-plugin-ui/plugin';
        import {{ DefaultPluginSpec }} from 'molstar/lib/mol-plugin-ui/spec';

        let plugin;

        async function init() {{
            const container = document.getElementById('molstar-root');
            plugin = await createPlugin(container, {{
                layout: {{
                    initial: {{ isExpanded: false, showControls: false }},
                }},
                spec: DefaultPluginSpec,
            }});

            const pdbData = `{pdb_b64}`;
            await plugin.builders.data.download({{
                url: 'data:text/plain;base64,' + pdbData,
                isBinary: true,
                format: 'pdb',
            }}, {{ state: {{ isHidden: true }}, representation: {{ params: {{}} }} }});

            plugin.build().toRoot();

            const structure = plugin.state.data.select('structure')[0];
            if (!structure) return;

            const modelData = structure.cell.obj?.data;
            if (!modelData) return;

            {hotspot_js}

            if (hotspotResidues.length > 0) {{
                const lociElements = [];
                for (const hs of hotspotResidues) {{
                    try {{
                        const l = modelData.location.label(hs.resId.toString(), hs.chain.toUpperCase());
                        if (l) lociElements.push(l.elements);
                    }} catch(e) {{}}
                }}
                if (lociElements.length > 0) {{
                    const combined = modelData.union(...lociElements);
                    plugin.managers.structureSelection.fromLoci('hotspot', combined);
                }}
            }}

            plugin.build().toRoot();
            plugin.managers.camera.resetSnapshot();

            window.resetView = () => plugin.managers.camera.resetSnapshot();
            window.toggleSpin = () => {{ const s = plugin.canvas3d?.props; if(s) s.spin = !s.spin; }};
            window.zoomFit = () => plugin.managers.camera.focus();
            window.takeScreenshot = () => plugin.canvas3d?.getImageData()?.toDataURL();
        }}

        init();
        </script>
    </body>
    </html>
    """
    return html


def render_hotspot_bar():
    current_val = ", ".join(f"{h['chain']}/{h['residue_id']}" for h in st.session_state.selected_hotspots)
    if not current_val:
        current_val = st.session_state.get("hotspot_manual_input", "")

    st.markdown("<div class='hotspot-input-section'><span class='hotspot-label-text'>热点</span><span style='color:#94a3b8;font-size:14px;margin-left:2px;'>ⓘ</span>", unsafe_allow_html=True)

    c_input, c_btn = st.columns([6, 4])
    with c_input:
        new_input = st.text_input(
            "",
            value=current_val,
            placeholder="输入残基编号：A/1 表示 A 链上的残基 1（例如：A/1、A/2、A/3）",
            label_visibility="collapsed",
            key="hotspot_manual_input_field"
        )
    with c_btn:
        pred_btn = st.button("🔬 预测热点 (DL)", use_container_width=True, type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if pred_btn:
        predict_hotspots(TOP_K)
        st.rerun()

    if new_input != current_val and new_input.strip():
        parsed = parse_manual_hotspot_input(new_input)
        if parsed:
            st.session_state.selected_hotspots = parsed
            st.rerun()


def render_pipeline_section():
    st.markdown("<div class='pipeline-section'>", unsafe_allow_html=True)

    steps_data = [
        ("① 热点预测", 0),
        ("② RFD3生成", 1),
        ("③ MPNN补序", 2),
        ("④ RF3验证", 3),
    ]

    step_cols = st.columns(4)
    for i, ((label, step), col) in enumerate(zip(steps_data, step_cols)):
        current = st.session_state.current_step
        status = "done" if step < current else ("active" if step == current else "pending")
        icon = "✅" if status == "done" else ("🔄" if status == "active" else "⏳")
        with col:
            st.markdown(f"<div class='pipe-step-item pipe-{status}'>{icon} {label}</div>", unsafe_allow_html=True)

    c_run, c_full, c_rst = st.columns([2, 2, 1])
    with c_run:
        binder_len = st.slider("Binder长度", 40, 150, 80, step=5, label_visibility="collapsed")
    with c_full:
        run_btn = st.button("🚀 运行全流程", type="primary", use_container_width=True)
    with c_rst:
        rst_btn = st.button("🔄 重置全部", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        if st.session_state.atom_array is None:
            st.error("请先上传目标蛋白结构文件")
        elif not st.session_state.selected_hotspots:
            st.warning("请先预测或选择热点残基")
        else:
            run_full_pipeline(binder_len, TOP_K, TOP_K, 2.0)

    if rst_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session()
        st.rerun()

    if st.session_state.rfd3_results:
        render_results()


def render_results():
    st.markdown("### 🧬 设计结果")
    rfd3_results = st.session_state.rfd3_results
    mpnn_results = st.session_state.mpnn_results
    rf3_results = st.session_state.rf3_results

    is_mock = rfd3_results.get("mock", False)
    if is_mock:
        st.warning("⚠️ RFD3/MPNN/RF3未安装，使用模拟数据展示流程")

    t1, t2, t3 = st.tabs(["② RFD3 主链生成", "③ MPNN 序列设计", "④ RF3 结构验证"])

    with t1:
        if rfd3_results.get("success"):
            designs = rfd3_results["designs"]

            st.markdown(f"<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>共生成 {len(designs)} 个Binder主链结构，选取 Top-{TOP_K}</div>", unsafe_allow_html=True)

            dc = st.columns(min(len(designs), 3))
            for i, d in enumerate(designs[:3]):
                with dc[i]:
                    st.markdown(render_rfd3_design_card(d, is_selected=(i == 0)), unsafe_allow_html=True)

            selected_idx = st.selectbox("选择Design查看3D结构", range(len(designs)), format_func=lambda x: f"Design {x+1}", key="rfd3_select")

            selected_design = designs[selected_idx]
            design_pdb_path = selected_design.get("pdb_path", "")

            if design_pdb_path and os.path.exists(design_pdb_path):
                target_pdb = st.session_state.pdb_content
                rfd3_html = render_rfd3_viewer(
                    design_pdb_path=design_pdb_path,
                    target_pdb_content=target_pdb,
                    plddt_data=None,
                    design_index=selected_idx,
                    rank=selected_design.get("rank", selected_idx + 1),
                    height=450
                )
                components.html(rfd3_html, height=480, scrolling=False)

            plddt_val = selected_design.get("plddt", 0)
            mock_plddt = [round(50 + (plddt_val - 50) * (0.7 + 0.6 * np.random.random()), 1) for _ in range(80)] if is_mock else None
            if mock_plddt:
                st.markdown(render_rfd3_plddt_chart(mock_plddt, selected_idx), unsafe_allow_html=True)
        else:
            st.error(f"RFD3生成失败: {rfd3_results.get('error', 'Unknown')}")

    with t2:
        if mpnn_results:
            st.markdown(render_mpnn_sequence_comparison(mpnn_results), unsafe_allow_html=True)
            st.markdown(render_mpnn_score_chart(mpnn_results), unsafe_allow_html=True)
            st.markdown(render_mpnn_legend(), unsafe_allow_html=True)

            df_d = []
            for m in mpnn_results:
                seq = m["sequence"]
                df_d.append({
                    "设计": f"Design {m['design_idx']+1}",
                    "序列": seq[:60] + "..." if len(seq) > 60 else seq,
                    "长度": len(seq),
                    "得分": f"{m.get('score', 0):.2f}",
                })
            with st.expander("📊 查看详细数据表"):
                st.dataframe(pd.DataFrame(df_d), use_container_width=True)
        else:
            st.info("MPNN序列设计未运行")

    with t3:
        if rf3_results:
            st.markdown(render_rf3_summary_metrics(rf3_results), unsafe_allow_html=True)

            selected_rf3_idx = st.selectbox("选择Design查看详情", range(len(rf3_results)), format_func=lambda x: f"Design {rf3_results[x].get('design_idx', x)+1}", key="rf3_select")

            selected_rf3 = rf3_results[selected_rf3_idx]
            st.markdown(render_rf3_result_card(selected_rf3, selected_rf3_idx), unsafe_allow_html=True)

            plddt_data = selected_rf3.get("plddt")
            if plddt_data:
                st.markdown(render_rf3_plddt_chart(plddt_data, selected_rf3.get("design_idx", 0)), unsafe_allow_html=True)

            pae_data = selected_rf3.get("pae")
            if pae_data:
                st.markdown(render_rf3_pae_heatmap(pae_data, selected_rf3.get("design_idx", 0)), unsafe_allow_html=True)

            per_res_rmsd = selected_rf3.get("per_res_rmsd")
            if per_res_rmsd:
                st.markdown(render_rf3_rmsd_chart(per_res_rmsd, selected_rf3.get("design_idx", 0)), unsafe_allow_html=True)

            with st.expander("📊 查看完整数据表"):
                df_r = []
                for r in rf3_results:
                    seq = r["sequence"]
                    status_txt = "✅ 通过" if r["passed"] else "❌ 未通过"
                    df_r.append({
                        "设计": f"Design {r['design_idx']+1}",
                        "序列": seq[:30] + "..." if len(seq) > 30 else seq,
                        "RMSD(Å)": f"{r['rmsd']:.3f}" if r['rmsd'] >= 0 else "N/A",
                        "pLDDT": f"{r.get('avg_plddt', 0):.1f}" if r.get('avg_plddt') else "N/A",
                        "状态": status_txt,
                    })
                st.dataframe(pd.DataFrame(df_r), use_container_width=True)
        else:
            st.info("RF3验证未运行")


def render_welcome():
    st.title("🧬 ODesign - 蛋白质Binder设计系统")
    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### 🚀 快速开始

        1. **上传目标蛋白结构** - 上方选择 PDB/CIF 文件
        2. **查看序列与3D** - 左侧单字母序列 + 右侧3D结构（点击高亮）
        3. **预测热点残基** - DL模型自动选取Top-3 或 手动输入
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

    with c2:
        st.markdown("""
        ### ⚙️ 系统要求

        - Python 3.10+
        - Streamlit 1.30+
        - Biotite 0.38+
        - GPU (可选, 用于DL模型)

        ### 💡 使用提示

        - 点击左侧序列中的**任意残基** → 右侧3D视图**高亮显示**
        - **蓝色标记**为已识别的热点残基
        - 底部输入框支持手动指定热点：`A/1, B/130`
        - 未安装RFD3/MPNN/RF3时自动使用模拟模式
        """)


def main():
    with st.sidebar:
        render_sidebar()

    render_main_page()


if __name__ == "__main__":
    main()
