"""
蛋白质Binder设计系统 - 主应用
集成: 热点残基预测(ML+DL) + RFD3 + MPNN + RF3 全流程
Top-K=3
"""
import sys
import os
import time
import tempfile
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

st.set_page_config(
    page_title="蛋白质Binder设计系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
.step-active { background: #e3f2fd; border-left: 4px solid #1976d2; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-weight: 600; }
.step-done { background: #e8f5e9; border-left: 4px solid #388e3c; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
.step-pending { background: #f5f5f5; border-left: 4px solid #bdbdbd; padding: 8px 12px; margin: 4px 0; border-radius: 4px; color: #757575; }
.metric-card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; text-align: center; }
.hotspot-high { background: #ffcdd2 !important; }
.hotspot-mid { background: #fff9c4 !important; }
.hotspot-low { background: #c8e6c9 !important; }
.pipeline-result { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 8px 0; }
.passed { border-left: 4px solid #4CAF50; }
.failed { border-left: 4px solid #f44336; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TOP_K = 3


def init_session():
    defaults = {
        "current_step": 0,
        "atom_array": None,
        "structure_summary": None,
        "hotspot_results_ml": None,
        "hotspot_results_dl": None,
        "selected_hotspots": [],
        "rfd3_results": None,
        "mpnn_results": None,
        "rf3_results": None,
        "job_id": None,
        "pdb_content": None,
        "structure_parser": None,
        "pipeline_running": False,
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


def render_sidebar():
    with st.sidebar:
        st.header("📁 文件上传")
        uploaded_file = st.file_uploader(
            "上传目标蛋白结构文件",
            type=["pdb", "cif"],
            help="支持PDB和CIF格式"
        )

        if uploaded_file:
            st.success(f"已上传: {uploaded_file.name}")

        st.markdown("---")
        st.header("🔥 热点残基预测")

        enable_hotspot = st.checkbox("启用热点残基预测", value=True)

        if enable_hotspot:
            method = st.radio(
                "预测方法",
                options=[
                    "两个模型都运行 (Top-3)",
                    "ppihotspotid (ML)",
                    "hotspot-prediction (DL)"
                ],
                index=0
            )

            top_k = st.slider("Top-K 热点残基", 1, 10, TOP_K, help="选取置信度最高的K个热点残基")

            if st.button("🎯 预测热点残基", type="primary", use_container_width=True):
                if st.session_state.atom_array is not None:
                    with st.spinner("正在预测热点残基..."):
                        predict_hotspots(method, top_k)
                else:
                    st.warning("请先上传蛋白质文件")

        st.markdown("---")
        st.header("⚙️ Binder设计参数")

        binder_length = st.slider("Binder长度 (aa)", 40, 150, 80, step=5)
        num_designs = st.number_input("RFD3生成数量", 1, 10, TOP_K)
        num_sequences = st.number_input("MPNN每设计序列数", 1, 8, TOP_K)
        rmsd_threshold = st.slider("RMSD阈值 (Å)", 0.5, 5.0, 2.0, 0.1)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 运行全流程", type="primary", use_container_width=True):
                if st.session_state.atom_array is None:
                    st.error("请先上传目标蛋白结构文件")
                elif not st.session_state.selected_hotspots:
                    st.warning("请先预测或选择热点残基")
                else:
                    run_full_pipeline(binder_length, num_designs, num_sequences, rmsd_threshold)
        with col2:
            if st.button("🔄 重置", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session()
                st.rerun()

        st.markdown("---")
        st.header("🔧 系统状态")

        rfd3 = RFD3Runner()
        mpnn = MPNNRunner()
        rf3 = RF3Runner()

        st.write(f"RFD3: {'✅ 可用' if rfd3.is_available() else '⚠️ 模拟模式'}")
        st.write(f"MPNN: {'✅ 可用' if mpnn.is_available() else '⚠️ 模拟模式'}")
        st.write(f"RF3: {'✅ 可用' if rf3.is_available() else '⚠️ 模拟模式'}")

    return uploaded_file, binder_length, num_designs, num_sequences, rmsd_threshold


def predict_hotspots(method, top_k):
    predictor = HotspotPredictor(top_k=top_k)
    pdb_string = st.session_state.structure_parser.to_pdb_string(st.session_state.atom_array)

    results_ml = None
    results_dl = None

    if "ML" in method or "两个" in method:
        with st.spinner("运行ML模型 (ppihotspotid)..."):
            try:
                results_ml = predictor.predict(
                    st.session_state.atom_array,
                    method="ml",
                    pdb_string=pdb_string,
                    top_k=top_k
                )
                st.session_state.hotspot_results_ml = results_ml
            except Exception as e:
                st.error(f"ML预测失败: {e}")

    if "DL" in method or "两个" in method:
        with st.spinner("运行DL模型 (hotspot-prediction)..."):
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

    merged_hotspots = merge_hotspot_results(results_ml, results_dl, top_k)
    st.session_state.selected_hotspots = merged_hotspots
    st.session_state.current_step = max(st.session_state.current_step, 1)
    st.success(f"预测完成！选取 Top-{top_k} 热点残基: {', '.join([h['label'] for h in merged_hotspots])}")


def merge_hotspot_results(results_ml, results_dl, top_k):
    all_scores = {}

    if results_ml and "all_scores" in results_ml:
        for label, score_info in results_ml["all_scores"].items():
            if isinstance(score_info, dict):
                ml_s = score_info.get("score", score_info.get("ml_score", 0))
            else:
                ml_s = float(score_info)
            if label not in all_scores:
                all_scores[label] = {"ml": 0, "dl": 0, "residue_name": "", "chain_id": "", "res_id": ""}
            all_scores[label]["ml"] = ml_s
            if isinstance(score_info, dict):
                all_scores[label]["residue_name"] = score_info.get("residue_name", "")
                all_scores[label]["chain_id"] = score_info.get("chain_id", "A")
                all_scores[label]["res_id"] = score_info.get("res_id", "")

    if results_dl and "all_scores" in results_dl:
        for label, score_info in results_dl["all_scores"].items():
            if isinstance(score_info, dict):
                dl_s = score_info.get("score", score_info.get("dl_score", 0))
            else:
                dl_s = float(score_info)
            if label not in all_scores:
                all_scores[label] = {"ml": 0, "dl": 0, "residue_name": "", "chain_id": "", "res_id": ""}
            all_scores[label]["dl"] = dl_s
            if isinstance(score_info, dict):
                if not all_scores[label].get("residue_name"):
                    all_scores[label]["residue_name"] = score_info.get("residue_name", "")
                if not all_scores[label].get("chain_id"):
                    all_scores[label]["chain_id"] = score_info.get("chain_id", "A")
                if not all_scores[label].get("res_id"):
                    all_scores[label]["res_id"] = score_info.get("res_id", "")

    combined = []
    for label, scores in all_scores.items():
        ml_s = scores.get("ml", 0)
        dl_s = scores.get("dl", 0)
        if ml_s > 0 and dl_s > 0:
            combined_score = 0.5 * ml_s + 0.5 * dl_s
        elif ml_s > 0:
            combined_score = ml_s
        else:
            combined_score = dl_s

        chain = scores.get("chain_id", "A")
        res_id = scores.get("res_id", "")
        if not res_id:
            for c in label:
                if c.isdigit():
                    res_id = label[label.index(c):]
                    chain = label[:label.index(c)]
                    break

        combined.append({
            "label": label,
            "chain": chain if chain else "A",
            "residue_id": str(res_id),
            "residue_name": scores.get("residue_name", ""),
            "ml_score": ml_s,
            "dl_score": dl_s,
            "combined_score": combined_score
        })

    combined.sort(key=lambda x: x["combined_score"], reverse=True)

    return combined[:top_k]


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


def render_main_area(uploaded_file):
    if uploaded_file is not None:
        try:
            with st.spinner("解析结构文件..."):
                if st.session_state.atom_array is None or st.session_state.pdb_content is None:
                    parser = StructureParser()
                    atom_array = parser.parse_uploaded_file(uploaded_file)
                    st.session_state.atom_array = atom_array
                    st.session_state.structure_parser = parser
                    st.session_state.structure_summary = parser.get_structure_summary(atom_array)
                    st.session_state.pdb_content = parser.to_pdb_string(atom_array)

            summary = st.session_state.structure_summary

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("原子数", summary["num_atoms"])
            with col2:
                st.metric("残基数", summary["num_residues"])
            with col3:
                st.metric("链数", summary["num_chains"])
            with col4:
                st.metric("序列长度", summary["sequence_length"])

            st.markdown("---")
            render_step_bar()
            st.markdown("---")

            tab1, tab2, tab3 = st.tabs(["🔥 热点预测", "🧬 设计结果", "🎯 3D可视化"])

            with tab1:
                render_hotspot_section()

            with tab2:
                render_pipeline_results()

            with tab3:
                render_3d_view()

        except Exception as e:
            st.error(f"解析文件时出错: {str(e)}")
            st.exception(e)
    else:
        render_welcome()


def render_welcome():
    st.title("🧬 蛋白质Binder设计系统")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 快速开始

        1. **上传目标蛋白结构** - 在左侧边栏上传PDB/CIF文件
        2. **预测热点残基** - 使用ML和DL两个模型预测，自动选取Top-3
        3. **运行全流程** - RFD3生成主链 → MPNN设计序列 → RF3验证结构
        4. **查看结果** - 3D可视化、RMSD分析、导出设计

        ### 📋 工作流程

        ```
        目标蛋白 → 热点预测(ML+DL) → RFD3(Top-3) → MPNN(Top-3) → RF3验证 → RMSD<2Å
        ```

        ### 🔧 集成模型

        | 模型 | 功能 | 说明 |
        |------|------|------|
        | ppihotspotid (ML) | 热点残基预测 | AutoGluon机器学习 |
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
        - GPU (可选, 用于DL模型)

        ### 💡 提示

        - 未安装RFD3/MPNN/RF3时自动使用模拟模式
        - Top-K=3 默认选取3个最优结果
        - RMSD < 2.0 Å 为通过标准
        """)


def render_hotspot_section():
    st.subheader("🔥 热点残基预测结果")

    hotspots = st.session_state.selected_hotspots

    if hotspots:
        col_info1, col_info2 = st.columns([3, 1])
        with col_info2:
            st.metric("检测到的热点残基", len(hotspots))

        cols = st.columns(min(len(hotspots), 5))
        for i, h in enumerate(hotspots[:5]):
            with cols[i % len(cols)]:
                score = h.get("combined_score", 0)
                css = "hotspot-high" if score > 0.7 else "hotspot-mid" if score > 0.4 else "hotspot-low"
                st.markdown(f"""
                <div class="metric-card {css}">
                    <div style="font-size:18px;font-weight:bold;">{h['label']}</div>
                    <div style="font-size:12px;">Chain {h.get('chain','A')} | {h.get('residue_name','')}</div>
                    <div style="font-size:14px;">综合: {score:.3f}</div>
                    <div style="font-size:11px;color:#666;">ML: {h.get('ml_score',0):.3f} | DL: {h.get('dl_score',0):.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📊 详细预测结果"):
            df_data = []
            for h in hotspots:
                df_data.append({
                    "残基": h["label"],
                    "链": h.get("chain", "A"),
                    "残基名": h.get("residue_name", ""),
                    "ML得分": f"{h.get('ml_score', 0):.4f}",
                    "DL得分": f"{h.get('dl_score', 0):.4f}",
                    "综合得分": f"{h.get('combined_score', 0):.4f}"
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        st.info(f"💡 已选取 Top-{len(hotspots)} 热点残基，可在侧边栏点击「🚀 运行全流程」开始设计")
    else:
        st.info("👈 请在侧边栏点击「🎯 预测热点残基」开始预测")


def render_pipeline_results():
    st.subheader("🧬 Binder设计结果")

    if not st.session_state.rfd3_results:
        st.info("请先运行全流程（侧边栏 → 🚀 运行全流程）")
        return

    rfd3_results = st.session_state.rfd3_results
    mpnn_results = st.session_state.mpnn_results
    rf3_results = st.session_state.rf3_results

    is_mock = rfd3_results.get("mock", False)
    if is_mock:
        st.warning("⚠️ RFD3/MPNN/RF3未安装，使用模拟数据展示流程")

    st.markdown("#### Step 2: RFD3 Binder主链生成")
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

    st.markdown("---")
    st.markdown("#### Step 3: MPNN序列设计")
    if mpnn_results:
        df_data = []
        for r in mpnn_results:
            seq = r["sequence"]
            df_data.append({
                "Design": f"D{r['design_idx']+1}",
                "序列长度": len(seq),
                "序列": seq[:50] + "..." if len(seq) > 50 else seq,
                "MPNN得分": f"{r['score']:.2f}"
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        st.caption(f"共设计 {len(mpnn_results)} 条序列，每设计选取 Top-{TOP_K}")
    else:
        st.info("MPNN未运行")

    st.markdown("---")
    st.markdown("#### Step 4: RF3结构验证与RMSD分析")
    if rf3_results:
        passed = [r for r in rf3_results if r["passed"]]
        failed = [r for r in rf3_results if not r["passed"] and r["rmsd"] >= 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总序列数", len(rf3_results))
        with col2:
            st.metric("通过验证 (RMSD<2Å)", len(passed), delta=f"{len(passed)} passed")
        with col3:
            valid_rmsd = [r["rmsd"] for r in rf3_results if r["rmsd"] >= 0]
            avg_rmsd = np.mean(valid_rmsd) if valid_rmsd else 0
            st.metric("平均RMSD", f"{avg_rmsd:.2f} Å")

        df_data = []
        for r in rf3_results:
            status = "✅ 通过" if r["passed"] else "❌ 未通过"
            df_data.append({
                "Design": f"D{r['design_idx']+1}",
                "序列长度": len(r["sequence"]),
                "RMSD (Å)": f"{r['rmsd']:.3f}" if r['rmsd'] >= 0 else "N/A",
                "pLDDT": f"{r.get('avg_plddt', 0):.1f}" if r.get('avg_plddt') else "N/A",
                "状态": status,
                "序列": r["sequence"][:30] + "..."
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        if passed:
            st.success(f"🎉 {len(passed)} 个设计通过验证！")

            best = min(passed, key=lambda x: x["rmsd"])
            st.markdown(f"**🏆 最佳设计**: Design {best['design_idx']+1}, RMSD = {best['rmsd']:.3f} Å, pLDDT = {best.get('avg_plddt', 'N/A')}")

            if best.get("per_res_rmsd"):
                rmsd_html = render_rmsd_chart(best["per_res_rmsd"])
                with st.expander("📊 最佳设计 - 每残基RMSD分布"):
                    components.html(rmsd_html, height=320)

            if best.get("plddt") and isinstance(best["plddt"], list):
                plddt_html = render_plddt_chart(best["plddt"])
                with st.expander("📊 最佳设计 - pLDDT置信度"):
                    components.html(plddt_html, height=280)

            with st.expander("📥 导出结果"):
                export_results(rf3_results, passed)
        else:
            st.warning("没有设计通过RMSD验证，建议调整参数后重试")
    else:
        st.info("RF3验证未运行")


def export_results(rf3_results, passed):
    if not passed:
        st.info("没有通过验证的设计")
        return

    best = min(passed, key=lambda x: x["rmsd"])

    st.markdown("**最佳设计序列**")
    st.code(best["sequence"], language="plaintext")

    st.markdown("**所有通过验证的设计**")
    for i, r in enumerate(passed):
        st.markdown(f"- Design {r['design_idx']+1}: RMSD={r['rmsd']:.3f} Å, pLDDT={r.get('avg_plddt', 'N/A')}")

    if best.get("rf3_pdb") and os.path.exists(best["rf3_pdb"]):
        with open(best["rf3_pdb"], "r") as f:
            pdb_content = f.read()
        st.download_button(
            "📥 下载最佳设计PDB",
            data=pdb_content,
            file_name=f"binder_design_best_rmsd{best['rmsd']:.2f}.pdb",
            mime="chemical/x-pdb"
        )


def render_3d_view():
    st.subheader("🎯 3D结构可视化")

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

    rfd3_pdb = None
    rf3_pdb = None
    rmsd_data = None

    if st.session_state.rf3_results:
        passed = [r for r in st.session_state.rf3_results if r["passed"]]
        if not passed:
            valid = [r for r in st.session_state.rf3_results if r["rmsd"] >= 0]
            if valid:
                passed = [min(valid, key=lambda x: x["rmsd"])]

        if passed:
            best = min(passed, key=lambda x: x["rmsd"])
            if best.get("rf3_pdb") and os.path.exists(best["rf3_pdb"]):
                try:
                    with open(best["rf3_pdb"], "r") as f:
                        rf3_pdb = f.read()
                except Exception:
                    rf3_pdb = None
            if best.get("backbone_pdb") and os.path.exists(best.get("backbone_pdb", "")):
                try:
                    with open(best["backbone_pdb"], "r") as f:
                        rfd3_pdb = f.read()
                except Exception:
                    rfd3_pdb = None
            rmsd_data = best.get("per_res_rmsd")

    molstar_html = render_molstar(
        pdb_content=pdb_content,
        hotspot_residues=hotspot_residues,
        rfd3_pdb=rfd3_pdb,
        rf3_pdb=rf3_pdb,
        rmsd_data=rmsd_data,
        height=650
    )
    components.html(molstar_html, height=670)

    if hotspot_residues:
        st.caption("🟢 Target | 🟠 RFD3 Binder | 🔵 RF3 Prediction | 🔴 Hotspot")


def main():
    uploaded_file, binder_length, num_designs, num_sequences, rmsd_threshold = render_sidebar()
    render_main_area(uploaded_file)


if __name__ == "__main__":
    main()
