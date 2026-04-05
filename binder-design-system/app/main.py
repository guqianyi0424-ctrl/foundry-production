"""
蛋白质Binder设计系统 - 主应用
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import py3Dmol
import numpy as np
from utils.structure_parser import StructureParser

import streamlit.components.v1 as components

st.set_page_config(
    page_title="蛋白质Binder设计系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 蛋白质Binder设计系统")
st.markdown("---")

if "structure_parser" not in st.session_state:
    st.session_state.structure_parser = StructureParser()
if "atom_array" not in st.session_state:
    st.session_state.atom_array = None
if "structure_summary" not in st.session_state:
    st.session_state.structure_summary = None
if "selected_residues" not in st.session_state:
    st.session_state.selected_residues = set()
if "hotspot_results" not in st.session_state:
    st.session_state.hotspot_results = None

with st.sidebar:
    st.header("📁 文件上传")
    
    uploaded_file = st.file_uploader(
        "上传目标蛋白结构文件",
        type=["pdb", "cif"],
        help="支持PDB和CIF格式"
    )
    
    if uploaded_file is not None:
        st.success(f"已上传: {uploaded_file.name}")
    
    st.markdown("---")
    
    st.header("🔥 热点残基预测")
    
    enable_hotspot = st.checkbox(
        "启用热点残基预测",
        value=False,
        help="使用机器学习/深度学习模型自动预测热点残基"
    )
    
    if enable_hotspot:
        hotspot_method = st.radio(
            "选择预测方法",
            options=["ppihotspotid (ML)", "DeepHotspot (DL)"],
            index=0,
            help="ML: 机器学习方法(快速), DL: 深度学习方法(准确)"
        )
        
        if st.button("开始预测热点残基", type="primary"):
            if st.session_state.atom_array is not None:
                with st.spinner("正在预测热点残基..."):
                    try:
                        from utils.hotspot_predictor import HotspotPredictor
                        
                        predictor = HotspotPredictor()
                        
                        method = "ml" if "ML" in hotspot_method else "dl"
                        results = predictor.predict(
                            st.session_state.atom_array,
                            method=method,
                            pdb_string=st.session_state.structure_parser.to_pdb_string(st.session_state.atom_array)
                        )
                        
                        st.session_state.hotspot_results = results
                        st.session_state.selected_residues.update(results.get("hotspots", []))
                        st.success(f"预测完成！发现 {len(results.get('hotspots', []))} 个热点残基")
                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")
            else:
                st.warning("请先上传蛋白质文件")
    
    st.markdown("---")
    
    st.header("⚙️ Binder参数")
    
    binder_length = st.slider(
        "Binder长度",
        min_value=40,
        max_value=120,
        value=60,
        step=5
    )
    
    num_designs = st.number_input(
        "生成数量",
        min_value=1,
        max_value=10,
        value=3
    )

if uploaded_file is not None:
    try:
        with st.spinner("正在解析结构文件..."):
            atom_array = st.session_state.structure_parser.parse_uploaded_file(uploaded_file)
            st.session_state.atom_array = atom_array
            
            summary = st.session_state.structure_parser.get_structure_summary(atom_array)
            st.session_state.structure_summary = summary
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("原子数", summary["num_atoms"])
        with col_info2:
            st.metric("残基数", summary["num_residues"])
        with col_info3:
            st.metric("链数", summary["num_chains"])
        with col_info4:
            st.metric("序列长度", summary["sequence_length"])
        
        st.markdown("---")
        
        col_seq, col_viz = st.columns([1, 2])
        
        with col_seq:
            st.subheader("🧪 氨基酸序列")
            
            sequence = summary["sequence"]
            residues = st.session_state.structure_parser.get_residue_info(atom_array)
            
            chain_info = {}
            for res in residues:
                chain_id = res.get("chain_id", "A")
                if chain_id not in chain_info:
                    chain_info[chain_id] = []
                chain_info[chain_id].append(res)
            
            for chain_id, chain_residues in chain_info.items():
                st.markdown(f"**{chain_id} protein**")
                
                seq_display = st.empty()
                
                residue_buttons = []
                cols_per_row = 10
                n_rows = (len(chain_residues) + cols_per_row - 1) // cols_per_row
                
                for row_idx in range(n_rows):
                    start_idx = row_idx * cols_per_row
                    end_idx = min(start_idx + cols_per_row, len(chain_residues))
                    
                    row_cols = st.columns(cols_per_row)
                    
                    for i, col in enumerate(row_cols):
                        idx = start_idx + i
                        if idx < len(chain_residues):
                            res = chain_residues[idx]
                            res_label = f"{res['res_name']}{res['res_id']}"
                            is_selected = res_label in st.session_state.selected_residues or idx in [r.get('index', -1) for r in (st.session_state.hotspot_results or {}).get('hotspots_detail', []) if r.get('residue_name') == res['res_name'] and r.get('residue_id') == res['res_id']]
                            
                            button_type = "primary" if is_selected else "secondary"
                            
                            if col.button(
                                res["res_name"],
                                key=f"seq_{chain_id}_{idx}",
                                use_container_width=True,
                                type=button_type
                            ):
                                if res_label in st.session_state.selected_residues:
                                    st.session_state.selected_residues.discard(res_label)
                                else:
                                    st.session_state.selected_residues.add(res_label)
                                st.rerun()
                            
                            if idx % 10 == 9 and idx < len(chain_residues) - 1:
                                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("")
            
            if st.session_state.hotspot_results:
                st.markdown("---")
                st.subheader("🔥 预测结果")
                
                hotspots = st.session_state.hotspot_results.get("hotspots", [])
                scores = st.session_state.hotspot_results.get("scores", {})
                
                st.info(f"检测到 **{len(hotspots)}** 个潜在热点残基")
                
                with st.expander("查看详细得分"):
                    for hotspot in sorted(hotspots)[:20]:
                        score = scores.get(hotspot, 0.0)
                        st.progress(min(score, 1.0), text=f"{hotspot}: {score:.3f}")
        
        with col_viz:
            st.subheader("🎯 3D结构可视化")
            
            pdb_string = st.session_state.structure_parser.to_pdb_string(atom_array)
            
            view = py3Dmol.view(width=700, height=550)
            view.addModel(pdb_string, "pdb")
            view.setStyle({"cartoon": {"color": "spectrum"}})
            
            if st.session_state.selected_residues:
                selected_list = list(st.session_state.selected_residues)
                selection_parts = []
                
                for sel in selected_list[:10]:
                    match = None
                    for res in residues:
                        label = f"{res['res_name']}{res['res_id']}"
                        if label == sel:
                            match = res
                            break
                    
                    if match:
                        chain = match.get("chain_id", "A")
                        res_id = match.get("res_id", "")
                        selection_parts.append((chain, res_id))
                
                if selection_parts:
                    selections = []
                    for chain, res_id in selection_parts[:5]:
                        selections.append({"chain": chain, "resi": int(res_id) if str(res_id).isdigit() else res_id})
                    
                    try:
                        view.setStyle(
                            {"or": selections},
                            {"stick": {"color": "red", "radius": 0.3}, "sphere": {"color": "red"}}
                        )
                    except Exception as e:
                        pass
            
            if st.session_state.hotspot_results:
                hotspots = st.session_state.hotspot_results.get("hotspots_detail", [])
                
                for hotspot in hotspots[:5]:
                    chain = hotspot.get("chain", "A")
                    res_id = hotspot.get("residue_id", "")
                    try:
                        view.setStyle(
                            {"chain": chain, "resi": int(res_id) if str(res_id).isdigit() else res_id},
                            {"stick": {"color": "orange", "radius": 0.25}}
                        )
                    except Exception:
                        pass
            
            view.zoomTo()
            components.html(view._make_html(), width=750, height=600)
            
            col_viz_btn1, col_viz_btn2, col_viz_btn3, col_viz_btn4 = st.columns(4)
            
            with col_viz_btn1:
                if st.button("🔄 载荷配点", use_container_width=True):
                    st.session_state.selected_residues.clear()
                    st.rerun()
            
            with col_viz_btn2:
                if st.button("🎯 指定热点", use_container_width=True):
                    st.info("请在左侧序列中点击选择热点残基")
            
            with col_viz_btn3:
                if st.button("↩️ 重置", use_container_width=True):
                    st.session_state.selected_residues.clear()
                    st.session_state.hotspot_results = None
                    st.rerun()
        
        st.markdown("---")
        st.subheader("📋 已选热点残基")
        
        if st.session_state.selected_residues:
            selected_list = sorted(list(st.session_state.selected_residues))
            st.code(", ".join(selected_list))
            
            st.markdown("**RFD3格式:**")
            rfd3_format = "{"
            for i, sel in enumerate(selected_list[:10]):
                parts = sel.split(")") if ")" in sel else [sel[:-1], sel[-1:]] if len(sel) > 1 else [sel, ""]
                res_name = "".join(filter(str.isalpha, sel))
                res_num = "".join(filter(str.isdigit, sel))
                if i > 0:
                    rfd3_format += ", "
                rfd3_format += f'"{res_name}{res_num}": "CD2,CZ"'
            rfd3_format += "}"
            st.code(rfd3_format)
        else:
            st.info("尚未选择热点残基")
        
        st.markdown("---")
        
        if st.button("🚀 开始Binder设计", type="primary", use_container_width=True):
            if st.session_state.atom_array is None:
                st.error("请先上传目标蛋白结构文件")
            elif not st.session_state.selected_residues and not st.session_state.hotspot_results:
                st.warning("建议选择或预测热点残基以获得更好的设计结果")
            else:
                st.info("Binder设计功能开发中...")
                st.markdown("""
                ### 设计流程
                1. RFD3生成Binder骨架
                2. MPNN设计氨基酸序列  
                3. RF3验证结构
                4. RMSD评估并筛选结果
                """)
    
    except Exception as e:
        st.error(f"解析文件时出错: {str(e)}")
        st.exception(e)

else:
    st.info("👈 请先在侧边栏上传目标蛋白结构文件")
    
    st.markdown("""
    ### 🎯 系统功能
    
    本系统整合多个深度学习模型，实现蛋白质Binder的自动化设计与验证：
    
    - **热点残基预测**: 基于机器学习和深度学习的热点残基识别
    - **Binder骨架生成**: 使用RFD3生成Binder骨架结构
    - **序列设计**: 使用MPNN设计Binder氨基酸序列
    - **结构验证**: 使用RF3预测并验证设计结构
    
    ### 📖 使用步骤
    
    1. 上传目标蛋白结构文件（PDB/CIF格式）
    2. 查看3D结构和氨基酸序列
    3. 点击序列选择热点残基（或启用自动预测）
    4. 配置Binder设计参数
    5. 开始设计并查看结果
    """)
