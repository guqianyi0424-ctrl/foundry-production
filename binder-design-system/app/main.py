"""
蛋白质Binder设计系统 - 主应用
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import py3Dmol
from utils.structure_parser import StructureParser

# 页面配置
st.set_page_config(
    page_title="蛋白质Binder设计系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("🧬 蛋白质Binder设计系统")
st.markdown("---")

# 初始化session state
if "structure_parser" not in st.session_state:
    st.session_state.structure_parser = StructureParser()
if "atom_array" not in st.session_state:
    st.session_state.atom_array = None
if "structure_summary" not in st.session_state:
    st.session_state.structure_summary = None

# 侧边栏
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
    
    st.header("ℹ️ 使用说明")
    st.markdown("""
    1. 上传目标蛋白结构文件（PDB/CIF格式）
    2. 查看3D结构和氨基酸序列
    3. 点击残基选择热点残基
    4. 配置Binder设计参数
    5. 开始设计
    """)

# 主内容区域
if uploaded_file is not None:
    try:
        # 解析文件
        with st.spinner("正在解析结构文件..."):
            atom_array = st.session_state.structure_parser.parse_uploaded_file(uploaded_file)
            st.session_state.atom_array = atom_array
            
            # 获取结构摘要
            summary = st.session_state.structure_parser.get_structure_summary(atom_array)
            st.session_state.structure_summary = summary
        
        # 显示结构摘要
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
        
        # 3D可视化和序列显示
        col_viz, col_seq = st.columns([2, 1])
        
        with col_viz:
            st.subheader("🎯 3D结构可视化")
            
            # py3Dmol可视化
            pdb_string = st.session_state.structure_parser.to_pdb_string(atom_array)
            
            view = py3Dmol.view(width=600, height=500)
            view.addModel(pdb_string, "pdb")
            view.setStyle({"cartoon": {"color": "spectrum"}})
            view.zoomTo()
            
            # 显示
            stmol.show_mol_viewer(view)
        
        with col_seq:
            st.subheader("🧪 氨基酸序列")
            
            # 显示序列
            sequence = summary["sequence"]
            st.code(sequence, language="text")
            
            # 显示链信息
            st.markdown("**链信息:**")
            for chain_id in summary["chains"]:
                st.markdown(f"- 链 {chain_id}")
            
            # 显示残基列表（可点击）
            st.markdown("---")
            st.markdown("**残基列表:**")
            
            residues = st.session_state.structure_parser.get_residue_info(atom_array)
            
            # 使用expander显示残基
            with st.expander(f"查看所有残基 ({len(residues)}个)", expanded=False):
                # 分列显示
                cols = st.columns(5)
                for i, residue in enumerate(residues[:100]):  # 限制显示前100个
                    col_idx = i % 5
                    with cols[col_idx]:
                        st.button(
                            residue["label"],
                            key=f"res_{i}",
                            help=f"点击选择残基 {residue['label']}"
                        )
        
        # 热点残基选择区域
        st.markdown("---")
        st.subheader("🔥 热点残基选择")
        
        col_input, col_display = st.columns([1, 1])
        
        with col_input:
            st.markdown("**手动输入热点残基:**")
            st.markdown("格式: `链ID:残基ID:原子列表`，多个用逗号分隔")
            st.markdown("示例: `E:64:CD2,CZ, E:88:CG,CZ`")
            
            hotspot_input = st.text_area(
                "热点残基",
                placeholder="E:64:CD2,CZ, E:88:CG,CZ",
                height=100
            )
            
            if st.button("确认热点残基", type="primary"):
                if hotspot_input:
                    st.session_state.hotspots = hotspot_input
                    st.success(f"已设置热点残基: {hotspot_input}")
                else:
                    st.warning("请输入热点残基")
        
        with col_display:
            st.markdown("**已选择的热点残基:**")
            if "hotspots" in st.session_state and st.session_state.hotspots:
                st.info(st.session_state.hotspots)
            else:
                st.info("尚未选择热点残基")
        
        # Binder设计参数配置
        st.markdown("---")
        st.subheader("⚙️ Binder设计参数")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            binder_length = st.slider(
                "Binder长度",
                min_value=40,
                max_value=120,
                value=60,
                step=5
            )
        
        with col_param2:
            num_designs = st.slider(
                "生成数量",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with col_param3:
            sequences_per_design = st.slider(
                "每个骨架生成序列数",
                min_value=1,
                max_value=5,
                value=2
            )
        
        # 开始设计按钮
        st.markdown("---")
        
        if st.button("🚀 开始Binder设计", type="primary", use_container_width=True):
            if st.session_state.atom_array is None:
                st.error("请先上传目标蛋白结构文件")
            elif "hotspots" not in st.session_state or not st.session_state.hotspots:
                st.warning("建议先选择热点残基，或继续使用默认设置")
            else:
                st.info("Binder设计功能开发中，敬请期待...")
                st.markdown("""
                **设计流程:**
                1. RFD3生成Binder骨架
                2. MPNN设计氨基酸序列
                3. RF3验证结构
                4. 评估并筛选结果
                """)
    
    except Exception as e:
        st.error(f"解析文件时出错: {str(e)}")
        st.exception(e)

else:
    # 未上传文件时显示欢迎信息
    st.info("👈 请先在侧边栏上传目标蛋白结构文件")
    
    st.markdown("""
    ### 🎯 系统功能
    
    本系统整合了多个深度学习模型，实现蛋白质Binder的自动化设计与验证：
    
    - **热点残基预测**: 基于机器学习和深度学习的热点残基识别
    - **Binder骨架生成**: 使用RFD3生成Binder骨架结构
    - **序列设计**: 使用MPNN设计Binder氨基酸序列
    - **结构验证**: 使用RF3预测并验证设计结构
    - **可视化展示**: 3D结构可视化和交互式界面
    
    ### 📖 使用步骤
    
    1. 上传目标蛋白结构文件（PDB/CIF格式）
    2. 查看3D结构和氨基酸序列
    3. 选择热点残基（可选）
    4. 配置Binder设计参数
    5. 开始设计并查看结果
    """)
