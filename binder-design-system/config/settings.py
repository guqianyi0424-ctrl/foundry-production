"""
系统配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 腾讯云COS配置
COS_CONFIG = {
    "secret_id": os.getenv("COS_SECRET_ID", ""),
    "secret_key": os.getenv("COS_SECRET_KEY", ""),
    "region": os.getenv("COS_REGION", "ap-beijing"),
    "bucket": os.getenv("COS_BUCKET", ""),
}

# RFD3配置
RFD3_CONFIG = {
    "default_length": 60,
    "min_length": 40,
    "max_length": 120,
    "diffusion_batch_size": 2,
}

# MPNN配置
MPNN_CONFIG = {
    "sequences_per_design": 3,
    "model_type": "ligand_mpnn",
}

# RF3配置
RF3_CONFIG = {
    "ckpt_path": "rf3",
}

# 筛选阈值
FILTER_THRESHOLDS = {
    "rmsd": 2.0,  # RMSD < 2.0 Å
    "plddt": 80,  # pLDDT > 80
}

# Streamlit配置
STREAMLIT_CONFIG = {
    "page_title": "蛋白质Binder设计系统",
    "page_icon": "🧬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
