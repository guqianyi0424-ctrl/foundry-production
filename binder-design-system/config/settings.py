"""
系统配置文件 - 云服务器部署
Top-K=5, 集成两个热点残基模型
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

COS_CONFIG = {
    "secret_id": os.getenv("COS_SECRET_ID", ""),
    "secret_key": os.getenv("COS_SECRET_KEY", ""),
    "region": os.getenv("COS_REGION", "ap-beijing"),
    "bucket": os.getenv("COS_BUCKET", ""),
}

TOP_K = 5

HOTSPOT_CONFIG = {
    "top_k": TOP_K,
    "ml_threshold": 0.3,
    "dl_threshold": 0.35,
    "combine_strategy": "average",
    "ml_model_path": os.getenv(
        "HOTSPOT_ML_PATH",
        str(PROJECT_ROOT.parent / "ppihotspotid-main" / "AutogluonModels" / "ag-20230915_030535")
    ),
    "dl_models_dir": os.getenv(
        "HOTSPOT_DL_DIR",
        str(PROJECT_ROOT.parent / "hotspot-prediction" / "models")
    ),
    "dl_model_folds": 5,
    "esm2_model": "facebook/esm2_t33_650M_UR50D",
}

RFD3_CONFIG = {
    "default_length": 80,
    "min_length": 40,
    "max_length": 150,
    "diffusion_batch_size": 2,
    "path_env": "RFD3_PATH",
    "default_path": str(PROJECT_ROOT.parent / "RFdiffusion"),
    "fallback_paths": [
        "/workspace/RFdiffusion",
        "/opt/RFdiffusion",
    ],
}

MPNN_CONFIG = {
    "sequences_per_design": TOP_K,
    "model_type": "ligand_mpnn",
    "path_env": "MPNN_PATH",
    "sampling_temp": 0.1,
    "default_path": str(PROJECT_ROOT.parent / "ProteinMPNN"),
    "fallback_paths": [
        "/workspace/ProteinMPNN",
        "/opt/ProteinMPNN",
    ],
}

RF3_CONFIG = {
    "ckpt_path": "rf3",
    "path_env": "RF3_PATH",
    "default_path": str(PROJECT_ROOT.parent / "RoseTTAFold3"),
    "fallback_paths": [
        "/workspace/RoseTTAFold3",
        "/opt/RoseTTAFold3",
    ],
}

FILTER_THRESHOLDS = {
    "rmsd": 2.0,
    "plddt": 80,
}

STREAMLIT_CONFIG = {
    "page_title": "蛋白质Binder设计系统",
    "page_icon": "🧬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

SERVER_CONFIG = {
    "port": int(os.getenv("STREAMLIT_PORT", 8501)),
    "address": os.getenv("STREAMLIT_ADDRESS", "0.0.0.0"),
    "headless": os.getenv("STREAMLIT_HEADLESS", "true").lower() == "true",
    "maxUploadSize": int(os.getenv("STREAMLIT_MAX_UPLOAD_MB", 200)),
}

LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "dir": str(OUTPUT_DIR / "logs"),
}
