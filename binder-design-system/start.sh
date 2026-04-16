#!/bin/bash

echo "=========================================="
echo "蛋白质Binder设计系统 - 快速启动"
echo "Top-K=3 | ML+DL | RFD3+MPNN+RF3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if ! pip list 2>/dev/null | grep -q streamlit; then
    echo "安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install -r requirements.txt
fi

echo "检查DGL兼容性..."
export CUDA_VISIBLE_DEVICES=""
export DGL_DOWNLOAD=1
export DGLBACKEND=pytorch

if ! python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
    echo "DGL不可用，尝试安装CPU版本..."

    pip uninstall dgl -y 2>/dev/null || true

    echo "尝试方式1: --no-index (仅从DGL仓库安装CPU版)..."
    pip install dgl --no-index -f https://data.dgl.ai/wheels/repo.html 2>/dev/null

    if ! python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
        echo "方式1失败，尝试方式2: 从DGL仓库+PyPI..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/repo.html 2>/dev/null || true
    fi

    if ! python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
        echo "方式2失败，尝试方式3: 直接从PyPI安装..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl 2>/dev/null || true
    fi

    if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
        echo "✅ DGL安装成功"
    else
        echo "⚠️ DGL安装失败，DL模型将不可用（ML模型仍可正常使用）"
    fi
else
    echo "DGL已可用"
fi

echo "检查XGBoost..."
if ! python -c "import xgboost; print('XGBoost版本:', xgboost.__version__)" 2>/dev/null; then
    echo "安装XGBoost..."
    pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

mkdir -p outputs data models

PORT=${STREAMLIT_PORT:-8501}

echo "启动应用: http://0.0.0.0:$PORT"
streamlit run app/main.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 200
