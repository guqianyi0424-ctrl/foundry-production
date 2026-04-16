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
if ! python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
    echo "DGL不可用，安装CPU版本..."
    pip uninstall dgl -y 2>/dev/null || true
    pip install dgl -f https://data.dgl.ai/wheels/repo.html -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
    pip install dgl -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
    echo "DGL安装完成，验证..."
    python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null || echo "⚠️ DGL安装失败，DL模型将不可用"
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
