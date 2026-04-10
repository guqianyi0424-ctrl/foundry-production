#!/bin/bash
set -e

echo "=========================================="
echo "蛋白质Binder设计系统 - 快速启动"
echo "Top-K=3 | ML+DL | RFD3+MPNN+RF3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

pip list 2>/dev/null | grep streamlit > /dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install -r requirements.txt
fi

mkdir -p outputs data models

PORT=${STREAMLIT_PORT:-8501}

echo "启动应用: http://0.0.0.0:$PORT"
streamlit run app/binder_app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 200
