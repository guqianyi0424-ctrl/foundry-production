#!/bin/bash
set -e

BINDER_DIR="binder-design-system"
PORT=8501

echo "=========================================="
echo "  蛋白质Binder设计系统 - 一键部署"
echo "  Top-K=3 | ML+DL热点预测 | RFD3+MPNN+RF3"
echo "=========================================="

if [ -d "/workspace" ]; then
    BASE="/workspace"
elif [ -d "/root" ]; then
    BASE="/root"
else
    BASE="$HOME"
fi

PROJECT_DIR="$BASE/foundry-production/$BINDER_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[1/5] 克隆项目..."
    cd "$BASE"
    if [ -d "$BASE/foundry-production" ]; then
        cd foundry-production && git pull && cd ..
    else
        git clone https://github.com/$(git config --get remote.origin.url 2>/dev/null || echo "your-repo/foundry-production.git") foundry-production 2>/dev/null || echo "请手动上传项目到 $BASE/foundry-production/"
    fi
else
    echo "[1/5] 项目已存在, 跳过克隆"
    cd "$BASE/foundry-production" && git pull 2>/dev/null || true
fi

echo "[2/5] 创建虚拟环境..."
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[3/5] 安装依赖..."
pip install --upgrade pip -q
pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install -r requirements.txt -q

echo "[4/5] 创建必要目录..."
mkdir -p outputs data models

echo "[5/5] 启动应用..."
echo ""
echo "=========================================="
echo "  系统已启动!"
echo "  访问地址: http://0.0.0.0:$PORT"
echo "  Top-K = 3"
echo "  热点模型: ppihotspotid(ML) + hotspot-prediction(DL)"
echo "  设计流程: RFD3 → MPNN → RF3 → RMSD验证"
echo "=========================================="
echo ""

streamlit run app/binder_app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 200
