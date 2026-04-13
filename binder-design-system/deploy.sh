#!/bin/bash
set -e

BINDER_DIR="binder-design-system"
PORT=8501

echo "=========================================="
echo "  蛋白质Binder设计系统 - 一键部署"
echo "  Top-K=3 | ML+DL热点预测 | RFD3+MPNN+RF3"
echo "  DL模型: 5折集成推理 (GAT+ESM-2)"
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
    echo "[1/6] 克隆项目..."
    cd "$BASE"
    if [ -d "$BASE/foundry-production" ]; then
        cd foundry-production && git pull && cd ..
    else
        git clone https://gitee.com/gu-qianyi0424/foundry-production.git foundry-production 2>/dev/null || echo "请手动上传项目到 $BASE/foundry-production/"
    fi
else
    echo "[1/6] 项目已存在, 跳过克隆"
    cd "$BASE/foundry-production" && git pull 2>/dev/null || true
fi

echo "[2/6] 创建虚拟环境..."
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[3/6] 安装依赖..."
pip install --upgrade pip -q
pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install -r requirements.txt -q

echo "[4/6] 检查模型权重..."

DL_MODELS_DIR="$BASE/foundry-production/hotspot-prediction/models"
ML_MODELS_DIR="$BASE/foundry-production/ppihotspotid-main/AutogluonModels/ag-20230915_030535"

DL_COUNT=0
for i in 1 2 3 4 5; do
    if [ -f "$DL_MODELS_DIR/best_model_fold${i}.pth" ]; then
        DL_COUNT=$((DL_COUNT + 1))
    fi
done

if [ -d "$ML_MODELS_DIR" ]; then
    echo "  ✅ ML模型 (ppihotspotid AutoGluon): 已就绪"
else
    echo "  ⚠️  ML模型: 未找到，将使用规则预测备用方案"
fi

if [ $DL_COUNT -gt 0 ]; then
    echo "  ✅ DL模型 (hotspot-prediction GAT): ${DL_COUNT}/5折权重已就绪"
else
    echo "  ⚠️  DL模型: 未找到权重，将使用规则预测备用方案"
fi

echo "[5/6] 创建必要目录..."
mkdir -p outputs data models

echo "[6/6] 启动应用..."
echo ""
echo "=========================================="
echo "  系统已启动!"
echo "  访问地址: http://0.0.0.0:$PORT"
echo "  Top-K = 3"
echo "  热点模型: ppihotspotid(ML) + hotspot-prediction(DL, ${DL_COUNT}折集成)"
echo "  设计流程: RFD3 → MPNN → RF3 → RMSD验证"
echo "=========================================="
echo ""

streamlit run app/binder_app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 200
