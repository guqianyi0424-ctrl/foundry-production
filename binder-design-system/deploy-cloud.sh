#!/bin/bash
set -e

echo "=========================================="
echo "  蛋白质Binder设计系统 - 云平台一键部署"
echo "  RFD3 + MPNN + RF3 | GPU推理 | React+FastAPI"
echo "=========================================="

# ==================== 配置 ====================
PROJECT_DIR="/workspace/foundry-production"
BINDER_DIR="$PROJECT_DIR/binder-design-system"
CONDA_ENV="foundry"
CHECKPOINT_DIR="$PROJECT_DIR/foundry-production/checkpoints"
BACKEND_PORT=8000
FRONTEND_PORT=3000

# ==================== 1. 系统依赖 ====================
echo ""
echo "[1/8] 安装系统依赖..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git wget curl \
    nginx supervisor \
    2>/dev/null || true

# 安装 Node.js 20
if ! command -v node &>/dev/null || [[ "$(node -v)" != v20* ]]; then
    echo "  安装 Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>/dev/null
    sudo apt-get install -y nodejs 2>/dev/null
fi
echo "  ✅ Node.js $(node -v)"

# 安装 Miniconda
if ! command -v conda &>/dev/null; then
    echo "  安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3 2>/dev/null
    rm -f /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash 2>/dev/null
fi
eval "$(conda shell.bash hook 2>/dev/null)"
echo "  ✅ Conda $(conda --version 2>/dev/null || echo 'OK')"

# ==================== 2. 创建 foundry 环境 ====================
echo ""
echo "[2/8] 创建 foundry 环境 (Python 3.12)..."

if ! conda env list 2>/dev/null | grep -q "^$CONDA_ENV "; then
    conda create -n $CONDA_ENV python=3.12 -y -q
fi

conda activate $CONDA_ENV 2>/dev/null
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✅ Python $PY_VER"

# ==================== 3. 安装 foundry 核心模型 ====================
echo ""
echo "[3/8] 安装 foundry 核心模型 (RFD3 + MPNN + RF3)..."

cd "$PROJECT_DIR/foundry-production" 2>/dev/null || {
    echo "  ⚠️ foundry-production 子目录不存在，尝试从 PyPI 安装..."
    pip install rc-foundry[all] -q 2>/dev/null || true
}

if [ -d "$PROJECT_DIR/foundry-production" ]; then
    cd "$PROJECT_DIR/foundry-production"
    pip install -e . -q 2>/dev/null || true
    pip install -e ".[rfd3]" -q 2>/dev/null || true
    pip install -e ".[rf3]" -q 2>/dev/null || true
fi

# 验证模型导入
RFD3_OK=false; MPNN_OK=false; RF3_OK=false; HOTSPOT_OK=false
python -c "from rfd3.engine import RFD3InferenceEngine" 2>/dev/null && RFD3_OK=true
python -c "from mpnn.inference_engines.mpnn import MPNNInferenceEngine" 2>/dev/null && MPNN_OK=true
python -c "from rf3.inference_engines.rf3 import RF3InferenceEngine" 2>/dev/null && RF3_OK=true

HOTSPOT_DIR="$PROJECT_DIR/hotspot-prediction"
if [ -d "$HOTSPOT_DIR" ]; then
    pip install dgl -q 2>/dev/null || pip install dgl -f https://data.dgl.ai/wheels/repo.html -q 2>/dev/null || true
    pip install torch-geometric -q 2>/dev/null || true
    pip install transformers -q 2>/dev/null || true
    pip install biopython pandas scipy scikit-learn imbalanced-learn -q 2>/dev/null || true
    python -c "import dgl; import torch; print('DGL OK')" 2>/dev/null && HOTSPOT_OK=true
fi

echo "  RFD3 API: $([ "$RFD3_OK" = true ] && echo '✅' || echo '❌')"
echo "  MPNN API: $([ "$MPNN_OK" = true ] && echo '✅' || echo '❌')"
echo "  RF3  API: $([ "$RF3_OK" = true ] && echo '✅' || echo '❌')"
echo "  Hotspot: $([ "$HOTSPOT_OK" = true ] && echo '✅' || echo '❌')"

# ==================== 4. 检查模型权重 ====================
echo ""
echo "[4/8] 检查模型权重..."

if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/*.ckpt "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1 | grep -q .; then
    echo "  ✅ 模型权重已存在: $CHECKPOINT_DIR"
    ls -lh "$CHECKPOINT_DIR"/*.ckpt "$CHECKPOINT_DIR"/*.pt 2>/dev/null | awk '{print "    " $NF, $5}'
else
    echo "  ⚠️ 未找到模型权重，请确认权重文件位于: $CHECKPOINT_DIR"
    echo "  如需下载，请手动运行:"
    echo "    conda activate foundry"
    echo "    foundry install rfd3 ligandmpnn rf3 --checkpoint-dir $CHECKPOINT_DIR"
fi

# ==================== 5. GPU 验证 ====================
echo ""
echo "[5/8] GPU 环境检查..."

python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  ⚠️ CUDA 不可用，将使用 mock 模式')
    print('  RFD3 和 RF3 需要 GPU 才能运行')
" 2>/dev/null || echo "  ⚠️ PyTorch 未安装"

# ==================== 6. 安装后端依赖 ====================
echo ""
echo "[6/8] 安装后端依赖..."

cd "$BINDER_DIR"
pip install fastapi uvicorn[standard] python-multipart pydantic numpy biotite -q 2>/dev/null

echo "  ✅ 后端依赖已安装"

# ==================== 7. 构建前端 ====================
echo ""
echo "[7/8] 构建前端..."

cd "$BINDER_DIR/frontend"
npm install --silent 2>/dev/null
npm run build 2>/dev/null

echo "  ✅ 前端构建完成"

# ==================== 8. 启动服务 ====================
echo ""
echo "[8/8] 启动服务..."

cd "$BINDER_DIR"

# 停止旧进程
pkill -f "uvicorn backend.main:app" 2>/dev/null || true
pkill -f "nginx" 2>/dev/null || true
sleep 1

# 配置 Nginx
cat > /tmp/odesign-nginx.conf << 'EOF'
server {
    listen 80;
    server_name _;
    client_max_body_size 200M;

    location / {
        root /workspace/foundry-production/binder-design-system/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
EOF

sudo cp /tmp/odesign-nginx.conf /etc/nginx/sites-available/odesign 2>/dev/null || true
sudo ln -sf /etc/nginx/sites-available/odesign /etc/nginx/sites-enabled/odesign 2>/dev/null || true
sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo nginx -t 2>/dev/null && sudo nginx 2>/dev/null || true

# 启动后端
export PYTHONPATH="$BINDER_DIR"
export FOUNDRY_CHECKPOINT_DIRS="$CHECKPOINT_DIR"
nohup conda run --no-banner -n foundry uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --timeout-keep-alive 3600 \
    > /tmp/odesign-backend.log 2>&1 &

BACKEND_PID=$!
echo "  后端 PID: $BACKEND_PID (端口 $BACKEND_PORT)"

# 等待后端启动
sleep 3

# 验证
HEALTH=$(curl -s http://localhost:$BACKEND_PORT/health 2>/dev/null || echo "failed")
if [[ "$HEALTH" == *"ok"* ]]; then
    echo "  ✅ 后端启动成功"
else
    echo "  ⚠️ 后端可能未就绪，查看日志: tail -f /tmp/odesign-backend.log"
fi

echo ""
echo "=========================================="
echo "  🎉 部署完成!"
echo "=========================================="
echo ""
echo "  访问地址: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost')"
echo "  后端API:  http://localhost:$BACKEND_PORT/health"
echo "  后端日志: tail -f /tmp/odesign-backend.log"
echo ""
echo "  GPU状态:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'    ✅ {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)')
else:
    print('    ⚠️ 无GPU，使用mock模式')
" 2>/dev/null || echo "    ⚠️ 无法检测GPU"
echo ""
echo "  模型状态:"
echo "    RFD3: $([ "$RFD3_OK" = true ] && echo '✅ 可用' || echo '❌ 不可用')"
echo "    MPNN: $([ "$MPNN_OK" = true ] && echo '✅ 可用' || echo '❌ 不可用')"
echo "    RF3:  $([ "$RF3_OK" = true ] && echo '✅ 可用' || echo '❌ 不可用')"
echo "    Hotspot: $([ "$HOTSPOT_OK" = true ] && echo '✅ 可用' || echo '❌ 不可用')"
echo ""
echo "  停止服务: pkill -f 'uvicorn backend.main:app'"
echo "=========================================="
