#!/bin/bash

echo "=========================================="
echo "蛋白质Binder设计系统 - 快速启动"
echo "Top-K=3 | ML+DL | RFD3+MPNN+RF3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="binder-design"
CONDA_PYTHON="$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin/python"
CONDA_PIP="$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin/pip"

if ! command -v conda &>/dev/null; then
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    else
        echo "安装Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" 2>/dev/null
        rm -f /tmp/miniconda.sh
        export PATH="$HOME/miniconda3/bin:$PATH"
        conda init bash 2>/dev/null
        echo "✅ Miniconda安装完成"
    fi
fi

eval "$(conda shell.bash hook 2>/dev/null)"

if ! conda env list 2>/dev/null | grep -q "^$CONDA_ENV_NAME "; then
    echo "创建Python 3.10环境: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y 2>/dev/null
fi

conda activate "$CONDA_ENV_NAME" 2>/dev/null

PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VER" != "3.10" ]; then
    echo "❌ Python版本错误: $PY_VER (需要3.10)"
    echo "删除旧环境并重新创建..."
    conda deactivate 2>/dev/null
    conda env remove -n "$CONDA_ENV_NAME" -y 2>/dev/null
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y 2>/dev/null
    conda activate "$CONDA_ENV_NAME" 2>/dev/null
    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
fi
echo "✅ Python版本: $PY_VER"

echo ""
echo "[1/4] 安装基础依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
pip install -r requirements.txt

echo ""
echo "[2/4] 安装AutoGluon 0.8.2..."
if python -c "import autogluon.tabular" 2>/dev/null; then
    echo "AutoGluon已可用"
else
    echo "使用--no-deps安装..."
    pip install "autogluon.tabular[all]==0.8.2" --no-deps 2>/dev/null || \
    pip install "autogluon.tabular==0.8.2" --no-deps 2>/dev/null

    if python -c "import autogluon.tabular" 2>/dev/null; then
        echo "✅ AutoGluon安装成功"
    else
        echo "⚠️ AutoGluon安装失败，ML模型将不可用"
    fi
fi

echo ""
echo "[3/4] 安装DGL..."
export DGL_DOWNLOAD=1
export DGLBACKEND=pytorch

if python -c "import dgl" 2>/dev/null; then
    echo "DGL已可用"
else
    CUDA_VER=""
    if python -c "import torch; v=torch.version.cuda; assert v" 2>/dev/null; then
        CUDA_VER=$(python -c "
import torch
cv = torch.version.cuda
if cv:
    major = int(cv.split('.')[0])
    minor = int(cv.split('.')[1])
    if major >= 12: print('cu121')
    elif major == 11 and minor >= 8: print('cu118')
    elif major == 11: print('cu117')
" 2>/dev/null)
    fi

    pip uninstall dgl -y 2>/dev/null || true
    DGL_INSTALLED=false

    if [ -n "$CUDA_VER" ]; then
        echo "PyTorch CUDA: $CUDA_VER, 安装匹配DGL..."
        pip install dgl -f "https://data.dgl.ai/wheels/$CUDA_VER/repo.html" --quiet 2>/dev/null
        python -c "import dgl" 2>/dev/null && DGL_INSTALLED=true
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        echo "尝试DGL CPU版本..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/repo.html --quiet 2>/dev/null || \
        pip install dgl --quiet 2>/dev/null || true
        python -c "import dgl" 2>/dev/null && DGL_INSTALLED=true
    fi

    if [ "$DGL_INSTALLED" = true ]; then
        echo "✅ DGL安装成功"
    else
        echo "⚠️ DGL安装失败，DL模型将不可用"
    fi
fi

echo ""
echo "[4/4] 启动应用..."
mkdir -p outputs data models

PORT=${STREAMLIT_PORT:-8501}

echo ""
echo "=========================================="
echo "  系统已启动!"
echo "  访问地址: http://0.0.0.0:$PORT"
echo "=========================================="

streamlit run app/main.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 200
