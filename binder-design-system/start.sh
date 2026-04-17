#!/bin/bash
set -e

echo "=========================================="
echo "蛋白质Binder设计系统 - 快速启动"
echo "Top-K=3 | ML+DL | RFD3+MPNN+RF3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="binder-design"

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
echo "系统Python版本: $PYTHON_VERSION"

NEED_CONDA=false
if python -c "import sys; exit(0 if sys.version_info < (3, 11) else 1)" 2>/dev/null; then
    :
else
    echo "⚠️ 系统Python ${PYTHON_VERSION} >= 3.11, AutoGluon 0.8.2 需要 Python < 3.11"
    NEED_CONDA=true
fi

if [ "$NEED_CONDA" = true ]; then
    if ! command -v conda &>/dev/null; then
        echo "❌ 需要Conda但未安装! AutoGluon 0.8.2需要Python 3.8-3.10"
        echo "请先安装Miniconda:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh -b"
        echo "  eval \"\$(~/miniconda3/bin/conda shell.bash hook)\""
        exit 1
    fi

    echo "使用Conda创建Python 3.10环境: $CONDA_ENV_NAME"

    if ! conda env list 2>/dev/null | grep -q "^$CONDA_ENV_NAME "; then
        echo "创建新环境..."
        conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    fi

    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$CONDA_ENV_NAME"

    NEW_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✅ Conda环境已激活: $CONDA_ENV_NAME (Python $NEW_VER)"
fi

echo ""
echo "[1/4] 安装基础依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
pip install -r requirements.txt

echo ""
echo "[2/4] 安装AutoGluon 0.8.2..."
if python -c "import autogluon.tabular; print('AutoGluon版本:', autogluon.tabular.__version__)" 2>/dev/null; then
    echo "AutoGluon已可用"
else
    echo "使用--no-deps安装 (绕过torch<1.14约束)..."
    pip install "autogluon.tabular[all]==0.8.2" --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
    pip install "autogluon.tabular[all]==0.8.2" --no-deps 2>/dev/null || \
    pip install "autogluon.tabular==0.8.2" --no-deps 2>/dev/null

    if python -c "import autogluon.tabular" 2>/dev/null; then
        echo "✅ AutoGluon安装成功"
    else
        echo "⚠️ AutoGluon安装失败，ML模型将不可用"
    fi
fi

echo ""
echo "[3/4] 检查DGL..."
export DGL_DOWNLOAD=1
export DGLBACKEND=pytorch

if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
    echo "DGL已可用"
else
    echo "DGL不可用，尝试安装..."

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
    else: print(f'cu{major}{minor}')
" 2>/dev/null)
    fi

    pip uninstall dgl -y 2>/dev/null || true
    DGL_INSTALLED=false

    if [ -n "$CUDA_VER" ]; then
        echo "检测到CUDA: $CUDA_VER"
        pip install dgl -f "https://data.dgl.ai/wheels/$CUDA_VER/repo.html" --quiet 2>/dev/null
        python -c "import dgl" 2>/dev/null && DGL_INSTALLED=true
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl --no-index -f https://data.dgl.ai/wheels/repo.html --quiet 2>/dev/null
        python -c "import dgl" 2>/dev/null && DGL_INSTALLED=true
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/repo.html --quiet 2>/dev/null || true
        python -c "import dgl" 2>/dev/null && DGL_INSTALLED=true
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        pip uninstall dgl -y 2>/dev/null || true
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
