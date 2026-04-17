#!/bin/bash

echo "=========================================="
echo "蛋白质Binder设计系统 - 快速启动"
echo "Top-K=3 | ML+DL | RFD3+MPNN+RF3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="binder-design"

if command -v conda &>/dev/null; then
    echo "检测到Conda，使用环境: $CONDA_ENV_NAME"

    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "Conda环境不存在，从environment.yml创建..."
        conda env create -f environment.yml -n "$CONDA_ENV_NAME" 2>/dev/null || {
            echo "environment.yml创建失败，尝试手动安装..."
            conda create -n "$CONDA_ENV_NAME" python=3.10 -y 2>/dev/null
        }
    fi

    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "Conda激活失败，使用系统Python"
    else
        echo "✅ Conda环境已激活: $CONDA_ENV_NAME"
    fi
else
    echo "未检测到Conda，使用系统Python"
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
fi

if ! pip list 2>/dev/null | grep -q streamlit; then
    echo "安装基础依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install -r requirements.txt
fi

echo "检查AutoGluon..."
if ! python -c "import autogluon.tabular; print('AutoGluon版本:', autogluon.tabular.__version__)" 2>/dev/null; then
    echo "AutoGluon未安装，安装0.8.2版本..."
    pip install "autogluon.tabular[all]==0.8.2" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
    pip install "autogluon.tabular[all]==0.8.2" 2>/dev/null || \
    pip install autogluon 2>/dev/null || true
    if python -c "import autogluon.tabular" 2>/dev/null; then
        echo "✅ AutoGluon安装成功"
    else
        echo "⚠️ AutoGluon安装失败，ML模型将不可用"
    fi
else
    echo "AutoGluon已可用"
fi

echo "检查DGL兼容性..."
export DGL_DOWNLOAD=1
export DGLBACKEND=pytorch

if ! python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
    echo "DGL不可用，尝试安装..."

    CUDA_VER=""
    if python -c "import torch; print(torch.version.cuda)" 2>/dev/null; then
        CUDA_VER=$(python -c "
import torch
cv = torch.version.cuda
if cv:
    major = int(cv.split('.')[0])
    minor = int(cv.split('.')[1])
    if major >= 12:
        print('cu121')
    elif major == 11 and minor >= 8:
        print('cu118')
    elif major == 11:
        print('cu117')
    else:
        print(f'cu{major}{minor}')
" 2>/dev/null)
    fi

    pip uninstall dgl -y 2>/dev/null || true

    DGL_INSTALLED=false

    if [ -n "$CUDA_VER" ]; then
        echo "检测到CUDA版本: $CUDA_VER，尝试安装匹配版DGL..."
        pip install dgl -f "https://data.dgl.ai/wheels/$CUDA_VER/repo.html" --quiet 2>/dev/null
        if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
            DGL_INSTALLED=true
        fi
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        echo "尝试安装DGL CPU版本..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl --no-index -f https://data.dgl.ai/wheels/repo.html --quiet 2>/dev/null
        if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
            DGL_INSTALLED=true
        fi
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        echo "尝试从DGL仓库+PyPI安装..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/repo.html --quiet 2>/dev/null || true
        if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
            DGL_INSTALLED=true
        fi
    fi

    if [ "$DGL_INSTALLED" = false ]; then
        echo "尝试从PyPI直接安装..."
        pip uninstall dgl -y 2>/dev/null || true
        pip install dgl --quiet 2>/dev/null || true
        if python -c "import dgl; print('DGL版本:', dgl.__version__)" 2>/dev/null; then
            DGL_INSTALLED=true
        fi
    fi

    if [ "$DGL_INSTALLED" = true ]; then
        echo "✅ DGL安装成功"
    else
        echo "⚠️ DGL安装失败，DL模型将不可用（ML模型仍可正常使用）"
    fi
else
    echo "DGL已可用"
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
