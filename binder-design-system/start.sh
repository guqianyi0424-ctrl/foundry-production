#!/bin/bash
set -e

echo "=========================================="
echo "  DeepBinder - 快速启动"
echo "  React + FastAPI"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="deepbinder"
BACKEND_PORT="${BACKEND_PORT:-8000}"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    if ! conda env list 2>/dev/null | grep -q "^$CONDA_ENV_NAME "; then
        conda env create -f environment.yml
    fi
    conda activate "$CONDA_ENV_NAME"
fi

cd "$SCRIPT_DIR/frontend"
npm install
npm run build

cd "$SCRIPT_DIR/backend"
export PYTHONPATH="$SCRIPT_DIR/backend:$SCRIPT_DIR"
export DEEPBINDER_PIPELINE_RF3_SLOTS="${DEEPBINDER_PIPELINE_RF3_SLOTS:-1}"
export DEEPBINDER_PIPELINE_MAX_RF3_CANDIDATES="${DEEPBINDER_PIPELINE_MAX_RF3_CANDIDATES:-5}"
if [ -z "${DEEPBINDER_DATABASE_URL:-}" ]; then
    echo "错误: 必须设置 PostgreSQL 连接 DEEPBINDER_DATABASE_URL"
    echo "示例: export DEEPBINDER_DATABASE_URL='postgresql+psycopg://deepbinder:password@127.0.0.1:5432/deepbinder'"
    exit 1
fi
python -m uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT"
