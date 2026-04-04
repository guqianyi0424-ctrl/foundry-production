#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "蛋白质Binder设计系统 - 启动脚本"
echo "=========================================="

# 进入项目目录
cd /root/binder-design-system

# 检查依赖
echo "检查依赖..."
pip list | grep streamlit > /dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 启动应用
echo "启动Streamlit应用..."
streamlit run app/main.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

echo "应用已启动！"
