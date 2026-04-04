# 云平台部署命令

## 🚀 Cloud Studio 部署（推荐）

### 完整部署命令（复制粘贴即可）

```bash
# 1. 进入工作目录
cd /root

# 2. 克隆项目
git clone https://gitee.com/gu-qianyi0424/foundry-production.git

# 3. 进入项目目录
cd foundry-production/binder-design-system

# 4. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 运行应用
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false
```

### 一键部署脚本

```bash
# 复制整段命令直接执行
cd /root && \
git clone https://gitee.com/gu-qianyi0424/foundry-production.git && \
cd foundry-production/binder-design-system && \
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false
```

---

## ☁️ Google Colab 部署

### 在Colab Notebook中运行

```python
# 1. 克隆项目
!git clone https://gitee.com/gu-qianyi0424/foundry-production.git

# 2. 进入项目目录
%cd foundry-production/binder-design-system

# 3. 安装依赖
!pip install -r requirements.txt -q

# 4. 安装pyngrok（用于公网访问）
!pip install pyngrok -q

# 5. 启动Streamlit并暴露公网
from pyngrok import ngrok
import subprocess
import time

# 启动Streamlit
process = subprocess.Popen(['streamlit', 'run', 'app/main.py', '--server.port', '8501', '--server.address', '0.0.0.0'])

# 等待启动
time.sleep(5)

# 创建ngrok隧道
public_url = ngrok.connect(8501)
print(f"访问地址: {public_url}")
```

---

## 🖥️ AutoDL 部署

### SSH连接后执行

```bash
# 1. 克隆项目
cd /root
git clone https://gitee.com/gu-qianyi0424/foundry-production.git

# 2. 进入项目目录
cd foundry-production/binder-design-system

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 运行应用
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0

# 5. 使用AutoDL自定义服务开放8501端口
# 在AutoDL控制台 → 自定义服务 → 开启端口8501
```

---

## 🐳 Docker 部署（高级）

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 克隆项目
RUN git clone https://gitee.com/gu-qianyi0424/foundry-production.git .

# 安装Python依赖
WORKDIR /app/binder-design-system
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t binder-design-system .

# 运行容器
docker run -p 8501:8501 binder-design-system
```

---

## 📦 Kaggle 部署

### 在Kaggle Notebook中运行

```python
# 1. 克隆项目
!git clone https://gitee.com/gu-qianyi0424/foundry-production.git

# 2. 进入项目目录
import os
os.chdir('foundry-production/binder-design-system')

# 3. 安装依赖
!pip install -r requirements.txt -q

# 4. 启动应用
!streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 &

# 5. 查看输出获取访问链接
```

---

## 🔧 常用维护命令

### 更新代码

```bash
cd /root/foundry-production/binder-design-system
git pull origin master
```

### 查看日志

```bash
# Streamlit日志
cat ~/.streamlit/logs/streamlit.log

# 实时查看
tail -f ~/.streamlit/logs/streamlit.log
```

### 重启应用

```bash
# 杀掉旧进程
pkill -f streamlit

# 重新启动
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

### 检查端口占用

```bash
# 查看8501端口
lsof -i:8501

# 或
netstat -tulpn | grep 8501
```

---

## 🎯 推荐部署流程

### 第一次部署

```bash
# Cloud Studio（最简单）
cd /root
git clone https://gitee.com/gu-qianyi0424/foundry-production.git
cd foundry-production/binder-design-system
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

### 后续更新

```bash
cd /root/foundry-production/binder-design-system
git pull
pip install -r requirements.txt
pkill -f streamlit
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

---

## ✅ 验证部署成功

部署成功后，访问应用应该看到：

1. ✅ 页面标题：🧬 蛋白质Binder设计系统
2. ✅ 左侧边栏：文件上传区域
3. ✅ 主区域：欢迎信息和使用说明
4. ✅ 上传PDB文件后显示3D结构

---

**选择适合你的平台，复制命令即可部署！** 🚀
