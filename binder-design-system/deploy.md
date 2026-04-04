# Cloud Studio 部署指南

## 第一步：创建Cloud Studio工作空间

### 1.1 注册登录

1. 访问 https://cloudstudio.net
2. 微信扫码登录
3. 完成实名认证

### 1.2 创建工作空间

1. 点击"新建工作空间"
2. 选择"高性能工作空间"
3. 选择"免费基础型"
4. 选择模板：**PyTorch** 或 **空白模板**
5. 配置：
   - 名称：binder-design-system
   - 描述：蛋白质Binder设计系统
6. 点击"创建"
7. 等待启动（2-5分钟）

---

## 第二步：上传项目文件

### 方式1：使用Git（推荐）

```bash
# 在Cloud Studio终端中执行
cd /root

# 克隆项目（如果有Git仓库）
git clone <your-repo-url>

# 或创建项目目录
mkdir binder-design-system
cd binder-design-system
```

### 方式2：直接上传

1. 在Cloud Studio文件管理器中
2. 右键 → 上传文件夹
3. 选择本地的 `binder-design-system` 文件夹

---

## 第三步：安装依赖

```bash
# 进入项目目录
cd /root/binder-design-system

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import py3Dmol; print('py3Dmol安装成功')"
python -c "import biotite; print(f'Biotite: {biotite.__version__}')"
```

---

## 第四步：运行应用

### 4.1 启动Streamlit

```bash
# 运行应用
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

### 4.2 访问应用

Cloud Studio会自动生成访问链接，格式如：
```
https://xxxxx-8501.app.cloudstudio.work
```

点击链接即可在浏览器中访问应用。

---

## 第五步：测试功能

### 5.1 准备测试文件

下载一个测试PDB文件：

```bash
# 下载PD-1结构文件作为测试
cd data
wget https://files.rcsb.org/download/5ZTN.pdb
```

### 5.2 测试上传

1. 在应用中点击"上传目标蛋白结构文件"
2. 选择 `5ZTN.pdb`
3. 查看是否正确显示3D结构和序列

---

## 常见问题

### Q1: 端口无法访问？

**解决方案**：
```bash
# 检查端口是否被占用
lsof -i:8501

# 如果被占用，杀掉进程
kill -9 <PID>

# 重新启动
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

### Q2: 依赖安装失败？

**解决方案**：
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: 文件上传失败？

**解决方案**：
- 检查文件格式（必须是PDB或CIF）
- 检查文件大小（不要超过200MB）
- 查看终端错误信息

### Q4: 3D可视化不显示？

**解决方案**：
- 检查浏览器是否支持WebGL
- 尝试刷新页面
- 检查控制台是否有JavaScript错误

---

## 后续步骤

完成基础部署后，可以继续：

1. **下载模型权重**
   ```bash
   foundry install rfd3 ligandmpnn rf3
   ```

2. **配置COS存储**
   - 创建COS存储桶
   - 设置环境变量
   - 测试数据上传下载

3. **集成热点预测模型**
   - 复制热点预测模型到项目中
   - 编写集成接口
   - 测试预测功能

4. **实现完整工作流**
   - RFD3骨架生成
   - MPNN序列设计
   - RF3结构验证
   - 结果评估导出

---

## 节省时长技巧

```bash
# 使用完毕后手动关机
# Cloud Studio控制台 → 工作空间列表 → 关机

# 定期检查剩余时长
# 控制台 → 个人中心 → 体验时长

# 批量处理任务
# 一次性上传多个文件，批量处理
```

---

## 备份数据

```bash
# 重要数据保存到COS
# 或下载到本地

# 下载结果文件
# Cloud Studio文件管理器 → 右键 → 下载
```

---

**祝你部署顺利！** 🎉
