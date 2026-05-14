# 基于深度学习的蛋白质Binder生成系统

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

本系统整合了多个深度学习模型，实现蛋白质Binder的自动化设计与验证。这是一个本科毕业设计项目，包含完整的Web界面、热点残基预测和Binder设计流程。

### 🎯 核心功能

- **热点残基预测**: 基于机器学习和深度学习的热点残基识别
- **Binder骨架生成**: 使用RFD3生成Binder骨架结构
- **序列设计**: 使用MPNN设计Binder氨基酸序列
- **结构验证**: 使用RF3预测并验证设计结构
- **可视化展示**: 3D结构可视化和交互式界面

---

## 📂 项目结构

```
foundry-production/
├── binder-design-system/       # DeepBinder Web应用
│   ├── backend/                # FastAPI API、服务层、测试
│   ├── frontend/               # React + Vite + TypeScript 前端
│   ├── utils/                  # 工具模块
│   │   ├── structure_parser.py # 蛋白质结构解析
│   │   └── ...
│   ├── config/                 # 配置文件
│   ├── data/                   # 测试数据
│   ├── outputs/                # 输出结果
│   ├── environment.yml         # Conda 环境
│   └── README.md              # 详细说明
│
├── hotspot-prediction/          # 深度学习热点预测模型
│   ├── models/                 # 训练好的模型权重
│   ├── dataset.py              # 数据集处理
│   ├── model.py                # 模型定义
│   ├── train.py                # 训练脚本
│   ├── main.py                 # 主程序
│   └── README.md              # 详细说明
│
├── ppihotspotid-main/           # 机器学习热点预测模型
│   └── ...                     # 相关代码和数据
│
├── docs/                        # 文档
│   ├── BINDER_DESIGN_SYSTEM_PRD.md          # 产品需求文档
│   ├── BUDGET_DEPLOYMENT_GUIDE.md           # 低成本部署指南
│   ├── CLOUD_DATA_STORAGE_GUIDE.md          # 云存储指南
│   └── CLOUD_STUDIO_COS_DEPLOYMENT_GUIDE.md # Cloud Studio部署指南
│
├── .gitignore                   # Git忽略文件
└── README.md                    # 项目说明（本文件）
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://gitee.com/gu-qianyi0424/foundry-production.git
cd foundry-production
```

### 2. 安装依赖

```bash
cd binder-design-system
conda env create -f environment.yml
conda activate deepbinder
```

### 3. 运行后端

```bash
cd binder-design-system/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 运行前端

```bash
cd binder-design-system/frontend
npm install
npm run dev
```

---

## ☁️ 云平台部署

### Cloud Studio（推荐）

```bash
export DEEPBINDER_ENV=production
export DEEPBINDER_SECRET_KEY="请替换为不少于32位的随机密钥"
export DEEPBINDER_CORS_ORIGINS="https://your-domain.example.edu"
export DEEPBINDER_ALLOW_MOCK=0
export DEEPBINDER_BOOTSTRAP_ADMIN_USERNAME="admin"
export DEEPBINDER_BOOTSTRAP_ADMIN_PASSWORD="请替换为强密码"

cd foundry-production/binder-design-system/frontend
npm install
npm run build

cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

详细部署说明请查看 [binder-design-system/CLOUD_DEPLOY_COMMANDS.md](binder-design-system/CLOUD_DEPLOY_COMMANDS.md)

---

## 🔬 技术栈

### 前端
- **React 18 / Vite / TypeScript**: Web应用框架
- **Molstar**: 3D分子可视化

### 后端
- **Python 3.12**: 主要编程语言
- **FastAPI**: API服务
- **Biotite**: 蛋白质结构处理
- **PyTorch**: 深度学习框架

### 深度学习模型
- **RFD3**: Binder骨架生成
- **MPNN**: 序列设计
- **RF3**: 结构预测验证

### 云服务
- **腾讯云COS**: 对象存储
- **Cloud Studio**: 在线开发环境

---

## 📊 系统架构

```
输入: 目标蛋白结构 (PDB/CIF)
    ↓
热点残基预测 (可选)
    ├── 深度学习模型 (hotspot-prediction)
    └── 机器学习模型 (ppihotspotid-main)
    ↓
RFD3骨架生成
    ↓
MPNN序列设计
    ↓
RF3结构验证
    ↓
输出: Binder候选结构
```

---

## 🎓 毕业设计信息

- **题目**: 基于深度学习的蛋白质Binder生成系统设计与实现
- **类型**: 本科毕业设计
- **时间**: 2024-2025学年

---

## 📝 开发进度

- [x] 项目结构搭建
- [x] 基础Streamlit应用
- [x] 文件上传与解析
- [x] 3D结构可视化
- [x] 热点残基预测模型训练
- [ ] 热点残基预测集成
- [ ] RFD3骨架生成
- [ ] MPNN序列设计
- [ ] RF3结构验证
- [ ] 结果评估与导出

## 🧹 仓库治理

模型权重、训练特征、推理输出、日志和大体积实验产物不建议直接进入 Git 历史。当前项目包含热点预测模型、特征数据和 AutoGluon 模型目录，后续维护建议：

- 将 `*.pth`、`*.pkl`、`*.npy`、训练日志和大体积结果迁移到 Git LFS、对象存储或发布包。
- 仓库内只保留小型示例数据、下载脚本和模型校验信息。
- 不直接重写公共分支历史；需要清理历史体积时，先和协作者确认窗口，再使用 `git filter-repo` 或仓库托管平台的 LFS 迁移工具。

---

## 📖 文档

- [产品需求文档](docs/BINDER_DESIGN_SYSTEM_PRD.md)
- [部署指南](binder-design-system/deploy.md)
- [云平台部署命令](binder-design-system/CLOUD_DEPLOY_COMMANDS.md)
- [低成本部署方案](docs/BUDGET_DEPLOYMENT_GUIDE.md)
- [云存储方案](docs/CLOUD_DATA_STORAGE_GUIDE.md)

---

## 🤝 贡献

这是一个毕业设计项目，暂不接受外部贡献。

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE.md) 文件

---

## 📧 联系方式

如有问题，请通过以下方式联系：

- 提交 Issue: https://gitee.com/gu-qianyi0424/foundry-production/issues
- 邮箱: (你的邮箱)

---

## 🙏 致谢

感谢以下开源项目：

- [rc-foundry](https://github.com/roscoe-ai/foundry-production) - RFD3/MPNN/RF3模型框架
- [Streamlit](https://streamlit.io/) - Web应用框架
- [py3Dmol](https://3dmol.csb.pitt.edu/) - 分子可视化

---

**⭐ 如果这个项目对你有帮助，请给一个Star！**
