# DeepBinder 蛋白质 Binder 生成系统

DeepBinder 是基于 React + FastAPI 的蛋白质 Binder 智能设计平台，整合热点残基预测、RFD3 骨架生成、MPNN 序列设计和 RF3 结构验证流程。

## 技术栈

- 前端: React 18, Vite, TypeScript, Zustand, Molstar
- 后端: FastAPI, Python 3.12
- 模型运行: PyTorch, rc-foundry, RFD3, MPNN, RF3
- 部署: conda + Nginx

## 项目结构

```text
binder-design-system/
├── backend/          # FastAPI API、服务层、测试
├── frontend/         # React 前端
├── utils/            # 结构解析、模型 runner、可视化工具
├── config/           # 模型路径与运行配置
├── data/             # 示例数据
├── outputs/          # 设计输出
├── environment.yml   # conda 环境
├── deploy-cloud.sh   # 云服务器部署脚本
└── deploy.md         # 部署说明
```

## 本地开发

后端：

```bash
cd binder-design-system/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd binder-design-system/frontend
npm install
npm run dev
```

## 生产构建

```bash
cd binder-design-system
conda env create -f environment.yml
conda activate deepbinder
cd frontend
npm install
npm run build
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

更多部署细节见 [deploy.md](deploy.md)。

## 验证

```bash
cd binder-design-system/backend
python -m pytest tests -v
cd ../frontend
npm run build
```
