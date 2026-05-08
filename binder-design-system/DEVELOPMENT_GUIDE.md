# DeepBinder 开发指南

## 架构

DeepBinder 使用 React 前端和 FastAPI 后端。

```text
frontend/        React + Vite + TypeScript + Zustand + Molstar
backend/         FastAPI routers + services + repositories + tests
utils/           模型 runner、结构解析和可视化辅助工具
config/          模型路径和运行配置
environment.yml  conda Python 3.12 环境
deploy-cloud.sh  云服务器部署脚本
```

## 本地开发

后端：

```bash
cd binder-design-system/backend
python -m uvicorn main:app --reload --port 8000
```

前端：

```bash
cd binder-design-system/frontend
npm install
npm run dev
```

Vite 开发服务器会把 `/api` 代理到 `http://localhost:8000`。

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

云服务器通过 Nginx 服务 `frontend/dist` 并把 `/api` 反向代理到后端。

## 后端边界

路由只负责请求解析和响应组装。业务逻辑放在 `backend/services/`，模型调用通过 `backend/adapters/`，实验记录通过 `backend/repositories/`。

主要入口：

- `backend/main.py`: FastAPI 应用和静态前端服务入口。
- `backend/routers/design.py`: 热点预测、RFD3、MPNN、RF3 和流水线接口。
- `backend/services/design_pipeline.py`: 设计流水线编排。
- `backend/adapters/model_adapters.py`: 模型 runner 适配。
- `backend/config/model_runtime.py`: `DEEPBINDER_*` 运行时环境变量。

## 前端边界

主要入口：

- `frontend/src/pages/NewDesignPage.tsx`: 新建设计工作流。
- `frontend/src/components/SequenceViewer/index.tsx`: 序列视图、靶标范围选择、热点红点。
- `frontend/src/components/MolstarViewer/index.tsx`: 3D 结构加载和持久高亮。
- `frontend/src/store/useAppStore.ts`: 全局状态。

选择状态由 Zustand 统一保存：

- `selectedRange`: 当前靶标范围。
- `selectedHotspots`: 当前热点残基。
- `focusedResidue`: 需要相机聚焦的残基。
- `hoveredResidue`: 临时 hover 残基。

## 测试

后端：

```bash
cd binder-design-system/backend
python -m pytest tests -v
```

前端：

```bash
cd binder-design-system/frontend
npm run build
```

## 部署注意

云服务器如果需要写入系统目录、安装 Nginx 或安装系统依赖，需要操作者手动执行带 `sudo` 的命令。项目自身部署路径以 conda 为准。
