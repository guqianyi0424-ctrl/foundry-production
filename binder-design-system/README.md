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

## 安全配置

生产环境建议显式设置以下变量：

```bash
export DEEPBINDER_ENV=production
export DEEPBINDER_SECRET_KEY="请替换为不少于32位的随机密钥"
export DEEPBINDER_CORS_ORIGINS="https://your-domain.example.edu"
export DEEPBINDER_MAX_LOGIN_FAILURES=5
export DEEPBINDER_LOGIN_FAILURE_WINDOW_SECONDS=300
```

其中 `DEEPBINDER_SECRET_KEY` 用于签发 JWT，生产环境不能使用默认开发密钥。`DEEPBINDER_CORS_ORIGINS` 用逗号分隔允许访问后端 API 的前端源。
`DEEPBINDER_MAX_LOGIN_FAILURES` 和 `DEEPBINDER_LOGIN_FAILURE_WINDOW_SECONDS` 用于控制登录失败限流。配置模板见 [.env.example](.env.example)。

## 验证

```bash
cd binder-design-system/backend
PYTHONPATH="$PWD/../..:$PWD/..:$PWD" python -m pytest tests/test_system_acceptance.py tests/test_security_robustness.py -v
cd ../frontend
npm test -- --run
npm run build
```
