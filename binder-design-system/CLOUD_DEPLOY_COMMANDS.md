# DeepBinder 云平台部署命令

## 基础部署

```bash
cd /root
git clone https://gitee.com/gu-qianyi0424/foundry-production.git
cd foundry-production/binder-design-system
conda env create -f environment.yml
conda activate deepbinder
cd frontend
npm install
npm run build
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 一键脚本

```bash
cd /root/foundry-production/binder-design-system
bash deploy-cloud.sh
```

`deploy-cloud.sh` 包含系统依赖、Nginx 配置和服务启动命令。需要 `sudo` 的命令请在云服务器上由操作者手动执行或授权执行。

## 常用维护

```bash
cd /root/foundry-production/binder-design-system
conda activate deepbinder
git pull
cd frontend && npm install && npm run build
cd ../backend
pkill -f 'uvicorn main:app' || true
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 健康检查

```bash
curl http://127.0.0.1:8000/health
```

期望服务名为 `deepbinder-api`。
