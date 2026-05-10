# DeepBinder 云平台部署命令

## 基础部署

```bash
cd /root
git clone https://gitee.com/gu-qianyi0424/foundry-production.git
cd foundry-production/binder-design-system
conda env create -f environment.yml
conda activate deepbinder
cp .env.example .env
# 生产环境必须替换为不少于32位的随机密钥，并把域名改成你的前端访问地址。
export DEEPBINDER_ENV=production
export DEEPBINDER_SECRET_KEY="replace-with-at-least-32-random-characters"
export DEEPBINDER_CORS_ORIGINS="https://your-domain.example.edu"
export DEEPBINDER_MAX_LOGIN_FAILURES=5
export DEEPBINDER_LOGIN_FAILURE_WINDOW_SECONDS=300
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
export DEEPBINDER_ENV=production
export DEEPBINDER_SECRET_KEY="replace-with-at-least-32-random-characters"
export DEEPBINDER_CORS_ORIGINS="https://your-domain.example.edu"
cd frontend && npm install && npm run build
cd ../backend
pkill -f 'uvicorn main:app' || true
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 安全配置检查

云平台训练或演示前建议确认：

- `DEEPBINDER_SECRET_KEY` 已替换，且长度不少于 32 位。
- `DEEPBINDER_CORS_ORIGINS` 只包含实际前端域名或本机调试地址，不使用 `*`。
- 登录失败限流默认 5 次/300 秒，可通过 `DEEPBINDER_MAX_LOGIN_FAILURES` 和 `DEEPBINDER_LOGIN_FAILURE_WINDOW_SECONDS` 调整。
- 管理员功能入口和 `/api/monitor/*` 监控接口只允许管理员 token 访问。

## 健康检查

```bash
curl http://127.0.0.1:8000/health
```

期望服务名为 `deepbinder-api`。
