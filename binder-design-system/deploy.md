# DeepBinder 云服务器部署指南

DeepBinder 云服务器部署使用 conda，不使用容器化部署。React 前端构建为静态文件，FastAPI 后端监听 8000 端口，Nginx 负责静态资源和 `/api` 反向代理。

## 环境

- Conda
- Python 3.12
- Node.js 20+
- Nginx
- FastAPI backend: 8000
- React frontend: `frontend/dist`

## 安装

```bash
cd /root/foundry-production/binder-design-system
conda env create -f environment.yml
conda activate deepbinder
cd frontend
npm install
npm run build
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Nginx

`deploy-cloud.sh` 是 Nginx 反向代理配置的参考脚本。涉及 `sudo` 的系统依赖、Nginx 站点文件写入和服务启动命令，需要由服务器操作者在云服务器上手动执行或授权执行。

核心反代关系：

```nginx
location / {
    root /root/foundry-production/binder-design-system/frontend/dist;
    try_files $uri $uri/ /index.html;
}

location /api/ {
    proxy_pass http://127.0.0.1:8000/api/;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

## 验证

```bash
curl http://127.0.0.1:8000/health
```

期望返回：

```json
{"status":"ok","service":"deepbinder-api","version":"2.0.0"}
```
