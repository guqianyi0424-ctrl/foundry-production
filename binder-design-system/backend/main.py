from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from routers import design, upload, jobs

app = FastAPI(
    title="ODesign API",
    description="蛋白质Binder设计系统后端API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["上传"])
app.include_router(design.router, prefix="/api", tags=["设计"])
app.include_router(jobs.router, prefix="/api", tags=["作业"])

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
frontend_dist = os.path.join(PROJECT_DIR, "frontend", "dist")

print(f"[Backend] PROJECT_DIR: {PROJECT_DIR}")
print(f"[Backend] frontend_dist: {frontend_dist}")
print(f"[Backend] dist exists: {os.path.exists(frontend_dist)}")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "odesign-api"}


if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"[Backend] ⚠️ 前端 dist 目录不存在: {frontend_dist}")
    print(f"[Backend] 请先构建前端: cd frontend && npm run build")
