from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
import os

from routers import design, upload, jobs
from routers import auth, experiments, monitor
from database import init_db
from logger import logger

app = FastAPI(
    title="ODesign API",
    description="蛋白质Binder设计系统后端API - 支持用户认证、实验记录管理、系统监控",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api", tags=["认证"])
app.include_router(upload.router, prefix="/api", tags=["上传"])
app.include_router(design.router, prefix="/api", tags=["设计"])
app.include_router(experiments.router, prefix="/api", tags=["实验记录"])
app.include_router(jobs.router, prefix="/api", tags=["作业"])
app.include_router(monitor.router, prefix="/api", tags=["系统监控"])

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
frontend_dist = os.path.join(PROJECT_DIR, "frontend", "dist")

logger.info("backend_starting", project_dir=PROJECT_DIR, frontend_dist=frontend_dist)


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("database_initialized")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "odesign-api", "version": "2.0.0"}


if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    logger.warning("frontend_dist_not_found", path=frontend_dist)
