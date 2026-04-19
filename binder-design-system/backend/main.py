from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "odesign-api"}
