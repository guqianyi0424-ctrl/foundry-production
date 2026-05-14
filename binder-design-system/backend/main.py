from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from sqlalchemy.exc import OperationalError
import os

from routers import design, upload, jobs
from routers import auth, experiments, monitor
from database import init_db
from logger import logger


def load_cors_origins() -> list[str]:
    raw = os.getenv("DEEPBINDER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:5173"]


app = FastAPI(
    title="DeepBinder API",
    description="蛋白质Binder设计系统后端API - 支持用户认证、实验记录管理、系统监控",
    version="2.0.0",
)


def error_code_for_status(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_error",
        503: "service_unavailable",
    }.get(status_code, "request_error")


def error_response(status_code: int, message: str, details=None) -> JSONResponse:
    payload = {
        "code": error_code_for_status(status_code),
        "message": message,
        "details": details,
        "detail": message,
    }
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
    details = None if isinstance(exc.detail, str) else exc.detail
    return error_response(exc.status_code, message, details)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(422, "请求参数校验失败", exc.errors())


@app.exception_handler(OperationalError)
async def database_operational_error_handler(request: Request, exc: OperationalError):
    message = str(exc).lower()
    if "database is locked" in message or "database is busy" in message:
        return error_response(503, "数据库繁忙，请稍后重试")
    logger.error("database_operational_error", path=request.url.path, error=str(exc))
    return error_response(500, "数据库操作失败")

app.add_middleware(
    CORSMiddleware,
    allow_origins=load_cors_origins(),
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
    auth.validate_secret_key_for_environment()
    init_db()
    logger.info("database_initialized")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deepbinder-api", "version": "2.0.0"}


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
