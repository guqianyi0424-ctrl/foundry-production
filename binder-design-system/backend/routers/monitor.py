from fastapi import APIRouter
import platform
import psutil
import time

router = APIRouter()

_start_time = time.time()


@router.get("/monitor/status", summary="系统状态")
async def system_status():
    gpu_info = "N/A"
    gpu_mem = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            gpu_mem = f"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
    except Exception:
        pass

    return {
        "status": "running",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "used_gb": round(psutil.virtual_memory().used / 1024**3, 1),
            "percent": psutil.virtual_memory().percent,
        },
        "disk": {
            "total_gb": round(psutil.disk_usage("/").total / 1024**3, 1),
            "used_gb": round(psutil.disk_usage("/").used / 1024**3, 1),
            "percent": psutil.disk_usage("/").percent,
        },
        "gpu": {
            "name": gpu_info,
            "memory": gpu_mem,
        },
    }


@router.get("/monitor/tasks", summary="当前运行任务")
async def current_tasks():
    from logger import task_history
    recent = task_history[-20:] if task_history else []
    return {"tasks": recent}


@router.get("/monitor/history", summary="历史任务统计")
async def task_history_stats():
    from logger import task_history
    total = len(task_history)
    success = sum(1 for t in task_history if t["status"] == "success")
    failed = total - success
    return {
        "total_tasks": total,
        "success": success,
        "failed": failed,
        "recent": task_history[-10:] if task_history else [],
    }
