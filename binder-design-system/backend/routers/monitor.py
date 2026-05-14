from fastapi import APIRouter, Depends
import platform
import psutil
import subprocess
import time

from database import User
from routers.auth import require_role

router = APIRouter()

_start_time = time.time()


def _parse_nvidia_smi_csv(output: str) -> dict | None:
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    if not first_line:
        return None
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 4:
        return None
    try:
        memory_used = int(float(parts[1].replace("MiB", "").strip()))
        memory_total = int(float(parts[2].replace("MiB", "").strip()))
        utilization = int(float(parts[3].replace("%", "").strip()))
    except ValueError:
        return None
    return {
        "name": parts[0],
        "memory": f"{memory_used} / {memory_total} MiB",
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "utilization_percent": utilization,
        "source": "nvidia-smi",
    }


def get_gpu_status() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            total_bytes = getattr(properties, "total_memory", getattr(properties, "total_mem", 0))
            total_mib = int(total_bytes / 1024**2)
            allocated_mib = int(torch.cuda.memory_allocated(0) / 1024**2)
            return {
                "name": torch.cuda.get_device_name(0),
                "memory": f"{allocated_mib} / {total_mib} MiB",
                "memory_used_mib": allocated_mib,
                "memory_total_mib": total_mib,
                "utilization_percent": None,
                "source": "torch",
            }
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=2,
            check=True,
        )
        parsed = _parse_nvidia_smi_csv(result.stdout)
        if parsed:
            return parsed
    except Exception:
        pass

    return {
        "name": "N/A",
        "memory": "N/A",
        "memory_used_mib": None,
        "memory_total_mib": None,
        "utilization_percent": None,
        "source": "unavailable",
    }


@router.get("/monitor/status", summary="系统状态")
async def system_status(current_user: User = Depends(require_role(["admin"]))):
    gpu = get_gpu_status()

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
        "gpu": gpu,
    }


@router.get("/monitor/tasks", summary="当前运行任务")
async def current_tasks(current_user: User = Depends(require_role(["admin"]))):
    from logger import task_history
    recent = task_history[-20:] if task_history else []
    return {"tasks": recent}


@router.get("/monitor/history", summary="历史任务统计")
async def task_history_stats(current_user: User = Depends(require_role(["admin"]))):
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
