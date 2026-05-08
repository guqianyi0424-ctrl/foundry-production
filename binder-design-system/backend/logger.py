import logging
import json
import time
from functools import wraps
from datetime import datetime


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg: str, **kwargs):
        self.logger.info(json.dumps({"level": "info", "msg": msg, "ts": datetime.utcnow().isoformat(), **kwargs}, default=str, ensure_ascii=False))

    def error(self, msg: str, **kwargs):
        self.logger.error(json.dumps({"level": "error", "msg": msg, "ts": datetime.utcnow().isoformat(), **kwargs}, default=str, ensure_ascii=False))

    def warning(self, msg: str, **kwargs):
        self.logger.warning(json.dumps({"level": "warning", "msg": msg, "ts": datetime.utcnow().isoformat(), **kwargs}, default=str, ensure_ascii=False))


logger = StructuredLogger("deepbinder")


def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info("task_completed", function=func.__name__, duration=f"{duration:.2f}s", status="success")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error("task_failed", function=func.__name__, duration=f"{duration:.2f}s", error=str(e))
            raise
    return wrapper


task_history: list = []


def record_task(task_name: str, status: str, duration: float, detail: dict = None):
    task_history.append({
        "name": task_name,
        "status": status,
        "duration": f"{duration:.2f}s",
        "timestamp": datetime.utcnow().isoformat(),
        "detail": detail or {},
    })
    if len(task_history) > 200:
        task_history.pop(0)
