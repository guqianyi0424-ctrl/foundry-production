from fastapi import APIRouter
from typing import List, Dict

router = APIRouter()

_jobs: List[Dict] = []

@router.get("/jobs")
async def list_jobs():
    return {"jobs": _jobs}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = next((j for j in _jobs if j["id"] == job_id), None)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="作业不存在")
    return job

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    global _jobs
    _jobs = [j for j in _jobs if j["id"] != job_id]
    return {"ok": True}
