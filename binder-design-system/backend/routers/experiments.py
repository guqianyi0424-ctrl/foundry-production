from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
import uuid
import json

from database import get_db, Experiment, ExperimentDesign, User, AuditLog
from routers.auth import require_login, get_current_user

router = APIRouter()


class ExperimentCreate(BaseModel):
    name: str
    input_pdb: Optional[str] = None
    target: Optional[str] = None
    hotspots: Optional[list] = None
    rfd3_config: Optional[dict] = None
    mpnn_config: Optional[dict] = None
    rf3_config: Optional[dict] = None


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    rfd3_results: Optional[dict] = None
    mpnn_results: Optional[dict] = None
    rf3_results: Optional[dict] = None
    duration_seconds: Optional[float] = None
    gpu_info: Optional[str] = None


class DesignCreate(BaseModel):
    design_name: Optional[str] = None
    sequence: Optional[str] = None
    pdb_content: Optional[str] = None
    plddt: Optional[float] = None
    rmsd: Optional[float] = None
    ranking_score: Optional[float] = None
    passed_validation: Optional[bool] = False


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    input_pdb: Optional[str] = None
    target: Optional[str] = None
    hotspots: Optional[list] = None
    rfd3_config: Optional[dict] = None
    mpnn_config: Optional[dict] = None
    rf3_config: Optional[dict] = None
    rfd3_results: Optional[dict] = None
    mpnn_results: Optional[dict] = None
    rf3_results: Optional[dict] = None
    duration_seconds: Optional[float] = None
    gpu_info: Optional[str] = None
    user_id: Optional[str] = None
    designs: Optional[List[dict]] = None

    class Config:
        from_attributes = True


@router.get("/experiments", summary="实验列表(分页+筛选)")
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    query = db.query(Experiment)
    if current_user and current_user.role != "admin":
        query = query.filter(Experiment.user_id == current_user.id)
    if status:
        query = query.filter(Experiment.status == status)
    if keyword:
        query = query.filter(Experiment.name.contains(keyword))
    query = query.order_by(desc(Experiment.created_at))
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    result = []
    for exp in items:
        exp_dict = {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
            "target": exp.target,
            "hotspots": exp.hotspots,
            "duration_seconds": exp.duration_seconds,
            "gpu_info": exp.gpu_info,
            "user_id": exp.user_id,
            "num_designs": len(exp.designs) if exp.designs else 0,
        }
        result.append(exp_dict)

    return {"total": total, "page": page, "page_size": page_size, "items": result}


@router.get("/experiments/{experiment_id}", summary="实验详情")
async def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="实验记录不存在")

    designs = []
    for d in exp.designs:
        designs.append({
            "id": d.id,
            "design_name": d.design_name,
            "sequence": d.sequence,
            "pdb_content": d.pdb_content,
            "plddt": d.plddt,
            "rmsd": d.rmsd,
            "ranking_score": d.ranking_score,
            "passed_validation": d.passed_validation,
        })

    return {
        "id": exp.id,
        "name": exp.name,
        "status": exp.status,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        "input_pdb": exp.input_pdb,
        "target": exp.target,
        "hotspots": exp.hotspots,
        "rfd3_config": exp.rfd3_config,
        "mpnn_config": exp.mpnn_config,
        "rf3_config": exp.rf3_config,
        "rfd3_results": exp.rfd3_results,
        "mpnn_results": exp.mpnn_results,
        "rf3_results": exp.rf3_results,
        "duration_seconds": exp.duration_seconds,
        "gpu_info": exp.gpu_info,
        "user_id": exp.user_id,
        "designs": designs,
    }


@router.post("/experiments", summary="创建实验记录")
async def create_experiment(
    req: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    exp = Experiment(
        id=str(uuid.uuid4()),
        name=req.name,
        status="created",
        input_pdb=req.input_pdb,
        target=req.target,
        hotspots=req.hotspots,
        rfd3_config=req.rfd3_config,
        mpnn_config=req.mpnn_config,
        rf3_config=req.rf3_config,
        user_id=current_user.id if current_user else None,
    )
    db.add(exp)

    if current_user:
        audit = AuditLog(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            action="create_experiment",
            target=exp.id,
            detail={"name": req.name},
        )
        db.add(audit)

    db.commit()
    db.refresh(exp)
    return {"id": exp.id, "name": exp.name, "status": exp.status, "created_at": exp.created_at.isoformat()}


@router.put("/experiments/{experiment_id}", summary="更新实验记录")
async def update_experiment(
    experiment_id: str,
    req: ExperimentUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="实验记录不存在")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(exp, key, value)
    exp.updated_at = datetime.utcnow()

    if current_user:
        audit = AuditLog(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            action="update_experiment",
            target=experiment_id,
            detail=update_data,
        )
        db.add(audit)

    db.commit()
    return {"ok": True, "id": exp.id}


@router.delete("/experiments/{experiment_id}", summary="删除实验")
async def delete_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_login),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="实验记录不存在")

    if current_user.role != "admin" and exp.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权删除此实验")

    audit = AuditLog(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action="delete_experiment",
        target=experiment_id,
        detail={"name": exp.name},
    )
    db.add(audit)
    db.delete(exp)
    db.commit()
    return {"ok": True}


@router.post("/experiments/{experiment_id}/designs", summary="添加设计结果")
async def add_design(
    experiment_id: str,
    req: DesignCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="实验记录不存在")

    design = ExperimentDesign(
        id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        design_name=req.design_name,
        sequence=req.sequence,
        pdb_content=req.pdb_content,
        plddt=req.plddt,
        rmsd=req.rmsd,
        ranking_score=req.ranking_score,
        passed_validation=req.passed_validation,
    )
    db.add(design)
    db.commit()
    return {"ok": True, "id": design.id}


@router.get("/experiments/{experiment_id}/export", summary="导出实验报告")
async def export_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="实验记录不存在")

    designs = []
    for d in exp.designs:
        designs.append({
            "design_name": d.design_name,
            "sequence": d.sequence,
            "plddt": d.plddt,
            "rmsd": d.rmsd,
            "ranking_score": d.ranking_score,
            "passed_validation": d.passed_validation,
        })

    report = {
        "experiment": {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "target": exp.target,
            "hotspots": exp.hotspots,
            "rfd3_config": exp.rfd3_config,
            "mpnn_config": exp.mpnn_config,
            "rf3_config": exp.rf3_config,
            "duration_seconds": exp.duration_seconds,
            "gpu_info": exp.gpu_info,
        },
        "results": {
            "rfd3": exp.rfd3_results,
            "mpnn": exp.mpnn_results,
            "rf3": exp.rf3_results,
        },
        "designs": designs,
    }
    return report


@router.post("/experiments/compare", summary="实验对比")
async def compare_experiments(
    experiment_ids: List[str],
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    experiments = db.query(Experiment).filter(Experiment.id.in_(experiment_ids)).all()
    if not experiments:
        raise HTTPException(status_code=404, detail="未找到实验记录")

    comparison = []
    for exp in experiments:
        rf3 = exp.rf3_results or {}
        comparison.append({
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "duration_seconds": exp.duration_seconds,
            "num_designs": len(exp.designs) if exp.designs else 0,
            "rf3_summary": rf3.get("summary") if isinstance(rf3, dict) else None,
            "rmsd": rf3.get("rmsd") if isinstance(rf3, dict) else None,
            "avg_plddt": rf3.get("avg_plddt") if isinstance(rf3, dict) else None,
            "passed": rf3.get("passed") if isinstance(rf3, dict) else None,
        })

    return {"comparison": comparison}
