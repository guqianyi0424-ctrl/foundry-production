from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Depends as DependsParam
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid

from database import SessionLocal, Experiment, ExperimentDesign, User, ensure_schema_compatibility
from routers.auth import get_current_user
from logger import logger, record_task
from schemas.domain import RFD3JobConfig
from adapters.model_adapters import sanitize_model_result
from services.factory import get_design_services

router = APIRouter()

_pipeline_executor = ThreadPoolExecutor(max_workers=2)
_pipeline_jobs: dict[str, dict] = {}
_pipeline_jobs_lock = Lock()


class HotspotRequest(BaseModel):
    pdb_content: str


class RFD3Request(BaseModel):
    pdb_content: str = ""
    target: Optional[str] = None
    hotspots: Optional[List[str]] = None
    binder_length: int = 80
    length_min: int = 40
    length_max: int = 120
    diffusion_batch_size: int = 2
    n_batches: int = 2
    experiment_id: Optional[str] = None
    task_name: Optional[str] = None
    target_filename: Optional[str] = None
    chain_type: Optional[str] = None


class MPNNRequest(BaseModel):
    backbone_pdb_content: Optional[str] = None
    backbone_pdb_path: Optional[str] = None
    batch_size: int = 10
    fixed_chains: Optional[List[str]] = None
    model_type: str = "ligand_mpnn"
    experiment_id: Optional[str] = None
    preview_only: bool = False


class RF3Request(BaseModel):
    mpnn_pdb_content: str
    rfd3_pdb_content: Optional[str] = None
    example_id: str = "binder_design"
    experiment_id: Optional[str] = None
    preview_only: bool = False


class PipelineRequest(BaseModel):
    pdb_content: str
    hotspots: list[dict]
    binder_length: int = 80
    target: Optional[str] = None
    length_min: int = 40
    length_max: int = 120
    diffusion_batch_size: int = 2
    n_batches: int = 2
    task_name: Optional[str] = None
    target_filename: Optional[str] = None
    chain_type: Optional[str] = None
    async_mode: bool = False


def _pipeline_result_response(result: object) -> dict:
    return {
        "job_id": result.job_id,
        "experiment_id": result.experiment_id,
        "status": result.status,
        "failed_step": result.failed_step,
        "rfd3_results": result.rfd3.data if result.rfd3 else None,
        "mpnn_results": result.mpnn.data if result.mpnn else None,
        "rf3_results": result.rf3.data if result.rf3 else None,
    }


def _set_pipeline_job(job_id: str, updates: dict):
    with _pipeline_jobs_lock:
        current = _pipeline_jobs.get(job_id, {"job_id": job_id})
        current.update(updates)
        _pipeline_jobs[job_id] = current
    persisted_updates = {
        key: value
        for key, value in updates.items()
        if key not in {"job_id", "future"}
    }
    if persisted_updates:
        _update_pipeline_job(job_id, **persisted_updates)


def _create_pipeline_job(job_id: str, status: str = "running"):
    get_design_services().experiments.create_pipeline_job(job_id, status)


def _update_pipeline_job(job_id: str, **updates):
    get_design_services().experiments.update_pipeline_job(job_id, **updates)


def _get_pipeline_job(job_id: str):
    return get_design_services().experiments.get_pipeline_job(job_id)


def _run_pipeline_job(job_id: str, call_args: dict):
    start = time.time()
    try:
        call_args = {
            **call_args,
            "progress_callback": lambda result: _set_pipeline_job(
                job_id,
                _pipeline_result_response(result),
            ),
        }
        result = get_design_services().pipeline.run_pipeline(**call_args)
        duration = time.time() - start
        record_task(
            "run_pipeline",
            "success" if result.status == "completed" else "failed",
            duration,
        )
        _set_pipeline_job(job_id, _pipeline_result_response(result))
    except Exception as e:
        duration = time.time() - start
        record_task("run_pipeline", "failed", duration, {"error": str(e)})
        _set_pipeline_job(
            job_id,
            {
                "status": "failed",
                "error": f"运行失败: {str(e)}",
                "failed_step": "pipeline",
            },
        )


def _save_experiment_step(experiment_id: str, step: str, results: dict, config: dict = None):
    if not experiment_id:
        return
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            return
        if step == "rfd3":
            exp.rfd3_results = results
            if config:
                exp.rfd3_config = config
            exp.status = "rfd3_completed"
        elif step == "mpnn":
            exp.mpnn_results = results
            if config:
                exp.mpnn_config = config
            exp.status = "mpnn_completed"
        elif step == "rf3":
            exp.rf3_results = results
            if config:
                exp.rf3_config = config
            exp.status = "completed"
        from datetime import datetime
        exp.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        logger.error("save_experiment_failed", experiment_id=experiment_id, step=step, error=str(e))
    finally:
        db.close()


def _save_designs(experiment_id: str, designs: list):
    if not experiment_id or not designs:
        return
    ensure_schema_compatibility()
    db = SessionLocal()
    try:
        for d in designs:
            design = ExperimentDesign(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                design_name=d.get("name", ""),
                sequence=d.get("sequence", ""),
                pdb_content=d.get("pdb_content", ""),
                plddt=d.get("plddt"),
                rmsd=d.get("rmsd"),
                ranking_score=d.get("ranking_score"),
                passed_validation=d.get("passed_validation", False),
                plddt_source=d.get("plddt_source"),
                ranking_source=d.get("ranking_source"),
                validation_status=d.get("validation_status"),
            )
            db.add(design)
        db.commit()
    except Exception as e:
        logger.error("save_designs_failed", experiment_id=experiment_id, error=str(e))
    finally:
        db.close()


def _upsert_mpnn_designs(
    experiment_id: str,
    sequences: list,
    backbone_pdb_content: str | None = None,
):
    if not experiment_id or not sequences:
        return
    ensure_schema_compatibility()
    db = SessionLocal()
    try:
        for index, sequence in enumerate(sequences):
            design = None
            backbone_pdb = sequence.get("backbone_pdb_content") or (
                backbone_pdb_content if index == 0 else None
            )
            if backbone_pdb:
                design = (
                    db.query(ExperimentDesign)
                    .filter(ExperimentDesign.experiment_id == experiment_id)
                    .filter(ExperimentDesign.pdb_content == backbone_pdb)
                    .first()
                )
            if not design and index == 0:
                design = (
                    db.query(ExperimentDesign)
                    .filter(ExperimentDesign.experiment_id == experiment_id)
                    .filter(ExperimentDesign.sequence == "")
                    .filter(ExperimentDesign.validation_status != "validated")
                    .order_by(ExperimentDesign.id.asc())
                    .first()
                )
            if not design:
                design = ExperimentDesign(
                    id=str(uuid.uuid4()),
                    experiment_id=experiment_id,
                    design_name=sequence.get("name", f"seq_{sequence.get('index', index)}"),
                    passed_validation=False,
                )
                db.add(design)

            design.design_name = sequence.get("name", design.design_name or f"seq_{index}")
            design.sequence = sequence.get("sequence", "")
            design.pdb_content = sequence.get("pdb_content", "")
            design.ranking_score = sequence.get("score")
            design.ranking_source = "mpnn" if sequence.get("score") is not None else "none"
            design.plddt_source = design.plddt_source or "none"
            design.validation_status = design.validation_status or "not_validated"
        db.commit()
    except Exception as e:
        logger.error("upsert_mpnn_designs_failed", experiment_id=experiment_id, error=str(e))
    finally:
        db.close()


def _update_validated_design(experiment_id: str, result: dict, mpnn_pdb_content: str | None = None):
    if not experiment_id:
        return
    ensure_schema_compatibility()
    db = SessionLocal()
    try:
        design = None
        if mpnn_pdb_content:
            design = (
                db.query(ExperimentDesign)
                .filter(ExperimentDesign.experiment_id == experiment_id)
                .filter(ExperimentDesign.pdb_content == mpnn_pdb_content)
                .first()
            )
        if not design:
            design = (
                db.query(ExperimentDesign)
                .filter(ExperimentDesign.experiment_id == experiment_id)
                .filter(ExperimentDesign.sequence != "")
                .filter(ExperimentDesign.validation_status != "validated")
                .order_by(ExperimentDesign.id.asc())
                .first()
            )
        if not design:
            design = (
                db.query(ExperimentDesign)
                .filter(ExperimentDesign.experiment_id == experiment_id)
                .order_by(ExperimentDesign.id.asc())
                .first()
            )
        if not design:
            return

        summary = result.get("summary") or {}
        avg_plddt = result.get("avg_plddt")
        ranking_score = summary.get("ranking_score")
        design.plddt = avg_plddt if avg_plddt is not None else design.plddt
        design.rmsd = result.get("rmsd")
        design.ranking_score = ranking_score if ranking_score is not None else design.ranking_score
        design.passed_validation = bool(result.get("passed", False))
        design.plddt_source = "rf3" if avg_plddt is not None else design.plddt_source or "none"
        design.ranking_source = "rf3" if ranking_score is not None else design.ranking_source or "none"
        design.validation_status = "validated" if result.get("success") else "failed"
        db.commit()
    except Exception as e:
        logger.error("update_validated_design_failed", experiment_id=experiment_id, error=str(e))
    finally:
        db.close()


def _create_rfd3_experiment(req: RFD3Request, config: dict, user_id: str | None = None) -> str:
    db = SessionLocal()
    try:
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=req.task_name or f"RFD3_{time.strftime('%Y%m%d_%H%M%S')}",
            status="running",
            input_pdb=req.pdb_content,
            target=req.target,
            hotspots=req.hotspots or [],
            rfd3_config=config,
            user_id=user_id,
        )
        db.add(experiment)
        db.commit()
        return experiment.id
    finally:
        db.close()


@router.post("/predict-hotspot", summary="热点残基预测", description="使用ESM-2+GAT模型预测蛋白质热点残基，返回Top-K热点列表")
async def predict_hotspot(req: HotspotRequest):
    start = time.time()
    try:
        result = get_design_services().hotspot_prediction.predict(req.pdb_content, top_k=3)
        if not result.success:
            raise RuntimeError(result.message or result.raw_error or "热点预测失败")
        payload = result.data or {}
        hotspots_detail = payload.get("hotspots_detail", [])
        duration = time.time() - start
        record_task("predict_hotspot", "success", duration)
        return {
            "hotspots": [
                {
                    "chain": h.get("chain", h.get("chain_id", "")),
                    "residue": h.get("residue_id", h.get("res_id", 0)),
                    "residue_name": h.get("residue_name", ""),
                    "score": h.get("score", h.get("combined_score", 0)),
                    "label": h.get("label", ""),
                }
                for h in hotspots_detail
            ],
            "num_hotspots": payload.get("num_hotspots", len(hotspots_detail)),
            "method": payload.get("method", "unknown"),
            "model_loaded": payload.get("model_loaded", False),
            "total_residues": payload.get("total_residues", 0),
        }
    except Exception as e:
        duration = time.time() - start
        record_task("predict_hotspot", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"热点预测失败: {str(e)}")


@router.post("/run-rfd3", summary="RFD3骨架生成", description="使用RFDiffusion3生成蛋白质Binder骨架结构，支持GPU推理和Mock回退")
async def run_rfd3(
    req: RFD3Request,
    current_user: User | None = Depends(get_current_user),
):
    start = time.time()
    try:
        job_id = f"rfd3_{int(time.time())}"
        config = {
            "task_type": "protein" if req.pdb_content else "de_novo",
            "task_id": job_id,
            "task_name": req.task_name,
            "target_filename": req.target_filename,
            "chain_type": req.chain_type or "proteinChain",
            "protein_chain": req.target,
            "target": req.target,
            "hotspots": req.hotspots,
            "binder_length": req.binder_length,
            "length_min": req.length_min,
            "length_max": req.length_max,
            "diffusion_batch_size": req.diffusion_batch_size,
            "n_batches": req.n_batches,
        }
        user_id = (
            current_user.id
            if current_user and not isinstance(current_user, DependsParam)
            else None
        )
        experiment_id = req.experiment_id or _create_rfd3_experiment(req, config, user_id)
        adapter_result = get_design_services().rfd3.run(
            RFD3JobConfig(
                pdb_content=req.pdb_content,
                target=req.target,
                hotspots=req.hotspots or [],
                binder_length=req.binder_length,
                length_min=req.length_min,
                length_max=req.length_max,
                diffusion_batch_size=req.diffusion_batch_size,
                n_batches=req.n_batches,
                job_id=job_id,
            )
        )
        result = adapter_result.data or {
            "success": False,
            "error": adapter_result.message or adapter_result.raw_error,
        }
        duration = time.time() - start
        record_task("run_rfd3", "success", duration)
        _save_experiment_step(experiment_id, "rfd3", sanitize_model_result(result), config)

        if result.get("success") and result.get("designs"):
            designs = []
            for d in result["designs"]:
                designs.append({
                    "name": d.get("name", f"design_{d.get('index', 0)}"),
                    "sequence": "",
                    "pdb_content": d.get("pdb_content", ""),
                    "plddt": d.get("plddt"),
                    "plddt_source": "rfd3" if d.get("plddt") is not None else "none",
                    "ranking_source": "none",
                    "validation_status": "not_validated",
                })
            _save_designs(experiment_id, designs)

        result["experiment_id"] = experiment_id

        return result
    except Exception as e:
        duration = time.time() - start
        record_task("run_rfd3", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"RFD3运行失败: {str(e)}")


@router.post("/run-mpnn", summary="MPNN序列设计", description="使用LigandMPNN基于骨架设计氨基酸序列，支持固定链和批量生成")
async def run_mpnn(req: MPNNRequest):
    start = time.time()
    try:
        job_id = f"mpnn_{int(time.time())}"
        config = {
            "batch_size": req.batch_size,
            "fixed_chains": req.fixed_chains,
            "model_type": req.model_type,
        }
        adapter_result = get_design_services().mpnn.run(
            backbone_pdb_content=req.backbone_pdb_content,
            backbone_pdb_path=req.backbone_pdb_path,
            batch_size=req.batch_size,
            fixed_chains=req.fixed_chains,
            model_type=req.model_type,
            job_id=job_id,
        )
        result = adapter_result.data or {
            "success": False,
            "error": adapter_result.message or adapter_result.raw_error,
        }
        duration = time.time() - start
        record_task("run_mpnn", "success", duration)
        if not req.preview_only:
            _save_experiment_step(req.experiment_id, "mpnn", sanitize_model_result(result), config)

            if result.get("success") and result.get("sequences"):
                _upsert_mpnn_designs(
                    req.experiment_id,
                    result["sequences"],
                    req.backbone_pdb_content,
                )

        return result
    except Exception as e:
        duration = time.time() - start
        record_task("run_mpnn", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"MPNN运行失败: {str(e)}")


@router.post("/run-rf3", summary="RF3结构验证", description="使用RoseTTAFold3验证设计结构的稳定性，计算RMSD、pLDDT等指标")
async def run_rf3(req: RF3Request):
    start = time.time()
    try:
        job_id = f"rf3_{int(time.time())}"
        config = {"example_id": req.example_id}
        adapter_result = get_design_services().rf3.run(
            mpnn_pdb_content=req.mpnn_pdb_content,
            rfd3_pdb_content=req.rfd3_pdb_content,
            example_id=req.example_id,
            job_id=job_id,
        )
        result = adapter_result.data or {
            "success": False,
            "error": adapter_result.message or adapter_result.raw_error,
        }
        duration = time.time() - start
        record_task("run_rf3", "success", duration)
        if not req.preview_only:
            _save_experiment_step(req.experiment_id, "rf3", sanitize_model_result(result), config)

        if result.get("success") and req.experiment_id and not req.preview_only:
            _update_validated_design(req.experiment_id, result, req.mpnn_pdb_content)
            db = SessionLocal()
            try:
                exp = db.query(Experiment).filter(Experiment.id == req.experiment_id).first()
                if exp:
                    exp.duration_seconds = duration
                    from datetime import datetime
                    exp.updated_at = datetime.utcnow()
                    db.commit()
            finally:
                db.close()

        return result
    except Exception as e:
        duration = time.time() - start
        record_task("run_rf3", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"RF3运行失败: {str(e)}")


@router.post("/run-pipeline", summary="完整设计流水线", description="自动执行RFD3→MPNN→RF3全流程")
async def run_pipeline(
    req: PipelineRequest,
    current_user: User | None = Depends(get_current_user),
):
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    start = time.time()

    try:
        call_args = {
            "pdb_content": req.pdb_content,
            "hotspots": req.hotspots,
            "binder_length": req.binder_length,
            "job_id": job_id,
            "user_id": (
                current_user.id
                if current_user and not isinstance(current_user, DependsParam)
                else None
            ),
            "target": req.target,
            "length_min": req.length_min,
            "length_max": req.length_max,
            "diffusion_batch_size": req.diffusion_batch_size,
            "n_batches": req.n_batches,
            "task_name": req.task_name,
            "target_filename": req.target_filename,
            "chain_type": req.chain_type,
        }
        if req.async_mode:
            _create_pipeline_job(job_id, "running")
            _set_pipeline_job(
                job_id,
                {
                    "status": "running",
                    "experiment_id": None,
                    "failed_step": None,
                    "rfd3_results": None,
                    "mpnn_results": None,
                    "rf3_results": None,
                },
            )
            future: Future = _pipeline_executor.submit(_run_pipeline_job, job_id, call_args)
            _set_pipeline_job(job_id, {"future": future})
            return {
                "job_id": job_id,
                "experiment_id": None,
                "status": "running",
                "failed_step": None,
                "rfd3_results": None,
                "mpnn_results": None,
                "rf3_results": None,
            }

        result = get_design_services().pipeline.run_pipeline(**call_args)
        duration = time.time() - start
        record_task(
            "run_pipeline",
            "success" if result.status == "completed" else "failed",
            duration,
        )

        return _pipeline_result_response(result)
    except Exception as e:
        duration = time.time() - start
        record_task("run_pipeline", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"运行失败: {str(e)}")


@router.get("/pipeline-jobs/{job_id}", summary="查询完整流水线作业状态")
async def get_pipeline_job(job_id: str):
    persisted = _get_pipeline_job(job_id)
    if persisted:
        return persisted
    with _pipeline_jobs_lock:
        job = _pipeline_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="流水线作业不存在")
        return {key: value for key, value in job.items() if key != "future"}
