from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hotspot_predictor import HotspotPredictor
from utils.rfd3_runner import RFD3Runner
from utils.mpnn_runner import MPNNRunner
from utils.rf3_runner import RF3Runner
from database import SessionLocal, Experiment, ExperimentDesign
from routers.auth import get_current_user
from logger import logger, record_task

router = APIRouter()

predictor = HotspotPredictor(top_k=5)


class HotspotRequest(BaseModel):
    pdb_content: str


class RFD3Request(BaseModel):
    pdb_content: str
    target: Optional[str] = None
    hotspots: Optional[List[str]] = None
    binder_length: int = 80
    length_min: int = 40
    length_max: int = 120
    diffusion_batch_size: int = 2
    n_batches: int = 2
    experiment_id: Optional[str] = None


class MPNNRequest(BaseModel):
    backbone_pdb_content: Optional[str] = None
    backbone_pdb_path: Optional[str] = None
    batch_size: int = 10
    fixed_chains: Optional[List[str]] = None
    model_type: str = "ligand_mpnn"
    experiment_id: Optional[str] = None


class RF3Request(BaseModel):
    mpnn_pdb_content: str
    rfd3_pdb_content: Optional[str] = None
    example_id: str = "binder_design"
    experiment_id: Optional[str] = None


class PipelineRequest(BaseModel):
    pdb_content: str
    hotspots: list[dict]
    binder_length: int = 80


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
            )
            db.add(design)
        db.commit()
    except Exception as e:
        logger.error("save_designs_failed", experiment_id=experiment_id, error=str(e))
    finally:
        db.close()


@router.post("/predict-hotspot", summary="热点残基预测", description="使用ESM-2+GAT模型预测蛋白质热点残基，返回Top-K热点列表")
async def predict_hotspot(req: HotspotRequest):
    start = time.time()
    try:
        result = predictor.predict_hotspots(req.pdb_content, top_k=5)
        hotspots_detail = result.get("hotspots_detail", [])
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
            "num_hotspots": result.get("num_hotspots", len(hotspots_detail)),
            "method": result.get("method", "unknown"),
            "model_loaded": result.get("model_loaded", False),
            "total_residues": result.get("total_residues", 0),
        }
    except Exception as e:
        duration = time.time() - start
        record_task("predict_hotspot", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"热点预测失败: {str(e)}")


@router.post("/run-rfd3", summary="RFD3骨架生成", description="使用RFDiffusion3生成蛋白质Binder骨架结构，支持GPU推理和Mock回退")
async def run_rfd3(req: RFD3Request):
    start = time.time()
    try:
        runner = RFD3Runner()
        job_id = f"rfd3_{int(time.time())}"
        config = {
            "target": req.target,
            "hotspots": req.hotspots,
            "binder_length": req.binder_length,
            "length_min": req.length_min,
            "length_max": req.length_max,
            "diffusion_batch_size": req.diffusion_batch_size,
            "n_batches": req.n_batches,
        }
        result = runner.run_rfd3(
            pdb_content=req.pdb_content,
            target=req.target,
            hotspots=req.hotspots,
            binder_length=req.binder_length,
            length_min=req.length_min,
            length_max=req.length_max,
            diffusion_batch_size=req.diffusion_batch_size,
            n_batches=req.n_batches,
            job_id=job_id,
        )
        duration = time.time() - start
        record_task("run_rfd3", "success", duration)
        _save_experiment_step(req.experiment_id, "rfd3", result, config)

        if result.get("success") and result.get("designs"):
            designs = []
            for d in result["designs"]:
                designs.append({
                    "name": d.get("name", f"design_{d.get('index', 0)}"),
                    "sequence": "",
                    "pdb_content": d.get("pdb_content", ""),
                    "plddt": d.get("plddt"),
                })
            _save_designs(req.experiment_id, designs)

        return result
    except Exception as e:
        duration = time.time() - start
        record_task("run_rfd3", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"RFD3运行失败: {str(e)}")


@router.post("/run-mpnn", summary="MPNN序列设计", description="使用LigandMPNN基于骨架设计氨基酸序列，支持固定链和批量生成")
async def run_mpnn(req: MPNNRequest):
    start = time.time()
    try:
        runner = MPNNRunner()
        job_id = f"mpnn_{int(time.time())}"
        config = {
            "batch_size": req.batch_size,
            "fixed_chains": req.fixed_chains,
            "model_type": req.model_type,
        }
        result = runner.run_mpnn(
            backbone_pdb_content=req.backbone_pdb_content,
            backbone_pdb_path=req.backbone_pdb_path,
            batch_size=req.batch_size,
            fixed_chains=req.fixed_chains,
            model_type=req.model_type,
            job_id=job_id,
        )
        duration = time.time() - start
        record_task("run_mpnn", "success", duration)
        _save_experiment_step(req.experiment_id, "mpnn", result, config)

        if result.get("success") and result.get("sequences"):
            designs = []
            for s in result["sequences"]:
                designs.append({
                    "name": s.get("name", f"seq_{s.get('index', 0)}"),
                    "sequence": s.get("sequence", ""),
                    "pdb_content": s.get("pdb_content", ""),
                    "ranking_score": s.get("score"),
                })
            _save_designs(req.experiment_id, designs)

        return result
    except Exception as e:
        duration = time.time() - start
        record_task("run_mpnn", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"MPNN运行失败: {str(e)}")


@router.post("/run-rf3", summary="RF3结构验证", description="使用RoseTTAFold3验证设计结构的稳定性，计算RMSD、pLDDT等指标")
async def run_rf3(req: RF3Request):
    start = time.time()
    try:
        runner = RF3Runner()
        job_id = f"rf3_{int(time.time())}"
        config = {"example_id": req.example_id}
        result = runner.run_rf3(
            mpnn_pdb_content=req.mpnn_pdb_content,
            rfd3_pdb_content=req.rfd3_pdb_content,
            example_id=req.example_id,
            job_id=job_id,
        )
        duration = time.time() - start
        record_task("run_rf3", "success", duration)
        _save_experiment_step(req.experiment_id, "rf3", result, config)

        if result.get("success") and req.experiment_id:
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
async def run_pipeline(req: PipelineRequest):
    job_id = f"job_{int(time.time())}"
    start = time.time()

    try:
        experiment_id = str(uuid.uuid4())
        db = SessionLocal()
        exp = Experiment(
            id=experiment_id,
            name=f"Pipeline_{time.strftime('%Y%m%d_%H%M%S')}",
            status="running",
            input_pdb=req.pdb_content,
            hotspots=req.hotspots,
            rfd3_config={"binder_length": req.binder_length},
        )
        db.add(exp)
        db.commit()
        db.close()

        rfd3_runner = RFD3Runner()
        rfd3_results = rfd3_runner.run_rfd3(
            pdb_content=req.pdb_content,
            hotspots=[f"{h['chain']}{h['residue']}" for h in req.hotspots],
            binder_length=req.binder_length,
            job_id=job_id,
        )
        _save_experiment_step(experiment_id, "rfd3", rfd3_results)

        mpnn_results = None
        if rfd3_results.get("success") and rfd3_results.get("first_backbone_pdb"):
            mpnn_runner = MPNNRunner()
            target_chains = list(set(h["chain"] for h in req.hotspots))
            mpnn_results = mpnn_runner.run_mpnn(
                backbone_pdb_content=rfd3_results["first_backbone_pdb"],
                batch_size=10,
                fixed_chains=target_chains,
                job_id=job_id,
            )
            _save_experiment_step(experiment_id, "mpnn", mpnn_results)

        rf3_results = None
        if mpnn_results and mpnn_results.get("success") and mpnn_results.get("first_sequence_pdb"):
            rf3_runner = RF3Runner()
            rf3_results = rf3_runner.run(
                pdb_content=mpnn_results["first_sequence_pdb"],
                example_id=f"binder_{job_id}",
                job_id=job_id,
            )
            _save_experiment_step(experiment_id, "rf3", rf3_results)

        duration = time.time() - start
        record_task("run_pipeline", "success", duration)

        db = SessionLocal()
        try:
            exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.duration_seconds = duration
                exp.status = "completed"
                from datetime import datetime
                exp.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

        return {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "status": "completed",
            "rfd3_results": rfd3_results,
            "mpnn_results": mpnn_results,
            "rf3_results": rf3_results,
        }
    except Exception as e:
        duration = time.time() - start
        record_task("run_pipeline", "failed", duration, {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"运行失败: {str(e)}")
