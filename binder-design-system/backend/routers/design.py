from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hotspot_predictor import HotspotPredictor
from utils.rfd3_runner import RFD3Runner
from utils.mpnn_runner import MPNNRunner
from utils.rf3_runner import RF3Runner

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


class MPNNRequest(BaseModel):
    backbone_pdb_content: Optional[str] = None
    backbone_pdb_path: Optional[str] = None
    batch_size: int = 10
    fixed_chains: Optional[List[str]] = None
    model_type: str = "ligand_mpnn"


class RF3Request(BaseModel):
    pdb_content: str
    example_id: str = "binder_design"


class PipelineRequest(BaseModel):
    pdb_content: str
    hotspots: list[dict]
    binder_length: int = 80


@router.post("/predict-hotspot")
async def predict_hotspot(req: HotspotRequest):
    try:
        result = predictor.predict_hotspots(req.pdb_content, top_k=5)
        hotspots_detail = result.get("hotspots_detail", [])
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
        raise HTTPException(status_code=500, detail=f"热点预测失败: {str(e)}")


@router.post("/run-rfd3")
async def run_rfd3(req: RFD3Request):
    try:
        runner = RFD3Runner()
        job_id = f"rfd3_{int(time.time())}"
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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RFD3运行失败: {str(e)}")


@router.post("/run-mpnn")
async def run_mpnn(req: MPNNRequest):
    try:
        runner = MPNNRunner()
        job_id = f"mpnn_{int(time.time())}"
        result = runner.run_mpnn(
            backbone_pdb_content=req.backbone_pdb_content,
            backbone_pdb_path=req.backbone_pdb_path,
            batch_size=req.batch_size,
            fixed_chains=req.fixed_chains,
            model_type=req.model_type,
            job_id=job_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MPNN运行失败: {str(e)}")


@router.post("/run-rf3")
async def run_rf3(req: RF3Request):
    try:
        runner = RF3Runner()
        job_id = f"rf3_{int(time.time())}"
        result = runner.run(
            sequence=None,
            pdb_content=req.pdb_content,
            example_id=req.example_id,
            job_id=job_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF3运行失败: {str(e)}")


@router.post("/run-pipeline")
async def run_pipeline(req: PipelineRequest):
    job_id = f"job_{int(time.time())}"

    try:
        rfd3_runner = RFD3Runner()
        rfd3_results = rfd3_runner.run_rfd3(
            pdb_content=req.pdb_content,
            hotspots=[f"{h['chain']}{h['residue']}" for h in req.hotspots],
            binder_length=req.binder_length,
            job_id=job_id,
        )

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

        rf3_results = None
        if mpnn_results and mpnn_results.get("success") and mpnn_results.get("first_sequence_pdb"):
            rf3_runner = RF3Runner()
            rf3_results = rf3_runner.run(
                pdb_content=mpnn_results["first_sequence_pdb"],
                example_id=f"binder_{job_id}",
                job_id=job_id,
            )

        return {
            "job_id": job_id,
            "status": "completed",
            "rfd3_results": rfd3_results,
            "mpnn_results": mpnn_results,
            "rf3_results": rf3_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行失败: {str(e)}")
