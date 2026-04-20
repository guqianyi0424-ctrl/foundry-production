from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import sys

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

@router.post("/run-pipeline")
async def run_pipeline(req: PipelineRequest):
    job_id = f"job_{__import__('time').time():.0f}"

    try:
        rfd3_runner = RFD3Runner()
        rfd3_results = await rfd3_runner.run(
            pdb_content=req.pdb_content,
            hotspots=req.hotspots,
            binder_length=req.binder_length,
            top_k=5,
        )

        mpnn_results = []
        if rfd3_results.get("success"):
            mpnn_runner = MPNNRunner()
            for design in rfd3_results["designs"][:5]:
                seq_result = await mpnn_runner.run(
                    pdb_path=design.get("pdb_path", ""),
                    num_sequences=1,
                )
                if seq_result.get("sequences"):
                    for seq in seq_result["sequences"]:
                        mpnn_results.append({
                            "design_idx": design["index"],
                            "sequence": seq["sequence"],
                            "score": seq.get("score", 0),
                            "seq_idx": 0,
                        })

        rf3_results = []
        rf3_runner = RF3Runner()
        for m in mpnn_results[:9]:
            result = await rf3_runner.run(
                sequence=m["sequence"],
                target_pdb=req.pdb_content,
                rmsd_threshold=2.0,
            )
            rf3_results.append({
                "design_idx": m["design_idx"],
                "sequence": m["sequence"],
                "rmsd": result.get("rmsd", -1),
                "avg_plddt": result.get("avg_plddt", 0),
                "passed": result.get("passed", False),
                "plddt": result.get("plddt"),
                "pae": result.get("pae"),
                "per_res_rmsd": result.get("per_res_rmsd"),
                "mock": result.get("mock", False),
            })

        return {
            "job_id": job_id,
            "status": "completed",
            "rfd3_results": rfd3_results,
            "mpnn_results": mpnn_results,
            "rf3_results": rf3_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"运行失败: {str(e)}")
