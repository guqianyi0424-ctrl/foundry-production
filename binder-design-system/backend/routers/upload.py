from fastapi import APIRouter, UploadFile, File, HTTPException

from utils.structure_parser import StructureParser

router = APIRouter()

@router.post("/upload")
async def upload_pdb(file: UploadFile = File(...)):
    if file.size > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过200MB")

    content = await file.read()
    pdb_str = content.decode("utf-8", errors="ignore")

    try:
        parser = StructureParser(pdb_str)
        chains = parser.get_chains_info()
        return {
            "chains": [
                {"chain_id": c["chain_id"], "sequence": c["sequence"], "length": c["length"]}
                for c in chains
            ],
            "pdb_content": pdb_str,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")
