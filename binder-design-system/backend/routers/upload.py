from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
from pathlib import Path

from utils.structure_parser import StructureParser

router = APIRouter()

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

@router.post("/upload")
async def upload_pdb(file: UploadFile = File(...)):
    if file.size is not None and file.size > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过200MB")

    content = await file.read()
    pdb_str = content.decode("utf-8", errors="ignore")

    suffix = Path(file.filename or "upload.pdb").suffix or ".pdb"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        parser = StructureParser()
        atom_array = parser.parse_file(tmp_path)
        chain_ids = parser.get_chain_ids(atom_array)
        residue_info = parser.get_residue_info(atom_array)
        chains = []
        for chain_id in chain_ids:
            chain_residues = [r for r in residue_info if r["chain_id"] == chain_id]
            chain_sequence = "".join(
                THREE_TO_ONE.get(str(residue["res_name"]).upper(), "X")
                for residue in chain_residues
            )
            chains.append({
                "chain_id": chain_id,
                "sequence": chain_sequence,
                "length": len(chain_residues),
            })
        return {
            "chains": [
                {"chain_id": c["chain_id"], "sequence": c["sequence"], "length": c["length"]}
                for c in chains
            ],
            "pdb_content": pdb_str,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
