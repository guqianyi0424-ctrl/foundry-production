"""
蛋白质结构解析工具
"""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import biotite.structure as struc
import numpy as np

try:
    from biotite.structure.io import load_structure, save_structure
    HAS_GENERIC_IO = True
except ImportError:
    HAS_GENERIC_IO = False
    import biotite.structure.io.pdb as pdb_io

try:
    import biotite.structure.io.cif as cif_io
    HAS_CIF = True
except ImportError:
    HAS_CIF = False


class StructureParser:
    """蛋白质结构解析器"""
    
    def __init__(self):
        self.structure = None
        self.atom_array = None
        self.file_path = None
    
    def parse_file(self, file_path: str) -> struc.AtomArray:
        """
        解析蛋白质结构文件
        
        Args:
            file_path: PDB/CIF文件路径
        
        Returns:
            AtomArray对象
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = self.file_path.suffix.lower()
        
        if HAS_GENERIC_IO:
            atom_array = load_structure(str(self.file_path))
        elif suffix == ".pdb":
            pdb_file = pdb_io.PDBFile.read(str(self.file_path))
            atom_array = pdb_file.get_structure(model=1)
        elif suffix == ".cif":
            if HAS_CIF:
                cif_file = cif_io.CIFFile.read(str(self.file_path))
                atom_array = cif_file.get_structure(model=1)
            else:
                raise ValueError(
                    f"CIF格式不支持。请升级biotite库: pip install --upgrade biotite\n"
                    f"或使用PDB格式的文件。"
                )
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .pdb 和 .cif")
        
        # 过滤掉非氨基酸残基（如水分子、配体等）
        try:
            atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        except Exception:
            pass
        
        self.atom_array = atom_array
        
        return self.atom_array
    
    def parse_uploaded_file(self, uploaded_file) -> struc.AtomArray:
        """
        解析Streamlit上传的文件
        
        Args:
            uploaded_file: Streamlit UploadedFile对象
        
        Returns:
            AtomArray对象
        """
        # 保存到临时文件
        suffix = Path(uploaded_file.name).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            atom_array = self.parse_file(tmp_path)
        finally:
            # 删除临时文件
            Path(tmp_path).unlink()
        
        return atom_array
    
    def get_sequence(self, atom_array: Optional[struc.AtomArray] = None) -> str:
        """
        获取氨基酸序列
        
        Args:
            atom_array: AtomArray对象，如果为None则使用self.atom_array
        
        Returns:
            氨基酸序列字符串
        """
        if atom_array is None:
            atom_array = self.atom_array
        
        if atom_array is None:
            raise ValueError("未加载结构数据")
        
        # 确保只包含氨基酸
        try:
            atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        except Exception:
            pass
        
        # 获取氨基酸序列
        try:
            sequence = struc.to_sequence(atom_array)[0]
            return str(sequence)
        except Exception as e:
            raise ValueError(f"无法提取氨基酸序列: {str(e)}")
    
    def get_chain_ids(self, atom_array: Optional[struc.AtomArray] = None) -> List[str]:
        """
        获取所有链ID
        
        Args:
            atom_array: AtomArray对象
        
        Returns:
            链ID列表
        """
        if atom_array is None:
            atom_array = self.atom_array
        
        if atom_array is None:
            raise ValueError("未加载结构数据")
        
        # 获取所有链ID
        chain_ids = np.unique(atom_array.chain_id)
        
        return chain_ids.tolist()
    
    def get_residue_info(
        self, 
        atom_array: Optional[struc.AtomArray] = None
    ) -> List[Dict]:
        """
        获取残基信息
        
        Args:
            atom_array: AtomArray对象
        
        Returns:
            残基信息列表
        """
        if atom_array is None:
            atom_array = self.atom_array
        
        if atom_array is None:
            raise ValueError("未加载结构数据")
        
        # 获取残基起始索引
        residue_starts = struc.get_residue_starts(atom_array)
        
        residues = []
        for i, start in enumerate(residue_starts):
            # 获取残基信息
            residue = atom_array[start]
            residue_id = residue.res_id
            residue_name = residue.res_name
            chain_id = residue.chain_id
            
            residues.append({
                "index": i,
                "res_id": residue_id,
                "res_name": residue_name,
                "chain_id": chain_id,
                "label": f"{chain_id}{residue_id} ({residue_name})"
            })
        
        return residues
    
    def get_structure_summary(
        self, 
        atom_array: Optional[struc.AtomArray] = None
    ) -> Dict:
        """
        获取结构摘要信息
        
        Args:
            atom_array: AtomArray对象
        
        Returns:
            结构摘要字典
        """
        if atom_array is None:
            atom_array = self.atom_array
        
        if atom_array is None:
            raise ValueError("未加载结构数据")
        
        # 获取基本信息
        num_atoms = len(atom_array)
        num_residues = struc.get_residue_count(atom_array)
        num_chains = len(np.unique(atom_array.chain_id))
        sequence = self.get_sequence(atom_array)
        
        summary = {
            "num_atoms": num_atoms,
            "num_residues": num_residues,
            "num_chains": num_chains,
            "sequence_length": len(sequence),
            "sequence": sequence,
            "chains": self.get_chain_ids(atom_array),
        }
        
        return summary
    
    def to_pdb_string(self, atom_array: Optional[struc.AtomArray] = None) -> str:
        """
        将结构转换为PDB格式字符串
        
        Args:
            atom_array: AtomArray对象
        
        Returns:
            PDB格式字符串
        """
        if atom_array is None:
            atom_array = self.atom_array
        
        if atom_array is None:
            raise ValueError("未加载结构数据")
        
        import io as io_module
        string_io = io_module.StringIO()
        
        # 直接使用PDBFile类，它可以写入StringIO
        if HAS_GENERIC_IO:
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(string_io)
        else:
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(string_io)
        
        return string_io.getvalue()
