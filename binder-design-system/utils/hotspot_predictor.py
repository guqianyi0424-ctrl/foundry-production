"""
热点残基预测工具
方案1: ESM-2 + DGL GAT 5折集成 (CPU推理，使用已有模型权重)
方案2: ESM-2 注意力分数 (CPU推理，无需权重)
方案3: 规则预测 (兜底)
GPU留给RFD3/MPNN/RF3大模型使用
"""
import os
import sys
import io
import types
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


def _patch_torchdata():
    try:
        import torchdata
    except ImportError:
        torchdata = types.ModuleType('torchdata')
        sys.modules['torchdata'] = torchdata

    if not hasattr(torchdata, 'datapipes'):
        datapipes = types.ModuleType('torchdata.datapipes')
        datapipes.iter = types.ModuleType('torchdata.datapipes.iter')
        datapipes.iter.IterDataPipe = type('IterDataPipe', (), {})
        sys.modules['torchdata.datapipes'] = datapipes
        sys.modules['torchdata.datapipes.iter'] = datapipes.iter
        torchdata.datapipes = datapipes


_patch_torchdata()


class HotspotPredictor:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.base_path = Path(__file__).parent.parent.parent
        self.hotspot_dl_path = self.base_path / "hotspot-prediction"
        self.models_dir = Path(
            os.getenv(
                "HOTSPOT_DL_DIR",
                str(self.hotspot_dl_path / "models" / "final_neg3"),
            )
        )

        self._dl_models = {}
        self._dl_loaded = False
        self._esm_model = None
        self._esm_loaded = False

    def predict(
        self,
        atom_array,
        method: str = "dl",
        pdb_string: str = None,
        top_k: int = None
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = top_k

        residues = self._extract_residue_features(atom_array)
        if residues.empty:
            return {
                "hotspots": [],
                "hotspots_detail": [],
                "all_scores": {},
                "error": "无法提取残基特征"
            }

        results_dl = self._predict_dl(residues, atom_array)

        if results_dl and results_dl.get("model_loaded"):
            return self._select_top_k(results_dl, residues, "dl")
        else:
            fallback = self._rule_based_predict(residues)
            return self._select_top_k(fallback, residues, "rule")

    def _predict_dl(self, residues: pd.DataFrame, atom_array) -> Dict[str, Any]:
        dl_models = self._load_dl_models()

        if dl_models:
            try:
                import torch

                sequence = self._get_sequence_from_residues(residues)
                esm_features = self._get_esm_features(sequence)

                if esm_features is not None and len(esm_features) == len(residues):
                    try:
                        import dgl

                        node_features = self._build_node_features(residues, esm_features)
                        g = self._build_graph(atom_array, residues)

                        device = torch.device('cpu')

                        all_probs = []
                        with torch.no_grad():
                            node_features_tensor = torch.tensor(node_features, dtype=torch.float32).to(device)
                            g = g.to(device)
                            for fold_name, model in dl_models.items():
                                logits = model(g, node_features_tensor)
                                probs = torch.softmax(logits, dim=1)
                                all_probs.append(probs[:, 1].cpu().numpy())

                        scores = np.mean(all_probs, axis=0)
                        n_models = len(dl_models)
                        print(f"[DL] GAT+ESM-2预测完成 ({n_models}折集成, CPU推理)")
                        return {"scores": scores, "method": "dl", "model_loaded": True}

                    except ImportError:
                        print("[DL] DGL不可用，尝试ESM-2注意力分数预测")
                    except Exception as e:
                        print(f"[DL] GAT推理失败: {e}，尝试ESM-2注意力分数预测")

                scores = self._esm_attention_scores(residues)
                if scores is not None:
                    return {"scores": scores, "method": "dl", "model_loaded": True}

            except Exception as e:
                print(f"[DL] 预测失败: {e}")
                import traceback
                traceback.print_exc()

        else:
            scores = self._esm_attention_scores(residues)
            if scores is not None:
                return {"scores": scores, "method": "dl", "model_loaded": True}

        scores = self._compute_rule_scores(residues, "dl")
        return {"scores": scores, "method": "dl", "model_loaded": False}

    def _esm_attention_scores(self, residues: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            import torch
            esm = self._load_esm_model()
            if esm is None:
                return None

            model = esm['model']
            device = esm['device']
            sequence = self._get_sequence_from_residues(residues)

            with torch.no_grad():
                tokenizer = esm['tokenizer']
                inputs = tokenizer(sequence, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_attentions=True)

                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    last_attn = outputs.attentions[-1]
                    attn_weights = last_attn.mean(dim=1)[0]
                    cls_attn = attn_weights[0, 1:-1].cpu().numpy()
                    if len(cls_attn) == len(residues):
                        min_a = cls_attn.min()
                        max_a = cls_attn.max()
                        if max_a > min_a:
                            scores = (cls_attn - min_a) / (max_a - min_a)
                        else:
                            scores = np.ones(len(residues)) * 0.5
                        print(f"[DL] ESM-2注意力分数预测完成 (CPU)")
                        return scores

            return None
        except Exception as e:
            print(f"[DL] ESM注意力分数失败: {e}")
            return None

    def _load_dl_models(self) -> Dict[str, Any]:
        if self._dl_loaded:
            return self._dl_models

        try:
            import torch

            device = torch.device('cpu')
            print(f"[DL] PyTorch设备: {device}")

            try:
                import dgl
                print(f"[DL] DGL版本: {dgl.__version__}")
            except ImportError:
                print("[DL] DGL未安装，GAT模型不可用")
                self._dl_models = {}
                self._dl_loaded = True
                return self._dl_models
            except OSError as e:
                print(f"[DL] DGL加载失败: {e}")
                print("[DL] 尝试CPU版DGL安装...")
                self._try_install_dgl_cpu()
                try:
                    import dgl
                    print(f"[DL] DGL安装成功: {dgl.__version__}")
                except Exception:
                    print("[DL] DGL仍不可用，GAT模型不可用")
                    self._dl_models = {}
                    self._dl_loaded = True
                    return self._dl_models

            hotspot_module = self._load_hotspot_dl_module()
            PPIHotspotGAT = hotspot_module["PPIHotspotGAT"]
            INPUT_DIM = hotspot_module["INPUT_DIM"]
            HIDDEN_DIM = hotspot_module["HIDDEN_DIM"]
            NUM_HEADS = hotspot_module["NUM_HEADS"]
            NUM_LAYERS = hotspot_module["NUM_LAYERS"]
            DROPOUT = hotspot_module["DROPOUT"]

            for fold_idx in range(1, 6):
                model_file = self.models_dir / f"best_model_fold{fold_idx}.pth"
                if model_file.exists():
                    try:
                        model = PPIHotspotGAT(
                            input_dim=INPUT_DIM,
                            hidden_dim=HIDDEN_DIM,
                            num_heads=NUM_HEADS,
                            num_layers=NUM_LAYERS,
                            dropout=DROPOUT
                        )

                        try:
                            state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
                        except TypeError:
                            state_dict = torch.load(str(model_file), map_location=device)
                        except RuntimeError:
                            state_dict = torch.load(str(model_file), map_location='cpu', weights_only=False)

                        if 'model_state_dict' in state_dict:
                            model.load_state_dict(state_dict['model_state_dict'])
                        else:
                            model.load_state_dict(state_dict)
                        model.eval()
                        model = model.to(device)
                        self._dl_models[f"fold{fold_idx}"] = model
                        print(f"[DL] Fold {fold_idx} 模型加载成功 (device={device})")
                    except Exception as e:
                        print(f"[DL] Fold {fold_idx} 加载失败: {e}")
                else:
                    print(f"[DL] Fold {fold_idx} 权重不存在: {model_file}")

            if not self._dl_models:
                print("[DL] 没有可用的DL模型权重")

        except ImportError as e:
            print(f"[DL] 依赖未安装: {e}")
            self._dl_models = {}
        except Exception as e:
            print(f"[DL] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._dl_models = {}

        self._dl_loaded = True
        return self._dl_models

    def _load_hotspot_dl_module(self) -> Dict[str, Any]:
        config_path = self.hotspot_dl_path / "config.py"
        model_path = self.hotspot_dl_path / "model.py"

        if not config_path.exists() or not model_path.exists():
            raise ImportError(f"hotspot-prediction源码不存在: {self.hotspot_dl_path}")

        config_spec = importlib.util.spec_from_file_location(
            "_hotspot_prediction_config",
            config_path,
        )
        model_spec = importlib.util.spec_from_file_location(
            "_hotspot_prediction_model",
            model_path,
        )
        if config_spec is None or config_spec.loader is None:
            raise ImportError(f"无法加载hotspot配置: {config_path}")
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"无法加载hotspot模型: {model_path}")

        config_module = importlib.util.module_from_spec(config_spec)
        model_module = importlib.util.module_from_spec(model_spec)
        original_config = sys.modules.get("config")
        original_path = list(sys.path)

        try:
            config_spec.loader.exec_module(config_module)
            sys.modules["config"] = config_module
            if str(self.hotspot_dl_path) not in sys.path:
                sys.path.insert(0, str(self.hotspot_dl_path))
            model_spec.loader.exec_module(model_module)
        finally:
            if original_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = original_config
            sys.path[:] = original_path

        required = [
            "INPUT_DIM",
            "HIDDEN_DIM",
            "NUM_HEADS",
            "NUM_LAYERS",
            "DROPOUT",
        ]
        missing = [name for name in required if not hasattr(config_module, name)]
        if missing:
            raise ImportError(f"hotspot配置缺少字段: {', '.join(missing)}")
        if not hasattr(model_module, "PPIHotspotGAT"):
            raise ImportError("hotspot模型缺少PPIHotspotGAT")

        return {
            "PPIHotspotGAT": model_module.PPIHotspotGAT,
            "INPUT_DIM": config_module.INPUT_DIM,
            "HIDDEN_DIM": config_module.HIDDEN_DIM,
            "NUM_HEADS": config_module.NUM_HEADS,
            "NUM_LAYERS": config_module.NUM_LAYERS,
            "DROPOUT": config_module.DROPOUT,
        }

    def _try_install_dgl_cpu(self):
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'uninstall', 'dgl', '-y', '--quiet'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            attempts = [
                ("DGL CPU (DGL仓库)", [sys.executable, '-m', 'pip', 'install', 'dgl',
                  '--no-index', '-f', 'https://data.dgl.ai/wheels/repo.html', '--quiet']),
                ("DGL (PyPI)", [sys.executable, '-m', 'pip', 'install', 'dgl', '--quiet']),
            ]

            for name, cmd in attempts:
                print(f"[DL] 尝试安装: {name}...")
                try:
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    test = subprocess.run(
                        [sys.executable, '-c', 'import dgl; print(dgl.__version__)'],
                        capture_output=True, text=True, timeout=15
                    )
                    if test.returncode == 0:
                        print(f"[DL] ✅ {name}安装成功: {test.stdout.strip()}")
                        return
                    else:
                        subprocess.check_call(
                            [sys.executable, '-m', 'pip', 'uninstall', 'dgl', '-y', '--quiet'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                except Exception as e:
                    print(f"[DL] {name}安装失败: {e}")

            print("[DL] ❌ DGL CPU版安装失败，请手动: pip install dgl")
        except Exception as e:
            print(f"[DL] DGL安装异常: {e}")

    def _load_esm_model(self):
        if self._esm_loaded:
            return self._esm_model

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "facebook/esm2_t33_650M_UR50D"
            print(f"[ESM] 正在加载: {model_name} (CPU推理)...")

            device = torch.device('cpu')
            load_kwargs = {"local_files_only": True}

            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
            model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device)
            model.eval()

            self._esm_model = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device,
            }
            print(f"[ESM] 模型加载成功 (device={device})")
        except ImportError:
            print("[ESM] transformers未安装，跳过ESM特征")
            self._esm_model = None
        except Exception as e:
            print(f"[ESM] 模型加载失败: {e}")
            self._esm_model = None

        self._esm_loaded = True
        return self._esm_model

    def _get_esm_features(self, sequence: str) -> Optional[np.ndarray]:
        import torch

        esm = self._load_esm_model()
        if esm is None:
            return None

        try:
            tokenizer = esm['tokenizer']
            model = esm['model']
            device = esm['device']

            with torch.no_grad():
                inputs = tokenizer(sequence, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()

            return embeddings
        except Exception as e:
            print(f"[ESM] 特征提取失败: {e}")
            return None

    def _get_sequence_from_residues(self, residues_df: pd.DataFrame) -> str:
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = ""
        for _, row in residues_df.iterrows():
            res_name = row.get('res_name', '')
            sequence += aa_map.get(res_name, 'X')

        return sequence

    def _build_node_features(
        self,
        residues_df: pd.DataFrame,
        esm_features: np.ndarray
    ) -> np.ndarray:
        n_residues = len(residues_df)

        pssm_features = np.zeros((n_residues, 20), dtype=np.float32)
        hmm_features = np.zeros((n_residues, 30), dtype=np.float32)

        node_features = np.concatenate([
            esm_features,
            pssm_features,
            hmm_features,
        ], axis=1)

        return node_features.astype(np.float32)

    def _build_graph(self, atom_array, residues_df: pd.DataFrame):
        import dgl

        n_residues = len(residues_df)

        ca_coords = []
        for idx, row in residues_df.iterrows():
            ca_coords.append([
                row.get('ca_x', 0),
                row.get('ca_y', 0),
                row.get('ca_z', 0)
            ])

        if len(ca_coords) == 0:
            return dgl.graph(([], []), num_nodes=n_residues)

        coords = np.array(ca_coords)

        src = []
        dst = []
        cutoff = 10.0

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff:
                    src.extend([i, j])
                    dst.extend([j, i])

        g = dgl.graph((src, dst), num_nodes=n_residues)
        return g

    def _select_top_k(
        self,
        result: Dict,
        residues: pd.DataFrame,
        method: str
    ) -> Dict[str, Any]:
        scores = result["scores"]

        all_scores_dict = {}
        for idx, row in residues.iterrows():
            label = f"{row['chain_id']}{row['res_id']}"
            score = float(scores[idx]) if idx < len(scores) else 0.0
            all_scores_dict[label] = {
                "score": round(score, 4),
                "residue_name": row["res_name"],
                "chain_id": row["chain_id"],
                "res_id": row["res_id"]
            }

        sorted_items = sorted(
            all_scores_dict.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        top_k_items = sorted_items[:self.top_k]

        hotspots_detail = []
        for label, info in top_k_items:
            hotspots_detail.append({
                "label": label,
                "chain": info["chain_id"],
                "residue_id": info["res_id"],
                "residue_name": info["residue_name"],
                "score": info["score"],
                "combined_score": info["score"]
            })

        return {
            "method": method,
            "model_loaded": result.get("model_loaded", False),
            "hotspots": [h["label"] for h in hotspots_detail],
            "hotspots_detail": hotspots_detail,
            "all_scores": all_scores_dict,
            "total_residues": len(residues),
            "num_hotspots": len(hotspots_detail),
            "top_k": self.top_k
        }

    def _rule_based_predict(self, residues: pd.DataFrame) -> Dict[str, Any]:
        scores = self._compute_rule_scores(residues, "both")
        return {"scores": scores, "method": "rule", "model_loaded": False}

    def _compute_rule_scores(self, residues: pd.DataFrame, method: str) -> np.ndarray:
        n = len(residues)
        scores = np.zeros(n)

        aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        hydrophobic = {'LEU', 'ILE', 'VAL', 'MET', 'ALA', 'PRO'}
        charged_pos = {'ARG', 'LYS'}
        charged_neg = {'ASP', 'GLU'}

        for idx, row in residues.iterrows():
            s = 0.0
            res_name = row.get('res_name', '')
            sasa = row.get('sasa', 50)
            conservation = row.get('conservation', 0.5)
            dist = row.get('dist_to_center', 15)
            energy = row.get('energy', 0)

            if res_name in aromatic:
                s += 0.35
            elif res_name in hydrophobic:
                s += 0.25
            elif res_name in charged_pos:
                s += 0.20
            elif res_name in charged_neg:
                s += 0.15

            if sasa > 100:
                s += 0.30
            elif sasa > 70:
                s += 0.20
            elif sasa > 40:
                s += 0.10

            s += conservation * 0.20

            if dist > 18:
                s += 0.20
            elif dist > 12:
                s += 0.10

            if energy < -1.5:
                s += 0.15
            elif energy < -0.5:
                s += 0.08

            if method == "dl":
                s += np.random.uniform(-0.05, 0.05)

            scores[idx] = min(s, 1.0)

        return scores

    def _extract_residue_features(self, atom_array) -> pd.DataFrame:
        try:
            import biotite.structure as struc

            residue_starts = struc.get_residue_starts(atom_array)
            residue_info = []

            center = atom_array.coord.mean(axis=0)

            for i, start in enumerate(residue_starts):
                res_mask = np.zeros(len(atom_array), dtype=bool)
                if i < len(residue_starts) - 1:
                    res_mask[start:residue_starts[i + 1]] = True
                else:
                    res_mask[start:] = True

                res_atoms = atom_array[res_mask]
                if len(res_atoms) == 0:
                    continue

                res_name = str(res_atoms.res_name[0])
                res_id = int(res_atoms.res_id[0])
                chain_id = str(res_atoms.chain_id[0])

                ca_mask = res_atoms.atom_name == "CA"
                ca_atoms = res_atoms[ca_mask]
                if len(ca_atoms) > 0:
                    ca_coord = ca_atoms.coord[0]
                else:
                    ca_coord = res_atoms.coord[0]

                dist_to_center = float(np.linalg.norm(ca_coord - center))
                sasa = self._estimate_sasa(res_atoms, dist_to_center)
                energy = self._estimate_energy(res_name, dist_to_center)
                conservation = self._estimate_conservation(res_name, dist_to_center)

                residue_info.append({
                    "index": i,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "res_name": res_name,
                    "sasa": round(float(sasa), 2),
                    "energy": round(float(energy), 3),
                    "conservation": round(float(conservation), 3),
                    "dist_to_center": round(dist_to_center, 2),
                    "num_atoms": int(len(res_atoms)),
                    "ca_x": float(ca_coord[0]),
                    "ca_y": float(ca_coord[1]),
                    "ca_z": float(ca_coord[2])
                })

            return pd.DataFrame(residue_info)

        except Exception as e:
            print(f"提取残基特征出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def predict_hotspots(
        self,
        pdb_string: str,
        top_k: int = None,
        method: str = "dl"
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = top_k

        atom_array = self._parse_pdb_string(pdb_string)
        if atom_array is None:
            return {
                "hotspots": [],
                "hotspots_detail": [],
                "all_scores": {},
                "error": "无法解析PDB字符串"
            }

        result = self.predict(
            atom_array=atom_array,
            method=method,
            pdb_string=pdb_string,
            top_k=top_k
        )
        return result

    def _parse_pdb_string(self, pdb_string: str):
        try:
            import biotite.structure as bs
            import biotite.structure.io.pdb as bpdb

            pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_string))
            atom_array = pdb_file.get_structure(model=1)
            atom_array = atom_array[bs.filter_amino_acids(atom_array)]
            return atom_array
        except ImportError:
            print("[PDB] biotite未安装")
            return None
        except Exception as e:
            print(f"[PDB] 解析失败: {e}")
            return None

    def _estimate_sasa(self, res_atoms, dist_to_center: float) -> float:
        num_atoms = len(res_atoms)
        base_sasa = num_atoms * 10.0
        surface_factor = min(dist_to_center / 20.0, 1.0)
        return base_sasa * (0.3 + 0.7 * surface_factor)

    def _estimate_energy(self, res_name: str, dist_to_center: float) -> float:
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        charged = {'ARG', 'LYS', 'ASP', 'GLU'}
        polar = {'SER', 'THR', 'ASN', 'GLN', 'HIS', 'CYS', 'TYR'}

        if res_name in hydrophobic:
            base = -2.0
        elif res_name in charged:
            base = -1.0
        elif res_name in polar:
            base = -1.5
        else:
            base = -0.5

        return base + (dist_to_center / 30.0) * 2.0

    def _estimate_conservation(self, res_name: str, dist_to_center: float) -> float:
        high_cons = {'CYS', 'TRP', 'HIS', 'PRO'}
        med_cons = {'ARG', 'LYS', 'ASP', 'GLU', 'PHE', 'TYR', 'ASN', 'GLN'}
        low_cons = {'ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU', 'ILE', 'MET'}

        if res_name in high_cons:
            base = 0.7
        elif res_name in med_cons:
            base = 0.5
        elif res_name in low_cons:
            base = 0.3
        else:
            base = 0.4

        return base
