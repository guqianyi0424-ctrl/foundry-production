"""
热点残基预测工具
集成 ppihotspotid (ML - AutoGluon 14模型集成) 和 hotspot-prediction (DL - GAT+ESM-2)
ML: TabularPredictor.load() 加载完整AutoGluon集成 (13 L1基础模型 + 1 L2加权集成 = 62子模型)
DL: 5折集成推理 (5-fold ensemble)
支持 Top-K 选择策略
"""
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


def _patch_autogluon_compat():
    _compat_defaults = {
        'passthrough': False,
        'passthrough_stage': 'first',
        'passthrough_types': None,
    }
    try:
        from autogluon.features.generators.abstract import AbstractFeatureGenerator

        if not getattr(AbstractFeatureGenerator, '_compat_patched', False):
            original_getattr = getattr(AbstractFeatureGenerator, '__getattr__', None)

            def _compat_getattr(self, name):
                if name in _compat_defaults:
                    return _compat_defaults[name]
                if original_getattr is not None:
                    return original_getattr(self, name)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

            AbstractFeatureGenerator.__getattr__ = _compat_getattr
            AbstractFeatureGenerator._compat_patched = True
            print("[ML] AutoGluon兼容性补丁已应用 (passthrough等属性)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[ML] AutoGluon兼容性补丁失败: {e}")


class HotspotPredictor:
    """热点残基预测器 - 集成ML和DL两个模型"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.base_path = Path(__file__).parent.parent.parent
        self.ppihotspotid_path = self.base_path / "ppihotspotid-main"
        self.hotspot_dl_path = self.base_path / "hotspot-prediction"

        self.ml_model_path = self.ppihotspotid_path / "AutogluonModels" / "ag-20230915_030535"

        self._ml_predictor = None
        self._ml_loaded = False
        self._dl_models = {}
        self._dl_loaded = False
        self._esm_model = None
        self._esm_loaded = False

    def predict(
        self,
        atom_array,
        method: str = "both",
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

        results_ml = None
        results_dl = None

        if method in ("ml", "both"):
            results_ml = self._predict_ml(residues)

        if method in ("dl", "both"):
            results_dl = self._predict_dl(residues, atom_array)

        if method == "both" and results_ml and results_dl:
            return self._merge_results(results_ml, results_dl, residues)
        elif method == "ml" and results_ml:
            return self._select_top_k(results_ml, residues, "ml")
        elif method == "dl" and results_dl:
            return self._select_top_k(results_dl, residues, "dl")
        else:
            fallback = self._rule_based_predict(residues)
            return self._select_top_k(fallback, residues, "rule")

    def _predict_ml(self, residues: pd.DataFrame) -> Dict[str, Any]:
        predictor = self._load_ml_predictor()

        if predictor is not None:
            try:
                pred_data = pd.DataFrame({
                    'Ty': residues['res_name'].values,
                    'cons': residues['conservation'].values,
                    'sasa': residues['sasa'].values,
                    'gas_e': residues['energy'].values,
                })

                predictions = predictor.predict(pred_data)

                label_map = predictor.class_labels_internal_dict
                pos_label = None
                for label, idx in label_map.items():
                    if str(label).upper() == 'P':
                        pos_label = idx
                        break

                if pos_label is None and len(label_map) == 2:
                    pos_label = 1

                if pos_label is not None:
                    try:
                        pred_proba = predictor.predict_proba(pred_data)
                        if pred_proba.shape[1] > pos_label:
                            scores = pred_proba.iloc[:, pos_label].values
                        else:
                            scores = (predictions == list(label_map.keys())[list(label_map.values()).index(pos_label)]).astype(float)
                    except Exception:
                        scores = (predictions == list(label_map.keys())[list(label_map.values()).index(pos_label)]).astype(float)
                else:
                    scores = predictions.astype(float).values

                n_models = len(predictor.model_names()) if hasattr(predictor, 'model_names') else 0
                print(f"[ML] AutoGluon预测完成 ({n_models}个模型集成)")
                return {"scores": scores, "method": "ml", "model_loaded": True}

            except Exception as e:
                print(f"[ML] AutoGluon预测失败: {e}")
                import traceback
                traceback.print_exc()

        scores = self._compute_rule_scores(residues, "ml")
        return {"scores": scores, "method": "ml", "model_loaded": False}

    def _predict_dl(self, residues: pd.DataFrame, atom_array) -> Dict[str, Any]:
        dl_models = self._load_dl_models()

        if dl_models:
            try:
                import torch
                import dgl

                sequence = self._get_sequence_from_residues(residues)
                esm_features = self._get_esm_features(sequence)

                if esm_features is not None and len(esm_features) == len(residues):
                    node_features = self._build_node_features(residues, esm_features)
                    g = self._build_graph(atom_array, residues)

                    device = next(iter(dl_models.values())).input_proj[0].weight.device

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
                    print(f"[DL] hotspot-prediction预测完成 ({n_models}折集成, GAT+ESM-2)")
                    return {"scores": scores, "method": "dl", "model_loaded": True}
                else:
                    print("[DL] ESM特征不可用，使用规则预测")

            except Exception as e:
                print(f"[DL] 预测失败: {e}")
                import traceback
                traceback.print_exc()

        scores = self._compute_rule_scores(residues, "dl")
        return {"scores": scores, "method": "dl", "model_loaded": False}

    def _merge_results(
        self,
        results_ml: Dict,
        results_dl: Dict,
        residues: pd.DataFrame
    ) -> Dict[str, Any]:
        ml_scores = results_ml["scores"]
        dl_scores = results_dl["scores"]

        ml_loaded = results_ml.get("model_loaded", False)
        dl_loaded = results_dl.get("model_loaded", False)

        if ml_loaded and dl_loaded:
            combined = 0.5 * ml_scores + 0.5 * dl_scores
        elif ml_loaded:
            combined = ml_scores
        elif dl_loaded:
            combined = dl_scores
        else:
            combined = 0.5 * ml_scores + 0.5 * dl_scores

        all_scores_dict = {}
        for idx, row in residues.iterrows():
            label = f"{row['chain_id']}{row['res_id']}"
            all_scores_dict[label] = {
                "ml_score": float(ml_scores[idx]) if idx < len(ml_scores) else 0.0,
                "dl_score": float(dl_scores[idx]) if idx < len(dl_scores) else 0.0,
                "combined_score": float(combined[idx]) if idx < len(combined) else 0.0,
                "residue_name": row["res_name"],
                "chain_id": row["chain_id"],
                "res_id": row["res_id"]
            }

        sorted_items = sorted(
            all_scores_dict.items(),
            key=lambda x: x[1]["combined_score"],
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
                "ml_score": round(info["ml_score"], 4),
                "dl_score": round(info["dl_score"], 4),
                "combined_score": round(info["combined_score"], 4),
                "score": round(info["combined_score"], 4)
            })

        return {
            "method": "both",
            "ml_model_loaded": ml_loaded,
            "dl_model_loaded": dl_loaded,
            "hotspots": [h["label"] for h in hotspots_detail],
            "hotspots_detail": hotspots_detail,
            "all_scores": all_scores_dict,
            "total_residues": len(residues),
            "num_hotspots": len(hotspots_detail),
            "top_k": self.top_k
        }

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

    def _load_ml_predictor(self):
        if self._ml_loaded:
            return self._ml_predictor

        _patch_autogluon_compat()

        try:
            from autogluon.tabular import TabularPredictor

            if not self.ml_model_path.exists():
                print(f"[ML] AutoGluon模型目录不存在: {self.ml_model_path}")
                self._ml_predictor = None
                self._ml_loaded = True
                return self._ml_predictor

            print(f"[ML] 加载AutoGluon集成模型: {self.ml_model_path}")
            self._ml_predictor = TabularPredictor.load(
                str(self.ml_model_path),
                require_version_match=False,
                require_py_version_match=False,
            )

            model_names = self._ml_predictor.model_names()
            n_models = len(model_names)
            print(f"[ML] AutoGluon模型加载成功 ({n_models}个模型: {', '.join(model_names)})")

        except ImportError:
            print("[ML] autogluon未安装，尝试安装...")
            self._try_install_autogluon()
            try:
                _patch_autogluon_compat()
                from autogluon.tabular import TabularPredictor
                self._ml_predictor = TabularPredictor.load(
                    str(self.ml_model_path),
                    require_version_match=False,
                    require_py_version_match=False,
                )
                model_names = self._ml_predictor.model_names()
                n_models = len(model_names)
                print(f"[ML] AutoGluon模型加载成功 ({n_models}个模型: {', '.join(model_names)})")
            except Exception as e2:
                print(f"[ML] AutoGluon安装后仍无法加载: {e2}")
                self._ml_predictor = None
        except Exception as e:
            print(f"[ML] AutoGluon模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._ml_predictor = None

        self._ml_loaded = True
        return self._ml_predictor

    def _try_install_autogluon(self):
        py_ver = sys.version_info
        if py_ver >= (3, 11):
            print(f"[ML] ❌ Python {py_ver.major}.{py_ver.minor} >= 3.11, AutoGluon 0.8.2 需要 Python 3.8-3.10")
            print("[ML] 请使用 conda 创建 Python 3.10 环境:")
            print("[ML]   conda create -n binder python=3.10 -y")
            print("[ML]   conda activate binder")
            return

        import subprocess

        def _pip_install(spec, no_deps=False, timeout=120):
            try:
                cmd = [sys.executable, '-m', 'pip', 'install', spec, '--quiet']
                if no_deps:
                    cmd.insert(4, '--no-deps')
                subprocess.check_call(cmd, timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                print(f"[ML] 安装 {spec} 超时")
                return False
            except Exception as e:
                print(f"[ML] 安装 {spec} 失败: {e}")
                return False

        try:
            import pkg_resources
        except ImportError:
            print("[ML] 安装 setuptools (提供 pkg_resources)...")
            if not _pip_install('setuptools'):
                print("[ML] ❌ setuptools 安装失败，AutoGluon 无法加载")
                return

        pkgs = [
            'autogluon.common==0.8.2',
            'autogluon.core==0.8.2',
            'autogluon.features==0.8.2',
            'autogluon.tabular==0.8.2',
        ]
        for pkg in pkgs:
            print(f"[ML] 安装 {pkg} (--no-deps)...")
            _pip_install(pkg, no_deps=True, timeout=300)

        compat_deps = [
            'pandas==1.5.3',
            'scipy==1.11.4',
            'scikit-learn==1.2.2',
            'boto3>=1.10,<2',
            'psutil>=5.7.3,<6',
        ]
        for dep in compat_deps:
            print(f"[ML] 安装兼容依赖 {dep}...")
            _pip_install(dep)

        print("[ML] AutoGluon子包安装完成")

    def _detect_cuda_version(self):
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version:
                    major = int(cuda_version.split('.')[0])
                    if major >= 12:
                        return 'cu121'
                    elif major == 11:
                        minor = int(cuda_version.split('.')[1])
                        if minor >= 8:
                            return 'cu118'
                        else:
                            return 'cu117'
                    return f'cu{major}{cuda_version.split(".")[1]}'
        except Exception:
            pass
        return None

    def _load_dl_models(self) -> Dict[str, Any]:
        if self._dl_loaded:
            return self._dl_models

        os.environ.setdefault('DGLBACKEND', 'pytorch')
        os.environ.setdefault('DGL_DOWNLOAD', '1')

        try:
            import torch

            has_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if has_cuda else 'cpu')
            print(f"[DL] PyTorch设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if has_cuda else ""))

            torch.set_num_threads(min(4, os.cpu_count() or 4))

            try:
                import dgl
                if has_cuda:
                    try:
                        _ = dgl.device('cuda:0')
                        print(f"[DL] DGL CUDA可用")
                    except Exception:
                        print(f"[DL] DGL CUDA不可用，切换到CPU模式")
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        device = torch.device('cpu')
                print(f"[DL] DGL版本: {dgl.__version__}")
            except OSError as e:
                err_msg = str(e)
                if 'libcusparseLt' in err_msg or 'cuda' in err_msg.lower() or 'cusparse' in err_msg.lower():
                    print(f"[DL] CUDA库缺失，尝试安装匹配版DGL...")
                    cuda_ver = self._detect_cuda_version()
                    self._try_dgl_install(cuda_ver)
                    try:
                        import dgl
                        print(f"[DL] DGL安装成功: {dgl.__version__}")
                    except OSError as e2:
                        print(f"[DL] DGL仍无法加载: {e2}")
                        self._dl_models = {}
                        self._dl_loaded = True
                        return self._dl_models
                    except ImportError:
                        print(f"[DL] DGL安装失败")
                        self._dl_models = {}
                        self._dl_loaded = True
                        return self._dl_models
                else:
                    raise

            sys.path.insert(0, str(self.hotspot_dl_path))
            from model import PPIHotspotGAT
            from config import INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT

            models_dir = self.hotspot_dl_path / "models"

            for fold_idx in range(1, 6):
                model_file = models_dir / f"best_model_fold{fold_idx}.pth"
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
                self._dl_models = {}

        except ImportError as e:
            err_msg = str(e)
            print(f"[DL] 依赖未安装: {e}")
            if 'torchdata' in err_msg or 'datapipes' in err_msg:
                print("[DL] 尝试安装torchdata...")
                try:
                    import subprocess
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install',
                        'torchdata==0.7.1', '--quiet',
                    ], timeout=120)
                    print("[DL] torchdata安装完成，重新加载DL模型...")
                    self._dl_loaded = False
                    return self._load_dl_models()
                except Exception as te:
                    print(f"[DL] torchdata安装失败: {te}")
            self._dl_models = {}
        except OSError as e:
            err_msg = str(e)
            if 'libcusparseLt' in err_msg or 'cuda' in err_msg.lower():
                print(f"[DL] CUDA库缺失，尝试安装匹配版DGL...")
                self._dl_models = {}
                cuda_ver = self._detect_cuda_version()
                self._try_dgl_install(cuda_ver)
            else:
                print(f"[DL] 系统库缺失: {e}")
                self._dl_models = {}
        except Exception as e:
            print(f"[DL] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._dl_models = {}

        self._dl_loaded = True
        return self._dl_models

    def _try_dgl_install(self, cuda_version=None):
        try:
            import subprocess

            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'uninstall', 'dgl', '-y', '--quiet'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            install_attempts = []

            if cuda_version:
                cuda_wheel_url = f'https://data.dgl.ai/wheels/{cuda_version}/repo.html'
                install_attempts.append(
                    (f"DGL {cuda_version} (DGL仓库)",
                     [sys.executable, '-m', 'pip', 'install', 'dgl',
                      '-f', cuda_wheel_url, '--quiet'])
                )
                install_attempts.append(
                    (f"DGL {cuda_version} (DGL仓库+PyPI)",
                     [sys.executable, '-m', 'pip', 'install', f'dgl+{cuda_version}',
                      '-f', cuda_wheel_url, '--quiet'])
                )

            install_attempts.append(
                ("DGL CPU (DGL仓库)",
                 [sys.executable, '-m', 'pip', 'install', 'dgl',
                  '--no-index', '-f', 'https://data.dgl.ai/wheels/repo.html', '--quiet'])
            )
            install_attempts.append(
                ("DGL CPU (DGL仓库+PyPI)",
                 [sys.executable, '-m', 'pip', 'install', 'dgl',
                  '-f', 'https://data.dgl.ai/wheels/repo.html', '--quiet'])
            )
            install_attempts.append(
                ("DGL (PyPI)",
                 [sys.executable, '-m', 'pip', 'install', 'dgl', '--quiet'])
            )

            installed = False
            for attempt_name, cmd in install_attempts:
                print(f"[DL] 尝试安装: {attempt_name}...")
                try:
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    test_env = {**os.environ, 'DGL_DOWNLOAD': '1'}
                    test = subprocess.run(
                        [sys.executable, '-c', 'import dgl; print(dgl.__version__)'],
                        capture_output=True, text=True, timeout=15,
                        env=test_env
                    )
                    if test.returncode == 0:
                        installed = True
                        print(f"[DL] ✅ {attempt_name}安装成功: {test.stdout.strip()}")
                        break
                    else:
                        err_output = test.stderr.strip() if test.stderr else ""
                        print(f"[DL] {attempt_name}安装后无法导入: {err_output[:100]}")
                        subprocess.check_call(
                            [sys.executable, '-m', 'pip', 'uninstall', 'dgl', '-y', '--quiet'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                except subprocess.TimeoutExpired:
                    print(f"[DL] {attempt_name}安装超时")
                except Exception as e:
                    print(f"[DL] {attempt_name}安装失败: {e}")

            if not installed:
                print("[DL] ❌ DGL所有安装方式均失败")
                print("[DL] 请手动运行以下命令安装:")
                if cuda_version:
                    print(f"[DL]   pip install dgl -f https://data.dgl.ai/wheels/{cuda_version}/repo.html")
                print("[DL]   pip install dgl --no-index -f https://data.dgl.ai/wheels/repo.html")
        except Exception as e:
            print(f"[DL] DGL安装过程异常: {e}")

    def _load_esm_model(self):
        if self._esm_loaded:
            return self._esm_model

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "facebook/esm2_t33_650M_UR50D"
            print(f"[ESM] 正在加载: {model_name}...")

            has_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if has_cuda else 'cpu')

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
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

        for i in range(n_residues - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

        g = dgl.graph((src, dst), num_nodes=n_residues)
        return g

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
