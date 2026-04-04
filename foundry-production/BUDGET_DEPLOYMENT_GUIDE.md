# 本科毕设低成本部署方案 - 蛋白Binder设计系统

**适用场景**: 本科毕业设计、个人学习、小规模研究  
**预算范围**: 0元 - 500元/月  
**更新日期**: 2025-04-04  

---

## 📋 目录

1. [方案对比](#1-方案对比)
2. [方案一: 免费云GPU平台](#2-方案一-免费云gpu平台)
3. [方案二: 国内GPU租用平台](#3-方案二-国内gpu租用平台)
4. [方案三: 学校实验室资源](#4-方案三-学校实验室资源)
5. [简化版系统架构](#5-简化版系统架构)
6. [推荐方案组合](#6-推荐方案组合)

---

## 1. 方案对比

### 1.1 四种方案总览

| 方案 | 成本 | GPU性能 | 使用时长 | 适合阶段 | 推荐度 |
|------|------|---------|----------|----------|--------|
| **Google Colab** | 免费 | T4 (16GB) | 每次最多12小时 | 开发测试 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | 免费 | P100 (16GB) | 每周30小时 | 开发测试 | ⭐⭐⭐⭐ |
| **AutoDL** | ¥1-2/小时 | RTX 3090 | 无限制 | 完整训练 | ⭐⭐⭐⭐⭐ |
| **学校实验室** | 免费 | 不确定 | 需申请 | 长期项目 | ⭐⭐⭐⭐ |

### 1.2 推荐方案组合

```
本科毕设推荐路线:

第一阶段 (第1-4周): 开发与调试
└─ 使用 Google Colab 免费 GPU
   - 编写代码
   - 调试模型
   - 小规模测试

第二阶段 (第5-8周): 完整实验
└─ 使用 AutoDL 租用 GPU (¥100-300)
   - 批量生成设计
   - 完整工作流测试
   - 收集实验数据

第三阶段 (第9-10周): 论文撰写
└─ 本地电脑 + 云端结果
   - 分析数据
   - 可视化结果
   - 撰写论文

总预算: ¥0 - ¥300
```

---

## 2. 方案一: 免费云GPU平台

### 2.1 Google Colab (强烈推荐)

#### 优势
- ✅ **完全免费**
- ✅ 无需配置，开箱即用
- ✅ 预装PyTorch、TensorFlow
- ✅ 支持Jupyter Notebook
- ✅ 可连接Google Drive存储

#### 限制
- ⚠️ 每次会话最多12小时
- ⚠️ 空闲超时会断开
- ⚠️ GPU类型随机（T4/P100）

#### 使用步骤

**Step 1: 访问Colab**

```
1. 打开浏览器访问: https://colab.research.google.com/
2. 使用Google账号登录
3. 点击"新建笔记本"
```

**Step 2: 启用GPU**

```python
# 在Colab中执行

# 1. 点击菜单: 运行时 → 更改运行时类型
# 2. 硬件加速器选择: GPU
# 3. 点击保存

# 验证GPU
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

**Step 3: 安装Foundry**

```python
# 在Colab中执行

# 安装Foundry
!pip install -q 'rc-foundry[all]'

# 下载模型权重 (首次需要10-15分钟)
!foundry install rfd3 ligandmpnn rf3

# 验证安装
import torch
from rfd3.engine import RFD3InferenceEngine

print("✓ Foundry安装成功")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

**Step 4: 运行Binder设计**

```python
# 在Colab中执行完整工作流

from lightning.fabric import seed_everything
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput
from biotite.structure import rmsd, superimpose
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
import numpy as np

# 设置随机种子
seed_everything(42)

# ========== Step 1: RFD3生成Binder骨架 ==========
print("Step 1: 生成Binder骨架...")

config = RFD3InferenceConfig(
    specification={
        'length': 60,  # 60残基的Binder
        'extra': {},
    },
    diffusion_batch_size=2,  # 生成2个结构
)

model = RFD3InferenceEngine(**config.__dict__)
rfd3_outputs = model.run(inputs=None, n_batches=1)

print(f"✓ 生成了 {len(rfd3_outputs)} 个Binder骨架")

# ========== Step 2: MPNN序列设计 ==========
print("\nStep 2: 设计序列...")

atom_array = rfd3_outputs[list(rfd3_outputs.keys())[0]][0].atom_array

mpnn_engine = MPNNInferenceEngine(
    model_type="ligand_mpnn",
    is_legacy_weights=True,
    out_directory=None,
    write_structures=False,
    write_fasta=False,
)

mpnn_outputs = mpnn_engine.run(
    input_dicts=[{"batch_size": 3}],  # 每个骨架生成3条序列
    atom_arrays=[atom_array]
)

print(f"✓ 设计了 {len(mpnn_outputs)} 条序列")

# ========== Step 3: RF3结构预测验证 ==========
print("\nStep 3: 验证结构...")

rf3_engine = RF3InferenceEngine(ckpt_path='rf3', verbose=False)
input_structure = InferenceInput.from_atom_array(
    mpnn_outputs[0].atom_array, 
    example_id="binder_1"
)
rf3_outputs = rf3_engine.run(inputs=input_structure)

print(f"✓ 结构预测完成")

# ========== Step 4: RMSD评估 ==========
print("\nStep 4: 评估设计质量...")

rf3_output = rf3_outputs["binder_1"][0]
aa_generated = atom_array
aa_refolded = rf3_output.atom_array

bb_generated = aa_generated[np.isin(aa_generated.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
bb_refolded = aa_refolded[np.isin(aa_refolded.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]

bb_refolded_fitted, _ = superimpose(bb_generated, bb_refolded)
rmsd_value = rmsd(bb_generated, bb_refolded_fitted)

print(f"\n{'='*50}")
print(f"设计结果:")
print(f"  RMSD: {rmsd_value:.2f} Å")
print(f"  pLDDT: {rf3_output.summary_confidences['overall_plddt']:.2f}")
print(f"  pTM: {rf3_output.summary_confidences['ptm']:.3f}")
print(f"  评价: {'优秀' if rmsd_value < 1.0 else '良好' if rmsd_value < 2.0 else '一般'}")
print(f"{'='*50}")
```

**Step 5: 保存结果到Google Drive**

```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存结果
import os
from atomworks.io.utils.io_utils import to_cif_file

result_dir = '/content/drive/MyDrive/binder_design_results'
os.makedirs(result_dir, exist_ok=True)

# 保存结构文件
to_cif_file(atom_array, f"{result_dir}/binder_backbone.cif")
to_cif_file(rf3_output.atom_array, f"{result_dir}/binder_predicted.cif")

# 保存序列
with open(f"{result_dir}/designed_sequences.txt", 'w') as f:
    for i, output in enumerate(mpnn_outputs):
        f.write(f">Sequence_{i+1}\n")
        f.write(f"{output.output_dict['designed_sequence']}\n")

print(f"✓ 结果已保存到: {result_dir}")
```

#### Colab使用技巧

```python
# 技巧1: 防止Colab断连
# 在控制台(F12)执行以下代码:
function KeepAlive() {
    console.log("保持连接...");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000);  // 每分钟点击一次

# 技巧2: 查看GPU使用情况
!nvidia-smi

# 技巧3: 清理显存
import torch
torch.cuda.empty_cache()

# 技巧4: 下载文件到本地
from google.colab import files
files.download('binder_backbone.cif')
```

---

### 2.2 Kaggle Kernels

#### 优势
- ✅ **完全免费**
- ✅ 每周30小时GPU时长
- ✅ P100 GPU (性能较好)
- ✅ 内置数据集

#### 使用步骤

```
1. 访问: https://www.kaggle.com/
2. 注册账号
3. 创建New Notebook
4. Settings → Accelerator → GPU P100
5. 运行代码
```

#### 示例代码

```python
# Kaggle中安装Foundry
!pip install -q 'rc-foundry[all]'
!foundry install rfd3 ligandmpnn rf3

# 后续代码与Colab相同
```

---

## 3. 方案二: 国内GPU租用平台

### 3.1 AutoDL (性价比最高)

#### 平台特点
- 💰 **价格便宜**: RTX 3090 约 ¥1.5/小时
- 🚀 **性能强劲**: 支持RTX 3090, A100等
- ⏱️ **按小时计费**: 用多少付多少
- 📦 **预装环境**: PyTorch镜像

#### 成本估算

| 任务 | GPU | 时长 | 费用 |
|------|-----|------|------|
| 环境配置 | RTX 3090 | 0.5小时 | ¥0.75 |
| 单次Binder设计 | RTX 3090 | 0.5小时 | ¥0.75 |
| 批量设计 (100个) | RTX 3090 | 5小时 | ¥7.5 |
| 完整毕设实验 | RTX 3090 | 50小时 | ¥75 |

**毕设总预算: ¥100-200**

#### 使用步骤

**Step 1: 注册充值**

```
1. 访问: https://www.autodl.com/
2. 注册账号
3. 充值 (建议充值 ¥100-200)
```

**Step 2: 租用实例**

```
1. 点击"算力市场"
2. 选择GPU: RTX 3090 (推荐)
3. 选择镜像: PyTorch 2.0 + Python 3.11
4. 点击"立即租用"
5. 选择按小时计费
```

**Step 3: 连接实例**

```bash
# 方式1: JupyterLab (推荐)
# 直接在网页中打开JupyterLab

# 方式2: SSH连接
ssh -p <端口> root@<地址>
# 密码在控制台查看
```

**Step 4: 配置环境**

```bash
# SSH连接后执行

# 更新pip
pip install --upgrade pip

# 安装Foundry
pip install 'rc-foundry[all]'

# 下载模型权重
foundry install rfd3 ligandmpnn rf3

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

**Step 5: 运行设计任务**

```bash
# 创建工作目录
mkdir -p ~/binder_design
cd ~/binder_design

# 上传你的代码或创建新文件
# 可以使用JupyterLab上传文件

# 运行设计脚本
python design_binder.py

# 或使用Jupyter Notebook
jupyter notebook --port=8888
```

**Step 6: 保存结果**

```bash
# 方式1: 下载到本地
# 在JupyterLab中右键文件 → Download

# 方式2: 使用SCP下载 (本地电脑执行)
scp -P <端口> root@<地址>:~/binder_design/results/* ./

# 方式3: 上传到云存储
# 使用阿里云OSS、腾讯云COS等
```

**Step 7: 关闭实例**

```
重要: 完成任务后立即关闭实例，避免继续计费！

1. 在控制台点击"关机"
2. 选择"关机并释放"
3. 确认关闭
```

#### AutoDL使用技巧

```bash
# 技巧1: 使用镜像市场
# 搜索"foundry"或"protein design"可能已有预装环境

# 技巧2: 保存自定义镜像
# 配置好环境后，可以保存为自定义镜像
# 下次租用时直接使用，节省配置时间

# 技巧3: 使用脚本自动化
# 创建启动脚本，实例启动时自动配置环境

# 技巧4: 监控GPU使用
watch -n 1 nvidia-smi
```

---

### 3.2 其他平台对比

| 平台 | GPU | 价格 | 特点 |
|------|-----|------|------|
| **矩池云** | RTX 3090 | ¥1.8/小时 | 国内老牌平台 |
| **恒源云** | RTX 3090 | ¥1.5/小时 | 价格便宜 |
| **阿里云PAI** | V100 | ¥10+/小时 | 企业级，贵 |
| **腾讯云** | T4 | ¥5+/小时 | 稳定但贵 |

**推荐**: AutoDL (性价比最高)

---

## 4. 方案三: 学校实验室资源

### 4.1 如何申请

```
申请流程:

1. 联系导师
   - 说明毕设需求
   - 询问实验室GPU资源
   
2. 联系实验室管理员
   - 提交申请表
   - 说明使用时长和目的
   
3. 签署使用协议
   - 遵守实验室规定
   - 不用于商业用途
   
4. 获取账号
   - SSH账号
   - VPN访问权限
```

### 4.2 使用学校服务器

```bash
# 通过VPN连接学校服务器
ssh username@server.university.edu.cn

# 加载环境模块 (如果使用module系统)
module load cuda/12.2
module load python/3.11

# 创建虚拟环境
python -m venv binder_env
source binder_env/bin/activate

# 安装依赖
pip install 'rc-foundry[all]'
foundry install rfd3 ligandmpnn rf3

# 运行任务
python design_binder.py
```

### 4.3 使用学校超算中心

```
如果学校有超算中心:

1. 申请账号
2. 提交作业脚本
3. 等待调度执行
4. 查看结果

优势:
- 免费或低成本
- 高性能GPU
- 可长时间运行

劣势:
- 需要排队等待
- 需要学习作业调度系统
```

---

## 5. 简化版系统架构

### 5.1 本科毕设版架构

对于本科毕设，不需要复杂的微服务架构，建议使用**单机版架构**：

```
┌─────────────────────────────────────────────────────────┐
│              简化版系统架构 (本科毕设版)                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    本地电脑 (开发端)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  代码编辑   │  │  文档撰写   │  │  数据分析   │   │
│  │  VS Code   │  │  Word/LaTeX│  │  Python    │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          │ 上传代码/下载结果
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              云GPU平台 (计算端)                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Jupyter Notebook / Python             │  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │
│  │  │   RFD3   │  │   MPNN   │  │   RF3    │      │  │
│  │  │  Engine  │  │  Engine  │  │  Engine  │      │  │
│  │  └──────────┘  └──────────┘  └──────────┘      │  │
│  │                                                   │  │
│  │  ┌──────────────────────────────────────────┐   │  │
│  │  │         工作流脚本 (design_binder.py)      │   │  │
│  │  └──────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  存储: Google Drive / 云盘 / 本地下载                   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 核心文件结构

```
binder_design_project/
│
├── design_binder.py          # 主设计脚本
├── utils.py                  # 工具函数
├── config.py                 # 配置文件
│
├── input/                    # 输入文件
│   └── target_protein.cif    # 目标蛋白结构
│
├── output/                   # 输出结果
│   ├── structures/           # 结构文件
│   ├── sequences/            # 序列文件
│   └── metrics/              # 评估指标
│
├── notebooks/                # Jupyter笔记本
│   ├── 01_test_env.ipynb     # 环境测试
│   ├── 02_design_binder.ipynb # Binder设计
│   └── 03_analyze_results.ipynb # 结果分析
│
├── docs/                     # 文档
│   ├── 毕设论文.docx
│   └── 使用说明.md
│
└── requirements.txt          # 依赖列表
```

### 5.3 核心脚本

#### design_binder.py

```python
"""
蛋白Binder设计主脚本
适用于本科毕设项目
"""

import os
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from lightning.fabric import seed_everything

from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput
from biotite.structure import rmsd, superimpose, get_residue_starts
from biotite.sequence import ProteinSequence
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
from atomworks.io.utils.io_utils import to_cif_file


class BinderDesigner:
    """简化版Binder设计器"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / "structures").mkdir(exist_ok=True)
        (self.output_dir / "sequences").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        print(f"✓ 输出目录: {self.output_dir.absolute()}")
    
    def design(
        self,
        target_file: str = None,
        binder_length: int = 60,
        num_designs: int = 5,
        sequences_per_design: int = 3,
        seed: int = 42
    ):
        """
        执行Binder设计
        
        Args:
            target_file: 目标蛋白文件路径 (None表示无条件生成)
            binder_length: Binder长度
            num_designs: 生成骨架数量
            sequences_per_design: 每个骨架生成序列数
            seed: 随机种子
        """
        seed_everything(seed)
        
        results = []
        
        print(f"\n{'='*60}")
        print(f"开始Binder设计")
        print(f"  目标蛋白: {target_file or '无条件生成'}")
        print(f"  Binder长度: {binder_length}")
        print(f"  骨架数量: {num_designs}")
        print(f"  序列数量: {sequences_per_design}")
        print(f"{'='*60}\n")
        
        # Step 1: 生成Binder骨架
        print("[Step 1/4] 生成Binder骨架...")
        backbones = self._generate_backbones(
            target_file=target_file,
            length=binder_length,
            num_designs=num_designs
        )
        print(f"  ✓ 生成了 {len(backbones)} 个骨架\n")
        
        # Step 2: 设计序列
        print("[Step 2/4] 设计序列...")
        designs = self._design_sequences(
            backbones=backbones,
            sequences_per_design=sequences_per_design
        )
        print(f"  ✓ 设计了 {len(designs)} 条序列\n")
        
        # Step 3: 结构预测验证
        print("[Step 3/4] 验证结构...")
        validated = self._validate_structures(designs)
        print(f"  ✓ 验证了 {len(validated)} 个设计\n")
        
        # Step 4: 评估和筛选
        print("[Step 4/4] 评估结果...")
        results = self._evaluate_designs(validated)
        print(f"  ✓ 完成 {len(results)} 个设计评估\n")
        
        # 保存结果
        self._save_results(results)
        
        print(f"{'='*60}")
        print(f"设计完成！结果保存在: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")
        
        return results
    
    def _generate_backbones(self, target_file, length, num_designs):
        """生成Binder骨架"""
        config = RFD3InferenceConfig(
            specification={'length': length, 'extra': {}},
            diffusion_batch_size=num_designs,
        )
        
        model = RFD3InferenceEngine(**config.__dict__)
        outputs = model.run(inputs=None, n_batches=1)
        
        backbones = []
        for key, data in outputs.items():
            for i, output in enumerate(data):
                backbones.append({
                    'id': f"backbone_{len(backbones)}",
                    'atom_array': output.atom_array,
                    'metadata': output.metadata
                })
        
        return backbones
    
    def _design_sequences(self, backbones, sequences_per_design):
        """设计序列"""
        mpnn_engine = MPNNInferenceEngine(
            model_type="ligand_mpnn",
            is_legacy_weights=True,
            out_directory=None,
            write_structures=False,
            write_fasta=False,
        )
        
        designs = []
        for backbone in backbones:
            outputs = mpnn_engine.run(
                input_dicts=[{"batch_size": sequences_per_design}],
                atom_arrays=[backbone['atom_array']]
            )
            
            for output in outputs:
                designs.append({
                    'id': f"design_{len(designs)}",
                    'backbone_id': backbone['id'],
                    'atom_array': output.atom_array,
                    'sequence': output.output_dict['designed_sequence'],
                    'sequence_recovery': output.output_dict['sequence_recovery']
                })
        
        return designs
    
    def _validate_structures(self, designs):
        """结构预测验证"""
        rf3_engine = RF3InferenceEngine(ckpt_path='rf3', verbose=False)
        
        validated = []
        for design in designs:
            input_struct = InferenceInput.from_atom_array(
                design['atom_array'],
                example_id=design['id']
            )
            
            outputs = rf3_engine.run(inputs=input_struct)
            output = outputs[design['id']][0]
            
            validated.append({
                **design,
                'predicted_structure': output.atom_array,
                'plddt': output.summary_confidences['overall_plddt'],
                'pae': output.summary_confidences['overall_pae'],
                'ptm': output.summary_confidences['ptm'],
                'ranking_score': output.summary_confidences['ranking_score']
            })
        
        return validated
    
    def _evaluate_designs(self, validated):
        """评估设计"""
        results = []
        
        for design in validated:
            # 计算RMSD
            aa_generated = design['atom_array']
            aa_predicted = design['predicted_structure']
            
            bb_generated = aa_generated[
                np.isin(aa_generated.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
            ]
            bb_predicted = aa_predicted[
                np.isin(aa_predicted.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
            ]
            
            bb_predicted_fitted, _ = superimpose(bb_generated, bb_predicted)
            rmsd_value = rmsd(bb_generated, bb_predicted_fitted)
            
            # 评估质量
            quality = '优秀' if rmsd_value < 1.0 else '良好' if rmsd_value < 2.0 else '一般'
            
            results.append({
                'id': design['id'],
                'sequence': design['sequence'],
                'rmsd': float(rmsd_value),
                'plddt': float(design['plddt']),
                'pae': float(design['pae']),
                'ptm': float(design['ptm']),
                'ranking_score': float(design['ranking_score']),
                'quality': quality
            })
        
        # 按RMSD排序
        results.sort(key=lambda x: x['rmsd'])
        
        return results
    
    def _save_results(self, results):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON
        with open(self.output_dir / "metrics" / f"results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存序列FASTA
        with open(self.output_dir / "sequences" / f"sequences_{timestamp}.fasta", 'w') as f:
            for r in results:
                f.write(f">{r['id']} | RMSD={r['rmsd']:.2f}Å | pLDDT={r['plddt']:.1f}\n")
                f.write(f"{r['sequence']}\n\n")
        
        # 保存CSV
        import csv
        with open(self.output_dir / "metrics" / f"results_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  ✓ 保存JSON: metrics/results_{timestamp}.json")
        print(f"  ✓ 保存FASTA: sequences/sequences_{timestamp}.fasta")
        print(f"  ✓ 保存CSV: metrics/results_{timestamp}.csv")


def main():
    """主函数"""
    designer = BinderDesigner(output_dir="./output")
    
    results = designer.design(
        target_file=None,        # 无条件生成
        binder_length=60,        # 60残基
        num_designs=3,           # 3个骨架
        sequences_per_design=2,  # 每个骨架2条序列
        seed=42
    )
    
    # 打印最佳设计
    print("\n最佳设计 (RMSD最小):")
    best = results[0]
    print(f"  ID: {best['id']}")
    print(f"  序列: {best['sequence']}")
    print(f"  RMSD: {best['rmsd']:.2f} Å")
    print(f"  pLDDT: {best['plddt']:.1f}")
    print(f"  质量: {best['quality']}")


if __name__ == "__main__":
    main()
```

### 5.4 requirements.txt

```txt
# 核心依赖
rc-foundry[all]
torch>=2.0.0
lightning>=2.0.0
biotite>=0.38.0

# 工具库
numpy
pandas
matplotlib
seaborn

# Jupyter (可选)
jupyter
ipykernel
```

---

## 6. 推荐方案组合

### 6.1 最佳实践路线

```
本科毕设推荐时间线:

第1-2周: 环境熟悉
├─ 使用 Google Colab
├─ 学习 RFD3/MPNN/RF3 基本用法
├─ 运行示例代码
└─ 成本: ¥0

第3-4周: 代码开发
├─ 使用 Google Colab
├─ 编写 design_binder.py
├─ 调试工作流
└─ 成本: ¥0

第5-6周: 小规模实验
├─ 使用 Google Colab / Kaggle
├─ 生成 10-20 个设计
├─ 分析结果
└─ 成本: ¥0

第7-8周: 大规模实验
├─ 租用 AutoDL GPU (RTX 3090)
├─ 批量生成 100+ 设计
├─ 收集实验数据
└─ 成本: ¥50-100

第9-10周: 论文撰写
├─ 本地电脑
├─ 数据分析
├─ 可视化
├─ 撰写论文
└─ 成本: ¥0

总预算: ¥50-100
```

### 6.2 快速开始清单

```bash
# 第一步: 在Colab中测试
1. 打开 https://colab.research.google.com/
2. 新建笔记本
3. 启用GPU
4. 运行以下代码:

!pip install -q 'rc-foundry[all]'
!foundry install rfd3 ligandmpnn rf3

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 第二步: 运行示例
# 复制 design_binder.py 代码到Colab运行

# 第三步: 保存结果到Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 第四步: 如果需要大规模实验
# 注册 AutoDL: https://www.autodl.com/
# 充值 ¥100
# 租用 RTX 3090
# 运行批量实验
```

---

## 📝 总结

### 推荐方案

| 阶段 | 推荐平台 | 理由 |
|------|----------|------|
| **开发测试** | Google Colab | 免费、方便、够用 |
| **小规模实验** | Google Colab | 免费、快速验证 |
| **大规模实验** | AutoDL | 便宜、高性能 |
| **论文撰写** | 本地电脑 | 无需GPU |

### 预算估算

- **最低预算**: ¥0 (全程使用免费平台)
- **推荐预算**: ¥100-200 (AutoDL租用50-100小时)
- **舒适预算**: ¥300-500 (更多实验次数)

### 关键建议

1. **先用免费平台**: Colab完全够用于开发和测试
2. **按需租用**: 只在大规模实验时租用GPU
3. **及时关机**: AutoDL用完立即关机
4. **保存结果**: 所有结果保存到云盘
5. **代码复用**: 写好脚本，避免重复配置环境

---

**祝你毕设顺利！** 🎓
