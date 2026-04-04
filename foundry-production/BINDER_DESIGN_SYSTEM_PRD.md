# 蛋白Binder设计系统 - 预备工作文档

**项目代号**: ProteinBinder Designer  
**版本**: v1.0  
**创建日期**: 2025-04-04  
**文档状态**: 预备阶段

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心功能模块](#3-核心功能模块)
4. [热点残基预测模型集成](#4-热点残基预测模型集成)
5. [技术栈与依赖](#5-技术栈与依赖)
6. [开发计划](#6-开发计划)
7. [资源需求](#7-资源需求)
8. [风险评估与应对](#8-风险评估与应对)
9. [附录](#9-附录)

---

## 1. 项目概述

### 1.1 项目背景

蛋白质-蛋白质相互作用（PPI）在生物医学研究和药物开发中具有重要意义。设计能够特异性结合目标蛋白的Binder蛋白，是蛋白质设计领域的核心挑战之一。本项目旨在构建一个端到端的深度学习蛋白Binder设计系统，整合多个先进的深度学习模型，实现从目标蛋白到可实验验证的Binder序列的全自动化设计流程。

### 1.2 项目目标

**主要目标**:

- 构建一个完整的蛋白Binder设计Web平台
- 整合RFD3、MPNN、RF3三大核心模型
- 引入热点残基预测模型，提升设计成功率
- 提供直观的可视化界面和交互式操作
- 支持批量设计和结果筛选

**技术目标**:

- 实现端到端自动化设计流程
- 提供RESTful API和异步任务处理
- 支持GPU加速推理
- 提供结构可视化和结果分析工具

### 1.3 核心价值

| 价值点     | 描述                         |
| ---------- | ---------------------------- |
| **自动化** | 全流程自动化，无需手动干预   |
| **智能化** | AI驱动的热点预测和Binder设计 |
| **可视化** | 直观的3D结构可视化和交互     |
| **可扩展** | 模块化设计，易于扩展新模型   |
| **高性能** | GPU加速，支持批量处理        |

---

## 2. 系统架构

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        蛋白Binder设计系统架构                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户层 (User Layer)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Web浏览器   │  │  移动端APP   │  │   API客户端   │  │  命令行工具   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            前端层 (Frontend Layer)                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    React + TypeScript + Vite                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  结构上传   │  │  参数配置   │  │  3D可视化   │  │  结果展示   │ │  │
│  │  │  组件      │  │  组件      │  │  组件      │  │  组件      │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  热点选择   │  │  任务监控   │  │  结果对比   │  │  文件下载   │ │  │
│  │  │  组件      │  │  组件      │  │  组件      │  │  组件      │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              Mol* / NGL Viewer                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API层 (API Layer)                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI + Uvicorn                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  设计API    │  │  文件API    │  │  任务API    │  │  用户API    │ │  │
│  │  │  /design   │  │  /files    │  │  /tasks    │  │  /users    │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │  认证中间件  │  │  限流中间件  │  │  日志中间件  │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              WebSocket Server                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          服务层 (Service Layer)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         业务逻辑服务                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  工作流引擎  │  │  文件服务   │  │  存储服务   │  │  通知服务   │ │  │
│  │  │  Workflow  │  │  FileStore │  │  Storage   │  │  Notify    │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        任务队列层 (Task Queue Layer)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       Celery + Redis                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  任务调度器  │  │  任务队列   │  │  结果后端   │  │  监控面板   │ │  │
│  │  │  Scheduler │  │  Queue     │  │  Backend   │  │  Flower    │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              Celery Workers (GPU)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         核心引擎层 (Core Engine Layer)                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        深度学习模型引擎                                │  │
│  │                                                                      │  │
│  │  ┌────────────────┐         ┌────────────────┐                      │  │
│  │  │  热点预测引擎   │         │  Binder设计引擎 │                      │  │
│  │  │                │         │                │                      │  │
│  │  │ ┌────────────┐ │         │ ┌────────────┐ │                      │  │
│  │  │ │ Hotspot    │ │         │ │   RFD3     │ │                      │  │
│  │  │ │ Model 1    │ │         │ │  Engine    │ │                      │  │
│  │  │ └────────────┘ │         │ └────────────┘ │                      │  │
│  │  │ ┌────────────┐ │    ┌───▶│ ┌────────────┐ │                      │  │
│  │  │ │ Hotspot    │ │    │    │ │   MPNN     │ │                      │  │
│  │  │ │ Model 2    │ │    │    │ │  Engine    │ │                      │  │
│  │  │ └────────────┘ │    │    │ └────────────┘ │                      │  │
│  │  └────────────────┘    │    │ ┌────────────┐ │                      │  │
│  │                        │    │ │   RF3      │ │                      │  │
│  │                        │    │ │  Engine    │ │                      │  │
│  │                        │    │ └────────────┘ │                      │  │
│  │                        │    └────────────────┘                      │  │
│  │                        │              │                             │  │
│  │                        │              ▼                             │  │
│  │                        │    ┌────────────────┐                      │  │
│  │                        └────│  验证评估引擎  │                      │  │
│  │                             │  Validation   │                      │  │
│  │                             └────────────────┘                      │  │
│  │                                                                      │  │
│  │                    统一数据格式: Biotite AtomArray                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  模型权重存储: foundry checkpoints (rfd3, ligandmpnn, rf3)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据层 (Data Layer)                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │ PostgreSQL  │  │    Redis    │  │    MinIO    │  │  本地文件    │ │  │
│  │  │  关系数据库  │  │  缓存/队列  │  │  对象存储   │  │  结构文件    │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  │                                                                      │  │
│  │  存储内容:                                                           │  │
│  │  - 用户信息和认证数据                                                │  │
│  │  - 任务状态和进度                                                    │  │
│  │  - 设计结果和元数据                                                  │  │
│  │  - PDB/CIF结构文件                                                   │  │
│  │  - 模型权重和配置                                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Binder设计数据流                                  │
└─────────────────────────────────────────────────────────────────────────────┘

用户输入                    系统处理                      输出结果
─────────                  ────────                      ────────

┌──────────┐
│ 目标蛋白  │
│ PDB/CIF  │
└─────┬────┘
      │
      ▼
┌──────────┐      ┌──────────────────────────────────────────────┐
│ 热点残基  │─────▶│ Step 0: 热点残基预测 (可选，推荐手动指定)     │
│  (可选)   │      │                                              │
└──────────┘      │  ⚠️ 本科毕设推荐: 手动指定热点残基            │
                  │                                              │
                  │  方式1: 手动指定 (推荐)                       │
                  │  ┌─────────────────────────────────────┐     │
                  │  │ hotspots = {                        │     │
                  │  │   "E64": "CD2,CZ",                  │     │
                  │  │   "E88": "CG,CZ",                   │     │
                  │  │   "E96": "CD1,CZ"                   │     │
                  │  │ }                                   │     │
                  │  └─────────────────────────────────────┘     │
                  │                                              │
                  │  方式2: 自动预测 (可选，需额外开发)           │
                  │  ┌─────────────┐    ┌─────────────┐         │
                  │  │ Hotspot     │    │ Hotspot     │         │
                  │  │ Model 1     │    │ Model 2     │         │
                  │  │ (界面预测)   │    │ (能量预测)   │         │
                  │  └──────┬──────┘    └──────┬──────┘         │
                  │         │                  │                 │
                  │         └────────┬─────────┘                 │
                  │                  ▼                           │
                  │         ┌─────────────┐                     │
                  │         │ 热点残基融合 │                     │
                  │         │ & 排序      │                     │
                  │         └──────┬──────┘                     │
                  └────────────────┼─────────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │ Step 1: RFD3 Binder骨架生成                   │
                  │                                              │
                  │  输入:                                        │
                  │  - 目标蛋白结构                               │
                  │  - contig: "40-120,/0,E6-155"                │
                  │  - select_hotspots: {"E64": "CD2,CZ", ...}   │
                  │  - infer_ori_strategy: "hotspots"            │
                  │                                              │
                  │  输出:                                        │
                  │  - Binder骨架结构 (AtomArray)                 │
                  │  - N个候选骨架                                │
                  └────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │ Step 2: MPNN序列设计                         │
                  │                                              │
                  │  输入:                                        │
                  │  - Binder骨架结构                             │
                  │  - batch_size: 每个骨架生成M条序列            │
                  │                                              │
                  │  输出:                                        │
                  │  - 设计序列 (氨基酸序列)                      │
                  │  - N × M 个候选设计                          │
                  └────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │ Step 3: RF3结构预测验证                      │
                  │                                              │
                  │  输入:                                        │
                  │  - 设计序列 + 骨架结构                        │
                  │                                              │
                  │  输出:                                        │
                  │  - 预测结构 (AtomArray)                       │
                  │  - 置信度指标 (pLDDT, PAE, pTM)               │
                  └────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │ Step 4: 验证与筛选                           │
                  │                                              │
                  │  计算:                                        │
                  │  - RMSD (预测结构 vs RFD3骨架)                │
                  │  - pLDDT (结构置信度)                         │
                  │  - PAE (预测对齐误差)                         │
                  │  - 界面接触分析                               │
                  │                                              │
                  │  筛选条件:                                     │
                  │  - RMSD < 阈值 (如 2.0 Å)                     │
                  │  - pLDDT > 阈值 (如 80)                       │
                  │  - 界面接触数 > 最小值                         │
                  └────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  最终结果    │
                            │             │
                            │ - 序列      │
                            │ - 结构文件  │
                            │ - 评估指标  │
                            │ - 可视化    │
                            └─────────────┘
```

---

## 3. 核心功能模块

### 3.1 功能模块清单

| 模块ID | 模块名称   | 功能描述                    | 优先级 | 说明                   |
| ------ | ---------- | --------------------------- | ------ | ---------------------- |
| M01    | 用户管理   | 用户注册、登录、权限管理    | P1     |                        |
| M02    | 文件管理   | PDB/CIF文件上传、解析、存储 | P0     | 核心                   |
| M03    | 热点预测   | 热点残基自动预测            | P2     | **可选**，推荐手动指定 |
| M04    | Binder设计 | RFD3骨架生成                | P0     | 核心                   |
| M05    | 序列设计   | MPNN序列设计                | P0     | 核心                   |
| M06    | 结构验证   | RF3结构预测与验证           | P0     |
| M07    | 结果评估   | RMSD计算、指标分析          | P0     |
| M08    | 可视化     | 3D结构可视化、交互          | P1     |
| M09    | 任务管理   | 异步任务调度、进度追踪      | P0     |
| M10    | 结果导出   | 文件下载、报告生成          | P1     |

### 3.2 核心工作流详细设计

#### 3.2.1 Binder设计工作流

```python
class BinderDesignWorkflow:
    """
    Binder设计完整工作流

    流程:
    1. 输入验证和预处理
    2. 热点残基预测 (可选)
    3. RFD3骨架生成
    4. MPNN序列设计
    5. RF3结构验证
    6. 结果评估和筛选
    """

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.hotspot_predictor = HotspotPredictorEngine()
        self.rfd3_engine = RFD3BinderEngine()
        self.mpnn_engine = MPNNSequenceEngine()
        self.rf3_engine = RF3ValidationEngine()
        self.validator = DesignValidator()

    async def run(self, input_data: DesignInput) -> DesignResult:
        """
        执行完整工作流
        """
        # Step 0: 预处理
        validated_input = await self._preprocess(input_data)

        # Step 1: 热点预测 (可选，推荐手动指定)
        # 如果用户已手动指定热点残基，跳过自动预测
        if not validated_input.hotspots:
            # 方式1: 使用默认热点 (推荐)
            # validated_input.hotspots = self._get_default_hotspots()

            # 方式2: 自动预测 (可选，需要额外开发)
            # hotspots = await self._predict_hotspots(validated_input)
            # validated_input.hotspots = hotspots

            # 如果既没有手动指定，也没有自动预测，使用空字典
            validated_input.hotspots = {}

        # Step 2: RFD3骨架生成
        backbones = await self._generate_backbones(validated_input)

        # Step 3: MPNN序列设计
        designs = await self._design_sequences(backbones)

        # Step 4: RF3验证
        validated_designs = await self._validate_structures(designs)

        # Step 5: 结果筛选
        final_results = self._filter_results(validated_designs)

        return final_results
```

#### 3.2.2 RFD3 Binder设计参数

```python
class RFD3BinderConfig:
    """
    RFD3 Binder设计配置

    核心参数说明:
    """

    # 输入文件
    input_file: str                    # 目标蛋白PDB/CIF文件路径

    # Contig定义
    contig: str                        # 格式: "binder_length,/0,target_chain_residues"
                                      # 示例: "40-120,/0,E6-155"
                                      # - "40-120": Binder长度范围
                                      # - "/0": 链断裂标记
                                      # - "E6-155": 目标蛋白E链6-155残基

    # 热点残基选择 (原子级别)
    select_hotspots: Dict[str, str]    # 格式: {"残基ID": "原子名列表"}
                                      # 示例: {"E64": "CD2,CZ", "E88": "CG,CZ"}

    # 取向策略
    infer_ori_strategy: str            # "hotspots" | "com"
                                      # 推荐: "hotspots" - 根据热点自动计算起始取向

    # 固定设置
    select_fixed_atoms: bool           # True: 目标蛋白原子固定在空间中
    select_unfixed_sequence: bool      # False: 目标蛋白序列固定

    # 生成参数
    diffusion_batch_size: int          # 每批生成的结构数量
    n_batches: int                     # 批次数

    # 高级参数
    plddt_enhanced: bool = True        # pLDDT增强
    symmetry: Optional[dict] = None    # 对称性设置
```

---

## 4. 热点残基预测模型集成 (可选模块)

> **⚠️ 重要说明**: 热点预测是**可选功能**，不影响核心Binder设计流程。
>
> - **推荐做法**: 在本科毕设中，建议**手动指定热点残基**（基于文献或经验）
> - **自动预测**: 如果时间充裕，可以后续添加自动预测功能
> - **两种方式**: 系统支持手动指定和自动预测两种方式

### 4.1 热点残基预测模型概述

热点残基（Hotspot Residues）是蛋白质-蛋白质界面中对结合能贡献最大的残基。准确预测热点残基对于Binder设计至关重要，可以：

- 指导Binder朝向正确的界面区域
- 提高Binder与目标蛋白的亲和力
- 减少无效设计尝试

**注意**: 对于本科毕设项目，可以采用以下简化方案：

```python
# 方案1: 手动指定热点残基 (推荐)
# 基于文献或经验，直接指定热点残基
hotspots = {
    "E64": "CD2,CZ",   # 文献报道的关键残基
    "E88": "CG,CZ",    # 基于结构分析选择
    "E96": "CD1,CZ"    # 界面中心残基
}

# 方案2: 简单规则选择
# 选择界面中心残基、高SASA残基等
# 无需复杂模型

# 方案3: 自动预测 (可选，需要额外开发)
# 使用热点预测模型自动识别
```

### 4.2 推荐的热点预测模型

#### 模型1: **Hotspot3D / KFC2**

**特点**:

- 基于结构特征的热点预测
- 考虑残基的溶剂可及性、能量贡献、保守性
- 提供残基级别的热点评分

**集成方案**:

```python
class Hotspot3DPredictor:
    """
    Hotspot3D热点预测模型封装

    输入: 蛋白质结构 (PDB/CIF)
    输出: 热点残基列表及评分
    """

    def predict(self, structure_file: str, chain_id: str) -> List[HotspotResidue]:
        """
        预测热点残基

        Returns:
            List[HotspotResidue]: 热点残基列表
                - residue_id: 残基ID
                - chain_id: 链ID
                - score: 热点评分 (0-1)
                - atoms: 推荐的原子列表
        """
        pass
```

#### 模型2: **DeepSite / PPI-HotspotDB**

**特点**:

- 深度学习模型
- 结合序列和结构特征
- 可预测界面残基和热点残基

**集成方案**:

```python
class DeepHotspotPredictor:
    """
    深度学习热点预测模型

    基于图神经网络的热点预测
    """

    def predict(
        self,
        structure: AtomArray,
        interface_chain: Optional[str] = None
    ) -> List[HotspotPrediction]:
        """
        预测界面热点残基

        Args:
            structure: 蛋白质结构
            interface_chain: 已知的界面链 (可选)

        Returns:
            热点预测结果，包含:
            - 残基位置
            - 热点概率
            - 推荐的原子
        """
        pass
```

### 4.3 热点预测模型集成架构

> **⚠️ 重要**: 以下架构为**可选功能**，本科毕设项目建议使用手动指定方式。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    热点残基预测模块架构 (可选功能)                            │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────┐
                         │   目标蛋白结构输入    │
                         │   (PDB/CIF/AtomArray)│
                         └───────────┬──────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      结构预处理与特征提取        │
                    │  - 溶剂可及性计算 (SASA)        │
                    │  - 残基保守性分析               │
                    │  - 界面残基识别                 │
                    │  - 能量计算                     │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐       ┌─────────────────────┐
        │   Hotspot Model 1   │       │   Hotspot Model 2   │
        │   (结构特征方法)     │       │   (深度学习方法)     │
        │                     │       │                     │
        │  - KFC2             │       │  - GNN-based        │
        │  - Hotspot3D        │       │  - DeepSite         │
        │  - FTMap            │       │  - PPI-Hotspot      │
        └──────────┬──────────┘       └──────────┬──────────┘
                   │                              │
                   │      ┌────────────────┐      │
                   └─────▶│  预测结果融合   │◀─────┘
                          │                │
                          │ - 加权平均     │
                          │ - 投票机制     │
                          │ - 置信度排序   │
                          └───────┬────────┘
                                  │
                                  ▼
                    ┌────────────────────────────────┐
                    │        热点残基输出             │
                    │                                │
                    │  输出格式:                      │
                    │  {                              │
                    │    "E64": {                    │
                    │      "atoms": ["CD2", "CZ"],   │
                    │      "score": 0.92,            │
                    │      "type": "hydrophobic"     │
                    │    },                          │
                    │    "E88": {                    │
                    │      "atoms": ["CG", "CZ"],    │
                    │      "score": 0.87,            │
                    │      "type": "aromatic"        │
                    │    }                           │
                    │  }                             │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │     传递给RFD3 Binder设计       │
                    │                                │
                    │  select_hotspots = {           │
                    │    "E64": "CD2,CZ",            │
                    │    "E88": "CG,CZ",             │
                    │    ...                         │
                    │  }                             │
                    │                                │
                    │  infer_ori_strategy = "hotspots"│
                    └────────────────────────────────┘
```

### 4.4 热点预测API设计

```python
# api/routes/hotspot.py

from fastapi import APIRouter, UploadFile, File
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/v1/hotspot", tags=["Hotspot Prediction"])

class HotspotPredictionRequest(BaseModel):
    chain_id: Optional[str] = None          # 目标链ID
    interface_chain: Optional[str] = None   # 已知界面链
    min_score: float = 0.5                  # 最小热点评分阈值
    top_k: int = 10                         # 返回top-k热点
    models: List[str] = ["model1", "model2"] # 使用的模型

class HotspotResidue(BaseModel):
    residue_id: str                         # 残基ID (如 "E64")
    chain_id: str                           # 链ID
    residue_name: str                       # 残基名称
    score: float                            # 热点评分
    atoms: List[str]                        # 推荐原子
    hotspot_type: str                       # 热点类型
    confidence: float                       # 置信度

class HotspotPredictionResponse(BaseModel):
    hotspots: List[HotspotResidue]
    select_hotspots: Dict[str, str]         # RFD3格式
    model_scores: Dict[str, float]          # 各模型评分

@router.post("/predict", response_model=HotspotPredictionResponse)
async def predict_hotspots(
    file: UploadFile = File(..., description="目标蛋白PDB/CIF文件"),
    request: HotspotPredictionRequest = None
):
    """
    预测热点残基

    支持多种模型:
    - model1: 结构特征方法 (KFC2/Hotspot3D)
    - model2: 深度学习方法 (GNN-based)

    返回格式兼容RFD3的select_hotspots参数
    """
    pass

@router.post("/visualize")
async def visualize_hotspots(
    file: UploadFile = File(...),
    hotspots: Dict[str, str] = None
):
    """
    在3D结构上可视化热点残基

    返回带有热点标记的结构文件
    """
    pass
```

### 4.5 热点预测模型实现示例

```python
# core/hotspot_predictor.py

import numpy as np
from typing import List, Dict, Tuple
from biotite.structure import AtomArray
from atomworks.ml.utils.structure import calculate_sasa

class HotspotPredictorEngine:
    """
    热点残基预测引擎

    整合多个热点预测模型
    """

    def __init__(self, model_paths: Dict[str, str]):
        """
        Args:
            model_paths: 模型路径字典
                {
                    "kfc2": "/path/to/kfc2/model",
                    "deep_hotspot": "/path/to/deep/model"
                }
        """
        self.models = {}
        if "kfc2" in model_paths:
            self.models["kfc2"] = KFC2Predictor(model_paths["kfc2"])
        if "deep_hotspot" in model_paths:
            self.models["deep_hotspot"] = DeepHotspotPredictor(model_paths["deep_hotspot"])

    def predict(
        self,
        atom_array: AtomArray,
        chain_id: str = None,
        interface_chain: str = None,
        min_score: float = 0.5,
        top_k: int = 10
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        执行热点预测

        Returns:
            hotspots: 热点残基详细信息列表
            select_hotspots: RFD3格式的热点选择字典
        """
        all_predictions = {}

        # 运行所有模型
        for model_name, model in self.models.items():
            predictions = model.predict(atom_array, chain_id, interface_chain)
            all_predictions[model_name] = predictions

        # 融合预测结果
        fused_hotspots = self._fuse_predictions(
            all_predictions,
            min_score=min_score,
            top_k=top_k
        )

        # 转换为RFD3格式
        select_hotspots = self._to_rfd3_format(fused_hotspots)

        return fused_hotspots, select_hotspots

    def _fuse_predictions(
        self,
        all_predictions: Dict[str, List[Dict]],
        min_score: float,
        top_k: int
    ) -> List[Dict]:
        """
        融合多个模型的预测结果

        策略:
        1. 加权平均评分
        2. 置信度加权
        3. 过滤低分残基
        4. 返回top-k
        """
        residue_scores = {}

        for model_name, predictions in all_predictions.items():
            weight = self._get_model_weight(model_name)
            for pred in predictions:
                res_key = f"{pred['chain_id']}{pred['residue_id']}"
                if res_key not in residue_scores:
                    residue_scores[res_key] = {
                        'scores': [],
                        'atoms': pred.get('atoms', []),
                        'info': pred
                    }
                residue_scores[res_key]['scores'].append(
                    pred['score'] * weight
                )

        # 计算加权平均
        fused = []
        for res_key, data in residue_scores.items():
            avg_score = np.mean(data['scores'])
            if avg_score >= min_score:
                data['info']['fused_score'] = avg_score
                fused.append(data['info'])

        # 排序并返回top-k
        fused.sort(key=lambda x: x['fused_score'], reverse=True)
        return fused[:top_k]

    def _to_rfd3_format(self, hotspots: List[Dict]) -> Dict[str, str]:
        """
        转换为RFD3的select_hotspots格式

        格式: {"E64": "CD2,CZ", "E88": "CG,CZ"}
        """
        select_hotspots = {}
        for hotspot in hotspots:
            res_key = f"{hotspot['chain_id']}{hotspot['residue_id']}"
            atoms = ",".join(hotspot.get('atoms', []))
            if atoms:
                select_hotspots[res_key] = atoms
        return select_hotspots

    def _get_model_weight(self, model_name: str) -> float:
        """获取模型权重"""
        weights = {
            "kfc2": 0.4,
            "deep_hotspot": 0.6
        }
        return weights.get(model_name, 0.5)


class KFC2Predictor:
    """
    KFC2热点预测模型

    基于结构特征的方法
    """

    def predict(
        self,
        atom_array: AtomArray,
        chain_id: str = None,
        interface_chain: str = None
    ) -> List[Dict]:
        """
        预测热点残基

        特征:
        - SASA (溶剂可及表面积)
        - 残基保守性
        - 能量贡献
        - 几何特征
        """
        predictions = []

        # 计算SASA
        sasa = calculate_sasa(atom_array)

        # 识别界面残基
        interface_residues = self._identify_interface(
            atom_array,
            chain_id,
            interface_chain
        )

        # 对每个界面残基计算热点评分
        for res_info in interface_residues:
            score = self._calculate_hotspot_score(
                atom_array,
                res_info,
                sasa
            )

            if score > 0.3:  # 初步过滤
                predictions.append({
                    'chain_id': res_info['chain_id'],
                    'residue_id': res_info['residue_id'],
                    'residue_name': res_info['res_name'],
                    'score': score,
                    'atoms': self._select_hotspot_atoms(res_info),
                    'type': self._classify_hotspot_type(res_info)
                })

        return predictions

    def _calculate_hotspot_score(
        self,
        atom_array: AtomArray,
        res_info: Dict,
        sasa: np.ndarray
    ) -> float:
        """
        计算热点评分

        综合考虑:
        - 相对溶剂可及性
        - 残基类型
        - 接触数量
        - 能量贡献
        """
        # 实现细节...
        pass


class DeepHotspotPredictor:
    """
    深度学习热点预测模型

    基于图神经网络
    """

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def predict(
        self,
        atom_array: AtomArray,
        chain_id: str = None,
        interface_chain: str = None
    ) -> List[Dict]:
        """
        使用深度学习模型预测热点
        """
        # 构建图表示
        graph = self._build_protein_graph(atom_array)

        # 模型推理
        predictions = self.model(graph)

        # 后处理
        return self._post_process(predictions, atom_array)
```

---

## 5. 技术栈与依赖

### 5.1 后端技术栈

| 类别           | 技术       | 版本   | 用途           |
| -------------- | ---------- | ------ | -------------- |
| **Web框架**    | FastAPI    | 0.104+ | REST API服务   |
| **ASGI服务器** | Uvicorn    | 0.24+  | 异步HTTP服务器 |
| **任务队列**   | Celery     | 5.3+   | 异步任务处理   |
| **消息代理**   | Redis      | 7.0+   | 任务队列后端   |
| **数据库**     | PostgreSQL | 15+    | 关系数据库     |
| **ORM**        | SQLAlchemy | 2.0+   | 数据库ORM      |
| **对象存储**   | MinIO      | -      | 文件存储       |
| **深度学习**   | PyTorch    | 2.0+   | 模型推理       |
| **蛋白质处理** | Biotite    | 0.38+  | 结构处理       |
| **蛋白质处理** | AtomWorks  | -      | 结构处理       |
| **模型**       | RFD3       | -      | Binder生成     |
| **模型**       | MPNN       | -      | 序列设计       |
| **模型**       | RF3        | -      | 结构预测       |

### 5.2 前端技术栈

| 类别           | 技术       | 版本 | 用途         |
| -------------- | ---------- | ---- | ------------ |
| **框架**       | React      | 18+  | UI框架       |
| **语言**       | TypeScript | 5.0+ | 类型安全     |
| **构建工具**   | Vite       | 5.0+ | 构建打包     |
| **状态管理**   | Zustand    | 4.4+ | 状态管理     |
| **UI组件**     | Ant Design | 5.0+ | UI组件库     |
| **3D可视化**   | Mol\*      | 4.0+ | 蛋白质可视化 |
| **图表**       | ECharts    | 5.4+ | 数据可视化   |
| **HTTP客户端** | Axios      | 1.6+ | API调用      |

### 5.3 核心依赖关系

```python
# requirements.txt

# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0

# 任务队列
celery==5.3.4
redis==5.0.1
flower==2.0.1

# 数据库
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1

# 认证
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# 深度学习
torch==2.1.0
lightning==2.1.0

# 蛋白质处理
biotite==0.38.0
atomworks @ git+https://github.com/RosettaCommons/atomworks

# Foundry模型
rc-foundry[all] @ git+https://github.com/RosettaCommons/foundry

# 工具库
numpy==1.26.2
scipy==1.11.4
pandas==2.1.3
toolz==0.12.0

# 文件处理
python-magic==0.4.27
minio==7.2.0

# 日志和监控
loguru==0.7.2
prometheus-client==0.19.0

# 测试
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

---

## 6. 开发计划

### 6.1 项目阶段划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           项目开发路线图                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 0: 准备阶段 (1周)
├── 环境搭建
│   ├── 开发环境配置
│   ├── GPU服务器准备
│   └── 模型权重下载
├── 技术调研
│   ├── RFD3 Binder设计验证
│   ├── 热点预测模型选型
│   └── 前端可视化方案确认
└── 文档完善
    ├── 详细设计文档
    └── API接口文档

Phase 1: 核心引擎开发 (3周)
├── Week 1: RFD3 Binder引擎
│   ├── Binder设计规格封装
│   ├── 热点选择集成
│   └── 批量生成优化
├── Week 2: MPNN & RF3引擎
│   ├── MPNN序列设计封装
│   ├── RF3结构预测封装
│   └── 工作流编排
└── Week 3: 热点预测集成
    ├── 模型1集成
    ├── 模型2集成
    └── 预测结果融合

Phase 2: 后端服务开发 (2周)
├── Week 4: API服务
│   ├── FastAPI项目搭建
│   ├── 核心API端点
│   ├── 认证与权限
│   └── 文件服务
└── Week 5: 任务队列
    ├── Celery配置
    ├── Worker开发
    ├── 任务监控
    └── 错误处理

Phase 3: 前端开发 (3周)
├── Week 6: 基础框架
│   ├── React项目搭建
│   ├── 路由配置
│   ├── 状态管理
│   └── API集成
├── Week 7: 核心功能
│   ├── 文件上传组件
│   ├── 参数配置面板
│   ├── 热点选择器
│   └── 任务监控
└── Week 8: 可视化与结果
    ├── Mol*集成
    ├── 3D结构可视化
    ├── 结果展示
    └── 文件下载

Phase 4: 测试与优化 (2周)
├── Week 9: 测试
│   ├── 单元测试
│   ├── 集成测试
│   ├── 端到端测试
│   └── 性能测试
└── Week 10: 优化与部署
    ├── 性能优化
    ├── Docker部署
    ├── 文档完善
    └── 用户培训

Phase 5: 上线与维护 (持续)
├── 系统上线
├── 监控告警
├── 用户反馈
└── 迭代优化
```

### 6.2 详细任务分解

#### Phase 1: 核心引擎开发

| 任务ID | 任务名称            | 负责人 | 预计工时 | 依赖      | 交付物                 |
| ------ | ------------------- | ------ | -------- | --------- | ---------------------- |
| T1.1   | RFD3 Binder规格封装 | -      | 2天      | -         | rfd3_binder_engine.py  |
| T1.2   | 热点选择功能集成    | -      | 1天      | T1.1      | hotspot_integration.py |
| T1.3   | 批量生成优化        | -      | 1天      | T1.1      | batch_generation.py    |
| T1.4   | MPNN引擎封装        | -      | 2天      | -         | mpnn_engine.py         |
| T1.5   | RF3引擎封装         | -      | 2天      | -         | rf3_engine.py          |
| T1.6   | 工作流编排器        | -      | 2天      | T1.1-T1.5 | workflow_engine.py     |
| T1.7   | 热点模型1集成       | -      | 3天      | -         | hotspot_model1.py      |
| T1.8   | 热点模型2集成       | -      | 3天      | -         | hotspot_model2.py      |
| T1.9   | 预测结果融合        | -      | 2天      | T1.7-T1.8 | hotspot_fusion.py      |

#### Phase 2: 后端服务开发

| 任务ID | 任务名称          | 负责人 | 预计工时 | 依赖       | 交付物           |
| ------ | ----------------- | ------ | -------- | ---------- | ---------------- |
| T2.1   | FastAPI项目初始化 | -      | 0.5天    | -          | 项目骨架         |
| T2.2   | 数据库模型设计    | -      | 1天      | -          | models.py        |
| T2.3   | 设计API端点       | -      | 2天      | T2.2       | design.py        |
| T2.4   | 文件API端点       | -      | 1天      | T2.2       | files.py         |
| T2.5   | 任务API端点       | -      | 1天      | T2.2       | tasks.py         |
| T2.6   | 用户认证系统      | -      | 2天      | T2.2       | auth.py          |
| T2.7   | Celery配置        | -      | 1天      | -          | celery_config.py |
| T2.8   | Worker任务定义    | -      | 2天      | T2.7, T1.6 | tasks.py         |
| T2.9   | 任务监控集成      | -      | 1天      | T2.8       | monitoring.py    |

#### Phase 3: 前端开发

| 任务ID | 任务名称        | 负责人 | 预计工时 | 依赖 | 交付物              |
| ------ | --------------- | ------ | -------- | ---- | ------------------- |
| T3.1   | React项目初始化 | -      | 0.5天    | -    | 项目骨架            |
| T3.2   | 路由与布局      | -      | 1天      | T3.1 | 路由配置            |
| T3.3   | 状态管理配置    | -      | 1天      | T3.1 | store配置           |
| T3.4   | API服务封装     | -      | 1天      | T3.1 | api.ts              |
| T3.5   | 文件上传组件    | -      | 2天      | T3.4 | FileUpload.tsx      |
| T3.6   | 参数配置面板    | -      | 2天      | T3.5 | ParameterPanel.tsx  |
| T3.7   | 热点选择器      | -      | 3天      | T3.5 | HotspotSelector.tsx |
| T3.8   | Mol\*集成       | -      | 2天      | T3.5 | ProteinViewer.tsx   |
| T3.9   | 任务监控组件    | -      | 2天      | T3.4 | TaskMonitor.tsx     |
| T3.10  | 结果展示组件    | -      | 2天      | T3.8 | ResultDisplay.tsx   |

---

## 7. 资源需求

### 7.1 硬件资源

| 资源类型      | 规格             | 数量  | 用途               |
| ------------- | ---------------- | ----- | ------------------ |
| **GPU服务器** | NVIDIA A100 40GB | 1-2台 | 模型推理           |
| **CPU服务器** | 32核 / 64GB内存  | 1台   | Web服务            |
| **存储**      | SSD 2TB          | -     | 模型权重、结果存储 |
| **网络**      | 1Gbps            | -     | 内网通信           |

### 7.2 软件资源

| 资源类型     | 说明                                   |
| ------------ | -------------------------------------- |
| **模型权重** | RFD3 (~2GB), MPNN (~500MB), RF3 (~2GB) |
| **热点模型** | 待确定 (约1-2GB)                       |
| **数据库**   | PostgreSQL 15                          |
| **缓存**     | Redis 7.0                              |
| **容器**     | Docker, Docker Compose                 |

### 7.3 人力资源

| 角色           | 人数  | 技能要求                    | 参与阶段  |
| -------------- | ----- | --------------------------- | --------- |
| **后端工程师** | 1-2人 | Python, FastAPI, Celery     | Phase 1-2 |
| **前端工程师** | 1人   | React, TypeScript, 3D可视化 | Phase 3   |
| **算法工程师** | 1人   | 深度学习, 蛋白质设计        | Phase 1   |
| **测试工程师** | 0.5人 | 自动化测试                  | Phase 4   |
| **DevOps**     | 0.5人 | Docker, K8s, CI/CD          | Phase 4-5 |

---

## 8. 风险评估与应对

### 8.1 技术风险

| 风险ID | 风险描述                  | 可能性 | 影响 | 应对措施                     |
| ------ | ------------------------- | ------ | ---- | ---------------------------- |
| R1     | RFD3 Binder生成质量不稳定 | 中     | 高   | 增加生成数量，优化筛选策略   |
| R2     | 热点预测模型准确率不足    | 中     | 中   | 多模型融合，允许手动调整     |
| R3     | GPU资源不足导致任务排队   | 高     | 中   | 任务优先级调度，资源弹性扩展 |
| R4     | 大文件上传处理超时        | 低     | 中   | 分片上传，断点续传           |
| R5     | 3D可视化性能问题          | 中     | 低   | 结构简化显示，LOD技术        |

### 8.2 项目风险

| 风险ID | 风险描述         | 可能性 | 影响 | 应对措施             |
| ------ | ---------------- | ------ | ---- | -------------------- |
| R6     | 模型权重下载失败 | 低     | 高   | 提前下载，本地备份   |
| R7     | 开发进度延期     | 中     | 中   | 敏捷开发，分阶段交付 |
| R8     | 需求变更         | 中     | 中   | 模块化设计，快速响应 |

### 8.3 应对策略

```
风险应对流程:

┌─────────────┐
│ 风险识别    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 风险评估    │ ◀─── 可能性 × 影响 = 风险等级
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 制定策略    │ ◀─── 规避 / 转移 / 缓解 / 接受
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 实施监控    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 效果评估    │
└─────────────┘
```

---

## 9. 附录

### 9.1 术语表

| 术语      | 英文                            | 解释                       |
| --------- | ------------------------------- | -------------------------- |
| Binder    | Binder蛋白                      | 设计的结合蛋白             |
| Hotspot   | 热点残基                        | 界面中对结合能贡献大的残基 |
| PPI       | Protein-Protein Interaction     | 蛋白质-蛋白质相互作用      |
| RMSD      | Root Mean Square Deviation      | 均方根偏差                 |
| pLDDT     | predicted LDDT                  | 预测的局部距离差异测试分数 |
| PAE       | Predicted Aligned Error         | 预测对齐误差               |
| pTM       | predicted TM-score              | 预测的TM分数               |
| SASA      | Solvent Accessible Surface Area | 溶剂可及表面积             |
| Contig    | Contig string                   | 序列拼接字符串             |
| AtomArray | -                               | Biotite的原子数组数据结构  |

### 9.2 参考文档

1. **RFD3 Documentation**: [foundry-production/models/rfd3/docs](file:///c:/Users/86133/Downloads/foundry-production/foundry-production/models/rfd3/docs)
2. **MPNN Documentation**: [foundry-production/models/mpnn/docs](file:///c:/Users/86133/Downloads/foundry-production/foundry-production/models/mpnn/docs)
3. **RF3 Documentation**: [foundry-production/models/rf3/docs](file:///c:/Users/86133/Downloads/foundry-production/foundry-production/models/rf3/docs)
4. **Example Notebook**: [examples/all.ipynb](file:///c:/Users/86133/Downloads/foundry-production/foundry-production/examples/all.ipynb)

### 9.3 API接口清单

#### 设计相关API

| 方法 | 路径                                                 | 描述               |
| ---- | ---------------------------------------------------- | ------------------ |
| POST | `/api/v1/design/binder`                              | 提交Binder设计任务 |
| GET  | `/api/v1/design/task/{task_id}`                      | 查询任务状态       |
| GET  | `/api/v1/design/task/{task_id}/results`              | 获取设计结果       |
| GET  | `/api/v1/design/task/{task_id}/download/{design_id}` | 下载设计文件       |

#### 热点预测API

| 方法 | 路径                        | 描述           |
| ---- | --------------------------- | -------------- |
| POST | `/api/v1/hotspot/predict`   | 预测热点残基   |
| POST | `/api/v1/hotspot/visualize` | 可视化热点残基 |

#### 文件相关API

| 方法   | 路径                      | 描述         |
| ------ | ------------------------- | ------------ |
| POST   | `/api/v1/files/upload`    | 上传结构文件 |
| GET    | `/api/v1/files/{file_id}` | 获取文件信息 |
| DELETE | `/api/v1/files/{file_id}` | 删除文件     |

#### 用户相关API

| 方法 | 路径                     | 描述             |
| ---- | ------------------------ | ---------------- |
| POST | `/api/v1/users/register` | 用户注册         |
| POST | `/api/v1/users/login`    | 用户登录         |
| GET  | `/api/v1/users/me`       | 获取当前用户信息 |

### 9.4 数据库表设计

#### design_tasks 表

```sql
CREATE TABLE design_tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(64) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),

    -- 输入参数
    target_file VARCHAR(512) NOT NULL,
    target_chain VARCHAR(10),
    target_residues VARCHAR(100),
    binder_length_range VARCHAR(50),
    hotspots JSONB,
    num_designs INTEGER DEFAULT 10,
    sequences_per_design INTEGER DEFAULT 5,

    -- 状态
    status VARCHAR(20) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    current_step VARCHAR(50),
    error_message TEXT,

    -- 时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- 索引
    INDEX idx_task_id (task_id),
    INDEX idx_user_id (user_id),
    INDEX idx_status (status)
);
```

#### design_results 表

```sql
CREATE TABLE design_results (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES design_tasks(id),
    design_index INTEGER NOT NULL,

    -- 序列信息
    sequence TEXT NOT NULL,

    -- 评估指标
    rmsd FLOAT,
    plddt FLOAT,
    pae FLOAT,
    ptm FLOAT,
    ranking_score FLOAT,

    -- 文件路径
    structure_file VARCHAR(512),
    trajectory_file VARCHAR(512),

    -- 元数据
    metadata JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_task_id (task_id),
    INDEX idx_rmsd (rmsd),
    INDEX idx_plddt (plddt)
);
```

### 9.5 配置文件示例

#### 后端配置 (config.py)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "ProteinBinder Designer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # 数据库配置
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/binder_db"

    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"

    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "binder-designs"

    # 模型配置
    RFD3_CHECKPOINT: str = "rfd3"
    MPNN_CHECKPOINT: str = "ligandmpnn"
    RF3_CHECKPOINT: str = "rf3"
    HOTSPOT_MODEL1_PATH: str = "/models/hotspot_model1"
    HOTSPOT_MODEL2_PATH: str = "/models/hotspot_model2"

    # GPU配置
    GPU_DEVICE: str = "cuda:0"
    MAX_GPU_MEMORY: int = 32  # GB

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 4
    TASK_TIMEOUT: int = 3600  # seconds

    # JWT配置
    JWT_SECRET_KEY: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 86400  # seconds

    class Config:
        env_file = ".env"

settings = Settings()
```

#### Celery配置 (celery_config.py)

```python
from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "binder_design",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours
    task_soft_time_limit=6600,  # 1.8 hours
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
)

# 定时任务
celery_app.conf.beat_schedule = {
    "cleanup-expired-tasks": {
        "task": "tasks.cleanup_expired_tasks",
        "schedule": crontab(hour=2, minute=0),  # 每天凌晨2点
    },
    "monitor-gpu-usage": {
        "task": "tasks.monitor_gpu_usage",
        "schedule": 300.0,  # 每5分钟
    },
}
```

---

## 📝 文档修订历史

| 版本 | 日期       | 修订人 | 修订内容                   |
| ---- | ---------- | ------ | -------------------------- |
| v1.0 | 2025-04-04 | -      | 初始版本，完成整体架构设计 |

---

**文档结束**

_本文档为预备工作文档，后续将根据实际开发情况进行更新和完善。_
