# 蛋白质 Binder 设计系统 - 开发指南

## 📋 项目概述

本项目是一个基于深度学习的**蛋白质Binder自动化设计与验证系统**，整合了多个AI模型实现从目标蛋白结构到最终Binder序列的完整流程。

### 核心功能模块

| 模块 | 功能描述 | 技术栈 |
|------|---------|--------|
| **热点预测** | 识别蛋白表面的关键结合位点 | ML/DL模型 |
| **RFD3骨架生成** | 基于扩散模型生成Binder主链结构 | RFdiffusion3 |
| **MPNN序列设计** | 为骨架设计最优氨基酸序列 | ProteinMPNN |
| **RF3结构验证** | 预测并验证设计结构的合理性 | RoseTTAFold3 |
| **3D可视化** | 交互式展示蛋白结构与设计结果 | Mol* |

---

## 🏗️ 项目架构

```
binder-design-system/
├── frontend/                    # React前端 (Vite + TypeScript)
│   ├── src/
│   │   ├── api/                # API调用层
│   │   │   └── index.ts        # Axios封装，定义所有后端接口
│   │   ├── components/         # UI组件
│   │   │   ├── DesignPanel/    # RFD3参数配置面板 (已修复✅)
│   │   │   ├── MolstarViewer/  # 3D分子可视化器
│   │   │   ├── SequenceViewer/ # 序列查看器与热点选择
│   │   │   ├── ResultTabs/     # 结果展示标签页
│   │   │   │   ├── RFD3Panel/  # RFD3骨架结果
│   │   │   │   ├── MPNNPanel/  # MPNN序列结果
│   │   │   │   └── RF3Panel/   # RF3验证结果
│   │   │   └── Sidebar/        # 导航侧边栏
│   │   ├── pages/              # 页面组件
│   │   │   ├── NewDesignPage.tsx   # 主设计页面
│   │   │   ├── JobCenterPage.tsx   # 作业中心
│   │   │   └── HelpPage.tsx        # 帮助页面
│   │   ├── store/
│   │   │   └── useAppStore.ts     # Zustand全局状态管理
│   │   ├── types/
│   │   │   └── index.ts           # TypeScript类型定义
│   │   ├── utils/
│   │   │   └── pdbParser.ts       # PDB文件解析工具
│   │   ├── App.tsx               # 应用入口
│   │   └── main.tsx              # React根组件
│   ├── package.json              # 前端依赖配置
│   └── vite.config.ts            # Vite构建配置
│
├── backend/                     # Python后端 (FastAPI)
│   ├── main.py                  # FastAPI应用主入口
│   ├── requirements.txt         # Python依赖
│   └── routers/                 # API路由
│       ├── design.py            # 设计流程核心API
│       ├── upload.py            # 文件上传处理
│       └── jobs.py              # 作业管理
│
├── utils/                       # 核心算法模块
│   ├── conda_bridge.py          # Conda环境桥接（跨环境调用foundry）
│   ├── hotspot_predictor.py     # 热点残基预测器
│   ├── rfd3_runner.py           # RFdiffusion3运行器 ⭐核心
│   ├── mpnn_runner.py           # ProteinMPNN运行器 ⭐核心
│   ├── rf3_runner.py            # RoseTTAFold3运行器 ⭐核心
│   ├── structure_parser.py      # 结构文件解析
│   ├── molstar_viewer.py        # Molstar可视化服务
│   ├── mpnn_visualizer.py       # MPNN结果可视化
│   └── rf3_visualizer.py        # RF3结果可视化
│
├── config/
│   └── settings.py              # 系统配置
│
├── data/                        # 数据目录
├── outputs/                     # 输出结果目录
│   ├── rfd3/                    # RFD3输出
│   ├── mpnn/                    # MPNN输出
│   └── rf3/                     # RF3输出
│
├── docker-compose.yml           # Docker编排配置
├── Dockerfile                   # Docker镜像构建
└── README.md                    # 项目说明
```

---

## 🔧 技术栈详情

### 前端技术栈
- **框架**: React 18.3 + TypeScript 5.6
- **构建工具**: Vite 6.0
- **状态管理**: Zustand 5.0
- **UI样式**: TailwindCSS 3.4
- **HTTP客户端**: Axios 1.7
- **图标库**: Lucide React 0.468
- **3D可视化**: Mol* 4.4 (专业分子可视化库)

### 后端技术栈
- **框架**: FastAPI (高性能异步Web框架)
- **语言**: Python 3.10+
- **深度学习**: PyTorch + rc-foundry (RFD3/MPNN/RF3)
- **数据处理**: NumPy, Biopython

### 关键依赖版本
```json
{
  "react": "^18.3.1",
  "zustand": "^5.0.2",
  "molstar": "^4.4.0",
  "axios": "^1.7.9"
}
```

---

## 🔄 核心工作流程

### 完整设计流水线 (Pipeline)

```
用户上传PDB → 解析结构 → 选择热点 → 运行Pipeline → 展示结果
```

#### 步骤详解：

**① 目标结构输入**
- 支持上传 `.pdb`, `.cif`, `.ent` 格式文件（最大200MB）
- 自动解析链信息、残基序列、残基编号
- 使用 `parseStructureFile()` 进行前端解析或后端解析备选

**② 热点残基选择**
- **自动预测**: 调用 `/api/predict-hotspot` 接口，基于DL模型预测top-3热点
- **手动指定**: 用户通过序列查看器点击选择或手动输入格式 `A/1,A/2,B/5`
- 热点数据存储在Zustand store: `selectedHotspots: HotspotResidue[]`

**③ RFD3骨架生成** ([rfd3_runner.py](utils/rfd3_runner.py))
```python
# 核心调用逻辑
rfd3_runner = RFD3Runner()
result = await rfd3_runner.run(
    target_pdb=pdb_content,
    hotspot_residues=["A/25", "A/28"],
    binder_length=80,          # Binder长度
    num_designs=3             # 生成数量
)
```
- 生成YAML配置文件，定义contig和热点约束
- 跨Conda环境调用: `binder`环境(Python3.10) → `foundry`环境(Python3.12)
- 输出: PDB/CIF格式的骨架结构文件
- **Mock模式**: 当foundry不可用时自动降级为模拟数据

**④ MPNN序列设计** ([mpnn_runner.py](utils/mpnn_runner.py))
```python
mpnn_runner = MPNNRunner()
seq_result = await mpnn_runner.run(
    backbone_pdb=design.pdb_path,
    num_sequences=1,
    sampling_temp=0.1
)
```
- 使用 `ligand_mpnn` 模型类型
- 基于骨架坐标生成氨基酸序列
- 按score排序，返回最优序列

**⑤ RF3结构验证** ([rf3_runner.py](utils/rf3_runner.py))
```python
rf3_runner = RF3Runner()
validation = await rf3_runner.validate_design(
    backbone_pdb=rfd3_output.pdb_path,
    sequence=mpnn_output.sequence,
    rmsd_threshold=2.0  # RMSD阈值(Å)
)
```
- 预测Binder的3D结构
- 计算与原始骨架的RMSD偏差
- 提供pLDDT置信度分数和PAE预测误差
- 判断是否通过验证 (`passed: bool`)

---

## 📊 数据流架构

### 状态管理 (Zustand Store)

**Store位置**: [useAppStore.ts](frontend/src/store/useAppStore.ts)

**核心状态字段**:
```typescript
interface AppState {
  // 文件与结构
  targetFile: File | null;
  pdbContent: string;
  chains: ChainInfo[];

  // 选择区域
  selectedRange: ResidueRange | null;
  selectedHotspots: HotspotResidue[];

  // RFD3配置
  rfd3Config: {
    targetEntityType: string;    // "蛋白"
    targetStructure: string;     // "A/6-36"
    hotspots: string;            // "A/1, A/2, A/3"
    conditionAtoms: string;      // JSON格式条件原子
    lengthMin: number;           // 40
    lengthMax: number;           // 120
    nBatches: number;            // 10
    diffusionBatchSize: number;  // 16
  };

  // 结果数据
  rfd3Results: RFD3Design[] | null;
  mpnnResults: MPNNResult[] | null;
  rf3Results: RF3Result[] | null;

  // 运行状态
  isRunning: boolean;
  jobHistory: Job[];
}
```

### API接口定义

**位置**: [api/index.ts](frontend/src/api/index.ts)

| 接口 | 方法 | 路径 | 功能 |
|------|------|------|------|
| `uploadPdb` | POST | `/api/upload` | 上传PDB文件并解析 |
| `predictHotspot` | POST | `/api/predict-hotspot` | 预测热点残基 |
| `runPipeline` | POST | `/api/run-pipeline` | 执行完整设计流水线 |

**Pipeline响应结构**:
```typescript
{
  job_id: string;
  status: 'completed' | 'failed';
  rfd3_results: {
    success: boolean;
    designs: Array<{
      index: number;
      rank: number;
      plddt: number;        // 平均pLDDT分数
      pdb_path: string;
      mock?: boolean;
    }>;
  };
  mpnn_results: Array<{
    design_idx: number;
    sequence: string;
    score: number;          // MPNN能量分数
    seq_idx: number;
  }>;
  rf3_results: Array<{
    design_idx: number;
    sequence: string;
    rmsd: number;           // RMSD偏差(Å)
    avg_plddt: number;      // 平均置信度
    passed: boolean;        // 是否通过验证
    per_res_rmsd: number[]; // 逐残基RMSD
  }>;
}
```

---

## 🎨 UI组件说明

### 主要页面

**1. 新建设计页** ([NewDesignPage.tsx](frontend/src/pages/NewDesignPage.tsx))
- 文件上传区域（支持拖拽）
- 任务类型选择（当前仅支持"蛋白"）
- 结构与序列查看区（双栏布局：SequenceViewer + MolstarViewer）
- RFD3设计参数配置面板
- 结果展示区（三标签页：RFD3/MPNN/RF3）

**2. DesignPanel组件** ([DesignPanel/index.tsx](frontend/src/components/DesignPanel/index.tsx))
- **已修复问题**: 将 `QuestionMarkCircle` 图标替换为 `HelpCircle` ✅
- 配置项：
  - 目标实体类型（蛋白）
  - 目标结构域（必填，如 `A/6-36`）
  - 热点残基（可选）
  - 条件原子（JSON格式，可选）
  - 设计长度范围（默认40-120）
  - 生成参数：n_batches × diffusion_batch_size

**3. 结果展示组件**

- **RFD3Panel**: 显示生成的骨架结构列表，包含pLDDT评分和排名
- **MPNNPanel**: 显示设计的氨基酸序列及MPNN score
- **RF3Panel**: 显示验证结果，包括RMSD、pLDDT、PAE热图等指标

---

## 🔌 后端API详细说明

### FastAPI路由结构

**主入口**: [backend/main.py](backend/main.py)

```python
app = FastAPI(title="ODesign API")
app.include_router(upload.router, prefix="/api")    # 上传相关
app.include_router(design.router, prefix="/api")    # 设计流程
app.include_router(jobs.router, prefix="/api")      # 作业管理
```

### 核心设计接口

**位置**: [backend/routers/design.py](backend/routers/design.py)

#### POST `/api/predict-hotspot`

**请求体**:
```json
{
  "pdb_content": "string (PDB文件内容)"
}
```

**响应**:
```json
{
  "hotspots": [
    {"chain": "A", "residue": 25, "score": 0.95},
    {"chain": "A", "residue": 28, "score": 0.87},
    {"chain": "A", "residue": 32, "score": 0.76}
  ]
}
```

#### POST `/api/run-pipeline`

**请求体**:
```json
{
  "pdb_content": "string",
  "hotspots": [
    {"chain": "A", "residue": 25},
    {"chain": "A", "residue": 28}
  ],
  "binder_length": 80
}
```

**处理流程**:
1. 调用 `RFD3Runner.run()` 生成骨架
2. 对每个骨架调用 `MPNNRunner.run()` 生成序列
3. 对每个序列调用 `RF3Runner.validate_design()` 验证结构
4. 汇总所有结果返回

---

## 🛠️ 开发环境搭建

### 前置要求

- Node.js >= 18.x
- Python >= 3.10
- Conda (用于管理foundry环境)
- Docker (可选，用于容器化部署)

### 安装步骤

#### 1. 克隆项目
```bash
cd foundry-production/binder-design-system
```

#### 2. 安装前端依赖
```bash
cd frontend
npm install
```

#### 3. 安装Python依赖
```bash
cd ../backend
pip install -r requirements.txt
```

#### 4. 配置Foundry环境（可选，用于真实模型推理）

```bash
# 创建foundry conda环境
conda create -n foundry python=3.12
conda activate foundry

# 安装rc-foundry
pip install rc-foundry

# 下载模型权重
foundry install rfd3 ligandmpnn rf3 --checkpoint-dir ./checkpoints
```

### 启动开发服务器

#### 方式一：分别启动前后端

**终端1 - 后端**:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**终端2 - 前端**:
```bash
cd frontend
npm run dev
```

访问: http://localhost:5173

#### 方式二：使用Docker Compose
```bash
docker-compose up --build
```

访问: http://localhost:8000

---

## 🧪 核心算法模块详解

### 1. RFD3 Runner ([utils/rfd3_runner.py](utils/rfd3_runner.py))

**功能**: 基于RFdiffusion3生成Binder主链骨架

**关键方法**:

```python
class RFD3Runner:
    def run(self, target_pdb, hotspot_residues, binder_length=60, num_designs=3):
        """
        主运行方法
        - 生成YAML配置
        - 调用foundry CLI执行RFD3
        - 收集并排序结果
        """
```

**Contig格式示例**:
```
80,/0,A1-999  # 生成80残基binder，绑定到A链1-999区域
```

**热点约束格式**:
```yaml
select_hotspots:
  A/25: ALL
  A/28: ALL
infer_ori_strategy: hotspots
```

**输出**: PDB文件数组，按pLDDT降序排列

### 2. MPNN Runner ([utils/mpnn_runner.py](utils/mpnn_runner.py))

**功能**: 基于ProteinMPNN为骨架设计序列

**特点**:
- 使用 `ligand_mpnn` 模型类型（针对配体/结合位点优化）
- 支持自定义采样温度（默认0.1，越低越确定性）
- 按能量分数排序返回最优序列

**输出**: FASTA格式的氨基酸序列

### 3. RF3 Runner ([utils/rf3_runner.py](utils/rf3_runner.py))

**功能**: 使用RoseTTAFold3进行结构预测和验证

**验证指标**:
- **pLDDT** (predicted LDDT): 预测的局部距离差异测试分数（0-100，越高越好）
- **PAE** (Predicted Aligned Error): 预测对齐误差矩阵
- **RMSD**: 与原始骨架的均方根偏差（Å），阈值默认2.0Å

**RMSD计算**:
```python
@staticmethod
def calculate_rmsd(pdb1_path, pdb2_path):
    # 提取CA原子坐标
    # Kabsch算法最优超对齐
    # 计算RMSD
```

### 4. Conda Bridge ([utils/conda_bridge.py](utils/conda_bridge.py))

**功能**: 解决跨Python环境调用问题

**原理**:
- 主应用运行在 `binder` 环境 (Python 3.10)
- Foundry模型需要 `foundry` 环境 (Python 3.12)
- 通过subprocess调用不同conda环境的CLI工具

---

## 📝 TypeScript类型系统

**位置**: [types/index.ts](frontend/src/types/index.ts)

```typescript
// 链信息
interface ChainInfo {
  chain_id: string;
  sequence: string;
  length: number;
  resSeqs: number[];
}

// 热点残基
interface HotspotResidue {
  chain: string;
  residue: number;
  score: number;
}

// RFD3设计结果
interface RFD3Design {
  index: number;
  rank: number;
  plddt: number;
  pdb_path?: string;
  pdb_content?: string;
}

// MPNN序列结果
interface MPNNResult {
  design_idx: number;
  sequence: string;
  score: number;
  seq_idx: number;
}

// RF3验证结果
interface RF3Result {
  design_idx: number;
  sequence: string;
  rmsd: number;
  avg_plddt: number;
  passed: boolean;
  plddt?: number[];
  pae?: number[][];
  per_res_rmsd?: number[];
  mock?: boolean;
}

// 设计作业
interface DesignJob {
  id: string;
  target_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  rfd3_results?: {...};
  mpnn_results?: MPNNResult[];
  rf3_results?: RF3Result[];
}
```

---

## 🚨 已知问题与解决方案

### 问题1: Lucide图标导入错误 ✅ **已修复**

**错误信息**:
```
error TS2305: Module '"lucide-react"' has no exported member 'QuestionMarkCircle'.
```

**原因**: lucide-react v0.468+ 版本中 `QuestionMarkCircle` 图标被重命名

**解决方案**:
```tsx
// 修改前
import { QuestionMarkCircle } from 'lucide-react'

// 修改后
import { HelpCircle } from 'lucide-react'
```

**修改文件**: [DesignPanel/index.tsx](frontend/src/components/DesignPanel/index.tsx) 第2行和第34行

### 问题2: Foundry环境依赖

**现象**: Pipeline运行时自动降级到Mock模式

**原因**:
- 未安装rc-foundry包
- 未下载模型权重文件
- Conda环境未正确配置

**解决步骤**:
```bash
# 1. 检查foundry可用性
python -c "from utils.conda_bridge import is_foundry_available; print(is_foundry_available())"

# 2. 如果返回False，执行：
conda create -n foundry python=3.12
conda activate foundry
pip install rc-foundry
foundry install rfd3 ligandmpnn rf3
```

### 问题3: PowerShell命令兼容性

**现象**: Windows下 `&&` 连接符报错

**解决方案**: 使用分号 `;` 替代 `&&`

---

## 🔮 未来开发方向

### 高优先级
- [ ] **JSON模式支持**: 当前UI显示"JSON模式"按钮但未实现
- [ ] **批量作业队列**: 支持提交多个设计任务排队执行
- [ ] **实时进度反馈**: WebSocket推送各阶段进度
- [ ] **结果导出功能**: 导出PDB、FASTA、CSV报告

### 中优先级
- [ ] **历史记录持久化**: 作业历史保存到数据库/本地存储
- [ ] **参数模板保存**: 保存常用配置为模板
- [ ] **对比分析工具**: 多个设计方案的并行对比视图
- [ ] **热图可视化增强**: PAE/pLDDT交互式热图

### 低优先级
- [ ] **用户认证系统**: 多用户支持和权限管理
- [ ] **云端部署优化**: K8s集群部署方案
- [ ] **GPU调度优化**: 多GPU并行推理加速
- [ ] **API限流与监控**: 生产环境稳定性保障

---

## 📚 关键代码索引

### 前端入口
- [App.tsx](frontend/src/App.tsx) - 应用根组件和路由
- [main.tsx](frontend/src/main.tsx) - React DOM渲染入口
- [useAppStore.ts](frontend/src/store/useAppStore.ts) - 全局状态管理（165行）

### 核心业务逻辑
- [NewDesignPage.tsx](frontend/src/pages/NewDesignPage.tsx) - 主设计页面（256行）
- [DesignPanel/index.tsx](frontend/src/components/DesignPanel/index.tsx) - 参数配置面板（101行）✅已修复
- [api/index.ts](frontend/src/api/index.ts) - API层封装（54行）

### 后端核心
- [main.py](backend/main.py) - FastAPI应用（32行）
- [design.py](backend/routers/design.py) - 设计流水线（95行）

### AI模型集成
- [rfd3_runner.py](utils/rfd3_runner.py) - RFD3调用（290行）
- [mpnn_runner.py](utils/mpnn_runner.py) - MPNN调用（179行）
- [rf3_runner.py](utils/rf3_runner.py) - RF3调用（278行）
- [conda_bridge.py](utils/conda_bridge.py) - 环境桥接

---

## 💡 开发最佳实践

### 1. 状态管理规范
- ✅ 使用Zustand的selector模式避免不必要的重渲染
- ✅ 相关状态更新放在同一个setter中（如 `setSelectedRange` 同时清空hotspots）
- ❌ 避免在组件内部直接修改store状态

### 2. 错误处理策略
- 前端: try-catch包裹异步操作，alert提示用户
- 后端: HTTPException返回详细错误信息
- 模型Runner: Mock降级机制确保流程不中断

### 3. 性能优化建议
- Mol* Viewer使用虚拟化渲染大分子
- API请求设置合理timeout（当前300秒）
- 考虑使用React.memo优化结果列表渲染

### 4. 代码风格
- 组件使用函数式组件 + Hooks
- TypeScript严格模式开启
- TailwindCSS原子类优先
- 中文注释用于业务逻辑，英文用于技术术语

---

## 📞 技术支持

### 常见问题排查清单

1. **TypeScript编译错误**: 运行 `npm run build` 或 `npx tsc --noEmit`
2. **API连接失败**: 检查后端是否启动在8000端口
3. **模型推理失败**: 确认foundry环境和模型权重
4. **文件上传失败**: 检查文件大小不超过200MB
5. **Mol*加载异常**: 检查浏览器控制台是否有CORS错误

### 日志查看

- **前端日志**: 浏览器F12开发者工具Console
- **后端日志**: 终端输出或uvicorn日志文件
- **模型日志**: foundry CLI的标准输出和错误流

---

## 📄 更新日志

### v1.0.1 (2026-04-20)
- ✅ 修复DesignPanel组件lucide-react图标导入错误
- ✅ 补充完整开发文档

### v1.0.0 (初始版本)
- 完成基础架构搭建
- 实现RFD3/MPNN/RF3三阶段流水线
- 前后端分离架构
- Mol* 3D可视化集成

---

**文档维护者**: AI Assistant
**最后更新**: 2026-04-20
**适用版本**: v1.0.1+
