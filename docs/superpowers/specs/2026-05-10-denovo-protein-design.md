# De Novo Protein Design 方案 A 设计

## 背景

当前 `binder-design-system` 的“新建设计”页面实际是 binder design 流程：上传靶点结构，选择热点，运行 RFD3 生成 binder 骨架，再用 MPNN 设计序列，最后用 RF3 验证结构。`foundry-production/examples/all.ipynb` 展示的是另一条 de novo 流程：不输入靶点结构，RFD3 通过长度条件直接生成新蛋白骨架，然后 MPNN 设计序列，RF3 预测并验证设计序列是否能折叠回目标骨架。

本次选择方案 A，目标是增加一个独立的 `De Novo Protein Design` 入口和页面，复用现有 RFD3、MPNN、RF3 后端能力，避免大规模重构现有 binder design 页面。

## 用户角色

研究人员：
- 登录后可以使用 Binder Design 和 De Novo Protein Design。
- 可以配置 de novo 生成参数，运行 RFD3、MPNN、RF3。
- 可以查看自己的实验结果、结构预览、序列、评分和验证指标。
- 可以导出或复制 PDB、FASTA、实验参数和结果摘要。

管理员：
- 具备研究人员全部能力。
- 可以访问系统监控、全部实验记录和用户列表。
- 可以删除异常实验记录，查看任务失败原因和 mock/fallback 状态。
- 可以后续扩展为管理模型路径、GPU 状态和队列策略。

权限策略：
- 前端侧边栏根据 `user.role` 展示管理员入口。`系统监控` 和后续 `用户管理` 面向管理员。
- 后端现有 `require_role(["admin"])` 用于管理员接口。De Novo 运行接口先允许已登录研究人员和管理员调用。

## 功能范围

新增侧边栏入口：
- `De Novo Design`，位于 `新建设计` 下方。
- `新建设计` 保持现有 binder design 语义，不改名以降低改动风险。

新增 De Novo 页面：
- 第一屏直接展示可运行的 de novo 工作台，不做营销页。
- 参数区包含：
  - protein length，默认 80，范围 40 到 200。
  - diffusion batch size，默认 2，范围 1 到 10。
  - n batches，默认 1，范围 1 到 10。
  - MPNN batch size，默认 10。
  - 是否自动运行 RF3 验证，默认关闭，避免一次点击触发过长链路。
- 流程区包含三个阶段：
  - RFD3 backbone generation。
  - MPNN sequence design。
  - RF3 structure validation。
- 结果区包含：
  - RFD3 生成骨架列表和 3D 预览。
  - MPNN 生成序列列表、序列长度、score。
  - RF3 pLDDT、RMSD、ranking score、是否通过验证。
  - RFD3 与 RF3 结构叠合视图。

## ProteinMPNN 可视化规则

ProteinMPNN 的职责是基于输入骨架生成氨基酸序列，不是进行结构预测。因此前端不能把 MPNN 结果表述为“预测结构”。

可视化处理规则：
- 如果真实 MPNN 输出包含 `pdb_content`，可以显示结构预览。该结构表示“输入骨架上替换为设计序列后的结构文件”，用于检查骨架和序列映射。
- 如果 MPNN 输出只有序列，没有 `pdb_content`，只展示序列、长度和 score，不显示结构查看器。
- 前端文案使用“序列映射结构”或“设计序列对应骨架”，不使用“MPNN 预测结构”。
- 最终折叠验证和结构可信度由 RF3 结果负责。

## Molstar 日志处理

当前页面多个位置通过 iframe `srcDoc` 直接加载 Molstar。实现时新增一个统一的静默结构预览组件，用于 RFD3、MPNN 和 RF3 结构预览。

要求：
- iframe 内部在加载 Molstar 前覆盖 `console.log`、`console.warn`、`console.info`、`console.debug`。
- iframe 内部捕获加载错误，只向父页面展示简短错误状态，不把 Molstar 原始日志直接显示到前端。
- 前端卡片只显示结构预览、加载状态和必要错误提示。
- 后端日志仍保留在服务端控制台或日志文件，便于排错。

## 后端接口

优先复用现有 `/run-rfd3`：
- De Novo 页面调用时不传 `pdb_content`、`target`、`hotspots`。
- 后端 `RFD3Runner` 已经在无 `pdb_content` 和 `target` 时构造 `specification={"length": binder_length, "extra": {}}`，与 `all.ipynb` 的 unconditional generation 对齐。

不新增轻量别名接口：
- 方案 A 直接复用 `/run-rfd3`，减少后端接口数量和测试范围。
- 前端 `runDeNovoRFD3` 可以只是 `runRFD3` 的语义包装，发送 `binder_length=length`，并且不发送 `pdb_content`、`target`、`hotspots`。

MPNN 与 RF3：
- De Novo 页面继续调用 `/run-mpnn`，输入 RFD3 选中骨架的 `pdb_content`。
- De Novo 页面继续调用 `/run-rf3`，输入 MPNN 选中序列的 `pdb_content`，并传 RFD3 原始骨架用于 RMSD 对比。

## 前端结构

新增文件：
- `frontend/src/pages/DeNovoDesignPage.tsx`
- `frontend/src/components/SilentStructureViewer/index.tsx`

修改文件：
- `frontend/src/App.tsx`：增加 `De Novo Design` 页面渲染。
- `frontend/src/components/Sidebar/index.tsx`：增加侧边栏入口。
- `frontend/src/api/index.ts`：新增 `runDeNovoRFD3` 前端包装函数，内部调用 `/run-rfd3`。

状态管理：
- De Novo 页面结果先使用页面局部 state，避免全局 store 膨胀。
- 后续若要加入后台任务恢复，再把 de novo 运行状态提升到 `useAppStore` 或后端 job store。

页面布局：
- 顶部为参数与运行按钮。
- 中部为 RFD3、MPNN、RF3 三段式工作流。
- 右侧或下方显示结构预览，不嵌套过多卡片。
- 移动端使用单列布局，避免按钮和序列文本溢出。

## 数据流

1. 研究人员进入 `De Novo Design`。
2. 设置 length、batch size、n batches。
3. 点击运行 RFD3。
4. 页面展示 backbone 列表，选中一个 backbone 后可送入 MPNN。
5. MPNN 返回序列列表。若返回 `pdb_content`，可显示“序列映射结构”；否则只显示序列。
6. 研究人员选择一个序列，点击 RF3 验证。
7. RF3 返回预测结构、pLDDT、RMSD、ranking score 和通过状态。
8. 页面展示验证结果和 RFD3/RF3 结构叠合。

## 缺失需求补齐

必须实现：
- 独立 De Novo 侧边栏入口。
- 独立 De Novo 页面。
- De Novo RFD3 参数表单。
- RFD3 结构列表和静默 3D 预览。
- MPNN 序列列表，不误导为结构预测。
- RF3 验证结果和结构对比。
- mock 模式提示。
- 研究人员和管理员角色说明在帮助页或页面说明中体现。

暂不实现：
- 长时间异步队列和后台任务恢复。
- 多用户配额、审批、GPU 排队。
- 复杂项目级权限模型。
- 新数据库表迁移。
- 完整用户管理页面。

## 测试计划

后端：
- 测试 de novo RFD3 请求可以在无 `pdb_content` 时调用 adapter。
- 测试 `/run-rfd3` 兼容旧 binder 请求。
- 测试 MPNN mock 结果无 `pdb_content` 时 API contract 不变。

前端：
- Sidebar 出现 `De Novo Design`。
- App 能渲染 `DeNovoDesignPage`。
- De Novo 页面无上传 PDB 也能发起 RFD3 请求。
- MPNN 无 `pdb_content` 时不显示结构预览。
- SilentStructureViewer 不把 Molstar 日志渲染到页面。

手工验证：
- `npm run build` 或前端类型检查通过。
- 后端 pytest contract 测试通过。
- 本地启动后可以从侧边栏进入 De Novo 页面。

## 验收标准

- 用户可以从侧边栏进入 `De Novo Design`。
- 不上传靶点结构也能运行 RFD3 de novo 生成。
- RFD3 mol 日志不在前端页面显示。
- ProteinMPNN 结果展示准确：序列为主，有 PDB 时才可视化为序列映射骨架。
- 研究人员和管理员角色边界在界面或帮助文档中清晰说明。
- 不破坏现有 binder design 页面。
