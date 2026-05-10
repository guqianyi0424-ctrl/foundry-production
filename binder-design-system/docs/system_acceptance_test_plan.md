# DeepBinder 系统验收测试补全计划

## 目标

补全系统验收测试，覆盖上传 PDB、实验记录、登录/权限、前端页面流程，以及一条 mock pipeline 端到端链路。测试结果同步整理为毕设可引用的测试用例表。

## 范围

1. 后端验收测试
   - 上传 PDB：验证链信息、序列、原始 PDB 内容返回。
   - 登录/权限：验证注册、登录、当前用户、研究人员受限、管理员放行。
   - 实验记录：验证创建、列表、更新、添加设计、详情、导出、删除。
   - mock pipeline：验证 RFD3、ProteinMPNN、RF3 三段结果贯通。

2. 前端验收测试
   - 匿名状态登录弹窗。
   - 研究人员和管理员侧边栏权限差异。
   - De Novo Design 页面 RFD3 -> MPNN -> RF3 mock 流程。
   - 实验记录页面列表加载。
   - 帮助页中用户角色和 ProteinMPNN 可视化边界说明。

3. 毕设文档
   - 输出测试用例表，字段包含：输入、步骤、预期结果、实际结果、是否通过、自动化用例。

## 实施方式

后端不依赖真实 GPU 和 Foundry 任务，使用内存 SQLite 与 fake design service，使验收测试能够稳定运行。后端验收测试通过 `httpx.ASGITransport` 请求真实 FastAPI 路由，覆盖 multipart 上传、认证 token、权限依赖、实验记录接口和 mock pipeline 接口。

前端使用 Vitest、jsdom 和 Testing Library，mock API 层与结构查看器，重点验证页面流程和权限展示。

## 验证命令

```bash
cd binder-design-system/backend
PYTHONPATH="$PWD/../..:$PWD/..:$PWD" python3.12 -m pytest tests/test_system_acceptance.py -v
```

```bash
cd binder-design-system/frontend
npm test -- --run
npm run build
```
