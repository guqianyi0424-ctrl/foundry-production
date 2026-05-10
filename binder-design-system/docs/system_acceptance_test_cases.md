# DeepBinder 系统验收测试用例表

| 编号 | 测试项 | 输入 | 步骤 | 预期结果 | 实际结果 | 是否通过 | 自动化用例 |
|---|---|---|---|---|---|---|---|
| SAT-001 | 上传 PDB 接口 | `mini.pdb`，包含 A 链 ALA/GLY 两个残基 | 1. 调用 `POST /api/upload` 上传 PDB 文件；2. 读取响应 JSON | 返回 200；`chains` 至少包含 A 链；链长度大于等于 2；返回原始 `pdb_content` | 自动化测试断言接口状态码、链信息和原始内容 | 是 | `backend/tests/test_system_acceptance.py::test_upload_pdb_returns_chain_summary_and_original_content` |
| SAT-002 | 用户注册与登录 | 用户名 `researcher1`，密码 `secret123`，邮箱 `researcher1@test.local` | 1. 调用 `POST /api/auth/register`；2. 调用 `POST /api/auth/login`；3. 使用 token 调用 `GET /api/auth/me` | 注册成功；登录返回 access token；当前用户角色为 `researcher` | 自动化测试断言注册、登录和当前用户信息 | 是 | `backend/tests/test_system_acceptance.py::test_login_and_role_permissions_for_researcher_and_admin` |
| SAT-003 | 研究人员权限限制 | 研究人员 token | 1. 使用研究人员 token 调用 `GET /api/users`；2. 不带 token 调用 `GET /api/auth/me` | 研究人员访问用户列表返回 403；匿名访问当前用户接口返回 401 | 自动化测试断言 403 和 401 | 是 | `backend/tests/test_system_acceptance.py::test_login_and_role_permissions_for_researcher_and_admin` |
| SAT-004 | 管理员权限 | 管理员账号 `admin/admin123` | 1. 调用登录接口获取管理员 token；2. 调用 `GET /api/users` | 返回 200；用户列表包含管理员和已注册研究人员 | 自动化测试断言管理员可访问用户列表 | 是 | `backend/tests/test_system_acceptance.py::test_login_and_role_permissions_for_researcher_and_admin` |
| SAT-005 | 实验记录创建与列表 | 实验名称 `系统验收实验`，靶点 `A/1-2`，热点 A1 | 1. 登录研究人员；2. 调用 `POST /api/experiments`；3. 调用 `GET /api/experiments` | 创建返回实验 ID；列表总数为 1；实验名称正确 | 自动化测试断言创建响应和列表内容 | 是 | `backend/tests/test_system_acceptance.py::test_experiment_record_lifecycle_acceptance` |
| SAT-006 | 实验记录更新与详情 | 实验状态 `completed`，RF3 结果 RMSD 1.4、pLDDT 90.0 | 1. 调用 `PUT /api/experiments/{id}` 更新结果；2. 调用详情接口 | 更新返回 `ok=true`；详情中状态、RF3 结果正确 | 自动化测试断言更新响应和详情字段 | 是 | `backend/tests/test_system_acceptance.py::test_experiment_record_lifecycle_acceptance` |
| SAT-007 | 添加设计结果与导出报告 | 设计名 `design_0`，序列 `ACDE`，验证通过 | 1. 调用 `POST /api/experiments/{id}/designs`；2. 调用 `GET /api/experiments/{id}/export` | 设计结果保存成功；导出报告包含实验信息和设计结果 | 自动化测试断言设计序列、通过状态和导出内容 | 是 | `backend/tests/test_system_acceptance.py::test_experiment_record_lifecycle_acceptance` |
| SAT-008 | 删除实验记录 | 已创建实验 ID，研究人员 token | 1. 调用 `DELETE /api/experiments/{id}`；2. 再次调用详情接口 | 删除返回 `ok=true`；再次查询返回 404 | 自动化测试断言删除和 404 | 是 | `backend/tests/test_system_acceptance.py::test_experiment_record_lifecycle_acceptance` |
| SAT-009 | 跨用户删除权限 | 用户 owner 创建实验，用户 other 删除该实验 | 1. owner 登录并创建实验；2. other 登录并调用删除接口 | 非管理员不能删除他人实验，返回 403 | 自动化测试断言返回 403 | 是 | `backend/tests/test_system_acceptance.py::test_researcher_cannot_delete_another_users_experiment` |
| SAT-010 | 完整 pipeline mock 端到端 | PDB 内容、热点 A1、binder 长度 80 | 1. 调用 `POST /api/run-pipeline`；2. mock 服务返回 RFD3、MPNN、RF3 结果 | 返回 `completed`；包含 backbone PDB、sequence PDB、RF3 验证通过结果 | 自动化测试断言三步结果和完成状态 | 是 | `backend/tests/test_system_acceptance.py::test_mock_pipeline_end_to_end_acceptance` |
| SAT-011 | 匿名前端入口 | 无 token 的浏览器状态 | 1. 打开前端应用；2. 查看默认页面和登录弹窗 | 显示登录弹窗；主设计页面仍在背景中；侧边栏包含 De Novo 入口 | 自动化测试断言登录弹窗、目标结构区和 De Novo 入口 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::shows login modal for anonymous users while keeping the new design page available behind it` |
| SAT-012 | 研究人员前端权限 | 研究人员登录成功响应 | 1. 在登录弹窗输入账号密码；2. 登录成功后查看侧边栏 | 登录弹窗关闭；显示研究员身份；显示 De Novo 入口；不显示系统监控入口 | 自动化测试断言研究员标签、De Novo 入口和系统监控隐藏 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::logs in a researcher and hides administrator-only system monitor` |
| SAT-013 | 管理员前端权限 | 管理员 token 和用户信息 | 1. 设置管理员登录状态；2. 打开前端应用 | 显示管理员身份；侧边栏显示系统监控入口 | 自动化测试断言管理员标签和系统监控入口 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::shows system monitor navigation for administrators` |
| SAT-014 | De Novo 前端流程 | mock RFD3、MPNN、RF3 API 响应 | 1. 点击 `De Novo Design`；2. 点击运行 RFD3；3. 选择结果并送入 MPNN；4. 送入 RF3 | 页面展示 RFD3 骨架、MPNN 序列、RF3 验证通过和 RMSD 指标 | 自动化测试断言三次 API 调用参数和页面结果 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::runs the De Novo RFD3 to MPNN to RF3 page flow with mocked APIs` |
| SAT-015 | 实验记录前端列表 | mock 实验列表返回 1 条已完成记录 | 1. 登录研究人员；2. 点击实验记录；3. 等待列表加载 | 表格展示实验名称、状态、靶点和设计数 | 自动化测试断言表格内容 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::loads experiment records through the experiments page` |
| SAT-016 | 帮助页角色和 ProteinMPNN 说明 | 登录研究人员状态 | 1. 点击帮助；2. 查看用户角色与结果解读部分 | 帮助页包含研究人员、管理员能力说明；说明 ProteinMPNN 负责序列设计而非结构预测 | 自动化测试断言角色文本和 MPNN 可视化说明 | 是 | `frontend/src/pages/__tests__/acceptance.test.tsx::documents roles and ProteinMPNN visualization limits in help page` |
| SAT-017 | 实验记录对象级权限 | owner 创建实验，other 使用自己的 token 访问 owner 实验 | 1. owner 创建实验；2. other 调用详情、更新、添加设计、导出、对比接口 | other 对 owner 实验的访问全部返回 403 | 自动化测试断言所有越权响应为 403 | 是 | `backend/tests/test_security_robustness.py::test_researcher_cannot_read_or_mutate_another_users_experiment` |
| SAT-018 | 实验接口匿名访问限制 | 无 token 请求实验详情、创建、更新、添加设计、导出、对比 | 1. 不携带 token 调用实验记录接口；2. 读取响应状态码 | 所有需要登录的实验接口返回 401 | 自动化测试断言所有匿名响应为 401 | 是 | `backend/tests/test_security_robustness.py::test_experiment_interfaces_require_login` |
| SAT-019 | 注册输入校验 | 短用户名、非法用户名、弱密码、非法邮箱 | 1. 调用注册接口提交非法输入；2. 读取响应状态码 | 非法输入均返回 422，不写入用户数据 | 自动化测试断言非法注册请求被拒绝 | 是 | `backend/tests/test_security_robustness.py::test_register_rejects_invalid_identity_inputs` |
| SAT-020 | 上传文件鲁棒性 | `.txt` 文件、空 `.pdb` 文件、超过大小限制的结构文件 | 1. 调用上传接口提交异常文件；2. 读取响应状态码 | 异常文件均返回 400 | 自动化测试断言非法扩展名、空文件和超限文件被拒绝 | 是 | `backend/tests/test_security_robustness.py::test_upload_rejects_invalid_file_types_empty_files_and_oversized_content` |
| SAT-021 | CORS 白名单配置 | `DEEPBINDER_CORS_ORIGINS=https://deepbinder.example.edu,http://localhost:5173` | 1. 设置环境变量；2. 重新加载后端应用；3. 读取 CORS 中间件配置 | CORS 只允许环境变量中列出的源 | 自动化测试断言 CORS allow_origins 等于配置列表 | 是 | `backend/tests/test_security_robustness.py::test_cors_origin_configuration_uses_environment_allowlist` |

## 执行命令

后端：

```bash
cd binder-design-system/backend
python3 -m pytest tests/test_system_acceptance.py tests/test_security_robustness.py -v
```

前端：

```bash
cd binder-design-system/frontend
npm install
npm test -- --run
```

构建验证：

```bash
cd binder-design-system/frontend
npm run build
```
