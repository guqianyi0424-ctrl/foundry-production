# DeepBinder 安全与鲁棒性补强实施计划

## 目标

补强系统的对象级权限、认证配置、CORS 策略和上传校验，并形成毕设可引用的安全与鲁棒性说明。

## 实施顺序

1. 对象级权限
   - 所有实验记录读写接口要求登录。
   - 研究人员只能访问自己的实验记录。
   - 管理员可访问全部实验记录。
   - 覆盖详情、更新、添加设计、导出、对比和删除接口。

2. JWT/CORS/上传校验
   - JWT 生产环境要求通过 `DEEPBINDER_SECRET_KEY` 提供强密钥。
   - CORS 默认只允许本地开发源，生产通过 `DEEPBINDER_CORS_ORIGINS` 配置。
   - 上传接口限制扩展名、空文件、实际读取大小和无有效链结构的文件。

3. 文档与测试
   - 补充系统验收测试中的安全用例。
   - 新增安全与鲁棒性设计文档。
   - 更新毕设测试用例表。

## 验证命令

```bash
cd binder-design-system/backend
PYTHONPATH="$PWD/../..:$PWD/..:$PWD" python -m pytest tests/test_system_acceptance.py tests/test_security_robustness.py -v
```

```bash
cd binder-design-system/frontend
npm test -- --run
npm run build
```
