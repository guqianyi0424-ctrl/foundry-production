# 基于深度学习的蛋白质Binder生成系统

## 项目简介

本系统整合了多个深度学习模型，实现蛋白质Binder的自动化设计与验证。

### 核心功能

- **热点残基预测**: 基于机器学习和深度学习的热点残基识别
- **Binder骨架生成**: 使用RFD3生成Binder骨架结构
- **序列设计**: 使用MPNN设计Binder氨基酸序列
- **结构验证**: 使用RF3预测并验证设计结构
- **可视化展示**: 3D结构可视化和交互式界面

### 技术栈

- **前端**: Streamlit
- **后端**: Python 3.11
- **深度学习**: PyTorch, rc-foundry
- **可视化**: py3Dmol
- **存储**: 腾讯云COS

## 项目结构

```
binder-design-system/
├── app/                    # Streamlit应用
│   ├── main.py            # 主应用
│   └── pages/             # 多页面应用
├── utils/                  # 工具模块
│   ├── structure_parser.py # 结构解析
│   ├── hotspot_predictor.py # 热点预测
│   └── cos_storage.py     # COS存储管理
├── models/                 # 模型权重
├── config/                 # 配置文件
│   └── settings.py        # 系统配置
├── data/                   # 数据文件
├── outputs/                # 输出结果
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型权重

```bash
foundry install rfd3 ligandmpnn rf3
```

### 3. 运行应用

```bash
streamlit run app/main.py
```

## 部署说明

详见 [deploy.md](deploy.md)

## 开发进度

- [x] 项目结构搭建
- [x] 基础Streamlit应用
- [ ] 文件上传与解析
- [ ] 3D结构可视化
- [ ] 热点残基预测集成
- [ ] RFD3骨架生成
- [ ] MPNN序列设计
- [ ] RF3结构验证
- [ ] 结果评估与导出

## 作者

本科毕业设计项目

## 许可证

MIT License
