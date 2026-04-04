# 测试数据获取

## 方式1：从PDB数据库下载

访问 https://www.rcsb.org/ 搜索并下载PDB文件

推荐测试蛋白：

1. **PD-1** (5ZTN)

   ```bash
   wget https://files.rcsb.org/download/5ZTN.pdb
   ```

2. **IL-2** (1M47)

   ```bash
   wget https://files.rcsb.org/download/1M47.pdb
   ```

3. **SARS-CoV-2 Spike** (6VSB)
   ```bash
   wget https://files.rcsb.org/download/6VSB.pdb
   ```

## 方式2：使用本地文件

将你的PDB/CIF文件复制到此目录

## 注意

- 文件格式：PDB或CIF
- 文件大小：建议<200MB
- 文件编码：UTF-8或ASCII
