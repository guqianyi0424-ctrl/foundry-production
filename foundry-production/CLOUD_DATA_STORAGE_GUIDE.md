# 云平台数据存储方案 - 蛋白Binder设计系统

**适用场景**: Google Colab / Kaggle / AutoDL等云平台  
**更新日期**: 2025-04-04

---

## 📋 目录

1. [方案对比](#1-方案对比)
2. [方案一: Google Drive存储](#2-方案一-google-drive存储)
3. [方案二: 云数据库 (Firebase)](#3-方案二-云数据库-firebase)
4. [方案三: GitHub存储](#4-方案三-github存储)
5. [完整代码示例](#5-完整代码示例)
6. [推荐方案](#6-推荐方案)

---

## 1. 方案对比

### 1.1 四种存储方案

| 方案             | 成本     | 容量 | 易用性     | 持久性 | 推荐度     |
| ---------------- | -------- | ---- | ---------- | ------ | ---------- |
| **Google Drive** | 免费15GB | 15GB | ⭐⭐⭐⭐⭐ | 永久   | ⭐⭐⭐⭐⭐ |
| **Firebase**     | 免费套餐 | 1GB  | ⭐⭐⭐⭐   | 永久   | ⭐⭐⭐⭐   |
| **GitHub**       | 免费     | 无限 | ⭐⭐⭐     | 永久   | ⭐⭐⭐     |
| **本地下载**     | 免费     | 无限 | ⭐⭐⭐⭐⭐ | 手动   | ⭐⭐⭐⭐   |

### 1.2 推荐组合

```
本科毕设推荐存储方案:

┌─────────────────────────────────────────────┐
│          数据存储层次结构                    │
└─────────────────────────────────────────────┘

第一层: 实时存储 (Google Drive)
├─ 设计结果文件 (.cif, .pdb)
├─ 序列文件 (.fasta)
├─ 评估指标 (.json, .csv)
└─ 用户操作日志 (.log)

第二层: 数据库存储 (Firebase, 可选)
├─ 用户信息
├─ 设计任务记录
├─ 查询统计
└─ 实验元数据

第三层: 版本控制 (GitHub)
├─ 代码版本
├─ 配置文件
├─ 重要结果快照
└─ 文档

第四层: 本地备份
├─ 定期下载关键结果
├─ 论文数据
└─ 最终成果
```

---

## 2. 方案一: Google Drive存储

### 2.1 优势

- ✅ **完全免费** (15GB)
- ✅ Colab原生支持
- ✅ 自动同步
- ✅ 可在任何设备访问
- ✅ 支持文件夹结构

### 2.2 使用方法

#### 挂载Google Drive

```python
# 在Colab中挂载Google Drive

from google.colab import drive
import os

# 挂载
drive.mount('/content/drive')

# 创建项目目录
project_dir = '/content/drive/MyDrive/binder_design_project'
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f'{project_dir}/structures', exist_ok=True)
os.makedirs(f'{project_dir}/sequences', exist_ok=True)
os.makedirs(f'{project_dir}/metrics', exist_ok=True)
os.makedirs(f'{project_dir}/logs', exist_ok=True)

print(f"✓ 项目目录: {project_dir}")
```

#### 保存设计结果

```python
import json
from datetime import datetime
from atomworks.io.utils.io_utils import to_cif_file

class GoogleDriveStorage:
    """Google Drive存储管理器"""

    def __init__(self, base_dir='/content/drive/MyDrive/binder_design_project'):
        self.base_dir = base_dir
        self.structures_dir = f"{base_dir}/structures"
        self.sequences_dir = f"{base_dir}/sequences"
        self.metrics_dir = f"{base_dir}/metrics"
        self.logs_dir = f"{base_dir}/logs"

        # 确保目录存在
        for dir_path in [self.structures_dir, self.sequences_dir,
                         self.metrics_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save_design_result(self, design_id, atom_array, sequence, metrics):
        """
        保存单个设计结果

        Args:
            design_id: 设计ID
            atom_array: 结构原子数组
            sequence: 氨基酸序列
            metrics: 评估指标字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存结构文件
        structure_file = f"{self.structures_dir}/{design_id}_{timestamp}.cif"
        to_cif_file(atom_array, structure_file)

        # 2. 保存序列文件
        sequence_file = f"{self.sequences_dir}/{design_id}_{timestamp}.fasta"
        with open(sequence_file, 'w') as f:
            f.write(f">{design_id}\n")
            f.write(f"{sequence}\n")

        # 3. 保存指标
        metrics_file = f"{self.metrics_dir}/{design_id}_{timestamp}.json"
        metrics['design_id'] = design_id
        metrics['timestamp'] = timestamp
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ 已保存设计 {design_id}")
        print(f"  结构: {structure_file}")
        print(f"  序列: {sequence_file}")
        print(f"  指标: {metrics_file}")

        return {
            'structure_file': structure_file,
            'sequence_file': sequence_file,
            'metrics_file': metrics_file
        }

    def save_batch_results(self, results):
        """
        批量保存结果

        Args:
            results: 结果列表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存汇总CSV
        import csv
        csv_file = f"{self.metrics_dir}/batch_results_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        print(f"✓ 批量结果已保存: {csv_file}")
        return csv_file

    def log_operation(self, operation, details):
        """
        记录操作日志

        Args:
            operation: 操作类型
            details: 操作详情
        """
        log_file = f"{self.logs_dir}/operations.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {operation}: {details}\n")

    def list_results(self):
        """列出所有已保存的结果"""
        results = []

        # 读取所有指标文件
        for filename in os.listdir(self.metrics_dir):
            if filename.endswith('.json'):
                with open(f"{self.metrics_dir}/{filename}", 'r') as f:
                    results.append(json.load(f))

        # 按时间排序
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return results

    def get_statistics(self):
        """获取统计信息"""
        results = self.list_results()

        if not results:
            return None

        import numpy as np

        stats = {
            'total_designs': len(results),
            'avg_rmsd': np.mean([r['rmsd'] for r in results if 'rmsd' in r]),
            'avg_plddt': np.mean([r['plddt'] for r in results if 'plddt' in r]),
            'best_rmsd': min([r['rmsd'] for r in results if 'rmsd' in r]),
            'best_plddt': max([r['plddt'] for r in results if 'plddt' in r])
        }

        return stats


# 使用示例
storage = GoogleDriveStorage()

# 保存单个结果
storage.save_design_result(
    design_id="binder_001",
    atom_array=atom_array,
    sequence="MKVLWAALLVTFLTC...",
    metrics={
        'rmsd': 1.2,
        'plddt': 85.3,
        'ptm': 0.75
    }
)

# 记录操作
storage.log_operation("DESIGN_START", "开始生成Binder，长度60残基")

# 查看统计
stats = storage.get_statistics()
print(f"总设计数: {stats['total_designs']}")
print(f"平均RMSD: {stats['avg_rmsd']:.2f} Å")
```

### 2.3 数据组织结构

```
Google Drive/
└── MyDrive/
    └── binder_design_project/
        ├── structures/              # 结构文件
        │   ├── binder_001_20250404_143022.cif
        │   ├── binder_002_20250404_143156.cif
        │   └── ...
        ├── sequences/               # 序列文件
        │   ├── binder_001_20250404_143022.fasta
        │   ├── binder_002_20250404_143156.fasta
        │   └── ...
        ├── metrics/                 # 评估指标
        │   ├── binder_001_20250404_143022.json
        │   ├── binder_002_20250404_143156.json
        │   ├── batch_results_20250404_143200.csv
        │   └── ...
        ├── logs/                    # 操作日志
        │   └── operations.log
        └── README.md                # 项目说明
```

---

## 3. 方案二: 云数据库 (Firebase)

### 3.1 Firebase Realtime Database

#### 优势

- ✅ **免费套餐**: 1GB存储，每日读写配额
- ✅ 实时同步
- ✅ 支持查询
- ✅ REST API访问
- ✅ 适合记录用户操作

#### 设置步骤

**Step 1: 创建Firebase项目**

```
1. 访问: https://console.firebase.google.com/
2. 点击"添加项目"
3. 项目名称: binder-design-system
4. 选择"不启用Google Analytics" (可选)
5. 创建项目
```

**Step 2: 创建Realtime Database**

```
1. 在项目控制台，点击"Realtime Database"
2. 点击"创建数据库"
3. 选择位置 (建议: asia-east1)
4. 选择"以测试模式启动" (开发阶段)
5. 创建
```

**Step 3: 获取配置信息**

```
1. 点击"项目设置"
2. 滚动到"您的应用"
3. 点击"</>" (Web应用)
4. 复制配置信息
```

#### 代码实现

```python
# 安装Firebase库
!pip install -q firebase-admin

import firebase_admin
from firebase_admin import credentials, db
import json

class FirebaseDatabase:
    """Firebase数据库管理器"""

    def __init__(self, database_url, service_account_key=None):
        """
        初始化Firebase

        Args:
            database_url: Firebase数据库URL
            service_account_key: 服务账号密钥 (可选，测试模式可省略)
        """
        # 初始化Firebase
        if not firebase_admin._apps:
            if service_account_key:
                cred = credentials.Certificate(service_account_key)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
            else:
                # 测试模式 (仅限开发阶段)
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })

        self.db = db.reference()

    def save_user(self, user_id, user_data):
        """
        保存用户信息

        Args:
            user_id: 用户ID
            user_data: 用户数据
        """
        ref = self.db.child('users').child(user_id)
        ref.set(user_data)
        print(f"✓ 用户 {user_id} 已保存")

    def save_design_task(self, task_id, task_data):
        """
        保存设计任务

        Args:
            task_id: 任务ID
            task_data: 任务数据
        """
        ref = self.db.child('tasks').child(task_id)
        ref.set(task_data)
        print(f"✓ 任务 {task_id} 已保存")

    def save_design_result(self, result_id, result_data):
        """
        保存设计结果

        Args:
            result_id: 结果ID
            result_data: 结果数据
        """
        ref = self.db.child('results').child(result_id)
        ref.set(result_data)
        print(f"✓ 结果 {result_id} 已保存")

    def get_user_tasks(self, user_id):
        """
        获取用户的所有任务

        Args:
            user_id: 用户ID

        Returns:
            任务列表
        """
        ref = self.db.child('tasks')
        tasks = ref.order_by_child('user_id').equal_to(user_id).get()
        return tasks

    def get_task_results(self, task_id):
        """
        获取任务的所有结果

        Args:
            task_id: 任务ID

        Returns:
            结果列表
        """
        ref = self.db.child('results')
        results = ref.order_by_child('task_id').equal_to(task_id).get()
        return results

    def update_task_status(self, task_id, status, progress=None):
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            progress: 进度 (可选)
        """
        ref = self.db.child('tasks').child(task_id)
        updates = {'status': status}
        if progress is not None:
            updates['progress'] = progress
        ref.update(updates)

    def get_statistics(self):
        """
        获取统计信息

        Returns:
            统计数据字典
        """
        users = self.db.child('users').get()
        tasks = self.db.child('tasks').get()
        results = self.db.child('results').get()

        stats = {
            'total_users': len(users) if users else 0,
            'total_tasks': len(tasks) if tasks else 0,
            'total_results': len(results) if results else 0
        }

        return stats


# 使用示例
# 注意: 需要先在Firebase控制台获取数据库URL
DATABASE_URL = "https://binder-design-system-default-rtdb.asia-southeast1.firebasedatabase.app/"

# 初始化
firebase_db = FirebaseDatabase(database_url=DATABASE_URL)

# 保存用户
firebase_db.save_user("user_001", {
    'name': '张三',
    'email': 'zhangsan@example.com',
    'created_at': '2025-04-04 14:30:00'
})

# 保存任务
firebase_db.save_design_task("task_001", {
    'user_id': 'user_001',
    'target_protein': 'example.pdb',
    'binder_length': 60,
    'status': 'running',
    'progress': 50,
    'created_at': '2025-04-04 14:30:00'
})

# 更新任务状态
firebase_db.update_task_status("task_001", "completed", progress=100)

# 保存结果
firebase_db.save_design_result("result_001", {
    'task_id': 'task_001',
    'sequence': 'MKVLWAALLVTFLTC...',
    'rmsd': 1.2,
    'plddt': 85.3,
    'created_at': '2025-04-04 14:35:00'
})

# 查询统计
stats = firebase_db.get_statistics()
print(f"总用户数: {stats['total_users']}")
print(f"总任务数: {stats['total_tasks']}")
print(f"总结果数: {stats['total_results']}")
```

### 3.2 Firebase数据结构

```json
{
  "users": {
    "user_001": {
      "name": "张三",
      "email": "zhangsan@example.com",
      "created_at": "2025-04-04 14:30:00"
    }
  },
  "tasks": {
    "task_001": {
      "user_id": "user_001",
      "target_protein": "example.pdb",
      "binder_length": 60,
      "status": "completed",
      "progress": 100,
      "created_at": "2025-04-04 14:30:00",
      "completed_at": "2025-04-04 14:35:00"
    }
  },
  "results": {
    "result_001": {
      "task_id": "task_001",
      "sequence": "MKVLWAALLVTFLTC...",
      "rmsd": 1.2,
      "plddt": 85.3,
      "ptm": 0.75,
      "created_at": "2025-04-04 14:35:00"
    }
  }
}
```

---

## 4. 方案三: GitHub存储

### 4.1 使用GitHub存储结果

```python
# 安装PyGithub
!pip install -q PyGithub

from github import Github
import base64

class GitHubStorage:
    """GitHub存储管理器"""

    def __init__(self, token, repo_name):
        """
        初始化GitHub存储

        Args:
            token: GitHub Personal Access Token
            repo_name: 仓库名称 (格式: username/repo)
        """
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def save_file(self, file_path, content, commit_message="Add file"):
        """
        保存文件到GitHub

        Args:
            file_path: 文件路径 (相对于仓库根目录)
            content: 文件内容
            commit_message: 提交信息
        """
        try:
            # 尝试获取现有文件
            self.repo.get_contents(file_path)
            # 如果存在，更新
            self.repo.update_file(
                file_path,
                commit_message,
                content,
                self.repo.get_contents(file_path).sha
            )
        except:
            # 如果不存在，创建
            self.repo.create_file(
                file_path,
                commit_message,
                content
            )

        print(f"✓ 文件已保存: {file_path}")

    def save_json(self, file_path, data, commit_message="Update JSON"):
        """保存JSON数据"""
        import json
        content = json.dumps(data, indent=2)
        self.save_file(file_path, content, commit_message)

    def get_file(self, file_path):
        """获取文件内容"""
        content = self.repo.get_contents(file_path)
        return content.decoded_content.decode('utf-8')


# 使用示例
# 需要先创建GitHub Personal Access Token
# 设置 → Developer settings → Personal access tokens → Generate new token

GITHUB_TOKEN = "your_github_token_here"
REPO_NAME = "your_username/binder-design-results"

github_storage = GitHubStorage(GITHUB_TOKEN, REPO_NAME)

# 保存结果
github_storage.save_json(
    "results/binder_001.json",
    {
        'design_id': 'binder_001',
        'sequence': 'MKVLWAALLVTFLTC...',
        'rmsd': 1.2,
        'plddt': 85.3
    },
    commit_message="Add binder_001 design result"
)
```

---

## 5. 完整代码示例

### 5.1 综合存储管理器

```python
"""
综合数据存储管理器
支持Google Drive + Firebase + 本地存储
"""

import os
import json
import csv
from datetime import datetime
from google.colab import drive

class DataManager:
    """数据管理器 - 统一管理所有存储方式"""

    def __init__(self,
                 use_drive=True,
                 use_firebase=False,
                 firebase_url=None,
                 project_name="binder_design_project"):
        """
        初始化数据管理器

        Args:
            use_drive: 是否使用Google Drive
            use_firebase: 是否使用Firebase
            firebase_url: Firebase数据库URL
            project_name: 项目名称
        """
        self.project_name = project_name

        # Google Drive
        if use_drive:
            self._init_drive()

        # Firebase
        if use_firebase and firebase_url:
            self._init_firebase(firebase_url)

        # 本地临时存储
        self.local_temp = "/tmp/binder_temp"
        os.makedirs(self.local_temp, exist_ok=True)

        print(f"✓ 数据管理器初始化完成")
        print(f"  Google Drive: {'启用' if use_drive else '禁用'}")
        print(f"  Firebase: {'启用' if use_firebase else '禁用'}")

    def _init_drive(self):
        """初始化Google Drive"""
        try:
            drive.mount('/content/drive', force_remount=True)
            self.drive_dir = f'/content/drive/MyDrive/{self.project_name}'

            # 创建目录结构
            for subdir in ['structures', 'sequences', 'metrics', 'logs', 'exports']:
                os.makedirs(f'{self.drive_dir}/{subdir}', exist_ok=True)

            print(f"  Google Drive目录: {self.drive_dir}")
        except Exception as e:
            print(f"  ⚠️ Google Drive挂载失败: {e}")
            self.drive_dir = None

    def _init_firebase(self, database_url):
        """初始化Firebase"""
        try:
            import firebase_admin
            from firebase_admin import db

            if not firebase_admin._apps:
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })

            self.firebase_db = db.reference()
            print(f"  Firebase已连接")
        except Exception as e:
            print(f"  ⚠️ Firebase连接失败: {e}")
            self.firebase_db = None

    def save_design(self, design_data):
        """
        保存设计结果到所有存储

        Args:
            design_data: 设计数据字典
                {
                    'design_id': str,
                    'atom_array': AtomArray,
                    'sequence': str,
                    'metrics': dict,
                    'task_id': str (可选),
                    'user_id': str (可选)
                }
        """
        design_id = design_data['design_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存到Google Drive
        if self.drive_dir:
            self._save_to_drive(design_data, timestamp)

        # 2. 保存到Firebase
        if self.firebase_db:
            self._save_to_firebase(design_data, timestamp)

        # 3. 保存到本地临时
        self._save_to_local(design_data, timestamp)

        print(f"✓ 设计 {design_id} 已保存到所有存储")

    def _save_to_drive(self, design_data, timestamp):
        """保存到Google Drive"""
        from atomworks.io.utils.io_utils import to_cif_file

        design_id = design_data['design_id']

        # 保存结构
        if 'atom_array' in design_data:
            structure_file = f"{self.drive_dir}/structures/{design_id}_{timestamp}.cif"
            to_cif_file(design_data['atom_array'], structure_file)

        # 保存序列
        if 'sequence' in design_data:
            sequence_file = f"{self.drive_dir}/sequences/{design_id}_{timestamp}.fasta"
            with open(sequence_file, 'w') as f:
                f.write(f">{design_id}\n{design_data['sequence']}\n")

        # 保存指标
        if 'metrics' in design_data:
            metrics_file = f"{self.drive_dir}/metrics/{design_id}_{timestamp}.json"
            metrics = design_data['metrics'].copy()
            metrics['design_id'] = design_id
            metrics['timestamp'] = timestamp
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

    def _save_to_firebase(self, design_data, timestamp):
        """保存到Firebase"""
        design_id = design_data['design_id']

        result_data = {
            'design_id': design_id,
            'sequence': design_data.get('sequence', ''),
            'metrics': design_data.get('metrics', {}),
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat()
        }

        if 'task_id' in design_data:
            result_data['task_id'] = design_data['task_id']

        if 'user_id' in design_data:
            result_data['user_id'] = design_data['user_id']

        self.firebase_db.child('results').child(design_id).set(result_data)

    def _save_to_local(self, design_data, timestamp):
        """保存到本地临时目录"""
        design_id = design_data['design_id']

        # 保存JSON汇总
        summary_file = f"{self.local_temp}/{design_id}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'design_id': design_id,
                'sequence': design_data.get('sequence', ''),
                'metrics': design_data.get('metrics', {}),
                'timestamp': timestamp
            }, f, indent=2)

    def export_results(self, output_format='csv'):
        """
        导出所有结果

        Args:
            output_format: 导出格式 ('csv', 'json', 'excel')
        """
        if not self.drive_dir:
            print("⚠️ Google Drive未挂载，无法导出")
            return

        # 收集所有结果
        results = []
        metrics_dir = f"{self.drive_dir}/metrics"

        for filename in os.listdir(metrics_dir):
            if filename.endswith('.json'):
                with open(f"{metrics_dir}/{filename}", 'r') as f:
                    results.append(json.load(f))

        if not results:
            print("⚠️ 没有找到结果")
            return

        # 按时间排序
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"{self.drive_dir}/exports"

        # 导出CSV
        if output_format in ['csv', 'all']:
            csv_file = f"{export_dir}/all_results_{timestamp}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"✓ CSV已导出: {csv_file}")

        # 导出JSON
        if output_format in ['json', 'all']:
            json_file = f"{export_dir}/all_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ JSON已导出: {json_file}")

        # 导出Excel
        if output_format in ['excel', 'all']:
            try:
                import pandas as pd
                excel_file = f"{export_dir}/all_results_{timestamp}.xlsx"
                df = pd.DataFrame(results)
                df.to_excel(excel_file, index=False)
                print(f"✓ Excel已导出: {excel_file}")
            except ImportError:
                print("⚠️ 需要安装pandas和openpyxl: !pip install pandas openpyxl")

    def get_statistics(self):
        """获取统计信息"""
        if not self.drive_dir:
            return None

        # 收集所有结果
        results = []
        metrics_dir = f"{self.drive_dir}/metrics"

        for filename in os.listdir(metrics_dir):
            if filename.endswith('.json'):
                with open(f"{metrics_dir}/{filename}", 'r') as f:
                    results.append(json.load(f))

        if not results:
            return None

        import numpy as np

        stats = {
            'total_designs': len(results),
            'avg_rmsd': np.mean([r['rmsd'] for r in results if 'rmsd' in r]),
            'std_rmsd': np.std([r['rmsd'] for r in results if 'rmsd' in r]),
            'avg_plddt': np.mean([r['plddt'] for r in results if 'plddt' in r]),
            'best_rmsd': min([r['rmsd'] for r in results if 'rmsd' in r]),
            'best_plddt': max([r['plddt'] for r in results if 'plddt' in r]),
            'excellent_count': len([r for r in results if r.get('rmsd', 999) < 1.0]),
            'good_count': len([r for r in results if 1.0 <= r.get('rmsd', 999) < 2.0])
        }

        return stats

    def create_report(self):
        """生成分析报告"""
        stats = self.get_statistics()

        if not stats:
            print("⚠️ 没有数据")
            return

        report = f"""
{'='*60}
蛋白Binder设计统计报告
{'='*60}

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

总体统计:
  总设计数: {stats['total_designs']}

RMSD统计:
  平均值: {stats['avg_rmsd']:.2f} Å
  标准差: {stats['std_rmsd']:.2f} Å
  最佳值: {stats['best_rmsd']:.2f} Å

pLDDT统计:
  平均值: {stats['avg_plddt']:.1f}
  最佳值: {stats['best_plddt']:.1f}

质量分布:
  优秀 (RMSD < 1.0 Å): {stats['excellent_count']} 个
  良好 (1.0 ≤ RMSD < 2.0 Å): {stats['good_count']} 个
  一般 (RMSD ≥ 2.0 Å): {stats['total_designs'] - stats['excellent_count'] - stats['good_count']} 个

{'='*60}
"""

        print(report)

        # 保存报告
        if self.drive_dir:
            report_file = f"{self.drive_dir}/exports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"✓ 报告已保存: {report_file}")


# 使用示例
data_manager = DataManager(
    use_drive=True,
    use_firebase=False,  # 可选
    project_name="binder_design_project"
)

# 保存设计结果
data_manager.save_design({
    'design_id': 'binder_001',
    'sequence': 'MKVLWAALLVTFLTC...',
    'metrics': {
        'rmsd': 1.2,
        'plddt': 85.3,
        'ptm': 0.75
    }
})

# 导出结果
data_manager.export_results(output_format='all')

# 生成报告
data_manager.create_report()
```

---

## 6. 推荐方案

### 6.1 本科毕设推荐配置

```python
# 推荐配置: Google Drive + 本地备份

from google.colab import drive
import os

# 1. 挂载Google Drive
drive.mount('/content/drive')

# 2. 创建项目目录
project_dir = '/content/drive/MyDrive/binder_design_project'
os.makedirs(project_dir, exist_ok=True)

# 3. 使用数据管理器
data_manager = DataManager(
    use_drive=True,
    use_firebase=False,  # 本科毕设不需要
    project_name="binder_design_project"
)

# 4. 保存设计结果
data_manager.save_design({
    'design_id': 'binder_001',
    'sequence': 'MKVLWAALLVTFLTC...',
    'metrics': {
        'rmsd': 1.2,
        'plddt': 85.3,
        'ptm': 0.75
    }
})

# 5. 定期导出和备份
data_manager.export_results(output_format='all')
data_manager.create_report()
```

### 6.2 数据持久化最佳实践

```
数据持久化三原则:

1. 多重备份
   ├─ Google Drive (主存储)
   ├─ 本地下载 (定期备份)
   └─ GitHub (重要结果)

2. 结构化存储
   ├─ 使用JSON/CSV格式
   ├─ 统一命名规范
   └─ 添加时间戳

3. 定期导出
   ├─ 每次实验后导出
   ├─ 生成统计报告
   └─ 保存到多个位置
```

---

## 📝 总结

### 推荐方案

| 数据类型     | 存储方式        | 理由                 |
| ------------ | --------------- | -------------------- |
| **设计结果** | Google Drive    | 免费、方便、自动同步 |
| **用户数据** | Firebase (可选) | 结构化、可查询       |
| **代码版本** | GitHub          | 版本控制             |
| **论文数据** | 本地备份        | 最终成果             |

### 快速开始

```python
# 最简单的方案: Google Drive

from google.colab import drive
drive.mount('/content/drive')

# 创建目录
!mkdir -p /content/drive/MyDrive/binder_project/{structures,sequences,metrics}

# 保存结果
import json
with open('/content/drive/MyDrive/binder_project/metrics/result.json', 'w') as f:
    json.dump({'rmsd': 1.2, 'plddt': 85.3}, f)

print("✓ 结果已保存到Google Drive")
```

---

**祝你数据管理顺利！** 💾
