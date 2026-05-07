#!/usr/bin/env python3
"""
创建服务器上传包
打包所有必要的文件和数据
"""
import os
import sys
import json
import zipfile
import gzip
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_upload_package(output_dir: str = "./upload_package"):
    """创建上传包"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"annual_report_assistant_upload_{timestamp}.zip"
    package_path = output_path / package_name
    
    logger.info("=" * 60)
    logger.info("创建服务器上传包")
    logger.info("=" * 60)
    logger.info(f"输出文件: {package_path}")
    
    # 需要包含的文件清单
    files_to_include = []
    
    # 1. 核心代码文件
    core_files = [
        "api/main.py",
        "api/__init__.py",
        "web/server.py",
        "web/index.html",
        "src/core/assistant.py",
        "src/core/__init__.py",
        "src/core/config.py",
        "src/core/exceptions.py",
        "src/core/logger.py",
        "src/core/retry.py",
        "src/rag/embedding.py",
        "src/rag/retriever.py",
        "src/rag/chunker.py",
        "src/rag/document_processor.py",
        "src/rag/vector_store.py",
        "src/rag/__init__.py",
        "src/rag/interfaces.py",
        "src/prompts/__init__.py",
        "src/prompts/templates.py",
        "client/llm_chat.py",
        "client/llm_chat_qwen3.py",
        "client/__init__.py",
    ]
    
    # 2. 配置文件
    config_files = [
        "config/config.yaml",
        "config/pai_distil_config.yaml",
    ]
    
    # 3. 数据处理脚本
    script_files = [
        "scripts/build_vector_db.py",
        "scripts/process_data_concurrent.py",
        "scripts/evaluate_quality.py",
        "scripts/check_models.py",
    ]
    
    # 4. 启动脚本
    shell_files = [
        "start_all_services.sh",
        "monitor_services.sh",
        "test_model_output.py",
    ]
    
    # 5. 数据文件（已压缩的）
    data_files = [
        "data/processed/knowledge_base/docs_raw_chunks.json.gz",
        "data/processed/training/train_dataset.zip",
        "data/processed/processing_stats.json",
    ]
    
    # 6. 其他必要文件
    other_files = [
        "requirements.txt",
        "README.md",
    ]
    
    all_files = core_files + config_files + script_files + shell_files + data_files + other_files
    
    # 收集存在的文件
    existing_files = []
    base_path = Path(".")
    
    for file_path in all_files:
        full_path = base_path / file_path
        if full_path.exists():
            existing_files.append(file_path)
            logger.info(f"✓ 包含: {file_path}")
        else:
            logger.warning(f"✗ 缺失: {file_path}")
    
    logger.info(f"\n找到 {len(existing_files)}/{len(all_files)} 个文件")
    
    # 创建压缩包
    logger.info("\n正在创建压缩包...")
    
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 添加文件
        for file_path in existing_files:
            full_path = base_path / file_path
            zf.write(full_path, file_path)
            logger.info(f"  添加: {file_path}")
        
        # 创建解压说明
        readme_content = """# 服务器上传包解压说明

## 文件清单

### 核心代码
- api/ - FastAPI 服务
- web/ - 前端服务
- src/ - 核心模块（RAG、Embedding等）
- client/ - LLM客户端

### 配置文件
- config/config.yaml - 主配置
- config/pai_distil_config.yaml - PAI训练配置

### 数据文件（已压缩）
- data/processed/knowledge_base/docs_raw_chunks.json.gz - Chunks数据
- data/processed/training/train_dataset.zip - 训练数据集
- data/processed/processing_stats.json - 处理统计

### 脚本
- scripts/ - 数据处理脚本
- *.sh - 启动脚本

## 解压步骤

1. 上传此压缩包到服务器
   ```bash
   scp annual_report_assistant_upload_*.zip root@<服务器IP>:/mnt/workspace/
   ```

2. 在服务器上解压
   ```bash
   cd /mnt/workspace
   unzip annual_report_assistant_upload_*.zip -d annual_report_assistant/
   cd annual_report_assistant
   ```

3. 解压数据文件
   ```bash
   cd data/processed
   gunzip knowledge_base/docs_raw_chunks.json.gz
   unzip training/train_dataset.zip -d training/
   cd ../..
   ```

4. 检查模型
   ```bash
   python scripts/check_models.py
   ```

5. 启动服务
   ```bash
   bash start_all_services.sh
   ```

## 注意事项

- 确保服务器已安装 Python 3.8+
- 确保有足够的磁盘空间（解压后约 5GB）
- 确保已下载 Qwen3 模型到正确路径
"""
        zf.writestr("UPLOAD_README.md", readme_content)
    
    # 获取压缩包信息
    package_size_mb = package_path.stat().st_size / (1024 * 1024)
    
    logger.info("\n" + "=" * 60)
    logger.info("压缩包创建完成!")
    logger.info("=" * 60)
    logger.info(f"文件名: {package_name}")
    logger.info(f"大小: {package_size_mb:.2f} MB")
    logger.info(f"包含文件数: {len(existing_files)}")
    logger.info(f"路径: {package_path.absolute()}")
    logger.info("=" * 60)
    
    # 创建上传脚本
    upload_script = output_path / f"upload_to_server_{timestamp}.sh"
    script_content = f"""#!/bin/bash
# 上传脚本 - 生成时间: {timestamp}

SERVER_IP="<你的服务器IP>"
SERVER_USER="root"
LOCAL_PACKAGE="{package_name}"
REMOTE_PATH="/mnt/workspace/"

echo "=========================================="
echo "上传年报分析助手到服务器"
echo "=========================================="
echo ""

# 检查参数
if [ "$1" != "" ]; then
    SERVER_IP=$1
fi

echo "服务器: $SERVER_USER@$SERVER_IP"
echo "本地文件: $LOCAL_PACKAGE"
echo "远程路径: $REMOTE_PATH"
echo ""

# 上传
echo "正在上传..."
scp "$LOCAL_PACKAGE" "$SERVER_USER@$SERVER_IP:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "✓ 上传成功!"
    echo ""
    echo "在服务器上执行以下命令解压:"
    echo "  cd $REMOTE_PATH"
    echo "  unzip $LOCAL_PACKAGE -d annual_report_assistant/"
    echo "  cd annual_report_assistant"
    echo "  bash setup_on_server.sh"
else
    echo "✗ 上传失败!"
    exit 1
fi
"""
    
    with open(upload_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 创建服务器端解压脚本
    server_setup = output_path / f"setup_on_server_{timestamp}.sh"
    setup_content = f"""#!/bin/bash
# 服务器端解压和设置脚本

echo "=========================================="
echo "年报分析助手 - 服务器端设置"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "1. 解压数据文件..."
cd data/processed

if [ -f "knowledge_base/docs_raw_chunks.json.gz" ]; then
    echo "  解压 chunks..."
    gunzip -k knowledge_base/docs_raw_chunks.json.gz
fi

if [ -f "training/train_dataset.zip" ]; then
    echo "  解压训练集..."
    unzip -o training/train_dataset.zip -d training/
fi

cd ../..

echo ""
echo "2. 检查模型..."
python scripts/check_models.py

echo ""
echo "3. 检查数据..."
if [ -f "data/processed/knowledge_base/docs_raw_chunks.json" ]; then
    echo "  ✓ Chunks 数据已就绪"
fi

if [ -f "data/processed/training/train_dataset.json" ]; then
    echo "  ✓ 训练集已就绪"
fi

echo ""
echo "=========================================="
echo "设置完成!"
echo "=========================================="
echo ""
echo "启动服务:"
echo "  bash start_all_services.sh"
echo ""
echo "监控服务:"
echo "  bash monitor_services.sh"
"""
    
    with open(server_setup, 'w', encoding='utf-8') as f:
        f.write(setup_content)
    
    # 设置执行权限
    os.chmod(upload_script, 0o755)
    os.chmod(server_setup, 0o755)
    
    logger.info(f"\n辅助脚本已创建:")
    logger.info(f"  - {upload_script.name}")
    logger.info(f"  - {server_setup.name}")
    
    return package_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="创建服务器上传包")
    parser.add_argument("--output-dir", type=str, default="./upload_package",
                       help="输出目录")
    
    args = parser.parse_args()
    
    package_path = create_upload_package(args.output_dir)
    
    print("\n" + "=" * 60)
    print("上传包创建完成!")
    print("=" * 60)
    print(f"\n文件位置: {package_path}")
    print("\n上传命令示例:")
    print(f"  scp {package_path.name} root@<服务器IP>:/mnt/workspace/")
    print("\n解压命令:")
    print(f"  cd /mnt/workspace && unzip {package_path.name} -d annual_report_assistant/")


if __name__ == "__main__":
    main()
