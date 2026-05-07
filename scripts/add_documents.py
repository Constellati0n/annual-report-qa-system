#!/usr/bin/env python3
"""
文档添加到知识库脚本
支持 PDF、TXT、Markdown 等格式
自动切分 chunk 并生成 embeddings
"""
import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/mnt/workspace/annual_report_assistant')

from src.core.assistant import AnnualReportAssistant


def add_single_file(file_path: str, company_name: str = None, year: str = None, stock_code: str = None):
    """添加单个文件到知识库"""
    print(f"添加文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 构建元数据
    metadata = {}
    if company_name:
        metadata["company_name"] = company_name
    if year:
        metadata["year"] = year
    if stock_code:
        metadata["stock_code"] = stock_code
    metadata["source"] = os.path.basename(file_path)
    
    print(f"元数据: {metadata}")
    
    try:
        # 初始化助手（只初始化 RAG 组件，不加载 LLM）
        assistant = AnnualReportAssistant(
            model_path=None,  # 不加载 LLM，只使用 RAG
            base_model=None,
            use_rag=True,
            load_in_4bit=False
        )
        
        # 添加文档
        assistant.add_documents_to_knowledge_base(
            documents_path=file_path,
            metadata=metadata if metadata else None
        )
        
        print(f"✅ 文件添加成功: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 添加失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_directory(dir_path: str):
    """添加整个目录到知识库"""
    print(f"添加目录: {dir_path}")
    
    if not os.path.exists(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        return False
    
    # 支持的文件类型
    supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
    
    # 遍历目录
    files = []
    for ext in supported_extensions:
        files.extend(Path(dir_path).glob(f"**/*{ext}"))
    
    if not files:
        print(f"⚠️ 目录中没有支持的文件类型: {supported_extensions}")
        return False
    
    print(f"找到 {len(files)} 个文件")
    
    # 逐个添加
    success_count = 0
    for file_path in files:
        # 尝试从文件名解析公司和年份
        filename = file_path.stem
        company_name = None
        year = None
        
        # 简单的文件名解析（可根据实际情况调整）
        if '_' in filename:
            parts = filename.split('_')
            company_name = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                year = parts[1]
        
        if add_single_file(str(file_path), company_name, year):
            success_count += 1
    
    print(f"\n总结: {success_count}/{len(files)} 个文件添加成功")
    return success_count > 0


def check_knowledge_base():
    """检查知识库状态"""
    try:
        # 直接使用向量存储，不加载 LLM
        sys.path.insert(0, '/mnt/workspace/annual_report_assistant')
        from src.rag.vector_store import ChromaVectorStore
        from src.rag.embedding import EmbeddingManager
        import torch

        # 初始化嵌入生成器
        embedding_generator = EmbeddingManager(
            model_name="BAAI/bge-large-zh-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 初始化向量存储
        vector_store = ChromaVectorStore(
            collection_name="annual_reports",
            persist_directory="./data/vector_db"
        )

        stats = vector_store.get_collection_stats()
        print("知识库状态:")
        print(f"  文档数量: {stats.get('document_count', 0)}")
        print(f"  集合名称: annual_reports")
        print(f"  存储路径: ./data/vector_db")
        return stats
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="添加文档到知识库")
    parser.add_argument("path", nargs="?", help="文件或目录路径")
    parser.add_argument("--company", "-c", help="公司名称")
    parser.add_argument("--year", "-y", help="年份")
    parser.add_argument("--stock-code", "-s", help="股票代码")
    parser.add_argument("--check", action="store_true", help="检查知识库状态")

    args = parser.parse_args()

    if args.check:
        check_knowledge_base()
        return

    if not args.path:
        print("请提供文件或目录路径")
        parser.print_help()
        return

    if os.path.isfile(args.path):
        # 添加单个文件
        add_single_file(args.path, args.company, args.year, args.stock_code)
    elif os.path.isdir(args.path):
        # 添加整个目录
        add_directory(args.path)
    else:
        print(f"❌ 路径不存在: {args.path}")

    # 最后检查知识库状态
    print("\n" + "="*60)
    check_knowledge_base()


if __name__ == "__main__":
    main()
