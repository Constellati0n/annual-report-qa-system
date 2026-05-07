#!/usr/bin/env python3
"""
本地文档处理脚本（并发版本）
- 解析 PDF
- 切分 chunk
- 多进程并发处理
- 保存为 JSON 格式
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import re
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import PyPDF2
    import pdfplumber
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("请先安装依赖: pip install PyPDF2 pdfplumber langchain")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取文本"""
    text = ""
    try:
        # 使用 pdfplumber（效果更好）
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        # 备用：使用 PyPDF2
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception:
            pass
    
    return text


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """将文本切分成 chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    
    return text_splitter.split_text(text)


def parse_pdf_info(pdf_file: Path) -> Dict[str, str]:
    """从 PDF 文件名解析信息"""
    parts = pdf_file.stem.split("__")
    stock_code = parts[0] if len(parts) > 0 else ""
    year = ""
    company_name = ""
    
    if len(parts) > 1:
        year_match = re.search(r'(\d{4})', parts[1])
        if year_match:
            year = year_match.group(1)
        company_name = parts[1].replace(f"{year}年年度报告", "").strip()
    
    return {
        "stock_code": stock_code,
        "year": year,
        "company_name": company_name or stock_code
    }


def process_single_pdf(pdf_file: Path) -> List[Dict[str, Any]]:
    """处理单个 PDF 文件（用于多进程）"""
    try:
        # 解析文件名信息
        info = parse_pdf_info(pdf_file)
        
        # 提取文本
        text = extract_text_from_pdf(str(pdf_file))
        if not text.strip():
            return []
        
        # 切分 chunks
        chunks = split_text_into_chunks(text)
        
        # 构建文档列表
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": hashlib.md5(f"{pdf_file}_{i}".encode()).hexdigest(),
                "content": chunk,
                "metadata": {
                    "source": pdf_file.name,
                    "company_name": info["company_name"],
                    "year": info["year"],
                    "stock_code": info["stock_code"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            documents.append(doc)
        
        print(f"✓ {pdf_file.name}: {len(chunks)} chunks")
        return documents
        
    except Exception as e:
        print(f"✗ {pdf_file.name}: {e}")
        return []


def process_directory_parallel(input_dir: str, output_dir: str, num_workers: int = None, limit: int = None):
    """并发处理整个目录"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 PDF 文件
    pdf_files = list(input_path.rglob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    total_files = len(pdf_files)
    print(f"找到 {total_files} 个 PDF 文件")
    print(f"使用 {num_workers or cpu_count()} 个进程并发处理\n")
    
    # 使用进程池并发处理
    all_documents = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_pdf, pdf_files)
        
        # 合并结果
        for docs in results:
            all_documents.extend(docs)
    
    # 保存为 JSON
    output_file = output_path / "processed_documents.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"  处理文件: {total_files}")
    print(f"  总 chunks: {len(all_documents)}")
    print(f"  输出文件: {output_file}")
    print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return all_documents


def main():
    parser = argparse.ArgumentParser(description="本地处理年报 PDF 文档（并发版）")
    parser.add_argument("--input", "-i", default="data/raw",
                       help="输入目录（包含 PDF 文件）")
    parser.add_argument("--output", "-o", default="data/processed_for_upload",
                       help="输出目录")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="并发进程数（默认使用所有 CPU 核心）")
    parser.add_argument("--limit", "-l", type=int, default=None,
                       help="限制处理文件数量（用于测试）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 处理文档
    process_directory_parallel(
        args.input,
        args.output,
        num_workers=args.workers,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
