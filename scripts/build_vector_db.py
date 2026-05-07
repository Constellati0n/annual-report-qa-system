"""
构建RAG向量数据库
将爬虫获取的年报数据导入向量数据库
"""
import os
import sys
from pathlib import Path
from typing import Optional
import argparse
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import get_embedding_manager, get_vector_store, Document
from src.rag.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vector_database(
    data_dir: str,
    vector_db_path: str = "./data/vector_db",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    batch_size: int = 32
):
    """
    构建向量数据库
    
    Args:
        data_dir: 年报数据目录
        vector_db_path: 向量数据库保存路径
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
        batch_size: 批处理大小
    """
    logger.info("=" * 60)
    logger.info("开始构建向量数据库")
    logger.info("=" * 60)
    
    # 初始化组件
    logger.info("初始化组件...")
    embedding_manager = get_embedding_manager()
    vector_store = get_vector_store(persist_directory=vector_db_path)
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 处理文档
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    logger.info(f"处理目录: {data_dir}")
    
    # 获取所有PDF文件
    pdf_files = list(data_path.rglob("*.pdf"))
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    if not pdf_files:
        logger.warning("未找到PDF文件，请检查数据目录")
        return
    
    # 处理每个文件
    total_chunks = 0
    total_documents = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] 处理: {pdf_file.name}")
        
        try:
            # 从文件名提取元数据
            metadata = extract_metadata_from_filename(pdf_file.name)
            
            # 处理PDF
            chunks = processor.process_pdf(pdf_file, metadata)
            
            if not chunks:
                logger.warning(f"  未提取到文本: {pdf_file.name}")
                continue
            
            logger.info(f"  提取 {len(chunks)} 个文本块")
            
            # 编码文本块
            texts = [chunk.content for chunk in chunks]
            embeddings = embedding_manager.encode(
                texts,
                batch_size=batch_size,
                show_progress=True
            )
            
            # 创建文档对象
            documents = []
            for j, chunk in enumerate(chunks):
                doc = Document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=embeddings[j]
                )
                documents.append(doc)
            
            # 添加到向量库
            vector_store.add_documents(documents)
            
            total_chunks += len(chunks)
            total_documents += 1
            
            logger.info(f"  成功添加 {len(documents)} 个文档")
            
        except Exception as e:
            logger.error(f"  处理失败: {e}")
            continue
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("向量数据库构建完成")
    logger.info("=" * 60)
    logger.info(f"处理文件数: {total_documents}")
    logger.info(f"总文本块数: {total_chunks}")
    
    stats = vector_store.get_stats()
    logger.info(f"向量库统计: {stats}")


def extract_metadata_from_filename(filename: str) -> dict:
    """
    从文件名提取元数据
    
    Args:
        filename: 文件名
        
    Returns:
        元数据字典
    """
    import re
    
    metadata = {"source": filename}
    
    # 提取股票代码（6位数字）
    stock_match = re.search(r'(\d{6})', filename)
    if stock_match:
        metadata["stock_code"] = stock_match.group(1)
    
    # 提取年份
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        metadata["year"] = year_match.group(1)
    
    # 提取公司名称（简化处理）
    # 实际应用中可以使用更复杂的NER或规则
    company_patterns = [
        r'([^_]+?)_\d{6}',  # 公司名_股票代码
        r'(\d{6})_([^_]+)',  # 股票代码_公司名
    ]
    
    for pattern in company_patterns:
        match = re.search(pattern, filename)
        if match:
            company_name = match.group(1 if '[^_]+' in pattern.split('_')[0] else 2)
            metadata["company_name"] = company_name
            break
    
    return metadata


def update_vector_database(
    data_dir: str,
    vector_db_path: str = "./data/vector_db",
    since: Optional[str] = None
):
    """
    增量更新向量数据库
    
    Args:
        data_dir: 年报数据目录
        vector_db_path: 向量数据库路径
        since: 只处理指定日期之后的文件
    """
    logger.info("增量更新向量数据库...")
    
    # 这里可以实现增量更新逻辑
    # 例如检查文件修改时间，只处理新文件
    
    build_vector_database(data_dir, vector_db_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建RAG向量数据库")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="年报数据目录")
    parser.add_argument("--vector-db-path", type=str, default="./data/vector_db",
                       help="向量数据库保存路径")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="文本分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=128,
                       help="分块重叠大小")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批处理大小")
    parser.add_argument("--update", action="store_true",
                       help="增量更新模式")
    
    args = parser.parse_args()
    
    if args.update:
        update_vector_database(
            data_dir=args.data_dir,
            vector_db_path=args.vector_db_path
        )
    else:
        build_vector_database(
            data_dir=args.data_dir,
            vector_db_path=args.vector_db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
