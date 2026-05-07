#!/usr/bin/env python3
"""
文档处理器 - 简化版
处理PDF文档，提取文本和关键信息
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️ pdfplumber未安装，PDF处理功能将不可用")


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict
    source: str


class TextSplitter:
    """文本切分器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """切分文本"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.text_splitter = TextSplitter()
    
    def process_pdf(self, pdf_path: str) -> Document:
        """处理PDF文件"""
        text = self._extract_text_from_pdf(pdf_path)
        
        metadata = {
            "source": pdf_path,
            "file_name": Path(pdf_path).name,
        }
        
        return Document(
            content=text,
            metadata=metadata,
            source=pdf_path
        )
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本"""
        if not HAS_PDFPLUMBER:
            print("⚠️ pdfplumber未安装，无法读取PDF")
            return ""
        
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:50]:  # 只读取前50页
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except:
                        continue
        except Exception as e:
            print(f"PDF读取错误: {e}")
        
        return text
    
    def extract_financial_data(self, text: str) -> Dict:
        """提取财务数据"""
        data = {}
        
        # 营业收入
        revenue_match = re.search(r'营业总收入[\s]*([\d,\.]+)[\s]*万元', text)
        if revenue_match:
            data['revenue'] = revenue_match.group(1)
        
        # 净利润
        profit_match = re.search(r'净利润[\s]*([\d,\.]+)[\s]*万元', text)
        if profit_match:
            data['net_profit'] = profit_match.group(1)
        
        return data
    
    def split_document(self, document: Document) -> List[Document]:
        """切分文档"""
        chunks = self.text_splitter.split_text(document.content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={**document.metadata, "chunk_index": i},
                source=document.source
            )
            documents.append(doc)
        
        return documents


# 便捷函数
def process_pdf(pdf_path: str) -> Document:
    """处理PDF文件"""
    processor = DocumentProcessor()
    return processor.process_pdf(pdf_path)


def split_text(text: str, chunk_size: int = 512) -> List[str]:
    """切分文本"""
    splitter = TextSplitter(chunk_size=chunk_size)
    return splitter.split_text(text)
