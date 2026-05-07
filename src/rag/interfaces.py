"""
RAG模块接口定义
使用抽象基类定义接口，便于替换实现
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Protocol
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int


class IVectorStore(ABC):
    """向量存储接口"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        pass


class IEmbedding(ABC):
    """Embedding模型接口"""
    
    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """编码文本"""
        pass
    
    @abstractmethod
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """编码查询"""
        pass
    
    @abstractmethod
    def encode_documents(self, documents: List[str], **kwargs) -> np.ndarray:
        """编码文档"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """获取embedding维度"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """释放资源"""
        pass


class IReranker(ABC):
    """重排序器接口"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """对文档进行重排序"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """释放资源"""
        pass


class IRetriever(ABC):
    """检索器接口"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def format_context(self, results: List[RetrievalResult]) -> str:
        """格式化检索结果为上下文"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class IDocumentProcessor(ABC):
    """文档处理器接口"""
    
    @abstractmethod
    def process_pdf(self, file_path: str, metadata: Optional[Dict] = None) -> List[Document]:
        """处理PDF文件"""
        pass
    
    @abstractmethod
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """处理纯文本"""
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分块"""
        pass
