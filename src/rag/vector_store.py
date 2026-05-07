"""
向量数据库管理器
支持ChromaDB、FAISS等多种向量数据库
"""
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class BaseVectorStore:
    """向量数据库基类"""
    
    def __init__(self, collection_name: str, **kwargs):
        self.collection_name = collection_name
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档"""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        raise NotImplementedError
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB向量数据库实现"""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/vector_db",
        distance_metric: str = "cosine",
        **kwargs
    ):
        """
        初始化ChromaDB向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            distance_metric: 距离度量（cosine/l2/ip）
        """
        super().__init__(collection_name)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            
            logger.info(f"ChromaDB集合 '{collection_name}' 初始化完成")
            
        except ImportError:
            logger.error("请先安装chromadb: pip install chromadb")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
        """
        if not documents:
            return
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding.tolist() for doc in documents if doc.embedding is not None]
        metadatas = [doc.metadata for doc in documents]
        
        # 分批添加（避免单次请求过大）
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size] if embeddings else None
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_contents,
                embeddings=batch_embeddings if batch_embeddings else None,
                metadatas=batch_metadatas
            )
        
        logger.info(f"成功添加 {len(documents)} 个文档")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            (文档, 相似度分数) 列表
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_dict
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'] else {},
            )
            distance = results['distances'][0][i]
            # ChromaDB返回的是距离，转换为相似度
            similarity = 1 - distance
            documents.append((doc, similarity))
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        self.collection.delete(ids=ids)
        logger.info(f"删除 {len(ids)} 个文档")
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory)
        }


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量数据库实现（适用于大规模数据）"""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/vector_db",
        distance_metric: str = "cosine",
        **kwargs
    ):
        """
        初始化FAISS向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            distance_metric: 距离度量
        """
        super().__init__(collection_name)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.distance_metric = distance_metric
        
        self.index = None
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        
        # 尝试加载已有索引
        self._load_index()
    
    def _load_index(self):
        """加载已有索引"""
        index_path = self.persist_directory / f"{self.collection_name}.faiss"
        metadata_path = self.persist_directory / f"{self.collection_name}_metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.id_to_index = data['id_to_index']
                logger.info(f"加载FAISS索引: {len(self.id_to_index)} 个文档")
            except Exception as e:
                logger.warning(f"加载索引失败: {e}")
    
    def _save_index(self):
        """保存索引"""
        if self.index is not None:
            import faiss
            index_path = self.persist_directory / f"{self.collection_name}.faiss"
            metadata_path = self.persist_directory / f"{self.collection_name}_metadata.json"
            
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'id_to_index': self.id_to_index,
                    'distance_metric': self.distance_metric
                }, f)
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档"""
        if not documents:
            return
        
        import faiss
        
        # 获取embedding维度
        dim = len(documents[0].embedding)
        
        # 初始化索引
        if self.index is None:
            if self.distance_metric == "cosine":
                # 归一化向量后使用内积
                self.index = faiss.IndexFlatIP(dim)
            elif self.distance_metric == "l2":
                self.index = faiss.IndexFlatL2(dim)
            else:
                self.index = faiss.IndexFlatIP(dim)
        
        # 准备embedding
        embeddings = []
        for doc in documents:
            self.documents[doc.id] = doc
            embeddings.append(doc.embedding)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # 归一化（用于余弦相似度）
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # 添加到索引
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # 更新id映射
        for i, doc in enumerate(documents):
            self.id_to_index[doc.id] = start_idx + i
        
        self._save_index()
        logger.info(f"成功添加 {len(documents)} 个文档")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        import faiss
        
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # 归一化查询向量
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
        
        # 搜索
        distances, indices = self.index.search(query, top_k)
        
        results = []
        index_to_id = {v: k for k, v in self.id_to_index.items()}
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            doc_id = index_to_id.get(int(idx))
            if doc_id and doc_id in self.documents:
                doc = self.documents[doc_id]
                distance = distances[0][i]
                # 转换距离为相似度
                if self.distance_metric == "cosine":
                    similarity = float(distance)
                else:
                    similarity = 1 / (1 + float(distance))
                results.append((doc, similarity))
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """删除文档（FAISS不支持直接删除，需要重建索引）"""
        logger.warning("FAISS不支持直接删除，标记为已删除")
        for doc_id in ids:
            if doc_id in self.documents:
                self.documents[doc_id].metadata['deleted'] = True
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "index_total": self.index.ntotal if self.index else 0,
            "distance_metric": self.distance_metric
        }


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(
        self,
        store_type: str = "chroma",
        collection_name: str = "annual_reports",
        persist_directory: str = "./data/vector_db",
        distance_metric: str = "cosine",
        **kwargs
    ):
        """
        初始化向量存储管理器
        
        Args:
            store_type: 存储类型（chroma/faiss）
            collection_name: 集合名称
            persist_directory: 持久化目录
            distance_metric: 距离度量
        """
        self.store_type = store_type
        
        if store_type == "chroma":
            self.store = ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                distance_metric=distance_metric,
                **kwargs
            )
        elif store_type == "faiss":
            self.store = FAISSVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                distance_metric=distance_metric,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的存储类型: {store_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档"""
        self.store.add_documents(documents)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        return self.store.search(query_embedding, top_k, filter_dict)
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        self.store.delete(ids)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.store.get_collection_stats()


# 全局向量存储实例
_vector_store = None

def get_vector_store(
    store_type: str = "chroma",
    collection_name: str = "annual_reports",
    **kwargs
) -> VectorStoreManager:
    """
    获取全局向量存储实例
    
    Args:
        store_type: 存储类型
        collection_name: 集合名称
        **kwargs: 其他参数
        
    Returns:
        VectorStoreManager实例
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager(
            store_type=store_type,
            collection_name=collection_name,
            **kwargs
        )
    return _vector_store
