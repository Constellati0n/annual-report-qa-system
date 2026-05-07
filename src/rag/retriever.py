"""
RAG检索器
整合向量检索、重排序等功能
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .vector_store import VectorStoreManager, Document, get_vector_store
from .embedding import EmbeddingManager, get_embedding_manager
from ..core.config import get_rag_config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int


class Reranker:
    """重排序器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        batch_size: int = 8
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
            device: 计算设备
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"加载重排序模型: {self.model_name}")
            
            # Qwen3-Reranker 使用 trust_remote_code
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("重排序模型加载完成")
            
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序文档列表
            top_k: 返回前k个结果
            
        Returns:
            排序后的(文档, 分数)列表
        """
        if not documents:
            return []
        
        import torch
        
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                
                # 构建输入对
                pairs = [[query, doc.content] for doc in batch_docs]
                
                # 编码
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 预测分数
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores)
        
        # 排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores


class RAGRetriever:
    """RAG检索器"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        reranker: Optional[Reranker] = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_reranker: bool = True
    ):
        """
        初始化RAG检索器
        
        Args:
            vector_store: 向量存储管理器
            embedding_manager: Embedding管理器
            reranker: 重排序器
            top_k: 向量检索返回数量
            rerank_top_k: 重排序后返回数量
            similarity_threshold: 相似度阈值
            use_reranker: 是否使用重排序
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold
        self.use_reranker = use_reranker
        
        # 如果启用重排序但没有提供reranker，则初始化
        if self.use_reranker and self.reranker is None:
            try:
                # 从配置获取模型路径
                rag_config = get_rag_config()
                rerank_model = getattr(rag_config, 'reranker_model', 'BAAI/bge-reranker-base')
                self.reranker = Reranker(model_name=rerank_model)
            except Exception as e:
                logger.warning(f"重排序器初始化失败，将不使用重排序: {e}")
                self.use_reranker = False
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        top_k = top_k or self.rerank_top_k
        
        # 1. 编码查询
        logger.debug(f"编码查询: {query[:50]}...")
        query_embedding = self.embedding_manager.encode_queries(query)
        
        # 2. 向量检索
        logger.debug(f"向量检索，top_k={self.top_k}")
        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_dict=filter_dict
        )
        
        # 过滤低相似度结果
        initial_results = [
            (doc, score) for doc, score in initial_results
            if score >= self.similarity_threshold
        ]
        
        if not initial_results:
            logger.debug("未找到相似文档")
            return []
        
        # 3. 重排序（如果启用）
        if self.use_reranker and self.reranker:
            logger.debug("执行重排序")
            documents = [doc for doc, _ in initial_results]
            reranked_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.rerank_top_k
            )
        else:
            reranked_results = initial_results[:self.rerank_top_k]
        
        # 4. 构建结果
        results = []
        for rank, (doc, score) in enumerate(reranked_results, 1):
            results.append(RetrievalResult(
                document=doc,
                score=score,
                rank=rank
            ))
        
        logger.info(f"检索完成，返回 {len(results)} 个结果")
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        context_window: int = 1
    ) -> List[RetrievalResult]:
        """
        检索并获取上下文
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件
            context_window: 上下文窗口大小（前后各取n个chunk）
            
        Returns:
            带上下文的检索结果
        """
        results = self.retrieve(query, filter_dict)
        
        # 如果需要上下文，可以在这里扩展
        # 例如根据chunk_id获取前后相邻的chunk
        
        return results
    
    def format_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        格式化检索结果为上下文字符串
        
        Args:
            results: 检索结果列表
            include_metadata: 是否包含元数据
            
        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "未找到相关参考资料。"
        
        context_parts = []
        
        for result in results:
            doc = result.document
            
            # 构建引用信息
            source_info = f"[来源: {doc.metadata.get('source', '未知')}"
            if 'page' in doc.metadata:
                source_info += f", 页码: {doc.metadata['page']}"
            if 'company_name' in doc.metadata:
                source_info += f", 公司: {doc.metadata['company_name']}"
            if 'year' in doc.metadata:
                source_info += f", 年份: {doc.metadata['year']}"
            source_info += "]"
            
            # 构建内容
            content = doc.content.strip()
            
            # 组合
            part = f"{source_info}\n{content}\n"
            context_parts.append(part)
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_dim": self.embedding_manager.get_embedding_dim(),
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "similarity_threshold": self.similarity_threshold,
            "use_reranker": self.use_reranker
        }


# 全局检索器实例
_retriever = None

def get_retriever(
    vector_store: Optional[VectorStoreManager] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
) -> RAGRetriever:
    """
    获取全局检索器实例
    
    Args:
        vector_store: 向量存储管理器
        embedding_manager: Embedding管理器
        **kwargs: 其他参数
        
    Returns:
        RAGRetriever实例
    """
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            **kwargs
        )
    return _retriever
