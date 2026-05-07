"""
Embedding模型实现
支持缓存、批处理、异步编码
"""
import torch
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import hashlib
import json

from .interfaces import IEmbedding, Document
from ..core.config import get_rag_config, get_model_config
from ..core.exceptions import EmbeddingException, ErrorCode
from ..core.logger import get_logger, log_execution_time
from ..core.retry import retry, get_circuit_breaker

logger = get_logger(__name__)


class EmbeddingCache:
    """Embedding缓存管理器"""
    
    def __init__(self, cache_dir: str = "./cache/embeddings", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._memory_cache: dict = {}
        self._access_count: dict = {}
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        key = f"{model_name}:{text}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """从缓存获取embedding"""
        cache_key = self._get_cache_key(text, model_name)
        
        # 先检查内存缓存
        if cache_key in self._memory_cache:
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._memory_cache[cache_key]
        
        # 再检查磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                # 放入内存缓存
                self._put_to_memory(cache_key, embedding)
                return embedding
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """保存embedding到缓存"""
        cache_key = self._get_cache_key(text, model_name)
        
        # 保存到内存
        self._put_to_memory(cache_key, embedding)
        
        # 保存到磁盘
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _put_to_memory(self, cache_key: str, embedding: np.ndarray):
        """放入内存缓存（LRU策略）"""
        # 清理过期缓存
        if len(self._memory_cache) >= self.max_size:
            # 移除最少访问的
            min_key = min(self._access_count, key=self._access_count.get)
            del self._memory_cache[min_key]
            del self._access_count[min_key]
        
        self._memory_cache[cache_key] = embedding
        self._access_count[cache_key] = 1
    
    def clear(self):
        """清空缓存"""
        self._memory_cache.clear()
        self._access_count.clear()
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingManager(IEmbedding):
    """Embedding模型管理器（改进版）"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_cache: bool = True,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        """
        初始化Embedding管理器
        
        Args:
            model_name: HuggingFace模型名称
            model_path: 本地模型路径（优先使用）
            device: 计算设备
            use_cache: 是否使用缓存
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        model_config = get_model_config()
        rag_config = get_rag_config()
        
        # 优先使用本地路径
        self.model_name = model_path or model_config.embedding_model_path or model_name or model_config.embedding_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 调试日志
        logger.info(f"Embedding配置: model_path={model_path}, config_path={model_config.embedding_model_path}, config_name={model_config.embedding_model_name}")
        logger.info(f"最终使用模型路径/名称: {self.model_name}")
        self.batch_size = batch_size or rag_config.batch_size
        self.max_length = max_length or rag_config.max_length
        
        self.model = None
        self.tokenizer = None
        self._embedding_dim: Optional[int] = None
        
        # 初始化缓存
        self.cache = EmbeddingCache() if use_cache else None
        
        # 熔断器
        self.circuit_breaker = get_circuit_breaker(
            "embedding",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        self._load_model()
    
    @retry(
        max_attempts=3,
        retry_exceptions=(Exception,),
        circuit_breaker=get_circuit_breaker("embedding_load")
    )
    def _load_model(self):
        """加载模型和分词器（带重试）"""
        import os
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"加载Embedding模型: {self.model_name}")
            
            # 检查本地路径是否存在
            if os.path.exists(self.model_name):
                logger.info(f"使用本地模型路径: {self.model_name}")
            else:
                logger.warning(f"本地模型路径不存在: {self.model_name}")
                # 如果路径不存在且看起来像本地路径，给出明确错误
                if self.model_name.startswith('/'):
                    raise EmbeddingException(
                        f"Embedding模型路径不存在: {self.model_name}. 请先下载模型到该路径，或配置正确的模型路径。",
                        ErrorCode.MODEL_LOAD_ERROR
                    )
            
            # 尝试加载本地模型
            try:
                # 加载分词器 - Qwen3-Embedding 使用 trust_remote_code
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # 加载模型 - Qwen3-Embedding 支持
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    local_files_only=True
                )
                
                if self.device == "cuda":
                    self.model = self.model.to('cuda')
                    
            except Exception as e:
                logger.error(f"本地模型加载失败: {e}")
                raise EmbeddingException(
                    f"Embedding模型加载失败: {e}",
                    ErrorCode.MODEL_LOAD_ERROR
                )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # 获取embedding维度
            self._embedding_dim = len(self.encode(["测试文本"])[0])
            
            logger.info(f"模型加载完成，Embedding维度: {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise EmbeddingException(
                f"Embedding模型加载失败: {e}",
                ErrorCode.MODEL_LOAD_ERROR
            )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        use_cache: Optional[bool] = None
    ) -> np.ndarray:
        """
        将文本编码为embedding向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
            use_cache: 是否使用缓存
            
        Returns:
            embedding向量数组
        """
        if self.model is None or self.tokenizer is None:
            raise EmbeddingException("模型未加载", ErrorCode.MODEL_LOAD_ERROR)
        
        # 确保是列表
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # 检查熔断器
        if not self.circuit_breaker.can_execute():
            raise EmbeddingException("服务熔断中", ErrorCode.MODEL_INFERENCE_ERROR)
        
        use_cache = use_cache if use_cache is not None else (self.cache is not None)
        batch_size = batch_size or self.batch_size
        
        # 检查缓存
        if use_cache and self.cache:
            embeddings = []
            texts_to_encode = []
            indices_to_encode = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
            
            if texts_to_encode:
                # 编码未缓存的文本
                new_embeddings = self._encode_batch(texts_to_encode, batch_size, show_progress)
                
                # 保存到缓存
                for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                    self.cache.set(text, self.model_name, emb)
                    embeddings.append((idx, emb))
            
            # 按原始顺序排序
            embeddings.sort(key=lambda x: x[0])
            result = np.vstack([emb for _, emb in embeddings])
        else:
            result = self._encode_batch(texts, batch_size, show_progress)
        
        # 记录成功
        self.circuit_breaker.record_success()
        
        if single_input:
            return result[0]
        return result
    
    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> np.ndarray:
        """批量编码（内部方法）"""
        all_embeddings = []
        
        # 进度条
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(texts), batch_size)
        
        try:
            with torch.no_grad():
                for i in iterator:
                    batch_texts = texts[i:i + batch_size]
                    
                    # 编码
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # 获取模型输出
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
                    embeddings = sum_embeddings / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                    
                    # 归一化
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu().numpy())
        
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise EmbeddingException(f"编码失败: {e}", ErrorCode.EMBEDDING_ERROR)
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码查询文本"""
        if isinstance(queries, str):
            queries = f"查询: {queries}"
        else:
            queries = [f"查询: {q}" for q in queries]
        
        return self.encode(queries, **kwargs)
    
    def encode_documents(self, documents: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文档文本"""
        if isinstance(documents, str):
            documents = f"文档: {documents}"
        else:
            documents = [f"文档: {d}" for d in documents]
        
        return self.encode(documents, **kwargs)
    
    def get_embedding_dim(self) -> int:
        """获取embedding维度"""
        if self._embedding_dim is None:
            raise EmbeddingException("模型未初始化", ErrorCode.MODEL_LOAD_ERROR)
        return self._embedding_dim
    
    def close(self):
        """释放资源"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Embedding模型资源已释放")


# 工厂函数
def create_embedding_manager(
    model_name: Optional[str] = None,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingManager:
    """创建Embedding管理器"""
    return EmbeddingManager(
        model_name=model_name,
        use_cache=use_cache,
        **kwargs
    )


# 全局实例（单例模式）
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """获取全局Embedding管理器实例"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = create_embedding_manager()
    return _embedding_manager
