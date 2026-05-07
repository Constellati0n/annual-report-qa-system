"""
统一配置管理系统
支持环境变量、配置文件、默认值多层配置
"""
import os
import yaml
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache


@dataclass
class ModelConfig:
    """模型配置"""
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_model_path: Optional[str] = None
    embedding_model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_model_path: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"  # auto, cuda, cpu


@dataclass
class RAGConfig:
    """RAG配置"""
    vector_store_type: str = "chroma"  # chroma, faiss
    vector_db_path: str = "./data/vector_db"
    collection_name: str = "annual_reports"
    distance_metric: str = "cosine"
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.5
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    max_length: int = 512


@dataclass
class APIConfig:
    """API服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # 限流配置
    rate_limit: int = 100  # 每分钟请求数
    rate_limit_burst: int = 10
    
    # 超时配置
    request_timeout: int = 120  # 秒
    
    # CORS配置
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    
    # 认证配置
    api_key_header: str = "X-API-Key"
    api_key: Optional[str] = None
    
    # 上传配置
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    upload_dir: str = "./uploads"


@dataclass
class CrawlerConfig:
    """爬虫配置"""
    base_url: str = "http://www.cninfo.com.cn"
    request_timeout: int = 30
    retry_times: int = 3
    retry_delay: float = 1.0
    delay_min: float = 0.5
    delay_max: float = 2.0
    max_concurrent: int = 5
    raw_data_dir: str = "./data/raw"
    start_year: int = 2020
    end_year: int = 2024


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./logs/app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # 结构化日志
    json_format: bool = False
    
    # 日志采样
    sampling_rate: float = 1.0


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # 健康检查
    health_check_interval: int = 30
    
    # 告警阈值
    latency_p99_threshold: float = 5.0  # 秒
    error_rate_threshold: float = 0.01  # 1%


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    cache_dir: str = "./cache"
    embedding_cache_size: int = 10000  # 最大缓存条目数
    ttl: int = 3600  # 秒


@dataclass
class ModelsConfig:
    """模型配置（新格式兼容）"""
    llm: Dict[str, Any] = field(default_factory=dict)
    embedding: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    type: str = "chroma"
    persist_directory: str = "./data/vector_db"
    collection_name: str = "annual_reports"
    distance_metric: str = "cosine"


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"


@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 512
    chunk_overlap: int = 128
    separator: str = "\n"


@dataclass
class RAGConfigNew:
    """RAG配置（新格式）"""
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass
class AppConfig:
    """应用主配置"""
    env: str = "development"  # development, testing, production
    debug: bool = False
    
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    api: APIConfig = field(default_factory=APIConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # 新格式配置
    models: ModelsConfig = field(default_factory=ModelsConfig)
    rag_new: RAGConfigNew = field(default_factory=RAGConfigNew)
    
    # 自定义配置
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """配置管理器"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """加载配置（优先级：环境变量 > 配置文件 > 默认值）"""
        config = AppConfig()
        
        # 1. 从配置文件加载
        config = self._load_from_file(config)
        
        # 2. 从环境变量加载（覆盖配置文件）
        config = self._load_from_env(config)
        
        return config
    
    def _load_from_file(self, config: AppConfig) -> AppConfig:
        """从YAML配置文件加载"""
        config_paths = [
            os.getenv("CONFIG_PATH"),
            "./config/config.yaml",
            "./config.yaml",
        ]
        
        for path in config_paths:
            if path and Path(path).exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if data:
                            config = self._merge_config(config, data)
                            print(f"✓ 加载配置文件: {path}")
                            break
                except Exception as e:
                    print(f"⚠ 配置文件加载失败 {path}: {e}")
        
        return config
    
    def _load_from_env(self, config: AppConfig) -> AppConfig:
        """从环境变量加载配置"""
        # 模型配置
        if os.getenv("LLM_MODEL_NAME"):
            config.model.llm_model_name = os.getenv("LLM_MODEL_NAME")
        if os.getenv("LLM_MODEL_PATH"):
            config.model.llm_model_path = os.getenv("LLM_MODEL_PATH")
        if os.getenv("EMBEDDING_MODEL_NAME"):
            config.model.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        if os.getenv("LOAD_IN_4BIT"):
            config.model.load_in_4bit = os.getenv("LOAD_IN_4BIT").lower() == "true"
        
        # RAG配置
        if os.getenv("VECTOR_DB_PATH"):
            config.rag.vector_db_path = os.getenv("VECTOR_DB_PATH")
        if os.getenv("VECTOR_STORE_TYPE"):
            config.rag.vector_store_type = os.getenv("VECTOR_STORE_TYPE")
        
        # API配置
        if os.getenv("API_HOST"):
            config.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))
        if os.getenv("API_KEY"):
            config.api.api_key = os.getenv("API_KEY")
        
        # 环境配置
        if os.getenv("APP_ENV"):
            config.env = os.getenv("APP_ENV")
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"
        
        return config
    
    def _merge_config(self, config: AppConfig, data: Dict) -> AppConfig:
        """合并配置数据（支持新旧两种配置格式）"""
        # 处理新的配置格式 (models, server, gpu 等)
        if "models" in data:
            models = data["models"]
            if "llm" in models:
                llm = models["llm"]
                if "name" in llm:
                    config.model.llm_model_name = llm["name"]
                if "path" in llm:
                    config.model.llm_model_path = llm["path"]
                if "base_model" in llm:
                    config.model.llm_model_name = llm["base_model"]
                if "load_in_4bit" in llm:
                    config.model.load_in_4bit = llm["load_in_4bit"]
                if "max_length" in llm:
                    config.model.max_new_tokens = llm["max_length"]
                if "temperature" in llm:
                    config.model.temperature = llm["temperature"]
                if "top_p" in llm:
                    config.model.top_p = llm["top_p"]
            if "embedding" in models:
                emb = models["embedding"]
                if "name" in emb:
                    config.model.embedding_model_name = emb["name"]
                if "path" in emb:
                    config.model.embedding_model_path = emb["path"]
                if "device" in emb:
                    config.model.device = emb["device"]
        
        # 处理 server 配置
        if "server" in data:
            server = data["server"]
            if "host" in server:
                config.api.host = server["host"]
            if "port" in server:
                config.api.port = server["port"]
            if "workers" in server:
                config.api.workers = server["workers"]
        
        # 处理 gpu 配置
        if "gpu" in data:
            gpu = data["gpu"]
            if "device" in gpu:
                config.model.device = gpu["device"]
        
        # 处理旧的配置格式
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # 处理 RAG 配置（新旧格式）
        if "rag" in data:
            rag = data["rag"]
            if "vector_store" in rag:
                vs = rag["vector_store"]
                if "type" in vs:
                    config.rag.vector_store_type = vs["type"]
                if "persist_directory" in vs:
                    config.rag.vector_db_path = vs["persist_directory"]
                if "collection_name" in vs:
                    config.rag.collection_name = vs["collection_name"]
                if "distance_metric" in vs:
                    config.rag.distance_metric = vs["distance_metric"]
            if "retrieval" in rag:
                ret = rag["retrieval"]
                if "top_k" in ret:
                    config.rag.top_k = ret["top_k"]
                if "similarity_threshold" in ret:
                    config.rag.similarity_threshold = ret["similarity_threshold"]
                if "rerank_enabled" in ret:
                    config.rag.use_reranker = ret["rerank_enabled"]
                if "rerank_model" in ret:
                    config.rag.reranker_model = ret["rerank_model"]
            if "chunking" in rag:
                chunk = rag["chunking"]
                if "chunk_size" in chunk:
                    config.rag.chunk_size = chunk["chunk_size"]
                if "chunk_overlap" in chunk:
                    config.rag.chunk_overlap = chunk["chunk_overlap"]
            # 旧格式
            for key, value in rag.items():
                if hasattr(config.rag, key):
                    setattr(config.rag, key, value)
        
        if "api" in data:
            for key, value in data["api"].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        if "crawler" in data:
            for key, value in data["crawler"].items():
                if hasattr(config.crawler, key):
                    setattr(config.crawler, key, value)
        
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        if "monitoring" in data:
            for key, value in data["monitoring"].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        if "cache" in data:
            for key, value in data["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)
        
        return config
    
    @property
    def config(self) -> AppConfig:
        """获取配置"""
        return self._config
    
    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()
        print("✓ 配置已重新加载")


# 全局配置访问函数
@lru_cache()
def get_config() -> AppConfig:
    """获取全局配置实例"""
    manager = ConfigManager()
    return manager.config


# 便捷访问函数
def get_model_config() -> ModelConfig:
    return get_config().model


def get_rag_config() -> RAGConfig:
    return get_config().rag


def get_api_config() -> APIConfig:
    return get_config().api


def get_crawler_config() -> CrawlerConfig:
    return get_config().crawler


def get_logging_config() -> LoggingConfig:
    """获取日志配置"""
    return get_config().logging


if __name__ == "__main__":
    # 测试配置加载
    config = get_config()
    print(f"\n环境: {config.env}")
    print(f"LLM模型: {config.model.llm_model_name}")
    print(f"向量存储: {config.rag.vector_store_type}")
    print(f"API端口: {config.api.port}")
