"""
统一配置管理系统
支持环境变量、配置文件、默认值多层配置
"""
import os
import yaml
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


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
    
    # 自定义扩展配置
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
                            logger.info("已加载配置文件: %s", path)
                            break
                except Exception as e:
                    logger.warning("配置文件加载失败 %s: %s", path, e)
        
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
        # 新格式: models.llm → model
        if "models" in data:
            models = data["models"]
            if "llm" in models:
                self._map_keys(models["llm"], config.model, {
                    "name": "llm_model_name",
                    "path": "llm_model_path",
                    "max_length": "max_new_tokens",
                })
            if "embedding" in models:
                self._map_keys(models["embedding"], config.model, {
                    "name": "embedding_model_name",
                    "path": "embedding_model_path",
                })
            # base_model 只有在 name 未设置时才生效
            llm = models.get("llm", {})
            if "base_model" in llm and "name" not in llm:
                config.model.llm_model_name = llm["base_model"]

        # 新格式: server → api, gpu → model.device
        if "server" in data:
            self._map_keys(data["server"], config.api,
                           {"host": "host", "port": "port", "workers": "workers"})
        if "gpu" in data:
            if "device" in data["gpu"]:
                config.model.device = data["gpu"]["device"]

        # 通用合并：model, api, crawler, logging, monitoring, cache
        for section, target in [
            ("model", config.model),
            ("api", config.api),
            ("crawler", config.crawler),
            ("logging", config.logging),
            ("monitoring", config.monitoring),
            ("cache", config.cache),
        ]:
            if section in data:
                for key, value in data[section].items():
                    if hasattr(target, key):
                        setattr(target, key, value)

        # RAG 配置：支持旧格式和新嵌套格式
        if "rag" in data:
            rag = data["rag"]
            # 新嵌套格式: rag.vector_store → config.rag
            for sub_section, key_map in [
                ("vector_store", {"type": "vector_store_type", "persist_directory": "vector_db_path",
                                  "collection_name": "collection_name", "distance_metric": "distance_metric"}),
                ("retrieval", {"top_k": "top_k", "similarity_threshold": "similarity_threshold",
                               "rerank_enabled": "use_reranker", "rerank_model": "reranker_model"}),
                ("chunking", {"chunk_size": "chunk_size", "chunk_overlap": "chunk_overlap"}),
            ]:
                if sub_section in rag:
                    self._map_keys(rag[sub_section], config.rag, key_map)
            # 旧格式: 直接映射
            for key, value in rag.items():
                if hasattr(config.rag, key):
                    setattr(config.rag, key, value)

        return config

    @staticmethod
    def _map_keys(source: dict, target: Any, key_map: Dict[str, str]):
        for src_key, dst_key in key_map.items():
            if src_key in source:
                setattr(target, dst_key, source[src_key])
    
    @property
    def config(self) -> AppConfig:
        """获取配置"""
        return self._config
    
    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()
        logger.info("配置已重新加载")


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
    logging.basicConfig(level=logging.INFO)
    config = get_config()
    logger.info("环境: %s", config.env)
    logger.info("LLM模型: %s", config.model.llm_model_name)
    logger.info("向量存储: %s", config.rag.vector_store_type)
    logger.info("API端口: %s", config.api.port)
