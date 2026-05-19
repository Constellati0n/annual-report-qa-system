import os
import tempfile
from pathlib import Path

import pytest

from src.core.config import (
    AppConfig,
    ModelConfig,
    RAGConfig,
    APIConfig,
    CrawlerConfig,
    LoggingConfig,
    MonitoringConfig,
    CacheConfig,
    ConfigManager,
)


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.llm_model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.llm_model_path is None
        assert config.load_in_4bit is True
        assert config.max_new_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.device == "auto"


class TestRAGConfig:
    def test_default_values(self):
        config = RAGConfig()
        assert config.vector_store_type == "chroma"
        assert config.collection_name == "annual_reports"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.top_k == 10
        assert config.use_reranker is True


class TestAPIConfig:
    def test_default_values(self):
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.rate_limit == 100
        assert config.request_timeout == 120
        assert config.cors_origins == ["*"]
        assert config.max_upload_size == 100 * 1024 * 1024


class TestAppConfig:
    def test_default_values(self):
        config = AppConfig()
        assert config.env == "development"
        assert config.debug is False
        assert config.model is not None
        assert config.rag is not None
        assert config.api is not None


class TestConfigManager:
    def test_singleton(self):
        m1 = ConfigManager()
        m2 = ConfigManager()
        assert m1 is m2

    def test_load_default_config(self, temp_dir):
        config_path = temp_dir / "empty_config.yaml"
        config_path.write_text("models: {}", encoding="utf-8")
        os.environ["CONFIG_PATH"] = str(config_path)
        manager = ConfigManager()
        manager._config = None
        manager._config = manager._load_config()
        assert manager.config.env == "development"
        assert manager.config.model.llm_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_load_from_env(self, temp_dir):
        os.environ["API_PORT"] = "9999"
        os.environ["APP_ENV"] = "production"
        os.environ["LLM_MODEL_NAME"] = "custom-model"

        manager = ConfigManager()
        manager._config = None
        manager._config = manager._load_config()

        assert manager.config.api.port == 9999
        assert manager.config.env == "production"
        assert manager.config.model.llm_model_name == "custom-model"

    def test_load_from_yaml_file(self, temp_dir, sample_config_yaml):
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(sample_config_yaml, encoding="utf-8")

        os.environ["CONFIG_PATH"] = str(config_path)

        manager = ConfigManager()
        manager._config = None
        manager._config = manager._load_config()

        assert manager.config.model.llm_model_name == "test-model"
        assert manager.config.rag.vector_store_type == "chroma"
        assert manager.config.rag.top_k == 3
        assert manager.config.rag.chunk_size == 256
        assert manager.config.api.host == "0.0.0.0"
        assert manager.config.api.port == 8000

    def test_env_overrides_file(self, temp_dir, sample_config_yaml):
        config_path = temp_dir / "override_config.yaml"
        config_path.write_text(sample_config_yaml, encoding="utf-8")

        os.environ["CONFIG_PATH"] = str(config_path)
        os.environ["API_PORT"] = "12345"

        manager = ConfigManager()
        manager._config = None
        manager._config = manager._load_config()

        assert manager.config.api.port == 12345

    def test_missing_config_file_uses_defaults(self):
        os.environ["CONFIG_PATH"] = "/nonexistent/path.yaml"
        manager = ConfigManager()
        manager._config = None
        manager._config = manager._load_config()

        assert manager.config.env == "development"