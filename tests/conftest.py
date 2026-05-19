import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_config_yaml():
    return """
server:
  host: "0.0.0.0"
  port: 8000

models:
  llm:
    name: "test-model"
    path: "/tmp/test-model"
    load_in_4bit: false
    max_length: 2048

  embedding:
    name: "test-embedding"
    path: "/tmp/test-embedding"

rag:
  vector_store:
    type: "chroma"
    persist_directory: "./test_vector_db"
    collection_name: "test_collection"

  retrieval:
    top_k: 3
    similarity_threshold: 0.5
    rerank_enabled: false

  chunking:
    chunk_size: 256
    chunk_overlap: 64
"""


@pytest.fixture(autouse=True)
def clean_env():
    vars_to_remove = [
        "LLM_MODEL_NAME", "LLM_MODEL_PATH", "EMBEDDING_MODEL_NAME",
        "LOAD_IN_4BIT", "VECTOR_DB_PATH", "API_HOST", "API_PORT",
        "API_KEY", "APP_ENV", "DEBUG", "CONFIG_PATH"
    ]
    saved = {}
    for var in vars_to_remove:
        if var in os.environ:
            saved[var] = os.environ.pop(var)
    yield
    for var, value in saved.items():
        os.environ[var] = value