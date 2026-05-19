import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeTorch:
    def __init__(self):
        self.__path__ = []
        self.__version__ = "2.0.0"
        self.__spec__ = MagicMock()

    @staticmethod
    def cuda():
        mock = MagicMock()
        mock.is_available.return_value = False
        return mock

    def __getattr__(self, name):
        return MagicMock()


sys.modules["torch"] = FakeTorch()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()


@pytest.fixture
def client():
    os.environ["MODEL_PATH"] = str(Path(__file__).parent.parent)

    mock_assistant = MagicMock()
    mock_assistant.model_name = "mock-model"
    mock_assistant.prompt_manager.detect_analysis_type.return_value = MagicMock(value="financial_analysis")
    mock_assistant.get_knowledge_base_stats.return_value = {"documents": 100}

    with patch("api.main.AnnualReportAssistant", return_value=mock_assistant), \
         patch("api.main.init_agent"):
        from api.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c


class TestRootAndHealth:
    def test_root_returns_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_analysis_types(self, client):
        response = client.get("/analysis-types")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) > 0


class TestAnalyzeValidation:
    def test_question_required(self, client):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_max_tokens_range_min(self, client):
        response = client.post("/analyze", json={"question": "test", "max_tokens": 0})
        assert response.status_code == 422

    def test_temperature_range(self, client):
        response = client.post("/analyze", json={"question": "test", "temperature": 3.0})
        assert response.status_code == 422

    def test_compare_min_companies(self, client):
        response = client.post("/analyze/compare", json={"companies": ["A"]})
        assert response.status_code == 422


class TestAgentRoutes:
    def test_agent_health(self, client):
        response = client.get("/agent/health")
        assert response.status_code == 200
        data = response.json()
        assert "agent_ready" in data

    def test_agent_tools(self, client):
        response = client.get("/agent/tools")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tools"]) == 3

    def test_clear_history(self, client):
        response = client.delete("/agent/history")
        assert response.status_code == 200