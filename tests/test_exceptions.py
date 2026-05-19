import os
import tempfile
from pathlib import Path

import pytest

from src.core.exceptions import (
    ErrorCode,
    AppException,
    ModelException,
    VectorStoreException,
    EmbeddingException,
    ValidationException,
    AuthenticationException,
    RateLimitException,
    NotFoundException,
    CrawlerException,
    RetryableException,
)


class TestErrorCode:
    def test_error_code_ranges(self):
        assert ErrorCode.SYSTEM_ERROR.value == 1000
        assert ErrorCode.MODEL_LOAD_ERROR.value == 2000
        assert ErrorCode.VECTOR_STORE_ERROR.value == 3000
        assert ErrorCode.VALIDATION_ERROR.value == 4000
        assert ErrorCode.CRAWLER_ERROR.value == 5000
        assert ErrorCode.COMPANY_NOT_FOUND.value == 6000

    def test_error_code_to_dict(self):
        exc = AppException("测试错误", code=ErrorCode.MODEL_NOT_FOUND)
        d = exc.to_dict()
        assert d["success"] is False
        assert d["error_code"] == 2002
        assert d["message"] == "测试错误"
        assert d["error_name"] == "MODEL_NOT_FOUND"


class TestAppException:
    def test_default_values(self):
        exc = AppException("错误消息")
        assert exc.message == "错误消息"
        assert exc.code == ErrorCode.SYSTEM_ERROR
        assert exc.status_code == 500
        assert exc.details == {}

    def test_custom_values(self):
        exc = AppException("自定义", code=ErrorCode.CONFIG_ERROR, status_code=400, details={"key": "value"})
        assert exc.code == ErrorCode.CONFIG_ERROR
        assert exc.status_code == 400
        assert exc.details == {"key": "value"}

    def test_to_dict(self):
        exc = AppException("测试", code=ErrorCode.VALIDATION_ERROR, details={"field": "name"})
        d = exc.to_dict()
        assert d == {
            "success": False,
            "error_code": 4000,
            "error_name": "VALIDATION_ERROR",
            "message": "测试",
            "details": {"field": "name"}
        }


class TestModelException:
    def test_default(self):
        exc = ModelException("模型加载失败")
        assert exc.code == ErrorCode.MODEL_INFERENCE_ERROR
        assert exc.status_code == 500


class TestValidationException:
    def test_default(self):
        exc = ValidationException("缺少必填字段")
        assert exc.code == ErrorCode.VALIDATION_ERROR
        assert exc.status_code == 400


class TestAuthenticationException:
    def test_default(self):
        exc = AuthenticationException()
        assert exc.message == "认证失败"
        assert exc.status_code == 401


class TestRateLimitException:
    def test_default(self):
        exc = RateLimitException()
        assert exc.retry_after == 60

    def test_custom(self):
        exc = RateLimitException("自定义", retry_after=120)
        assert exc.retry_after == 120


class TestNotFoundException:
    def test_default(self):
        exc = NotFoundException("资源不存在", resource_type="company")
        assert exc.status_code == 404
        assert exc.details["resource_type"] == "company"


class TestCrawlerException:
    def test_default(self):
        exc = CrawlerException("下载失败", code=ErrorCode.DOWNLOAD_ERROR)
        assert exc.code == ErrorCode.DOWNLOAD_ERROR


class TestRetryableException:
    def test_default(self):
        exc = RetryableException("网络超时", code=ErrorCode.DOWNLOAD_ERROR)
        assert exc.max_retries == 3
        assert exc.retry_delay == 1.0