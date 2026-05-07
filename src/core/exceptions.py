"""
统一异常处理模块
定义应用级别的异常类和错误处理
"""
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """错误码定义"""
    # 系统级错误 (1000-1999)
    SYSTEM_ERROR = 1000
    CONFIG_ERROR = 1001
    INITIALIZATION_ERROR = 1002
    
    # 模型相关错误 (2000-2999)
    MODEL_LOAD_ERROR = 2000
    MODEL_INFERENCE_ERROR = 2001
    MODEL_NOT_FOUND = 2002
    
    # RAG相关错误 (3000-3999)
    VECTOR_STORE_ERROR = 3000
    EMBEDDING_ERROR = 3001
    RETRIEVAL_ERROR = 3002
    DOCUMENT_PROCESS_ERROR = 3003
    
    # API相关错误 (4000-4999)
    VALIDATION_ERROR = 4000
    AUTHENTICATION_ERROR = 4001
    AUTHORIZATION_ERROR = 4002
    RATE_LIMIT_ERROR = 4003
    NOT_FOUND = 4004
    
    # 爬虫相关错误 (5000-5999)
    CRAWLER_ERROR = 5000
    DOWNLOAD_ERROR = 5001
    PARSE_ERROR = 5002
    
    # 业务逻辑错误 (6000-6999)
    COMPANY_NOT_FOUND = 6000
    REPORT_NOT_FOUND = 6001
    ANALYSIS_ERROR = 6002


class AppException(Exception):
    """应用基础异常类"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "success": False,
            "error_code": self.code.value,
            "error_name": self.code.name,
            "message": self.message,
            "details": self.details
        }


class ModelException(AppException):
    """模型相关异常"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.MODEL_INFERENCE_ERROR, details: Optional[Dict] = None):
        super().__init__(message, code, 500, details)


class VectorStoreException(AppException):
    """向量存储异常"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.VECTOR_STORE_ERROR, details: Optional[Dict] = None):
        super().__init__(message, code, 500, details)


class EmbeddingException(AppException):
    """Embedding异常"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.EMBEDDING_ERROR, details: Optional[Dict] = None):
        super().__init__(message, code, 500, details)


class ValidationException(AppException):
    """参数验证异常"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, 400, details)


class AuthenticationException(AppException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR, 401)


class RateLimitException(AppException):
    """限流异常"""
    
    def __init__(self, message: str = "请求过于频繁，请稍后再试", retry_after: int = 60):
        super().__init__(message, ErrorCode.RATE_LIMIT_ERROR, 429)
        self.retry_after = retry_after


class NotFoundException(AppException):
    """资源不存在异常"""
    
    def __init__(self, message: str, resource_type: str = "resource"):
        super().__init__(message, ErrorCode.NOT_FOUND, 404, {"resource_type": resource_type})


class CrawlerException(AppException):
    """爬虫异常"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.CRAWLER_ERROR, details: Optional[Dict] = None):
        super().__init__(message, code, 500, details)


class RetryableException(AppException):
    """可重试的异常"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(message, code, 500)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
