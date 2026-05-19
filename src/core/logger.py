"""
结构化日志模块
支持JSON格式、日志采样、上下文追踪
"""
import json
import logging
import sys
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from datetime import datetime, timezone

from .config import get_logging_config

# 请求上下文
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加上下文信息
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """添加上下文信息的过滤器"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
        self._config = get_logging_config()
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        config = self._config
        
        # 设置日志级别
        self._logger.setLevel(getattr(logging, config.level.upper()))
        
        # 清除已有处理器
        self._logger.handlers.clear()
        
        # 添加上下文过滤器
        self._logger.addFilter(ContextFilter())
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        if config.json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(config.format))
        self._logger.addHandler(console_handler)
        
        # 文件处理器
        if config.file_path:
            log_dir = Path(config.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            if config.json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(config.format))
            self._logger.addHandler(file_handler)
    
    def _should_log(self) -> bool:
        """日志采样检查"""
        import random
        return random.random() < self._config.sampling_rate
    
    def _log(self, level: int, message: str, extra: Optional[Dict] = None, **kwargs):
        """内部日志方法"""
        if not self._should_log():
            return
        
        extra_data = extra or {}
        extra_data.update(kwargs)
        
        if extra_data:
            # 创建LogRecord并添加额外数据
            record = self._logger.makeRecord(
                self.name, level, "", 0, message, (), None
            )
            record.extra_data = extra_data
            self._logger.handle(record)
        else:
            self._logger.log(level, message)
    
    def debug(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self._log(logging.DEBUG, message, extra, **kwargs)
    
    def info(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self._log(logging.INFO, message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self._log(logging.WARNING, message, extra, **kwargs)
    
    def error(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self._log(logging.ERROR, message, extra, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self._log(logging.CRITICAL, message, extra, **kwargs)
    
    def exception(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """记录异常信息"""
        if not self._should_log():
            return
        
        extra_data = extra or {}
        extra_data.update(kwargs)
        self._logger.exception(message, extra={"extra_data": extra_data})


def get_logger(name: str) -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """设置请求ID"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    """获取当前请求ID"""
    return request_id_var.get()


def set_user_id(user_id: str):
    """设置用户ID"""
    user_id_var.set(user_id)


def get_user_id() -> str:
    """获取当前用户ID"""
    return user_id_var.get()


def clear_context():
    """清除上下文"""
    request_id_var.set('')
    user_id_var.set('')


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id or ''
        self.token_request_id = None
        self.token_user_id = None
    
    def __enter__(self):
        self.token_request_id = request_id_var.set(self.request_id)
        if self.user_id:
            self.token_user_id = user_id_var.set(self.user_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.reset(self.token_request_id)
        if self.token_user_id:
            user_id_var.reset(self.token_user_id)


def log_execution_time(logger: Optional[StructuredLogger] = None):
    """记录函数执行时间的装饰器"""
    import functools
    import time
    
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"函数 {func.__name__} 执行成功",
                    extra={"execution_time": f"{execution_time:.3f}s", "function": func.__name__}
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"函数 {func.__name__} 执行失败",
                    extra={"execution_time": f"{execution_time:.3f}s", "error": str(e), "function": func.__name__}
                )
                raise
        
        return wrapper
    return decorator
