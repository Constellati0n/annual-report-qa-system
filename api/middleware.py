"""
API中间件
包含认证、限流、日志、错误处理等
"""
import time
import uuid
from typing import Optional, Callable
from functools import wraps

from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import get_api_config
from ..core.logger import get_logger, set_request_id, get_request_id, LogContext
from ..core.exceptions import (
    AppException, AuthenticationException, RateLimitException, 
    ValidationException, NotFoundException
)

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求上下文中间件 - 设置请求ID和日志上下文"""
    
    async def dispatch(self, request: Request, call_next):
        # 生成或获取请求ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # 设置日志上下文
        with LogContext(request_id=request_id):
            # 记录请求开始
            start_time = time.time()
            logger.info(
                f"请求开始: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            )
            
            # 处理请求
            try:
                response = await call_next(request)
                
                # 计算耗时
                process_time = time.time() - start_time
                
                # 添加响应头
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(process_time)
                
                # 记录请求完成
                logger.info(
                    f"请求完成: {request.method} {request.url.path}",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "process_time": f"{process_time:.3f}s"
                    }
                )
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(
                    f"请求异常: {request.method} {request.url.path}",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e),
                        "process_time": f"{process_time:.3f}s"
                    }
                )
                raise


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app: ASGIApp, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.config = get_api_config()
        self.exclude_paths = exclude_paths or ["/docs", "/openapi.json", "/health", "/"]
    
    async def dispatch(self, request: Request, call_next):
        # 检查是否需要认证
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # 如果没有配置API Key，跳过认证
        if not self.config.api_key:
            return await call_next(request)
        
        # 获取API Key
        api_key = request.headers.get(self.config.api_key_header)
        
        if not api_key:
            logger.warning(f"缺少API Key: {request.url.path}")
            raise HTTPException(status_code=401, detail="缺少API Key")
        
        if api_key != self.config.api_key:
            logger.warning(f"无效的API Key: {request.url.path}")
            raise HTTPException(status_code=401, detail="无效的API Key")
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件 - 基于滑动窗口"""
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: dict = {}  # client_ip -> [(timestamp, count)]
    
    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用API Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # 其次使用IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def _is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        import time
        
        now = time.time()
        window_start = now - 60  # 1分钟窗口
        
        # 清理旧记录
        if client_id in self.requests:
            self.requests[client_id] = [
                (ts, count) for ts, count in self.requests[client_id]
                if ts > window_start
            ]
        
        # 计算当前窗口内的请求数
        current_count = sum(
            count for ts, count in self.requests.get(client_id, [])
        )
        
        # 检查是否超过限制
        if current_count >= self.requests_per_minute:
            return False
        
        # 记录请求
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id].append((now, 1))
        
        return True
    
    async def dispatch(self, request: Request, call_next):
        client_id = self._get_client_id(request)
        
        if not self._is_allowed(client_id):
            logger.warning(f"限流触发: {client_id} - {request.url.path}")
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件 - 统一异常处理"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except AppException as e:
            # 应用自定义异常
            logger.error(
                f"业务异常: {e.message}",
                extra={"error_code": e.code.value, "path": request.url.path}
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
            
        except HTTPException as e:
            # FastAPI HTTP异常
            raise
            
        except Exception as e:
            # 未处理的异常
            logger.exception(f"未处理的异常: {str(e)}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": 1000,
                    "message": "服务器内部错误",
                    "details": {"error": str(e)} if get_api_config().debug else {}
                }
            )


class TimingMiddleware(BaseHTTPMiddleware):
    """性能计时中间件"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 慢请求警告
        if process_time > 5.0:
            logger.warning(
                f"慢请求: {request.method} {request.url.path} 耗时 {process_time:.2f}s"
            )
        
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        return response


def setup_middleware(app):
    """设置所有中间件"""
    config = get_api_config()
    
    # 错误处理（最先添加，最后执行）
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 请求上下文
    app.add_middleware(RequestContextMiddleware)
    
    # 认证
    app.add_middleware(AuthenticationMiddleware)
    
    # 限流
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.rate_limit,
        burst_size=config.rate_limit_burst
    )
    
    # 计时
    app.add_middleware(TimingMiddleware)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info("✓ 中间件设置完成")


# 依赖函数
def get_request_id_dependency() -> str:
    """获取当前请求ID的依赖"""
    return get_request_id()
