"""
重试机制模块
支持指数退避、熔断器模式
"""
import time
import random
import functools
from typing import Callable, Optional, Type, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


class RetryStrategy(Enum):
    """重试策略"""
    FIXED = "fixed"           # 固定间隔
    EXPONENTIAL = "exponential"  # 指数退避
    LINEAR = "linear"         # 线性增长


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0
    jitter: bool = True  # 添加随机抖动
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[Exception, int], None]] = None


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("熔断器进入半开状态")
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self):
        """记录成功"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._reset()
                logger.info("熔断器关闭，服务恢复正常")
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"熔断器打开，失败次数: {self.failure_count}")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"熔断器打开，失败次数: {self.failure_count}")
    
    def _reset(self):
        """重置熔断器"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""
    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """计算重试延迟"""
    if config.strategy == RetryStrategy.FIXED:
        delay = config.base_delay
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.base_delay * attempt
    else:  # EXPONENTIAL
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    
    # 限制最大延迟
    delay = min(delay, config.max_delay)
    
    # 添加抖动，避免惊群效应
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """重试装饰器"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        retry_exceptions=retry_exceptions,
        on_retry=on_retry
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查熔断器
            if circuit_breaker and not circuit_breaker.can_execute():
                raise CircuitBreakerOpen(f"服务熔断中，请稍后重试")
            
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # 记录成功
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                    
                except config.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        # 记录失败
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        
                        logger.error(
                            f"函数 {func.__name__} 重试 {config.max_attempts} 次后仍然失败",
                            extra={"error": str(e), "function": func.__name__}
                        )
                        raise
                    
                    # 计算延迟
                    delay = calculate_delay(attempt, config)
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt} 次尝试失败，{delay:.2f}s 后重试",
                        extra={"error": str(e), "attempt": attempt, "delay": delay}
                    )
                    
                    # 回调
                    if config.on_retry:
                        config.on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            # 不应该到达这里
            raise last_exception or Exception("重试失败")
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """异步重试装饰器"""
    import asyncio
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        retry_exceptions=retry_exceptions,
        on_retry=on_retry
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 检查熔断器
            if circuit_breaker and not circuit_breaker.can_execute():
                raise CircuitBreakerOpen(f"服务熔断中，请稍后重试")
            
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # 记录成功
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                    
                except config.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        # 记录失败
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        
                        logger.error(
                            f"函数 {func.__name__} 重试 {config.max_attempts} 次后仍然失败",
                            extra={"error": str(e), "function": func.__name__}
                        )
                        raise
                    
                    # 计算延迟
                    delay = calculate_delay(attempt, config)
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt} 次尝试失败，{delay:.2f}s 后重试",
                        extra={"error": str(e), "attempt": attempt, "delay": delay}
                    )
                    
                    # 回调
                    if config.on_retry:
                        config.on_retry(e, attempt)
                    
                    await asyncio.sleep(delay)
            
            # 不应该到达这里
            raise last_exception or Exception("重试失败")
        
        return wrapper
    return decorator


# 全局熔断器实例
circuit_breakers: dict = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """获取或创建熔断器"""
    if name not in circuit_breakers:
        circuit_breakers[name] = CircuitBreaker(**kwargs)
    return circuit_breakers[name]
