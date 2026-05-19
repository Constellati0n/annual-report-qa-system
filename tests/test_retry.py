import time
import pytest

from src.core.retry import (
    RetryStrategy,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpen,
    calculate_delay,
    retry,
)


class TestCalculateDelay:
    def test_fixed_strategy(self):
        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=2.0, jitter=False)
        delay = calculate_delay(3, config)
        assert delay == 2.0

    def test_exponential_strategy(self):
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0, exponential_base=2.0, jitter=False)
        assert calculate_delay(1, config) == 1.0
        assert calculate_delay(2, config) == 2.0
        assert calculate_delay(3, config) == 4.0
        assert calculate_delay(4, config) == 8.0

    def test_linear_strategy(self):
        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=1.0, jitter=False)
        assert calculate_delay(1, config) == 1.0
        assert calculate_delay(2, config) == 2.0
        assert calculate_delay(3, config) == 3.0

    def test_max_delay_cap(self):
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, base_delay=10.0, max_delay=30.0, jitter=False)
        delay = calculate_delay(5, config)
        assert delay <= 30.0 + 0.01

    def test_jitter_adds_variation(self):
        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=1.0, jitter=True)
        delays = [calculate_delay(1, config) for _ in range(20)]
        assert any(d != 1.0 for d in delays)
        assert all(0.5 <= d <= 1.0 for d in delays)


class TestRetryDecorator:
    def test_successful_call(self):
        call_count = [0]

        @retry(max_attempts=3)
        def succeed():
            call_count[0] += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        call_count = [0]

        @retry(max_attempts=3, base_delay=0.01)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("临时错误")
            return "recovered"

        result = fail_twice()
        assert result == "recovered"
        assert call_count[0] == 3

    def test_exhaust_retries(self):
        call_count = [0]

        @retry(max_attempts=2, base_delay=0.01)
        def always_fail():
            call_count[0] += 1
            raise RuntimeError("永久错误")

        with pytest.raises(RuntimeError, match="永久错误"):
            always_fail()
        assert call_count[0] == 2

    def test_retry_specific_exceptions(self):
        @retry(max_attempts=3, base_delay=0.01, retry_exceptions=(ValueError,))
        def specific_fail():
            raise TypeError("不被重试")

        with pytest.raises(TypeError):
            specific_fail()

    def test_on_retry_callback(self):
        records = []

        def on_retry(exc, attempt):
            records.append((exc, attempt))

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        def fail_once():
            if not records:
                raise ValueError("第一次失败")
            return "ok"

        result = fail_once()
        assert result == "ok"
        assert len(records) == 1
        assert records[0][1] == 1


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.can_execute() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_close_after_half_open_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.can_execute() is True
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reopen_on_half_open_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestRetryWithCircuitBreaker:
    def test_blocks_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()

        @retry(max_attempts=2, base_delay=0.01, circuit_breaker=cb)
        def should_not_run():
            return "ran"

        with pytest.raises(CircuitBreakerOpen):
            should_not_run()