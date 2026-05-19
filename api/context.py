"""
应用上下文管理
提供线程安全的全局单例管理，替代模块级可变全局变量
"""
import threading
from typing import Any, Optional


class AppContext:
    """应用级上下文，管理全局单例"""

    __slots__ = ("_lock", "_instances")

    def __init__(self):
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_instances", {})

    def get(self, name: str) -> Optional[Any]:
        with self._lock:
            return self._instances.get(name)

    def set(self, name: str, instance: Any):
        with self._lock:
            self._instances[name] = instance

    def remove(self, name: str):
        with self._lock:
            instance = self._instances.pop(name, None)
            if instance and hasattr(instance, "close"):
                try:
                    instance.close()
                except Exception:
                    pass

    def clear(self):
        with self._lock:
            for instance in self._instances.values():
                if hasattr(instance, "close"):
                    try:
                        instance.close()
                    except Exception:
                        pass
            self._instances.clear()


_context: Optional[AppContext] = None
_context_lock = threading.Lock()


def get_app_context() -> AppContext:
    global _context
    if _context is None:
        with _context_lock:
            if _context is None:
                _context = AppContext()
    return _context