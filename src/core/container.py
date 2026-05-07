"""
依赖注入容器
使用依赖注入模式管理组件生命周期
"""
from typing import Optional, Type, TypeVar, Dict, Any, Callable
from functools import wraps
import threading

T = TypeVar('T')


class Container:
    """依赖注入容器"""
    
    def __init__(self):
        self._registrations: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._lock = threading.Lock()
    
    def register_instance(self, interface: Type[T], instance: T):
        """注册实例"""
        self._singletons[interface] = instance
    
    def register_factory(self, interface: Type[T], factory: Callable[..., T]):
        """注册工厂函数"""
        self._factories[interface] = factory
    
    def register_singleton(self, interface: Type[T], factory: Callable[..., T]):
        """注册单例（延迟初始化）"""
        self._factories[interface] = factory
        self._registrations[interface] = "singleton"
    
    def resolve(self, interface: Type[T]) -> T:
        """解析依赖"""
        # 1. 检查已有实例
        if interface in self._singletons:
            return self._singletons[interface]
        
        # 2. 检查工厂函数
        if interface in self._factories:
            factory = self._factories[interface]
            
            # 如果是单例，需要线程安全地创建
            if interface in self._registrations and self._registrations[interface] == "singleton":
                with self._lock:
                    # 双重检查
                    if interface in self._singletons:
                        return self._singletons[interface]
                    
                    instance = factory()
                    self._singletons[interface] = instance
                    return instance
            else:
                return factory()
        
        raise KeyError(f"未注册的接口: {interface}")
    
    def clear(self):
        """清空容器"""
        with self._lock:
            # 清理单例资源
            for instance in self._singletons.values():
                if hasattr(instance, 'close'):
                    try:
                        instance.close()
                    except:
                        pass
            
            self._singletons.clear()
            self._factories.clear()
            self._registrations.clear()


# 全局容器实例
_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """获取全局容器实例"""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def reset_container():
    """重置容器（主要用于测试）"""
    global _container
    with _container_lock:
        if _container:
            _container.clear()
        _container = Container()


# 依赖注入装饰器
def inject(**dependencies):
    """依赖注入装饰器
    
    用法:
        @inject(vector_store=IVectorStore, embedding=IEmbedding)
        def my_function(vector_store, embedding, other_arg):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            
            # 注入依赖
            for name, interface in dependencies.items():
                if name not in kwargs:
                    kwargs[name] = container.resolve(interface)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 初始化函数
def initialize_container():
    """初始化容器，注册所有组件"""
    from ..rag.interfaces import IVectorStore, IEmbedding, IRetriever, IReranker
    from ..rag.embedding import EmbeddingManager
    from ..rag.vector_store import ChromaVectorStore, FAISSVectorStore
    from ..rag.retriever import RAGRetriever
    from ..rag.reranker import BGEReranker
    from .config import get_rag_config
    
    container = get_container()
    config = get_rag_config()
    
    # 注册Embedding
    container.register_singleton(
        IEmbedding,
        lambda: EmbeddingManager()
    )
    
    # 注册VectorStore
    def create_vector_store():
        if config.vector_store_type == "chroma":
            return ChromaVectorStore(
                collection_name=config.collection_name,
                persist_directory=config.vector_db_path
            )
        else:
            return FAISSVectorStore(
                collection_name=config.collection_name,
                persist_directory=config.vector_db_path
            )
    
    container.register_singleton(IVectorStore, create_vector_store)
    
    # 注册Reranker
    container.register_singleton(
        IReranker,
        lambda: BGEReranker() if config.use_reranker else None
    )
    
    # 注册Retriever
    container.register_singleton(
        IRetriever,
        lambda: RAGRetriever(
            vector_store=container.resolve(IVectorStore),
            embedding=container.resolve(IEmbedding),
            reranker=container.resolve(IReranker)
        )
    )
    
    print("✓ 依赖注入容器初始化完成")


# 便捷获取函数
def get_embedding() -> 'IEmbedding':
    """获取Embedding管理器"""
    return get_container().resolve(IEmbedding)


def get_vector_store() -> 'IVectorStore':
    """获取向量存储"""
    return get_container().resolve(IVectorStore)


def get_retriever() -> 'IRetriever':
    """获取检索器"""
    return get_container().resolve(IRetriever)
