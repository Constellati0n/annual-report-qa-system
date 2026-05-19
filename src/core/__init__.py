"""
核心模块
提供年报分析助手的主要功能
"""
__all__ = ['AnnualReportAssistant']


def __getattr__(name):
    if name == 'AnnualReportAssistant':
        from .assistant import AnnualReportAssistant
        return AnnualReportAssistant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")