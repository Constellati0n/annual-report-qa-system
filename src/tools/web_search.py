#!/usr/bin/env python3
"""
联网搜索工具 - 让Agent能够搜索最新公司信息和行业动态
"""

import os
import json
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any
from langchain.tools import tool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """搜索工具输入参数"""
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量")


class WebSearchTool:
    """网页搜索工具 - 支持多种搜索API"""

    def __init__(self):
        # 可以配置多个搜索引擎API
        self.api_key = os.getenv("SEARCH_API_KEY", "")
        self.search_engine = os.getenv("SEARCH_ENGINE", "duckduckgo")

    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """使用DuckDuckGo搜索（无需API Key）"""
        try:
            import requests
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            return results
        except Exception as e:
            return [{"error": f"搜索失败: {str(e)}"}]

    def search_bing(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """使用Bing搜索API"""
        if not self.api_key:
            return [{"error": "未配置Bing API Key"}]

        try:
            import requests

            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {
                "q": query,
                "count": num_results,
                "textDecorations": False,
                "textFormat": "HTML"
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            if "webPages" in data and "value" in data["webPages"]:
                for item in data["webPages"]["value"]:
                    results.append({
                        "title": item.get("name", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    })

            return results
        except Exception as e:
            return [{"error": f"Bing搜索失败: {str(e)}"}]

    def search(self, query: str, num_results: int = 5) -> str:
        """
        执行搜索

        Args:
            query: 搜索关键词
            num_results: 返回结果数量

        Returns:
            格式化的搜索结果
        """
        # 优先使用DuckDuckGo（无需API Key）
        try:
            results = self.search_duckduckgo(query, num_results)
        except:
            results = [{"error": "搜索服务暂时不可用"}]

        if not results:
            return "未找到相关搜索结果"

        # 格式化输出
        output = f"【搜索: {query}】\n\n"

        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"错误: {result['error']}\n"
            else:
                output += f"{i}. {result.get('title', '无标题')}\n"
                output += f"   链接: {result.get('link', '无链接')}\n"
                output += f"   摘要: {result.get('snippet', '无摘要')[:200]}...\n\n"

        return output


class CompanyInfoSearchTool:
    """公司信息专用搜索工具"""

    def __init__(self):
        self.web_search = WebSearchTool()

    def search_company_news(self, company_name: str, num_results: int = 5) -> str:
        """搜索公司最新新闻"""
        query = f"{company_name} 最新新闻 2024 2025"
        return self.web_search.search(query, num_results)

    def search_company_finance(self, company_name: str, num_results: int = 5) -> str:
        """搜索公司财务信息"""
        query = f"{company_name} 财务报告 业绩 营收 利润"
        return self.web_search.search(query, num_results)

    def search_industry_info(self, industry: str, num_results: int = 5) -> str:
        """搜索行业信息"""
        query = f"{industry} 行业分析 发展趋势 2024 2025"
        return self.web_search.search(query, num_results)

    def search_stock_price(self, stock_code: str, num_results: int = 3) -> str:
        """搜索股价信息"""
        query = f"{stock_code} 股价 行情 最新"
        return self.web_search.search(query, num_results)


@tool
def search_company_info(company_name: str, info_type: str = "news") -> str:
    """
    搜索公司的最新信息和动态

    使用此工具可以获取公司最新新闻、财务信息、行业动态等，补充年报中的历史数据。

    Args:
        company_name: 公司名称或股票代码，如"美的集团"、"000333"
        info_type: 信息类型，可选值：
            - "news": 最新新闻（默认）
            - "finance": 财务信息
            - "industry": 所属行业信息
            - "stock": 股价行情

    Returns:
        搜索结果，包含标题、链接和摘要

    Example:
        search_company_info("美的集团", "news")  # 搜索美的集团最新新闻
        search_company_info("000333", "finance")  # 搜索美的集团财务信息
    """
    searcher = CompanyInfoSearchTool()

    if info_type == "news":
        return searcher.search_company_news(company_name)
    elif info_type == "finance":
        return searcher.search_company_finance(company_name)
    elif info_type == "industry":
        # 需要先从公司名推断行业
        return searcher.search_industry_info(f"{company_name} 所在行业")
    elif info_type == "stock":
        return searcher.search_stock_price(company_name)
    else:
        return searcher.web_search.search(f"{company_name} {info_type}")


@tool
def search_industry_analysis(industry_name: str) -> str:
    """
    搜索行业分析和市场研究报告

    Args:
        industry_name: 行业名称，如"家电行业"、"新能源汽车"

    Returns:
        行业分析报告搜索结果
    """
    searcher = CompanyInfoSearchTool()
    return searcher.search_industry_info(industry_name)


# 为LangChain Agent准备的工具列表
web_search_tools = [search_company_info, search_industry_analysis]
