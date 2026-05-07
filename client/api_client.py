"""
年报分析助手 - 客户端SDK
用于连接阿里云PAI DSW部署的API服务
"""
import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    sources: List[Dict]
    model_used: str
    success: bool
    error: str = None


class AnnualReportClient:
    """年报分析助手客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        初始化客户端
        
        Args:
            base_url: API服务地址
            api_key: API密钥（如果需要）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_rag: bool = True,
        stream: bool = False
    ) -> QueryResult:
        """
        查询问题
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_rag: 是否使用RAG
            stream: 是否流式输出
            
        Returns:
            查询结果
        """
        try:
            if stream:
                # 流式查询
                return self._query_stream(question, top_k, use_rag)
            
            # 普通查询
            response = requests.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json={
                    "question": question,
                    "top_k": top_k,
                    "use_rag": use_rag
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return QueryResult(
                    answer=data.get("answer", ""),
                    sources=data.get("sources", []),
                    model_used=data.get("model_used", ""),
                    success=True
                )
            else:
                return QueryResult(
                    answer="",
                    sources=[],
                    model_used="",
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        
        except Exception as e:
            return QueryResult(
                answer="",
                sources=[],
                model_used="",
                success=False,
                error=str(e)
            )
    
    def _query_stream(
        self,
        question: str,
        top_k: int = 5,
        use_rag: bool = True
    ) -> QueryResult:
        """流式查询"""
        try:
            response = requests.post(
                f"{self.base_url}/query/stream",
                headers=self.headers,
                json={
                    "question": question,
                    "top_k": top_k,
                    "use_rag": use_rag
                },
                stream=True,
                timeout=60
            )
            
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            if "answer" in json_data:
                                full_answer = json_data["answer"]
                        except:
                            pass
            
            return QueryResult(
                answer=full_answer,
                sources=[],
                model_used="",
                success=True
            )
        
        except Exception as e:
            return QueryResult(
                answer="",
                sources=[],
                model_used="",
                success=False,
                error=str(e)
            )
    
    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                headers=self.headers,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


# 示例用法
if __name__ == "__main__":
    # 创建客户端（替换为你的PAI DSW实例地址）
    client = AnnualReportClient(
        base_url="http://your-pai-dsw-instance:8000"
    )
    
    # 健康检查
    health = client.health_check()
    print(f"服务状态: {health}")
    
    # 查询问题
    result = client.query(
        question="平安银行2023年的净利润是多少？",
        top_k=5,
        use_rag=True
    )
    
    if result.success:
        print(f"\n回答: {result.answer}")
        print(f"\n参考来源:")
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source['content'][:100]}...")
    else:
        print(f"查询失败: {result.error}")
