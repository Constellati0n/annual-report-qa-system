#!/usr/bin/env python3
"""
ReAct Agent API服务
提供基于ReAct架构的年报分析接口
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agent.react_agent import create_agent, AnnualReportAssistant


# 创建FastAPI应用
app = FastAPI(
    title="年报分析ReAct Agent API",
    description="基于ReAct架构的年报分析助手，支持PDF阅读和联网搜索",
    version="2.0.0"
)

# 全局Agent实例
agent = None
assistant = None


class AnalyzeRequest(BaseModel):
    """分析请求"""
    question: str = Field(..., description="分析问题", example="请分析美的集团2023年的财务状况")
    use_history: bool = Field(default=False, description="是否使用对话历史")
    stream: bool = Field(default=False, description="是否流式输出")


class AnalyzeResponse(BaseModel):
    """分析响应"""
    success: bool = Field(..., description="是否成功")
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="分析回答")
    intermediate_steps: Optional[List] = Field(default=None, description="中间步骤")
    error: Optional[str] = Field(default=None, description="错误信息")


class BatchAnalyzeRequest(BaseModel):
    """批量分析请求"""
    company_codes: List[str] = Field(..., description="公司股票代码列表", example=["000333", "000651"])
    questions: List[str] = Field(..., description="问题列表")


@app.on_event("startup")
async def startup_event():
    """启动时初始化Agent"""
    global agent, assistant

    try:
        model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        print(f"正在初始化ReAct Agent，模型路径: {model_path}")

        agent = create_agent(model_path=model_path)
        assistant = AnnualReportAssistant(agent=agent)

        print("✅ ReAct Agent初始化成功")
    except Exception as e:
        print(f"⚠️ Agent初始化失败: {e}")
        print("API将以降级模式运行（仅提供健康检查）")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "年报分析ReAct Agent API",
        "version": "2.0.0",
        "architecture": "ReAct + Tool Calling",
        "tools": ["read_annual_report", "search_company_info", "search_industry_analysis"]
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "tools": ["read_annual_report", "search_company_info", "search_industry_analysis"]
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    单问题分析

    使用ReAct Agent分析问题，自动调用工具获取信息
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    try:
        if request.use_history and assistant:
            answer = assistant.ask(request.question, use_history=True)
            return AnalyzeResponse(
                success=True,
                question=request.question,
                answer=answer
            )
        else:
            result = agent.analyze(request.question)
            return AnalyzeResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    流式分析

    以流式方式返回分析过程和结果
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    async def event_generator():
        """生成SSE事件"""
        # 发送开始事件
        yield f"data: {json.dumps({'type': 'start', 'message': '开始分析'}, ensure_ascii=False)}\n\n"

        try:
            # 这里简化处理，实际应该实现真正的流式推理
            result = agent.analyze(request.question)

            # 模拟中间步骤输出
            if result.get("intermediate_steps"):
                for step in result["intermediate_steps"]:
                    yield f"data: {json.dumps({'type': 'step', 'data': str(step)}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)

            # 发送最终结果
            yield f"data: {json.dumps({'type': 'complete', 'answer': result.get('answer', '')}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/batch_analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    批量分析

    对多家公司批量执行相同的分析问题
    """
    if assistant is None:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    try:
        results = assistant.batch_analyze(
            company_codes=request.company_codes,
            questions=request.questions
        )
        return {
            "success": True,
            "total_companies": len(request.company_codes),
            "total_questions": len(request.questions),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools():
    """列出可用工具"""
    return {
        "tools": [
            {
                "name": "read_annual_report",
                "description": "读取上市公司年报PDF文件内容，获取财务数据、经营情况等",
                "parameters": {
                    "company_code": "股票代码，如'000333'",
                    "year": "年份，如'2023'（可选）"
                }
            },
            {
                "name": "search_company_info",
                "description": "搜索公司最新新闻、财务信息、股价等",
                "parameters": {
                    "company_name": "公司名称或股票代码",
                    "info_type": "信息类型：news/finance/industry/stock"
                }
            },
            {
                "name": "search_industry_analysis",
                "description": "搜索行业分析和市场研究报告",
                "parameters": {
                    "industry_name": "行业名称，如'家电行业'"
                }
            }
        ]
    }


@app.delete("/history")
async def clear_history():
    """清空对话历史"""
    if assistant:
        assistant.clear_history()
        return {"message": "对话历史已清空"}
    return {"message": "无对话历史可清空"}


# 运行服务
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    print(f"启动ReAct Agent API服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
