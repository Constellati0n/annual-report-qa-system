"""
ReAct Agent API 路由
提供基于ReAct架构的年报分析接口，以 APIRouter 方式集成到主应用中
"""
import os
import sys
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.context import get_app_context

router = APIRouter(prefix="/agent", tags=["ReAct Agent"])


class AgentAnalyzeRequest(BaseModel):
    question: str = Field(..., description="分析问题", example="请分析美的集团2023年的财务状况")
    use_history: bool = Field(default=False, description="是否使用对话历史")
    stream: bool = Field(default=False, description="是否流式输出")


class AgentAnalyzeResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="分析回答")
    intermediate_steps: Optional[List] = Field(default=None, description="中间步骤")
    error: Optional[str] = Field(default=None, description="错误信息")


class BatchAnalyzeRequest(BaseModel):
    company_codes: List[str] = Field(..., description="公司股票代码列表", example=["000333", "000651"])
    questions: List[str] = Field(..., description="问题列表")


def get_agent():
    return get_app_context().get("agent")


def get_assistant():
    return get_app_context().get("agent_assistant")


def init_agent():
    ctx = get_app_context()
    try:
        model_path = os.getenv("AGENT_MODEL_PATH", os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"))
        logger.info("正在初始化ReAct Agent，模型路径: %s", model_path)

        from src.agent.react_agent import create_agent, AnnualReportAssistant
        _agent = create_agent(model_path=model_path)
        _assistant = AnnualReportAssistant(agent=_agent)
        ctx.set("agent", _agent)
        ctx.set("agent_assistant", _assistant)

        logger.info("ReAct Agent 初始化成功")
    except Exception as e:
        logger.warning("Agent初始化失败: %s", e)


@router.get("/health")
async def agent_health_check():
    return {
        "status": "healthy",
        "agent_ready": get_agent() is not None,
        "tools": ["read_annual_report", "search_company_info", "search_industry_analysis"]
    }


@router.post("/analyze", response_model=AgentAnalyzeResponse)
async def agent_analyze(request: AgentAnalyzeRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    try:
        if request.use_history:
            assistant = get_assistant()
            if assistant:
                answer = assistant.ask(request.question, use_history=True)
                return AgentAnalyzeResponse(
                    success=True,
                    question=request.question,
                    answer=answer
                )
        result = agent.analyze(request.question)
        return AgentAnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/stream")
async def agent_analyze_stream(request: AgentAnalyzeRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'message': '开始分析'}, ensure_ascii=False)}\n\n"

        try:
            result = agent.analyze(request.question)

            if result.get("intermediate_steps"):
                for step in result["intermediate_steps"]:
                    yield f"data: {json.dumps({'type': 'step', 'data': str(step)}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'type': 'complete', 'answer': result.get('answer', '')}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/batch_analyze")
async def agent_batch_analyze(request: BatchAnalyzeRequest):
    assistant = get_assistant()
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


@router.get("/tools")
async def list_tools():
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


@router.delete("/history")
async def clear_history():
    assistant = get_assistant()
    if assistant:
        assistant.clear_history()
        return {"message": "对话历史已清空"}
    return {"message": "无对话历史可清空"}