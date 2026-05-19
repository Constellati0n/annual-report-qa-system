"""
API服务主入口
提供年报分析助手的RESTful API接口
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
import json

from src.core.assistant import AnnualReportAssistant
from src.prompts import AnalysisType
from src.core.config import get_config
from api.react_agent_api import router as agent_router, init_agent
from api.context import get_app_context


def get_assistant() -> Optional[AnnualReportAssistant]:
    return get_app_context().get("assistant")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载配置
config = get_config()


class AnalyzeRequest(BaseModel):
    """分析请求模型 - 支持 Qwen3 特性"""
    question: str = Field(..., description="用户问题", min_length=1)
    company: Optional[str] = Field(None, description="公司名称")
    year: Optional[str] = Field(None, description="年份")
    analysis_type: Optional[str] = Field(None, description="分析类型")
    stream: bool = Field(False, description="是否流式输出")
    # Qwen3 特性
    enable_thinking: Optional[bool] = Field(None, description="是否启用思考模式 (Qwen3)")
    max_tokens: int = Field(2048, description="最大生成token数", ge=1, le=8192)
    temperature: float = Field(0.7, description="温度", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Top-p采样", ge=0.0, le=1.0)
    tools: Optional[List[Dict]] = Field(None, description="工具定义列表 (Function Calling)")


class AnalyzeResponse(BaseModel):
    """分析响应模型"""
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    analysis_type: Optional[str] = None
    retrieved_documents: Optional[int] = None


class FinancialAnalysisRequest(BaseModel):
    """财务分析请求"""
    company: str = Field(..., description="公司名称")
    year: Optional[str] = Field(None, description="年份")
    metrics: Optional[List[str]] = Field(None, description="分析指标")


class CompareRequest(BaseModel):
    """对比分析请求"""
    companies: List[str] = Field(..., description="公司名称列表", min_length=2)
    metrics: Optional[List[str]] = Field(None, description="对比指标")


class KnowledgeBaseStats(BaseModel):
    """知识库统计"""
    document_count: int
    embedding_dim: int
    top_k: int
    use_reranker: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - Qwen3 版本"""
    ctx = get_app_context()
    
    # 启动时初始化助手
    logger.info("初始化年报分析助手 (Qwen3)...")
    try:
        # Qwen3 配置
        model_path = os.getenv("MODEL_PATH")
        base_model = os.getenv("BASE_MODEL")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_db")
        use_rag = os.getenv("USE_RAG", "true").lower() == "true"
        load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
        max_length = int(os.getenv("MAX_LENGTH", "32768"))  # Qwen3 支持 32K
        enable_thinking = os.getenv("ENABLE_THINKING", "true").lower() == "true"

        if not model_path and not base_model:
            logger.error("未配置模型路径，请设置环境变量 MODEL_PATH 或 BASE_MODEL")
            raise ValueError("未配置模型路径，请设置环境变量 MODEL_PATH 或 BASE_MODEL")
        
        logger.info(f"模型路径: {model_path}")
        logger.info(f"基础模型: {base_model}")
        logger.info(f"向量存储: {vector_store_path}")
        logger.info(f"使用RAG: {use_rag}")
        logger.info(f"4bit量化: {load_in_4bit}")
        logger.info(f"最大长度: {max_length}")
        logger.info(f"思考模式: {enable_thinking}")
        
        # 验证模型路径 - 如果finetuned路径不存在，使用基础模型
        if model_path and not os.path.exists(model_path):
            logger.warning(f"微调模型路径不存在: {model_path}")
            model_path = None
        if not model_path and base_model:
            if os.path.exists(base_model):
                logger.info(f"使用基础模型: {base_model}")
                model_path = base_model
            else:
                logger.error(f"基础模型路径不存在: {base_model}")
                raise FileNotFoundError(f"模型路径不存在: {base_model}")
        if not model_path:
            raise FileNotFoundError("未找到可用的模型路径")
        
        # 使用 Qwen3 客户端
        from client.llm_chat_qwen3 import Qwen3ChatClient
        
        ctx.set("assistant", AnnualReportAssistant(
            model_path=model_path,
            base_model=base_model,
            vector_store_path=vector_store_path,
            use_rag=use_rag,
            load_in_4bit=load_in_4bit,
            max_length=max_length,
            enable_thinking=enable_thinking
        ))
        logger.info("年报分析助手 (Qwen3) 初始化完成")

        # 初始化 ReAct Agent（可选，失败不影响主服务）
        try:
            init_agent()
        except Exception as e:
            logger.warning("ReAct Agent 初始化失败（不影响主服务）: %s", e)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("关闭服务...")


# 创建FastAPI应用
app = FastAPI(
    title="年报分析助手API",
    description="基于RAG和微调LLM的年报分析服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 ReAct Agent 路由
app.include_router(agent_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "年报分析助手API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "assistant_ready": get_assistant() is not None
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    通用分析接口
    
    分析年报相关问题，支持多种分析类型
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 执行分析
        result = get_assistant().analyze(
            question=request.question,
            company=request.company,
            year=request.year,
            analysis_type=request.analysis_type,
            stream=request.stream
        )
        
        # 检测分析类型
        analysis_type = request.analysis_type
        if not analysis_type:
            analysis_type = get_assistant().prompt_manager.detect_analysis_type(request.question).value
        
        return AnalyzeResponse(
            success=True,
            data=result,
            analysis_type=analysis_type,
            retrieved_documents=None  # 可以从retriever获取
        )
    
    except Exception as e:
        logger.error(f"分析失败: {e}")
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )


@app.post("/analyze/financial")
async def analyze_financial(request: FinancialAnalysisRequest):
    """
    财务分析接口
    
    对公司进行全面的财务分析
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = get_assistant().analyze_financial(
            company=request.company,
            year=request.year,
            metrics=request.metrics
        )
        
        return {
            "success": True,
            "data": result,
            "analysis_type": "financial_analysis"
        }
    
    except Exception as e:
        logger.error(f"财务分析失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/analyze/compare")
async def compare_companies(request: CompareRequest):
    """
    公司对比分析接口
    
    对比多家公司的财务和经营情况
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = get_assistant().compare_companies(
            companies=request.companies,
            metrics=request.metrics
        )
        
        return {
            "success": True,
            "data": result,
            "analysis_type": "industry_comparison"
        }
    
    except Exception as e:
        logger.error(f"对比分析失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/analyze/risk")
async def assess_risk(
    company: str,
    year: Optional[str] = None
):
    """
    风险评估接口
    
    评估投资风险
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = get_assistant().assess_risk(company=company, year=year)
        
        return {
            "success": True,
            "data": result,
            "analysis_type": "risk_assessment"
        }
    
    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/analyze/summary")
async def summarize_report(
    company: str,
    year: Optional[str] = None
):
    """
    年报摘要接口
    
    提取年报关键信息摘要
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = get_assistant().summarize_report(company=company, year=year)
        
        return {
            "success": True,
            "data": result,
            "analysis_type": "summary"
        }
    
    except Exception as e:
        logger.error(f"摘要提取失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/analyze/advice")
async def investment_advice(
    company: str,
    year: Optional[str] = None
):
    """
    投资建议接口
    
    提供投资建议和估值分析
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = get_assistant().investment_advice(company=company, year=year)
        
        return {
            "success": True,
            "data": result,
            "analysis_type": "investment_advice"
        }
    
    except Exception as e:
        logger.error(f"投资建议生成失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """
    获取知识库统计信息
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        stats = get_assistant().get_knowledge_base_stats()
        return {
            "success": True,
            "data": stats
        }
    
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/knowledge-base/documents")
async def add_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_name: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    stock_code: Optional[str] = Form(None)
):
    """
    添加文档到知识库
    
    上传年报PDF文件
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 保存上传的文件
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 构建元数据
        metadata = {
            "company_name": company_name,
            "year": year,
            "stock_code": stock_code,
            "source": file.filename
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # 后台任务：处理文档
        def process_document():
            get_assistant().add_documents_to_knowledge_base(
                documents_path=str(file_path),
                metadata=metadata
            )
        
        background_tasks.add_task(process_document)
        
        return {
            "success": True,
            "message": "文档已接收，正在后台处理",
            "file_path": str(file_path)
        }
    
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/analysis-types")
async def get_analysis_types():
    """
    获取支持的分析类型列表
    """
    try:
        descriptions = {
            "financial_analysis": "财务分析 - 分析财务报表和指标",
            "business_review": "经营回顾 - 解读经营情况",
            "risk_assessment": "风险评估 - 评估投资风险",
            "industry_comparison": "行业对比 - 公司对比分析",
            "trend_prediction": "趋势预测 - 预测发展趋势",
            "summary": "摘要提取 - 提取关键信息",
            "investment_advice": "投资建议 - 估值和投资建议",
            "general": "通用问答 - 一般性问题"
        }
        
        data = []
        for t in AnalysisType:
            data.append({
                "type": t.value,
                "name": t.name,
                "description": descriptions.get(t.value, "未知类型")
            })
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"获取分析类型失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": []
        }


@app.post("/chat")
async def chat(request: AnalyzeRequest):
    """
    对话接口（流式输出）
    
    支持SSE流式输出
    """
    if get_assistant() is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 检查是否请求流式输出
        if request.stream:
            # 执行分析并获取生成器
            generator = get_assistant().analyze(
                question=request.question,
                company=request.company,
                year=request.year,
                analysis_type=request.analysis_type,
                stream=True,
                enable_thinking=request.enable_thinking,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                tools=request.tools
            )
            
            # SSE 生成器函数
            async def sse_generator():
                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()

                def _produce():
                    try:
                        for chunk in generator:
                            queue.put_nowait(chunk)
                        queue.put_nowait(None)  # 结束信号
                    except Exception as e:
                        queue.put_nowait(("error", str(e)))

                loop.run_in_executor(None, _produce)

                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            yield "data: [DONE]\n\n"
                            break
                        if isinstance(chunk, tuple) and chunk[0] == "error":
                            logger.error("流式输出异常: %s", chunk[1])
                            yield f"data: {json.dumps({'error': chunk[1]})}\n\n"
                            break
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                except Exception as e:
                    logger.error("流式输出异常: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")
        
        else:
            # 非流式输出
            result = get_assistant().analyze(
                question=request.question,
                company=request.company,
                year=request.year,
                analysis_type=request.analysis_type,
                stream=False,
                enable_thinking=request.enable_thinking,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                tools=request.tools
            )
            
            return {
                "success": True,
                "data": result
            }
    
    except Exception as e:
        logger.error(f"对话失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="年报分析助手API服务")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000,
                       help="服务端口")
    parser.add_argument("--workers", type=int, default=1,
                       help="工作进程数")
    
    args = parser.parse_args()
    
    # 启动服务
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )


if __name__ == "__main__":
    main()
