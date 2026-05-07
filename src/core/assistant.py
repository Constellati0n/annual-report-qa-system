"""
年报分析助手核心模块
整合RAG检索、模型生成、提示词工程
"""
import os
import sys
from typing import Optional, Dict, List, Any, AsyncGenerator
from pathlib import Path
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 延迟导入避免循环导入
# from src.rag import get_retriever, get_embedding_manager, get_vector_store
# from src.rag.document_processor import DocumentProcessor

# 导入分析类型
from src.prompts import AnalysisType

# 导入 LLM 客户端
try:
    from client.llm_chat_qwen3 import Qwen3ChatClient as LLMChatClient
except ImportError:
    from client.llm_chat import LLMChatClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnualReportAssistant:
    """年报分析助手 - Qwen3-8B 版本"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "/mnt/workspace/models/llm/qwen/Qwen3-8B",
        vector_store_path: str = "./data/vector_db",
        use_rag: bool = True,
        load_in_4bit: bool = False,  # Qwen3 默认使用 bfloat16
        max_length: int = 32768,  # Qwen3 支持 32K 上下文
        enable_thinking: bool = True,  # Qwen3 思考模式
        enable_tool_call: bool = True
    ):
        """
        初始化年报分析助手
        
        Args:
            model_path: 微调模型路径
            base_model: 基础模型名称
            vector_store_path: 向量数据库路径
            use_rag: 是否使用RAG
            load_in_4bit: 是否使用4bit量化
        """
        self.use_rag = use_rag
        self.model_path = model_path
        self.base_model = base_model
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.enable_tool_call = enable_tool_call
        
        logger.info("=" * 60)
        logger.info("初始化年报分析助手 (Qwen3)")
        logger.info("=" * 60)
        logger.info(f"最大上下文: {max_length}")
        logger.info(f"思考模式: {enable_thinking}")
        
        # 初始化Qwen3 LLM客户端
        logger.info("加载 Qwen3 模型...")
        from client.llm_chat_qwen3 import Qwen3ChatClient
        self.llm_client = Qwen3ChatClient(
            model_path=model_path,
            base_model=base_model,
            load_in_4bit=load_in_4bit,
            max_length=max_length,
            enable_thinking=enable_thinking,
            enable_tool_call=enable_tool_call
        )
        
        # 初始化RAG组件（延迟导入避免循环导入）
        if use_rag:
            logger.info("初始化RAG组件...")
            from src.rag import get_embedding_manager, get_vector_store, get_retriever
            from src.prompts import get_prompt_manager
            self.embedding_manager = get_embedding_manager()
            self.vector_store = get_vector_store(persist_directory=vector_store_path)
            self.retriever = get_retriever()
            self.prompt_manager = get_prompt_manager()
            logger.info("RAG组件初始化完成")
        
        logger.info("=" * 60)
        logger.info("年报分析助手初始化完成")
        logger.info("=" * 60)
    
    def analyze(
        self,
        question: str,
        company: Optional[str] = None,
        year: Optional[str] = None,
        analysis_type: Optional[str] = None,
        stream: bool = False,
        enable_thinking: Optional[bool] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        tools: Optional[List[Dict]] = None
    ) -> Any:
        """
        分析年报问题 - 支持 Qwen3 特性
        
        Args:
            question: 用户问题
            company: 公司名称（用于过滤）
            year: 年份（用于过滤）
            analysis_type: 分析类型
            stream: 是否流式输出
            enable_thinking: 是否启用思考模式 (Qwen3)
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            tools: 工具定义列表
            
        Returns:
            分析结果 (若 stream=True 则返回生成器)
        """
        # 1. 确定分析类型
        if analysis_type:
            try:
                analysis_type_enum = AnalysisType(analysis_type)
            except ValueError:
                analysis_type_enum = self.prompt_manager.detect_analysis_type(question)
        else:
            if self.use_rag:
                analysis_type_enum = self.prompt_manager.detect_analysis_type(question)
            else:
                analysis_type_enum = AnalysisType.GENERAL
        
        logger.info(f"分析类型: {analysis_type_enum.value}")
        
        # 2. 构建过滤条件
        filter_dict = {}
        if company:
            filter_dict["company_name"] = company
        if year:
            filter_dict["year"] = year
        
        # 3. RAG检索（如果启用）
        context = ""
        if self.use_rag:
            logger.info("执行RAG检索...")
            
            # 先尝试带过滤条件的检索
            retrieval_results = self.retriever.retrieve(
                query=question,
                filter_dict=filter_dict if filter_dict else None
            )
            
            # 如果带过滤条件没有结果，尝试不带过滤条件
            if not retrieval_results and filter_dict:
                logger.info("带过滤条件未检索到结果，尝试全局检索...")
                retrieval_results = self.retriever.retrieve(
                    query=question,
                    filter_dict=None
                )
            
            if retrieval_results:
                context = self.retriever.format_context(retrieval_results)
                logger.info(f"检索到 {len(retrieval_results)} 条相关文档")
            else:
                logger.warning("未检索到相关文档")
                context = "未找到相关参考资料，将基于一般知识回答。"
        
        # 4. 构建提示词 - 使用简洁格式
        system_prompt = "你是一位专业的企业年报分析专家，擅长财务分析、行业洞察和风险评估。请基于提供的信息给出专业、准确的回答。"

        # 构建简洁的用户提示词
        if context and context != "未找到相关参考资料，将基于一般知识回答。":
            user_prompt = f"问题：{question}\n\n相关信息：{context}\n\n请回答上述问题。"
        else:
            user_prompt = f"问题：{question}\n\n请回答上述问题。"
        
        # 5. 调用模型生成回答
        logger.info("生成回答...")
        
        # Qwen3 参数
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        
        # 生成回答
        response = self.llm_client.chat(
            message=user_prompt,
            system_prompt=system_prompt,
            stream=stream,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=thinking,
            tools=tools
        )
        
        return response
    
    def analyze_financial(
        self,
        company: str,
        metrics: Optional[List[str]] = None,
        year: Optional[str] = None
    ) -> str:
        """
        财务分析专用接口
        
        Args:
            company: 公司名称
            metrics: 要分析的指标列表
            year: 年份
            
        Returns:
            财务分析报告
        """
        metrics = metrics or ["盈利能力", "偿债能力", "运营效率", "成长能力"]
        
        question = f"请对{company}" + (f"{year}年" if year else "") + "的财务状况进行全面分析，重点关注：" + "、".join(metrics)
        
        return self.analyze(
            question=question,
            company=company,
            year=year,
            analysis_type=AnalysisType.FINANCIAL_ANALYSIS.value
        )
    
    def compare_companies(
        self,
        companies: List[str],
        metrics: Optional[List[str]] = None
    ) -> str:
        """
        公司对比分析
        
        Args:
            companies: 公司名称列表
            metrics: 对比指标
            
        Returns:
            对比分析报告
        """
        if len(companies) < 2:
            return "请提供至少两家公司进行对比"
        
        metrics = metrics or ["营收规模", "盈利能力", "成长性", "估值水平"]
        
        question = f"请将{companies[0]}与{', '.join(companies[1:])}进行对比分析，对比维度包括：" + "、".join(metrics)
        
        return self.analyze(
            question=question,
            analysis_type=AnalysisType.INDUSTRY_COMPARISON.value
        )
    
    def assess_risk(
        self,
        company: str,
        year: Optional[str] = None
    ) -> str:
        """
        风险评估
        
        Args:
            company: 公司名称
            year: 年份
            
        Returns:
            风险评估报告
        """
        question = f"请评估投资{company}" + (f"{year}年" if year else "") + "的风险"
        
        return self.analyze(
            question=question,
            company=company,
            year=year,
            analysis_type=AnalysisType.RISK_ASSESSMENT.value
        )
    
    def summarize_report(
        self,
        company: str,
        year: Optional[str] = None
    ) -> str:
        """
        年报摘要
        
        Args:
            company: 公司名称
            year: 年份
            
        Returns:
            年报摘要
        """
        question = f"请提取{company}" + (f"{year}年" if year else "") + "年报的关键信息摘要"
        
        return self.analyze(
            question=question,
            company=company,
            year=year,
            analysis_type=AnalysisType.SUMMARY.value
        )
    
    def investment_advice(
        self,
        company: str,
        year: Optional[str] = None
    ) -> str:
        """
        投资建议
        
        Args:
            company: 公司名称
            year: 年份
            
        Returns:
            投资建议报告
        """
        question = f"请对{company}" + (f"{year}年" if year else "") + "给出投资建议"
        
        return self.analyze(
            question=question,
            company=company,
            year=year,
            analysis_type=AnalysisType.INVESTMENT_ADVICE.value
        )
    
    def add_documents_to_knowledge_base(
        self,
        documents_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        添加文档到知识库
        
        Args:
            documents_path: 文档路径（文件或目录）
            metadata: 元数据
        """
        if not self.use_rag:
            logger.warning("RAG未启用，无法添加文档")
            return
        
        logger.info(f"添加文档到知识库: {documents_path}")
        
        # 处理文档（延迟导入）
        from src.rag.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        path = Path(documents_path)
        if path.is_file():
            if path.suffix == '.pdf':
                chunks = processor.process_pdf(path, metadata)
            else:
                text = path.read_text(encoding='utf-8')
                chunks = processor.process_text(text, metadata)
        elif path.is_dir():
            chunks = processor.process_directory(path)
        else:
            logger.error(f"无效路径: {documents_path}")
            return
        
        # 编码并添加到向量库
        logger.info(f"编码 {len(chunks)} 个文本块...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.encode(texts, show_progress=True)
        
        # 创建文档对象
        from src.rag import Document
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=chunk.id,
                content=chunk.content,
                metadata={**chunk.metadata, **(metadata or {})},
                embedding=embeddings[i]
            )
            documents.append(doc)
        
        # 添加到向量库
        self.vector_store.add_documents(documents)
        
        logger.info(f"成功添加 {len(documents)} 个文档到知识库")
    
    def get_knowledge_base_stats(self) -> Dict:
        """获取知识库统计信息"""
        if not self.use_rag:
            return {"error": "RAG未启用"}
        
        return {
            "vector_store": self.vector_store.get_stats(),
            "retriever": self.retriever.get_stats()
        }
    
    def interactive_mode(self):
        """交互式对话模式"""
        print("\n" + "=" * 60)
        print("年报分析助手 - 交互式模式")
        print("=" * 60)
        print("支持的命令：")
        print("  /financial <公司名> [年份] - 财务分析")
        print("  /compare <公司1> <公司2> ... - 公司对比")
        print("  /risk <公司名> [年份] - 风险评估")
        print("  /summary <公司名> [年份] - 年报摘要")
        print("  /advice <公司名> [年份] - 投资建议")
        print("  /stats - 知识库统计")
        print("  /quit - 退出")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "/quit":
                    print("再见！")
                    break
                
                if user_input.lower() == "/stats":
                    stats = self.get_knowledge_base_stats()
                    print("\n知识库统计：")
                    print(stats)
                    continue
                
                # 解析命令
                if user_input.startswith("/"):
                    parts = user_input.split()
                    command = parts[0].lower()
                    args = parts[1:]
                    
                    if command == "/financial" and len(args) >= 1:
                        company = args[0]
                        year = args[1] if len(args) > 1 else None
                        print(f"\n助手: 正在进行{company}" + (f"{year}年" if year else "") + "的财务分析...")
                        response = self.analyze_financial(company, year=year)
                        print(f"\n{response}")
                    
                    elif command == "/compare" and len(args) >= 2:
                        companies = args
                        print(f"\n助手: 正在对比分析{', '.join(companies)}...")
                        response = self.compare_companies(companies)
                        print(f"\n{response}")
                    
                    elif command == "/risk" and len(args) >= 1:
                        company = args[0]
                        year = args[1] if len(args) > 1 else None
                        print(f"\n助手: 正在评估{company}的投资风险...")
                        response = self.assess_risk(company, year=year)
                        print(f"\n{response}")
                    
                    elif command == "/summary" and len(args) >= 1:
                        company = args[0]
                        year = args[1] if len(args) > 1 else None
                        print(f"\n助手: 正在提取{company}年报摘要...")
                        response = self.summarize_report(company, year=year)
                        print(f"\n{response}")
                    
                    elif command == "/advice" and len(args) >= 1:
                        company = args[0]
                        year = args[1] if len(args) > 1 else None
                        print(f"\n助手: 正在生成{company}的投资建议...")
                        response = self.investment_advice(company, year=year)
                        print(f"\n{response}")
                    
                    else:
                        print("\n助手: 未知命令或参数不足，请查看支持的命令列表")
                
                else:
                    # 普通问答
                    print("\n助手: ", end="", flush=True)
                    response = self.analyze(user_input)
                    print(response)
            
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error(f"错误: {e}")
                print(f"\n助手: 发生错误: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="年报分析助手 - Qwen3-8B")
    parser.add_argument("--model-path", type=str, default=None,
                       help="微调模型路径（PEFT adapter）")
    parser.add_argument("--base-model", type=str, default="/mnt/workspace/models/llm/qwen/Qwen3-8B",
                       help="基础模型路径")
    parser.add_argument("--vector-store", type=str, default="./data/vector_db",
                       help="向量数据库路径")
    parser.add_argument("--no-rag", action="store_true",
                       help="不使用RAG")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="使用4bit量化（Qwen3默认使用bfloat16）")

    args = parser.parse_args()

    # 创建助手
    assistant = AnnualReportAssistant(
        model_path=args.model_path,
        base_model=args.base_model,
        vector_store_path=args.vector_store,
        use_rag=not args.no_rag,
        load_in_4bit=args.load_in_4bit
    )

    # 启动交互模式
    assistant.interactive_mode()


if __name__ == "__main__":
    main()
