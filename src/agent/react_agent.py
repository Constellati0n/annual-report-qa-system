#!/usr/bin/env python3
"""
ReAct Agent - 实现思考-行动-观察循环的年报分析助手

ReAct (Reasoning + Acting) 核心机制:
1. Thought: 分析用户问题，规划解决步骤
2. Action: 调用工具获取信息
3. Observation: 观察工具返回结果
4. 循环直到获得足够信息
5. Final Answer: 给出最终回答
"""

import os
import json
from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool, Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLanguageModel

# 导入自定义工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.pdf_reader import read_annual_report, pdf_tools
from tools.web_search import search_company_info, search_industry_analysis, web_search_tools


# ReAct Agent的Prompt模板
REACT_PROMPT_TEMPLATE = """你是一个专业的年报分析助手，擅长分析上市公司财务报表、经营情况和行业趋势。

你可以使用以下工具来帮助用户分析问题:

{tools}

工具名称: {tool_names}

请使用以下格式进行思考和行动:

Question: 用户提出的问题
Thought: 我需要分析这个问题，并决定使用什么工具
Action: 要使用的工具名称（必须是上面列出的工具之一）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (这个Thought/Action/Action Input/Observation可以重复多次)
Thought: 我现在有足够的信息来回答问题了
Final Answer: 给用户的最终回答

重要提示:
1. 每次只能使用一个工具
2. 必须严格按照格式输出
3. 如果一次工具调用没有获得足够信息，可以继续调用其他工具
4. 回答要专业、准确、有条理

开始!

Question: {input}
Thought:{agent_scratchpad}
"""


class AnnualReportReActAgent:
    """
    年报分析ReAct Agent

    实现完整的思考-行动-观察循环，能够:
    1. 理解用户关于年报的分析需求
    2. 自动调用PDF阅读工具获取年报数据
    3. 调用搜索工具补充最新信息
    4. 综合分析并给出专业回答
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[Tool]] = None,
        verbose: bool = True
    ):
        """
        初始化ReAct Agent

        Args:
            llm: 语言模型，如果为None则尝试从环境变量加载
            tools: 工具列表，如果为None则使用默认工具
            verbose: 是否打印详细日志
        """
        self.verbose = verbose

        # 初始化LLM
        if llm is None:
            self.llm = self._create_default_llm()
        else:
            self.llm = llm

        # 初始化工具
        if tools is None:
            self.tools = self._create_default_tools()
        else:
            self.tools = tools

        # 创建Agent
        self.agent_executor = self._create_agent()

    def _create_default_llm(self) -> BaseLanguageModel:
        """创建默认的LLM（使用本地Qwen模型）"""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            )

            return HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            print(f"加载本地模型失败: {e}")
            print("请确保已设置MODEL_PATH环境变量或传入llm参数")
            raise

    def _create_default_tools(self) -> List[Tool]:
        """创建默认工具列表"""
        tools = [
            Tool(
                name="read_annual_report",
                func=read_annual_report,
                description="""读取上市公司年报PDF文件内容。
                输入: 公司股票代码(如"000333")和可选的年份(如"2023")
                输出: 年报的财务数据、经营情况、风险因素等关键内容
                使用场景: 需要分析具体公司的年报数据时"""
            ),
            Tool(
                name="search_company_info",
                func=search_company_info,
                description="""搜索公司的最新信息和动态。
                输入: 公司名称或股票代码，以及信息类型(news/finance/industry/stock)
                输出: 搜索结果，包含标题、链接和摘要
                使用场景: 需要补充年报之外的最新信息时使用"""
            ),
            Tool(
                name="search_industry_analysis",
                func=search_industry_analysis,
                description="""搜索行业分析和市场研究报告。
                输入: 行业名称(如"家电行业"、"新能源汽车")
                输出: 行业分析报告
                使用场景: 需要进行行业对比或分析行业趋势时使用"""
            ),
        ]
        return tools

    def _create_agent(self) -> AgentExecutor:
        """创建ReAct Agent执行器"""

        # 创建Prompt
        prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

        # 创建Agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # 创建执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=10,  # 最大迭代次数
            handle_parsing_errors=True,
        )

        return agent_executor

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        分析用户查询

        Args:
            query: 用户的分析问题

        Returns:
            包含回答和中间过程的字典
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "success": True,
                "query": query,
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "answer": f"分析过程中出现错误: {str(e)}"
            }

    def chat(self, query: str) -> str:
        """
        简单的对话接口，只返回答案

        Args:
            query: 用户问题

        Returns:
            回答文本
        """
        result = self.analyze(query)
        return result.get("answer", "抱歉，无法处理您的请求")


class AnnualReportAssistant:
    """
    年报分析助手 - 高级封装

    提供更友好的接口，支持:
    - 多轮对话
    - 上下文记忆
    - 批量分析
    """

    def __init__(self, agent: Optional[AnnualReportReActAgent] = None):
        self.agent = agent or AnnualReportReActAgent()
        self.history: List[Dict[str, str]] = []

    def ask(self, question: str, use_history: bool = True) -> str:
        """
        提问并获取回答

        Args:
            question: 问题
            use_history: 是否使用历史上下文

        Returns:
            回答
        """
        # 构建带上下文的查询
        if use_history and self.history:
            context = self._build_context()
            full_query = f"{context}\n\n新问题: {question}"
        else:
            full_query = question

        # 获取回答
        result = self.agent.analyze(full_query)
        answer = result.get("answer", "")

        # 保存到历史
        self.history.append({
            "question": question,
            "answer": answer,
            "success": result.get("success", False)
        })

        return answer

    def _build_context(self) -> str:
        """构建历史上下文"""
        context = "之前的对话:\n"
        for i, item in enumerate(self.history[-5:], 1):  # 只保留最近5轮
            context += f"Q{i}: {item['question']}\n"
            context += f"A{i}: {item['answer'][:200]}...\n\n"
        return context

    def clear_history(self):
        """清空对话历史"""
        self.history = []

    def batch_analyze(
        self,
        company_codes: List[str],
        questions: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        批量分析多家公司

        Args:
            company_codes: 公司股票代码列表
            questions: 问题模板列表

        Returns:
            分析结果字典
        """
        results = {}

        for code in company_codes:
            company_results = []
            for question in questions:
                # 替换问题中的占位符
                query = question.replace("{company}", code)
                result = self.agent.analyze(query)
                company_results.append({
                    "question": question,
                    "result": result
                })
            results[code] = company_results

        return results


# 便捷函数
def create_agent(
    model_path: Optional[str] = None,
    use_web_search: bool = True
) -> AnnualReportReActAgent:
    """
    快速创建Agent

    Args:
        model_path: 模型路径
        use_web_search: 是否启用联网搜索

    Returns:
        ReAct Agent实例
    """
    if model_path:
        os.environ["MODEL_PATH"] = model_path

    tools = None
    if not use_web_search:
        # 只使用PDF阅读工具
        tools = [Tool(
            name="read_annual_report",
            func=read_annual_report,
            description="读取上市公司年报PDF文件内容"
        )]

    return AnnualReportReActAgent(tools=tools)


def quick_analyze(question: str, model_path: Optional[str] = None) -> str:
    """
    快速分析函数

    Args:
        question: 分析问题
        model_path: 模型路径（可选）

    Returns:
        分析结果
    """
    agent = create_agent(model_path=model_path)
    return agent.chat(question)


# 示例用法
if __name__ == "__main__":
    # 创建Agent
    agent = create_agent()

    # 测试问题
    test_questions = [
        "请分析美的集团2023年的财务状况",
        "格力电器和美的集团相比，哪家公司的盈利能力更强？",
        "最近有哪些关于比亚迪的重要新闻？",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        answer = agent.chat(question)
        print(f"回答: {answer}")
