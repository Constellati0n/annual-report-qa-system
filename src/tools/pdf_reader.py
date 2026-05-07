#!/usr/bin/env python3
"""
PDF阅读工具 - 让Agent能够读取本地年报PDF文件
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import pdfplumber
from langchain.tools import tool
from pydantic import BaseModel, Field


class PDFReaderInput(BaseModel):
    """PDF阅读工具输入参数"""
    company_code: str = Field(description="公司股票代码，如'000333'")
    year: Optional[str] = Field(default=None, description="年报年份，如'2023'，不指定则读取最新")


class PDFReaderTool:
    """PDF年报阅读工具"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def find_pdf(self, company_code: str, year: Optional[str] = None) -> Optional[Path]:
        """查找公司年报PDF文件"""
        company_dir = self.data_dir / company_code

        if not company_dir.exists():
            return None

        pdf_files = list(company_dir.glob("*.pdf"))

        if not pdf_files:
            return None

        if year:
            # 查找指定年份
            for pdf in pdf_files:
                if year in pdf.name:
                    return pdf

        # 返回最新的年报（按文件名中的年份排序）
        def extract_year(filename):
            match = re.search(r'(20\d{2})', str(filename))
            return match.group(1) if match else "0000"

        pdf_files.sort(key=lambda x: extract_year(x), reverse=True)
        return pdf_files[0]

    def extract_text(self, pdf_path: Path, max_pages: int = 50) -> str:
        """从PDF提取文本内容"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 读取前max_pages页（通常包含主要财务数据）
                for i, page in enumerate(pdf.pages):
                    if i >= max_pages:
                        break
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception:
                        continue
        except Exception as e:
            return f"PDF读取错误: {str(e)}"

        return text

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """提取关键章节内容"""
        sections = {}

        # 定义关键章节模式
        section_patterns = {
            "财务数据": [
                r'主要会计数据和财务指标(.*?)第二节',
                r'主要会计数据(.*?)第三节',
            ],
            "经营情况": [
                r'经营情况讨论与分析(.*?)第四节',
                r'管理层讨论与分析(.*?)第四节',
            ],
            "风险因素": [
                r'可能面对的风险(.*?)第四节',
                r'风险因素(.*?)第四节',
            ],
            "主营业务": [
                r'主营业务分析(.*?)第三节',
                r'收入与成本(.*?)第三节',
            ]
        }

        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # 限制长度
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    sections[section_name] = content
                    break

        return sections

    def read(self, company_code: str, year: Optional[str] = None) -> str:
        """
        读取公司年报内容

        Args:
            company_code: 股票代码
            year: 年份（可选）

        Returns:
            年报内容摘要
        """
        # 查找PDF
        pdf_path = self.find_pdf(company_code, year)

        if not pdf_path:
            available_years = self.get_available_years(company_code)
            if available_years:
                return f"未找到{'指定年份' if year else ''}年报。可用年份: {', '.join(available_years)}"
            return f"未找到公司 {company_code} 的年报文件"

        # 提取文本
        text = self.extract_text(pdf_path)

        if not text:
            return f"无法读取PDF内容: {pdf_path.name}"

        # 提取关键章节
        sections = self.extract_key_sections(text)

        # 构建输出
        result = f"【{company_code} {pdf_path.name} 年报内容】\n\n"

        if sections:
            for section_name, content in sections.items():
                result += f"=== {section_name} ===\n{content}\n\n"
        else:
            # 如果没有提取到特定章节，返回前2000字符
            result += text[:2000] + "..."

        return result

    def get_available_years(self, company_code: str) -> List[str]:
        """获取可用的年报年份"""
        company_dir = self.data_dir / company_code

        if not company_dir.exists():
            return []

        years = []
        for pdf in company_dir.glob("*.pdf"):
            match = re.search(r'(20\d{2})', pdf.name)
            if match:
                years.append(match.group(1))

        return sorted(set(years), reverse=True)


@tool
def read_annual_report(company_code: str, year: Optional[str] = None) -> str:
    """
    读取上市公司年报PDF文件内容

    使用此工具可以获取公司的年度财务数据、经营情况、风险因素等信息。

    Args:
        company_code: 公司股票代码，如"000333"(美的集团)、"000858"(五粮液)
        year: 年报年份，如"2023"。如果不指定，默认读取最新年份的年报

    Returns:
        年报的关键内容，包括财务数据、经营分析、风险提示等

    Example:
        read_annual_report("000333", "2023")  # 读取美的集团2023年年报
        read_annual_report("000333")  # 读取美的集团最新年报
    """
    reader = PDFReaderTool()
    return reader.read(company_code, year)


# 为LangChain Agent准备的工具列表
pdf_tools = [read_annual_report]
