"""
PDF年报解析模块
支持文本提取、表格提取和结构化处理
"""
import os
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import PyPDF2
from PyPDF2 import PdfReader


class PDFParser:
    """PDF年报解析器"""
    
    def __init__(self):
        self.supported_sections = {
            "重要提示": "important_notes",
            "公司基本情况": "company_info",
            "会计数据": "financial_data",
            "经营情况": "business_operation",
            "董事会报告": "board_report",
            "重要事项": "significant_events",
            "股份变动": "share_changes",
            "股东情况": "shareholders",
            "董事监事高管": "management",
            "公司治理": "corporate_governance",
            "财务报告": "financial_statements",
            "资产负债表": "balance_sheet",
            "利润表": "income_statement",
            "现金流量表": "cash_flow"
        }
    
    def extract_text(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        提取PDF文本内容
        
        Args:
            pdf_path: PDF文件路径
            max_pages: 最大提取页数
            
        Returns:
            提取的文本内容
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            pages_to_read = len(reader.pages)
            if max_pages:
                pages_to_read = min(max_pages, pages_to_read)
            
            for i, page in enumerate(reader.pages[:pages_to_read]):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {i+1} ---\n"
                        text += page_text
                except Exception as e:
                    print(f"解析第{i+1}页失败: {e}")
                    continue
            
            return text
            
        except Exception as e:
            print(f"PDF解析失败 {pdf_path}: {e}")
            return ""
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        提取PDF元数据
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            元数据字典
        """
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata
            
            return {
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "subject": metadata.subject if metadata else "",
                "creator": metadata.creator if metadata else "",
                "producer": metadata.producer if metadata else "",
                "pages": len(reader.pages),
                "file_size": os.path.getsize(pdf_path)
            }
            
        except Exception as e:
            print(f"提取元数据失败 {pdf_path}: {e}")
            return {"pages": 0, "file_size": 0}
    
    def parse_annual_report(self, pdf_path: str) -> Dict:
        """
        解析年报，提取结构化信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            结构化年报数据
        """
        # 提取完整文本
        full_text = self.extract_text(pdf_path)
        
        # 提取元数据
        metadata = self.extract_metadata(pdf_path)
        
        # 解析基本信息
        basic_info = self._extract_basic_info(full_text)
        
        # 提取财务数据
        financial_data = self._extract_financial_data(full_text)
        
        # 提取章节内容
        sections = self._extract_sections(full_text)
        
        return {
            "file_path": pdf_path,
            "metadata": metadata,
            "basic_info": basic_info,
            "financial_data": financial_data,
            "sections": sections,
            "full_text": full_text[:50000]  # 限制存储大小
        }
    
    def _extract_basic_info(self, text: str) -> Dict:
        """提取年报基本信息"""
        info = {
            "stock_code": "",
            "stock_name": "",
            "report_year": "",
            "company_name": ""
        }
        
        # 提取股票代码
        code_patterns = [
            r"股票代码[：:]\s*(\d{6})",
            r"证券代码[：:]\s*(\d{6})",
            r"(?:公司|股份).*?(\d{6})"
        ]
        for pattern in code_patterns:
            match = re.search(pattern, text)
            if match:
                info["stock_code"] = match.group(1)
                break
        
        # 提取股票简称
        name_patterns = [
            r"股票简称[：:]\s*([^\n]{2,20})",
            r"证券简称[：:]\s*([^\n]{2,20})"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                info["stock_name"] = match.group(1).strip()
                break
        
        # 提取报告年份
        year_patterns = [
            r"(\d{4})\s*年度报告",
            r"(\d{4})\s*年年度报告",
            r"(\d{4})\s*年度"
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                info["report_year"] = match.group(1)
                break
        
        # 提取公司全称
        company_patterns = [
            r"公司名称[：:]\s*([^\n]{5,50})",
            r"公司全称[：:]\s*([^\n]{5,50})",
            r"(^[^\n]{5,50}股份有限公司)"
        ]
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                info["company_name"] = match.group(1).strip()
                break
        
        return info
    
    def _extract_financial_data(self, text: str) -> Dict:
        """提取关键财务数据"""
        financial = {
            "total_revenue": None,
            "net_profit": None,
            "total_assets": None,
            "net_assets": None,
            "eps": None  # 每股收益
        }
        
        # 营业收入
        revenue_patterns = [
            r"营业收入[\s\w]*?[：:]\s*([\d,\.]+)",
            r"营业总收入[\s\w]*?[：:]\s*([\d,\.]+)"
        ]
        for pattern in revenue_patterns:
            match = re.search(pattern, text)
            if match:
                financial["total_revenue"] = self._parse_number(match.group(1))
                break
        
        # 净利润
        profit_patterns = [
            r"归属于.*净利润[\s\w]*?[：:]\s*([\d,\.]+)",
            r"净利润[\s\w]*?[：:]\s*([\d,\.]+)"
        ]
        for pattern in profit_patterns:
            match = re.search(pattern, text)
            if match:
                financial["net_profit"] = self._parse_number(match.group(1))
                break
        
        # 总资产
        assets_patterns = [
            r"资产总计[\s\w]*?[：:]\s*([\d,\.]+)",
            r"总资产[\s\w]*?[：:]\s*([\d,\.]+)"
        ]
        for pattern in assets_patterns:
            match = re.search(pattern, text)
            if match:
                financial["total_assets"] = self._parse_number(match.group(1))
                break
        
        # 净资产
        net_assets_patterns = [
            r"归属于.*所有者权益[\s\w]*?[：:]\s*([\d,\.]+)",
            r"净资产[\s\w]*?[：:]\s*([\d,\.]+)"
        ]
        for pattern in net_assets_patterns:
            match = re.search(pattern, text)
            if match:
                financial["net_assets"] = self._parse_number(match.group(1))
                break
        
        # 每股收益
        eps_patterns = [
            r"基本每股收益[\s\w]*?[：:]\s*([\d,\.]+)",
            r"每股收益[\s\w]*?[：:]\s*([\d,\.]+)"
        ]
        for pattern in eps_patterns:
            match = re.search(pattern, text)
            if match:
                financial["eps"] = self._parse_number(match.group(1))
                break
        
        return financial
    
    def _extract_sections(self, text: str) -> Dict:
        """提取各章节内容"""
        sections = {}
        
        # 基于目录结构提取章节
        section_patterns = {
            "重要提示": r"重要提示[\s\S]{0,2000}(?=第一节|章|\d+\s+[^\d])",
            "公司简介": r"(?:第一节|第1节|一)[\s]*公司简介[\s\S]{0,3000}(?=第二节|第2节|二)",
            "会计数据": r"(?:主要会计数据|主要财务指标)[\s\S]{0,5000}(?=第三节|第3节|三)",
            "管理层讨论": r"(?:管理层讨论|经营情况)[\s\S]{0,10000}(?=重要事项|公司治理)",
            "重要事项": r"(?:重要事项|重大事项)[\s\S]{0,8000}(?=股份变动|股东情况)",
            "股东情况": r"(?:股东情况|股本变动)[\s\S]{0,5000}(?=董事|监事|高管)"
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text)
            if match:
                sections[section_name] = match.group(0).strip()[:3000]
        
        return sections
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """解析数字字符串"""
        try:
            # 移除逗号
            num_str = num_str.replace(",", "").replace("，", "")
            # 处理单位
            if "万" in num_str:
                return float(num_str.replace("万", "")) * 10000
            elif "亿" in num_str:
                return float(num_str.replace("亿", "")) * 100000000
            else:
                return float(num_str)
        except:
            return None
    
    def batch_parse(self, pdf_dir: str, output_dir: str) -> List[Dict]:
        """
        批量解析PDF文件
        
        Args:
            pdf_dir: PDF文件目录
            output_dir: 输出目录
            
        Returns:
            解析结果列表
        """
        results = []
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 遍历PDF文件
        for root, dirs, files in os.walk(pdf_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    print(f"正在解析: {file}")
                    
                    try:
                        result = self.parse_annual_report(pdf_path)
                        results.append(result)
                        
                        # 保存解析结果
                        output_file = os.path.join(
                            output_dir,
                            file.replace(".pdf", "_parsed.json")
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                            
                    except Exception as e:
                        print(f"解析失败 {file}: {e}")
        
        print(f"\n成功解析 {len(results)} 个文件")
        return results


def main():
    """测试解析功能"""
    parser = PDFParser()
    
    # 测试单个文件
    test_pdf = "data/raw/000001/000001_平安银行_2023年年度报告.pdf"
    if os.path.exists(test_pdf):
        result = parser.parse_annual_report(test_pdf)
        print("\n基本信息:")
        print(json.dumps(result["basic_info"], ensure_ascii=False, indent=2))
        print("\n财务数据:")
        print(json.dumps(result["financial_data"], ensure_ascii=False, indent=2))
    else:
        print(f"测试文件不存在: {test_pdf}")


if __name__ == "__main__":
    main()
