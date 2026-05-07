#!/usr/bin/env python3
"""
本地数据处理流程
1. PDF 文本提取
2. Chunk 切分
3. 训练集构建
4. 质量评估
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalDataProcessor:
    """本地数据处理器"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_pdfs": 0,
            "processed_pdfs": 0,
            "total_chunks": 0,
            "training_samples": 0
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """从 PDF 提取文本"""
        try:
            import pdfplumber
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"提取页面失败 {pdf_path}: {e}")
                        continue
            
            return text
        except ImportError:
            logger.error("pdfplumber 未安装，请运行: pip install pdfplumber")
            raise
        except Exception as e:
            logger.error(f"PDF 读取失败 {pdf_path}: {e}")
            return ""
    
    def parse_filename(self, pdf_path: Path) -> Dict[str, str]:
        """从文件名解析元数据"""
        filename = pdf_path.name
        
        # 匹配股票代码
        stock_code_match = re.match(r'(\d{6})', filename)
        stock_code = stock_code_match.group(1) if stock_code_match else ""
        
        # 匹配年份
        year_match = re.search(r'(20\d{2})', filename)
        year = year_match.group(1) if year_match else ""
        
        return {
            "stock_code": stock_code,
            "year": year,
            "filename": filename
        }
    
    def chunk_text(self, text: str, metadata: Dict, 
                   chunk_size: int = 800, 
                   chunk_overlap: int = 100) -> List[Dict]:
        """
        切分文本为 chunks
        使用改进的切分策略：按章节和语义切分
        """
        chunks = []
        
        # 首先尝试按年报章节切分
        sections = self._split_by_sections(text)
        
        for section_name, section_content in sections:
            section_metadata = {**metadata, "section": section_name}
            
            # 如果章节太长，进一步切分
            if len(section_content) > chunk_size:
                sub_chunks = self._recursive_split(
                    section_content, 
                    chunk_size, 
                    chunk_overlap
                )
                for i, chunk_content in enumerate(sub_chunks):
                    chunk_metadata = {
                        **section_metadata,
                        "chunk_index": i,
                        "chunk_id": f"{metadata.get('stock_code', 'unknown')}_{metadata.get('year', 'unknown')}_{section_name}_{i}"
                    }
                    chunks.append({
                        "content": chunk_content,
                        "metadata": chunk_metadata
                    })
            else:
                chunk_metadata = {
                    **section_metadata,
                    "chunk_id": f"{metadata.get('stock_code', 'unknown')}_{metadata.get('year', 'unknown')}_{section_name}_0"
                }
                chunks.append({
                    "content": section_content,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[tuple]:
        """按年报章节切分"""
        # 年报章节关键词
        section_patterns = [
            ("重要提示", r"重要提示|董事会声明"),
            ("公司简介", r"公司简介|公司基本情况|法定中文名称"),
            ("财务数据", r"主要会计数据|主要财务指标|会计数据和财务指标"),
            ("经营情况", r"管理层讨论|经营情况讨论|公司业务概要"),
            ("重要事项", r"重要事项|重大事项"),
            ("股份变动", r"股份变动|股本变动"),
            ("股东情况", r"股东情况|前十名股东"),
            ("董事监事", r"董事、监事|高级管理人员|董监高"),
            ("公司治理", r"公司治理|股东大会|董事会"),
            ("财务报告", r"财务报告|审计报告|财务报表"),
            ("资产负债表", r"资产负债表|合并资产负债表"),
            ("利润表", r"利润表|合并利润表"),
            ("现金流量", r"现金流量表|合并现金流量表"),
        ]
        
        sections = []
        current_pos = 0
        
        # 查找所有章节位置
        section_positions = []
        for section_name, pattern in section_patterns:
            for match in re.finditer(pattern, text):
                section_positions.append((match.start(), section_name, match.group()))
        
        # 按位置排序
        section_positions.sort(key=lambda x: x[0])
        
        # 提取章节内容
        for i, (pos, section_name, _) in enumerate(section_positions):
            start = pos
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)
            
            content = text[start:end].strip()
            if content and len(content) > 50:  # 过滤太短的章节
                sections.append((section_name, content))
        
        # 如果没有识别到章节，整体作为一个 chunk
        if not sections:
            sections.append(("全文", text))
        
        return sections
    
    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """递归切分长文本"""
        separators = ["\n\n", "\n", "。", ".", " ", ""]
        
        def split_recursive(text: str, separators: List[str]) -> List[str]:
            if len(text) <= chunk_size or not separators:
                return [text] if len(text) <= chunk_size else [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            separator = separators[0]
            parts = text.split(separator)
            
            result = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk) + len(part) + len(separator) <= chunk_size:
                    current_chunk = separator.join([current_chunk, part]) if current_chunk else part
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    if len(part) > chunk_size:
                        result.extend(split_recursive(part, separators[1:]))
                    else:
                        current_chunk = part
            
            if current_chunk:
                result.append(current_chunk)
            
            return result
        
        chunks = split_recursive(text, separators)
        
        # 添加重叠
        if chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
                overlapped.append(overlap_text + current_chunk)
            chunks = overlapped
        
        return chunks
    
    def extract_financial_data(self, text: str) -> Dict[str, Any]:
        """提取财务数据"""
        data = {}
        
        # 营业收入
        revenue_patterns = [
            r'营业总收入[\s]*([\d,\.]+)[\s]*万元',
            r'营业收入[\s]*([\d,\.]+)[\s]*万元',
        ]
        for pattern in revenue_patterns:
            match = re.search(pattern, text)
            if match:
                data['revenue'] = match.group(1).replace(',', '')
                break
        
        # 净利润
        profit_patterns = [
            r'净利润[\s]*([\d,\.]+)[\s]*万元',
            r'归属于.*?净利润[\s]*([\d,\.]+)[\s]*万元',
        ]
        for pattern in profit_patterns:
            match = re.search(pattern, text)
            if match:
                data['net_profit'] = match.group(1).replace(',', '')
                break
        
        # 每股收益
        eps_match = re.search(r'每股收益[\s]*([\d\.]+)', text)
        if eps_match:
            data['eps'] = eps_match.group(1)
        
        # ROE
        roe_match = re.search(r'净资产收益率[\s]*([\d\.]+)', text)
        if roe_match:
            data['roe'] = roe_match.group(1)
        
        return data
    
    def generate_training_samples(self, chunks: List[Dict], financial_data: Dict) -> List[Dict]:
        """生成训练样本"""
        samples = []
        
        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk["metadata"]
            
            stock_code = metadata.get("stock_code", "")
            year = metadata.get("year", "")
            section = metadata.get("section", "")
            
            # 跳过太短的 chunks
            if len(content) < 100:
                continue
            
            # 1. 财务数据提取任务
            if section in ["财务数据", "财务报告", "利润表"]:
                if financial_data.get("revenue") or financial_data.get("net_profit"):
                    samples.append({
                        "instruction": f"从以下{stock_code}年报文本中提取关键财务指标",
                        "input": f"{stock_code} {year}年年报\n{content[:1000]}",
                        "output": self._format_financial_output(financial_data),
                        "category": "extraction",
                        "source": f"{stock_code}_{year}"
                    })
            
            # 2. 问答任务
            if section == "经营情况":
                samples.append({
                    "instruction": f"请简要介绍{stock_code}的{year}年度业绩表现",
                    "input": content[:1500],
                    "output": self._generate_summary(content, financial_data),
                    "category": "qa",
                    "source": f"{stock_code}_{year}"
                })
            
            # 3. 财务分析任务
            if section in ["财务数据", "财务报告"]:
                samples.append({
                    "instruction": f"分析{stock_code}的盈利能力",
                    "input": content[:1500],
                    "output": self._generate_analysis(content, financial_data),
                    "category": "financial_analysis",
                    "source": f"{stock_code}_{year}"
                })
        
        return samples
    
    def _format_financial_output(self, data: Dict) -> str:
        """格式化财务数据输出"""
        lines = []
        if data.get("revenue"):
            lines.append(f"营业收入: {data['revenue']}万元")
        if data.get("net_profit"):
            lines.append(f"净利润: {data['net_profit']}万元")
        if data.get("eps"):
            lines.append(f"每股收益: {data['eps']}元")
        if data.get("roe"):
            lines.append(f"ROE: {data['roe']}%")
        return "\n".join(lines) if lines else "未找到财务数据"
    
    def _generate_summary(self, content: str, financial_data: Dict) -> str:
        """生成业绩摘要"""
        summary_parts = []
        
        if financial_data.get("revenue"):
            summary_parts.append(f"{financial_data['revenue']}万元营业收入")
        if financial_data.get("net_profit"):
            summary_parts.append(f"{financial_data['net_profit']}万元净利润")
        
        if summary_parts:
            return f"公司实现了{'，'.join(summary_parts)}。"
        
        # 从内容中提取关键信息
        key_points = []
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 20 and len(line) < 200:
                key_points.append(line)
        
        return " ".join(key_points) if key_points else "业绩表现良好，具体数据请查看财务报表。"
    
    def _generate_analysis(self, content: str, financial_data: Dict) -> str:
        """生成财务分析"""
        analysis = []
        
        if financial_data.get("revenue") and financial_data.get("net_profit"):
            revenue = float(financial_data["revenue"])
            profit = float(financial_data["net_profit"])
            margin = (profit / revenue * 100) if revenue > 0 else 0
            analysis.append(f"公司实现营业收入{revenue:.0f}万元，净利润{profit:.0f}万元，净利率为{margin:.2f}%。")
        
        if financial_data.get("roe"):
            analysis.append(f"净资产收益率为{financial_data['roe']}%，")
        
        return " ".join(analysis) if analysis else "公司盈利能力良好。"
    
    def process_all_pdfs(self, limit: int = None):
        """处理所有 PDF 文件"""
        logger.info("=" * 60)
        logger.info("开始处理 PDF 文件")
        logger.info("=" * 60)
        
        # 查找所有 PDF
        pdf_files = list(self.raw_dir.rglob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
        
        if limit:
            pdf_files = pdf_files[:limit]
            logger.info(f"限制处理前 {limit} 个文件")
        
        all_chunks = []
        all_training_samples = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n处理 [{i}/{len(pdf_files)}]: {pdf_path.name}")
            
            try:
                # 1. 提取文本
                text = self.extract_text_from_pdf(pdf_path)
                if not text:
                    logger.warning(f"  未能提取文本，跳过")
                    continue
                
                logger.info(f"  提取文本: {len(text)} 字符")
                
                # 2. 解析元数据
                metadata = self.parse_filename(pdf_path)
                logger.info(f"  股票代码: {metadata['stock_code']}, 年份: {metadata['year']}")
                
                # 3. 提取财务数据
                financial_data = self.extract_financial_data(text)
                logger.info(f"  财务数据: {financial_data}")
                
                # 4. Chunk 切分
                chunks = self.chunk_text(text, metadata)
                logger.info(f"  生成 {len(chunks)} 个 chunks")
                
                # 5. 生成训练样本
                samples = self.generate_training_samples(chunks, financial_data)
                logger.info(f"  生成 {len(samples)} 个训练样本")
                
                all_chunks.extend(chunks)
                all_training_samples.extend(samples)
                
                self.stats["processed_pdfs"] += 1
                
            except Exception as e:
                logger.error(f"  处理失败: {e}")
                continue
        
        self.stats["total_pdfs"] = len(pdf_files)
        self.stats["total_chunks"] = len(all_chunks)
        self.stats["training_samples"] = len(all_training_samples)
        
        # 保存结果
        self._save_results(all_chunks, all_training_samples)
        
        return all_chunks, all_training_samples
    
    def _save_results(self, chunks: List[Dict], samples: List[Dict]):
        """保存处理结果"""
        logger.info("\n" + "=" * 60)
        logger.info("保存处理结果")
        logger.info("=" * 60)
        
        # 保存 chunks
        chunks_file = self.processed_dir / "knowledge_base" / "docs_raw_chunks.json"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Chunks 已保存: {chunks_file} ({len(chunks)} 个)")
        
        # 保存训练集
        train_file = self.processed_dir / "training" / "train_dataset.json"
        train_file.parent.mkdir(parents=True, exist_ok=True)
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 训练集已保存: {train_file} ({len(samples)} 个样本)")
        
        # 保存统计信息
        stats_file = self.processed_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 统计信息已保存: {stats_file}")
        
        # 打印统计
        logger.info("\n处理统计:")
        logger.info(f"  PDF 文件: {self.stats['processed_pdfs']}/{self.stats['total_pdfs']}")
        logger.info(f"  Chunks: {self.stats['total_chunks']}")
        logger.info(f"  训练样本: {self.stats['training_samples']}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="本地数据处理流程")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="数据目录")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的 PDF 数量（用于测试）")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = LocalDataProcessor(args.data_dir)
    
    # 处理所有 PDF
    processor.process_all_pdfs(limit=args.limit)
    
    logger.info("\n" + "=" * 60)
    logger.info("数据处理完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
