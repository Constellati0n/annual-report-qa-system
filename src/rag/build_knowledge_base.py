"""
构建多层知识库
整合：原始文本Chunks + 结构化数据 + AI分析摘要
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from rag.chunker import chunk_documents, Chunk
from rag.extractor import StructuredDataExtractor
from rag.analyzer import RuleBasedAnalyzer


class MultiLayerKnowledgeBase:
    """多层知识库构建器"""
    
    def __init__(
        self,
        parsed_dir: str = None,
        output_dir: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        self.project_root = Path(__file__).parent.parent.parent
        self.parsed_dir = parsed_dir or str(self.project_root / "data" / "processed" / "parsed")
        self.output_dir = output_dir or str(self.project_root / "data" / "processed" / "knowledge_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.extractor = StructuredDataExtractor()
        self.analyzer = RuleBasedAnalyzer()
    
    def process_report(self, parsed_file: str) -> Dict:
        """
        处理单份年报，生成三层数据
        
        Args:
            parsed_file: 解析后的年报JSON文件路径
            
        Returns:
            包含三层数据的字典
        """
        print(f"\n处理文件: {Path(parsed_file).name}")
        
        # 加载解析数据
        with open(parsed_file, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        text = parsed_data.get('full_text', '')
        basic_info = parsed_data.get('basic_info', {})
        
        result = {
            "file_path": parsed_file,
            "basic_info": basic_info,
            "layers": {}
        }
        
        # Layer 1: 原始文本 Chunks
        print("  [Layer 1] 生成文本Chunks...")
        from rag.chunker import AnnualReportChunker
        chunker = AnnualReportChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.split_text(text, basic_info)
        result["layers"]["raw_chunks"] = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata
            }
            for c in chunks
        ]
        print(f"    生成 {len(chunks)} 个chunks")
        
        # Layer 2: 结构化数据
        print("  [Layer 2] 提取结构化数据...")
        structured_data = self.extractor.extract(text, basic_info)
        result["layers"]["structured_data"] = structured_data.to_dict()
        print(f"    提取完成")
        
        # Layer 3: AI分析摘要
        print("  [Layer 3] 生成分析摘要...")
        analysis = self.analyzer.analyze(structured_data.to_dict())
        result["layers"]["analysis"] = {
            "text_format": analysis.to_text(),
            "json_format": analysis.to_dict()
        }
        print(f"    分析完成")
        
        return result
    
    def build(self):
        """构建完整知识库"""
        print("=" * 70)
        print("构建多层知识库")
        print("=" * 70)
        print("\n数据层级:")
        print("  Layer 1: 原始文本Chunks (用于细节检索)")
        print("  Layer 2: 结构化数据 (用于精确查询)")
        print("  Layer 3: AI分析摘要 (用于快速洞察)")
        print("=" * 70)
        
        # 查找所有解析文件
        import glob
        parsed_files = glob.glob(os.path.join(self.parsed_dir, "*_parsed.json"))
        print(f"\n找到 {len(parsed_files)} 个解析文件")
        
        if not parsed_files:
            print("没有可用的文档")
            return
        
        # 处理每个文件
        all_reports = []
        for file_path in parsed_files:
            try:
                report_data = self.process_report(file_path)
                all_reports.append(report_data)
            except Exception as e:
                print(f"处理失败 {file_path}: {e}")
        
        # 保存完整知识库
        kb_path = os.path.join(self.output_dir, "knowledge_base.json")
        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        print(f"\n知识库已保存: {kb_path}")
        
        # 生成统计信息
        stats = self._generate_stats(all_reports)
        stats_path = os.path.join(self.output_dir, "kb_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存: {stats_path}")
        
        # 打印统计
        self._print_stats(stats)
        
        return all_reports, stats
    
    def _generate_stats(self, reports: List[Dict]) -> Dict:
        """生成统计信息"""
        total_raw_chunks = sum(len(r["layers"]["raw_chunks"]) for r in reports)
        
        # 收集所有分析文本用于向量化
        analysis_texts = []
        for r in reports:
            analysis = r["layers"].get("analysis", {})
            if "text_format" in analysis:
                analysis_texts.append({
                    "stock_code": r["basic_info"].get("stock_code", ""),
                    "stock_name": r["basic_info"].get("stock_name", ""),
                    "report_year": r["basic_info"].get("report_year", ""),
                    "content": analysis["text_format"]
                })
        
        return {
            "total_reports": len(reports),
            "total_raw_chunks": total_raw_chunks,
            "avg_chunks_per_report": total_raw_chunks / len(reports) if reports else 0,
            "analysis_documents": len(analysis_texts),
            "reports": [
                {
                    "stock_code": r["basic_info"].get("stock_code", ""),
                    "stock_name": r["basic_info"].get("stock_name", ""),
                    "report_year": r["basic_info"].get("report_year", ""),
                    "raw_chunks_count": len(r["layers"]["raw_chunks"])
                }
                for r in reports
            ]
        }
    
    def _print_stats(self, stats: Dict):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("知识库构建完成！")
        print("=" * 70)
        print(f"\n统计信息:")
        print(f"  - 年报总数: {stats['total_reports']}")
        print(f"  - 原始文本Chunks: {stats['total_raw_chunks']} 个")
        print(f"  - 平均每份年报Chunks: {stats['avg_chunks_per_report']:.1f} 个")
        print(f"  - 分析摘要文档: {stats['analysis_documents']} 个")
        print(f"\n数据文件位置: {self.output_dir}")
        print("=" * 70)
    
    def export_for_vector_db(self):
        """
        导出为向量数据库格式
        生成三种类型的文档用于向量化
        """
        kb_path = os.path.join(self.output_dir, "knowledge_base.json")
        if not os.path.exists(kb_path):
            print("知识库不存在，请先运行build()")
            return
        
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        # 准备三种文档集合
        documents = {
            "raw_chunks": [],      # 原始文本
            "structured": [],      # 结构化数据（转为文本）
            "analysis": []         # 分析摘要
        }
        
        for report in kb_data:
            stock_code = report["basic_info"].get("stock_code", "")
            stock_name = report["basic_info"].get("stock_name", "")
            report_year = report["basic_info"].get("report_year", "")
            
            base_metadata = {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "report_year": report_year
            }
            
            # Layer 1: 原始文本
            for chunk in report["layers"]["raw_chunks"]:
                documents["raw_chunks"].append({
                    "content": chunk["content"],
                    "metadata": {
                        **base_metadata,
                        "layer": "raw",
                        "chunk_id": chunk["chunk_id"],
                        "section": chunk["metadata"].get("section", "")
                    }
                })
            
            # Layer 2: 结构化数据（转为描述性文本）
            structured = report["layers"]["structured_data"]
            structured_text = self._structured_to_text(structured)
            documents["structured"].append({
                "content": structured_text,
                "metadata": {
                    **base_metadata,
                    "layer": "structured"
                }
            })
            
            # Layer 3: 分析摘要
            analysis_text = report["layers"]["analysis"]["text_format"]
            documents["analysis"].append({
                "content": analysis_text,
                "metadata": {
                    **base_metadata,
                    "layer": "analysis"
                }
            })
        
        # 保存
        for doc_type, docs in documents.items():
            output_path = os.path.join(self.output_dir, f"docs_{doc_type}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
            print(f"已导出 {len(docs)} 个 {doc_type} 文档: {output_path}")
        
        return documents
    
    def _structured_to_text(self, structured: Dict) -> str:
        """将结构化数据转为文本"""
        lines = []
        
        # 基本信息
        stock_name = structured.get("stock_name", "")
        year = structured.get("report_year", "")
        lines.append(f"{stock_name} {year}年度年报关键数据：")
        lines.append("")
        
        # 财务数据
        financials = structured.get("financials", {})
        lines.append("【财务指标】")
        if financials.get("total_revenue"):
            lines.append(f"营业收入: {self._format_number(financials['total_revenue'])}")
        if financials.get("net_profit"):
            lines.append(f"净利润: {self._format_number(financials['net_profit'])}")
        if financials.get("eps"):
            lines.append(f"每股收益: {financials['eps']}元")
        if financials.get("roe"):
            lines.append(f"净资产收益率(ROE): {financials['roe']}%")
        lines.append("")
        
        # 分红
        dividend = structured.get("dividend", {})
        if dividend.get("dividend_per_share"):
            lines.append(f"【分红方案】每股派息{dividend['dividend_per_share']}元")
            lines.append("")
        
        # 业务摘要
        if structured.get("business_summary"):
            lines.append("【业务概要】")
            lines.append(structured["business_summary"][:200])
            lines.append("")
        
        # 重大事项
        if structured.get("major_events"):
            lines.append("【重大事项】")
            for event in structured["major_events"][:3]:
                lines.append(f"- {event}")
            lines.append("")
        
        # 风险因素
        if structured.get("risk_factors"):
            lines.append("【风险提示】")
            for risk in structured["risk_factors"][:3]:
                lines.append(f"- {risk}")
        
        return "\n".join(lines)
    
    def _format_number(self, num: float) -> str:
        """格式化数字"""
        if num >= 100000000:
            return f"{num/100000000:.2f}亿元"
        elif num >= 10000:
            return f"{num/10000:.2f}万元"
        else:
            return f"{num:.2f}元"


def main():
    """主函数"""
    builder = MultiLayerKnowledgeBase(
        chunk_size=800,
        chunk_overlap=100
    )
    
    # 构建知识库
    kb_data, stats = builder.build()
    
    # 导出为向量数据库格式
    print("\n导出向量数据库格式...")
    documents = builder.export_for_vector_db()
    
    # 显示示例
    if kb_data:
        print("\n" + "=" * 70)
        print("示例 - Layer 3 分析摘要")
        print("=" * 70)
        sample_analysis = kb_data[0]["layers"]["analysis"]["text_format"]
        print(sample_analysis[:800] + "...")


if __name__ == "__main__":
    main()
