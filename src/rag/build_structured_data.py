"""
构建结构化数据流程
从已解析的年报中提取结构化信息
"""
import os
import json
import glob
from typing import List, Dict
from pathlib import Path

from extractor import AnnualReportExtractor, StructuredAnnualReport


class StructuredDataBuilder:
    """结构化数据构建器"""
    
    def __init__(
        self,
        parsed_dir: str = None,
        output_dir: str = None
    ):
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        self.parsed_dir = parsed_dir or str(project_root / "data" / "processed" / "parsed")
        self.output_dir = output_dir or str(project_root / "data" / "processed" / "structured")
        
        # 确保目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化提取器
        self.extractor = AnnualReportExtractor()
    
    def load_parsed_reports(self) -> List[Dict]:
        """加载所有解析后的年报数据"""
        reports = []
        
        json_files = glob.glob(os.path.join(self.parsed_dir, "*_parsed.json"))
        print(f"找到 {len(json_files)} 个解析文件")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取必要信息
                text = data.get('full_text', '')
                basic_info = data.get('basic_info', {})
                
                if text:
                    reports.append({
                        'text': text,
                        'stock_code': basic_info.get('stock_code', ''),
                        'stock_name': basic_info.get('stock_name', ''),
                        'report_year': basic_info.get('report_year', ''),
                        'file_path': file_path
                    })
                    
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        return reports
    
    def extract_structured_data(self, reports: List[Dict]) -> List[StructuredAnnualReport]:
        """从报告中提取结构化数据"""
        structured_reports = []
        
        print(f"\n开始提取 {len(reports)} 份报告的结构化数据...")
        
        for i, report in enumerate(reports):
            print(f"[{i+1}/{len(reports)}] 处理 {report['stock_code']} {report['stock_name']} {report['report_year']}年报...")
            
            try:
                structured = self.extractor.extract(
                    text=report['text'],
                    stock_code=report['stock_code'],
                    stock_name=report['stock_name'],
                    report_year=report['report_year']
                )
                
                structured_reports.append(structured)
                
            except Exception as e:
                print(f"提取失败: {e}")
                continue
        
        print(f"\n提取完成，成功处理 {len(structured_reports)} 份报告")
        return structured_reports
    
    def save_structured_data(self, reports: List[StructuredAnnualReport]):
        """保存结构化数据"""
        # 保存为单个JSON文件
        all_data = [report.to_dict() for report in reports]
        
        output_path = os.path.join(self.output_dir, "structured_reports.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n结构化数据已保存: {output_path}")
        
        # 生成统计信息
        stats = self._generate_statistics(reports)
        
        stats_path = os.path.join(self.output_dir, "extraction_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存: {stats_path}")
        
        return stats
    
    def _generate_statistics(self, reports: List[StructuredAnnualReport]) -> Dict:
        """生成提取统计信息"""
        stats = {
            "total_reports": len(reports),
            "companies": list(set(r.stock_code for r in reports)),
            "years": list(set(r.report_year for r in reports)),
            "extraction_coverage": {}
        }
        
        # 统计各字段提取成功率
        field_counts = {
            "company_name": sum(1 for r in reports if r.company_info.company_full_name),
            "industry": sum(1 for r in reports if r.company_info.industry),
            "total_revenue": sum(1 for r in reports if r.financial_indicators.total_revenue),
            "net_profit": sum(1 for r in reports if r.financial_indicators.net_profit),
            "eps": sum(1 for r in reports if r.financial_indicators.eps),
            "roe": sum(1 for r in reports if r.financial_indicators.roe),
            "total_assets": sum(1 for r in reports if r.financial_indicators.total_assets),
            "controlling_shareholder": sum(1 for r in reports if r.shareholder_info.controlling_shareholder),
            "dividend_plan": sum(1 for r in reports if r.major_events.dividend_plan),
            "executive_summary": sum(1 for r in reports if r.executive_summary),
        }
        
        total = len(reports)
        stats["extraction_coverage"] = {
            field: f"{count}/{total} ({count/total*100:.1f}%)"
            for field, count in field_counts.items()
        }
        
        return stats
    
    def build(self):
        """执行完整的构建流程"""
        print("=" * 60)
        print("构建结构化数据")
        print("=" * 60)
        
        # 1. 加载解析后的年报
        print("\n【步骤1】加载解析后的年报数据")
        reports = self.load_parsed_reports()
        
        if not reports:
            print("没有可用的文档")
            return
        
        # 2. 提取结构化数据
        print("\n【步骤2】提取结构化数据")
        structured_reports = self.extract_structured_data(reports)
        
        # 3. 保存数据
        print("\n【步骤3】保存结构化数据")
        stats = self.save_structured_data(structured_reports)
        
        print("\n" + "=" * 60)
        print("结构化数据构建完成！")
        print("=" * 60)
        print(f"\n统计信息:")
        print(f"- 总报告数: {stats['total_reports']}")
        print(f"- 公司数量: {len(stats['companies'])}")
        print(f"- 数据覆盖:")
        for field, coverage in stats['extraction_coverage'].items():
            print(f"  - {field}: {coverage}")
        
        return structured_reports


def main():
    """主函数"""
    builder = StructuredDataBuilder()
    
    reports = builder.build()
    
    if reports:
        print("\n示例结构化数据:")
        report = reports[0]
        print(f"\n公司: {report.stock_name} ({report.stock_code})")
        print(f"年度: {report.report_year}")
        print(f"\n财务指标:")
        print(f"  - 营业收入: {report.financial_indicators.total_revenue}")
        print(f"  - 净利润: {report.financial_indicators.net_profit}")
        print(f"  - 每股收益: {report.financial_indicators.eps}")
        print(f"  - ROE: {report.financial_indicators.roe}%")
        print(f"\n公司信息:")
        print(f"  - 全称: {report.company_info.company_full_name}")
        print(f"  - 行业: {report.company_info.industry}")
        print(f"  - 注册地址: {report.company_info.registered_address}")


if __name__ == "__main__":
    main()
