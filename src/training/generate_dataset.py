"""
生成微调训练数据集
基于已有的三层知识库生成问答对
"""
import json
import random
from pathlib import Path
from typing import List, Dict


class DatasetGenerator:
    """训练数据集生成器"""
    
    def __init__(self, knowledge_base_dir: str = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.kb_dir = Path(knowledge_base_dir) if knowledge_base_dir else self.project_root / "data" / "processed" / "knowledge_base"
        
        # 加载知识库
        self.kb_data = self._load_knowledge_base()
        
        # 问题模板
        self.question_templates = {
            "financial": [
                "{stock_name}{year}年的营业收入是多少？",
                "{stock_name}{year}年净利润增长情况如何？",
                "{stock_name}的每股收益(EPS)是多少？",
                "{stock_name}{year}年的ROE是多少？",
                "{stock_name}的资产负债情况如何？",
            ],
            "dividend": [
                "{stock_name}{year}年的分红方案是什么？",
                "{stock_name}每股派息多少？",
                "{stock_name}的分红率如何？",
            ],
            "business": [
                "{stock_name}的主营业务是什么？",
                "{stock_name}{year}年的经营情况如何？",
                "{stock_name}的业务模式是什么？",
            ],
            "risk": [
                "{stock_name}面临的主要风险有哪些？",
                "投资{stock_name}需要注意什么风险？",
                "{stock_name}{year}年有哪些风险因素？",
            ],
            "events": [
                "{stock_name}{year}年有哪些重大事项？",
                "{stock_name}报告期内发生了什么重要事件？",
            ],
            "analysis": [
                "{stock_name}{year}年的业绩表现如何？",
                "请分析{stock_name}的投资价值",
                "{stock_name}的投资亮点有哪些？",
                "对{stock_name}的投资建议是什么？",
            ],
            "comparison": [
                "{stock_name}{year1}年和{year2}年的业绩对比如何？",
                "{stock_name}的盈利能力有什么变化？",
            ]
        }
    
    def _load_knowledge_base(self) -> List[Dict]:
        """加载知识库"""
        kb_file = self.kb_dir / "knowledge_base.json"
        if not kb_file.exists():
            raise FileNotFoundError(f"知识库不存在: {kb_file}")
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_financial_qa(self) -> List[Dict]:
        """生成财务数据问答对"""
        qa_pairs = []
        
        for report in self.kb_data:
            basic = report["basic_info"]
            structured = report["layers"]["structured_data"]
            financials = structured.get("financials", {})
            
            stock_name = basic.get("stock_name", "该公司")
            year = basic.get("report_year", "")
            
            # 营收问答
            if financials.get("total_revenue"):
                revenue = financials["total_revenue"]
                revenue_str = self._format_number(revenue)
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年的营业收入是多少？",
                    "input": "",
                    "output": f"{stock_name}{year}年营业收入为{revenue_str}。",
                    "metadata": {
                        "type": "financial",
                        "metric": "revenue",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # 净利润问答
            if financials.get("net_profit"):
                profit = financials["net_profit"]
                profit_str = self._format_number(profit)
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年净利润是多少？",
                    "input": "",
                    "output": f"{stock_name}{year}年实现净利润{profit_str}。",
                    "metadata": {
                        "type": "financial",
                        "metric": "profit",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # EPS问答
            if financials.get("eps"):
                eps = financials["eps"]
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年的每股收益(EPS)是多少？",
                    "input": "",
                    "output": f"{stock_name}{year}年每股收益(EPS)为{eps}元。",
                    "metadata": {
                        "type": "financial",
                        "metric": "eps",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # ROE问答
            if financials.get("roe"):
                roe = financials["roe"]
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年的净资产收益率(ROE)是多少？",
                    "input": "",
                    "output": f"{stock_name}{year}年净资产收益率(ROE)为{roe}%，盈利能力{'优秀' if roe > 15 else '良好' if roe > 10 else '一般'}。",
                    "metadata": {
                        "type": "financial",
                        "metric": "roe",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
        
        return qa_pairs
    
    def generate_dividend_qa(self) -> List[Dict]:
        """生成分红相关问答对"""
        qa_pairs = []
        
        for report in self.kb_data:
            basic = report["basic_info"]
            structured = report["layers"]["structured_data"]
            dividend = structured.get("dividend", {})
            
            stock_name = basic.get("stock_name", "该公司")
            year = basic.get("report_year", "")
            
            if dividend.get("dividend_per_share"):
                dps = dividend["dividend_per_share"]
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年的分红方案是什么？",
                    "input": "",
                    "output": f"{stock_name}{year}年分红方案为每股派息{dps}元（含税），体现了公司对股东的回报。",
                    "metadata": {
                        "type": "dividend",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
        
        return qa_pairs
    
    def generate_analysis_qa(self) -> List[Dict]:
        """生成分析摘要问答对"""
        qa_pairs = []
        
        for report in self.kb_data:
            basic = report["basic_info"]
            analysis = report["layers"]["analysis"]["json_format"]
            
            stock_name = basic.get("stock_name", "该公司")
            year = basic.get("report_year", "")
            
            # 核心观点
            takeaways = analysis.get("key_takeaways", [])
            if takeaways:
                output = "\n".join([f"{i+1}. {point}" for i, point in enumerate(takeaways[:3])])
                qa_pairs.append({
                    "instruction": f"{stock_name}{year}年的核心投资观点是什么？",
                    "input": "",
                    "output": f"{stock_name}{year}年的核心观点如下：\n{output}",
                    "metadata": {
                        "type": "analysis",
                        "subtype": "takeaways",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # 投资亮点
            highlights = analysis.get("investment_highlights", [])
            if highlights:
                output = "\n".join([f"{i+1}. {point}" for i, point in enumerate(highlights)])
                qa_pairs.append({
                    "instruction": f"{stock_name}的投资亮点有哪些？",
                    "input": "",
                    "output": f"{stock_name}的投资亮点包括：\n{output}",
                    "metadata": {
                        "type": "analysis",
                        "subtype": "highlights",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # 风险提示
            risks = analysis.get("risk_warnings", [])
            if risks:
                output = "\n".join([f"{i+1}. {risk}" for i, risk in enumerate(risks)])
                qa_pairs.append({
                    "instruction": f"投资{stock_name}需要注意哪些风险？",
                    "input": "",
                    "output": f"投资{stock_name}的主要风险包括：\n{output}",
                    "metadata": {
                        "type": "analysis",
                        "subtype": "risks",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
            
            # 投资建议
            suggestion = analysis.get("investment_suggestion", "")
            if suggestion:
                qa_pairs.append({
                    "instruction": f"对{stock_name}的投资建议是什么？",
                    "input": "",
                    "output": suggestion,
                    "metadata": {
                        "type": "analysis",
                        "subtype": "suggestion",
                        "stock_code": basic.get("stock_code", ""),
                        "year": year
                    }
                })
        
        return qa_pairs
    
    def generate_comprehensive_qa(self) -> List[Dict]:
        """生成综合分析问答对"""
        qa_pairs = []
        
        for report in self.kb_data:
            basic = report["basic_info"]
            structured = report["layers"]["structured_data"]
            analysis = report["layers"]["analysis"]["json_format"]
            
            stock_name = basic.get("stock_name", "该公司")
            year = basic.get("report_year", "")
            
            # 综合分析
            financials = structured.get("financials", {})
            revenue = financials.get("total_revenue")
            profit = financials.get("net_profit")
            roe = financials.get("roe")
            
            output_parts = [f"{stock_name}{year}年业绩表现："]
            
            if revenue:
                output_parts.append(f"营业收入{self._format_number(revenue)}")
            if profit:
                output_parts.append(f"净利润{self._format_number(profit)}")
            if roe:
                output_parts.append(f"ROE为{roe}%")
            
            takeaways = analysis.get("key_takeaways", [])
            if takeaways:
                output_parts.append(f"核心观点：{takeaways[0]}")
            
            output = "。".join(output_parts)
            
            qa_pairs.append({
                "instruction": f"请简要分析{stock_name}{year}年的业绩表现",
                "input": "",
                "output": output,
                "metadata": {
                    "type": "comprehensive",
                    "stock_code": basic.get("stock_code", ""),
                    "year": year
                }
            })
        
        return qa_pairs
    
    def generate_all(self) -> Dict:
        """生成所有类型的数据集"""
        print("=" * 70)
        print("生成微调训练数据集")
        print("=" * 70)
        
        datasets = {
            "financial_qa": self.generate_financial_qa(),
            "dividend_qa": self.generate_dividend_qa(),
            "analysis_qa": self.generate_analysis_qa(),
            "comprehensive_qa": self.generate_comprehensive_qa()
        }
        
        # 合并所有数据
        all_data = []
        for name, data in datasets.items():
            print(f"{name}: {len(data)} 条")
            all_data.extend(data)
        
        print(f"\n总计: {len(all_data)} 条训练数据")
        
        return {
            "datasets": datasets,
            "all": all_data
        }
    
    def save(self, data: Dict, output_dir: str = None):
        """保存数据集"""
        if output_dir is None:
            output_dir = self.project_root / "data" / "processed" / "training"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分类数据集
        for name, dataset in data["datasets"].items():
            file_path = output_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"已保存: {file_path}")
        
        # 保存合并数据集（用于训练）
        all_path = output_dir / "train_dataset.json"
        with open(all_path, 'w', encoding='utf-8') as f:
            json.dump(data["all"], f, ensure_ascii=False, indent=2)
        print(f"已保存: {all_path}")
        
        # 保存为Alpaca格式
        alpaca_path = output_dir / "alpaca_format.json"
        with open(alpaca_path, 'w', encoding='utf-8') as f:
            json.dump(data["all"], f, ensure_ascii=False, indent=2)
        print(f"已保存: {alpaca_path}")
        
        # 生成统计信息
        stats = {
            "total_samples": len(data["all"]),
            "by_type": {name: len(ds) for name, ds in data["datasets"].items()},
            "by_stock": {}
        }
        
        for item in data["all"]:
            stock = item["metadata"].get("stock_code", "unknown")
            stats["by_stock"][stock] = stats["by_stock"].get(stock, 0) + 1
        
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存: {stats_path}")
        
        return output_dir
    
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
    generator = DatasetGenerator()
    
    # 生成数据集
    data = generator.generate_all()
    
    # 保存
    output_dir = generator.save(data)
    
    print("\n" + "=" * 70)
    print("数据集生成完成！")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print("\n示例数据:")
    for i, item in enumerate(data["all"][:3]):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {item['instruction']}")
        print(f"  回答: {item['output'][:100]}...")


if __name__ == "__main__":
    main()
