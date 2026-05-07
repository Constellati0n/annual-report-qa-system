"""
AI分析摘要生成模块
使用LLM或规则生成年报分析摘要
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ReportAnalysis:
    """年报分析摘要"""
    stock_code: str = ""
    stock_name: str = ""
    report_year: str = ""
    
    # 核心观点
    key_takeaways: List[str] = None
    
    # 业绩点评
    performance_review: str = ""
    
    # 投资亮点
    investment_highlights: List[str] = None
    
    # 主要风险
    risk_warnings: List[str] = None
    
    # 估值分析
    valuation_analysis: str = ""
    
    # 投资建议
    investment_suggestion: str = ""
    
    # 同业对比
    peer_comparison: str = ""
    
    def __post_init__(self):
        if self.key_takeaways is None:
            self.key_takeaways = []
        if self.investment_highlights is None:
            self.investment_highlights = []
        if self.risk_warnings is None:
            self.risk_warnings = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_text(self) -> str:
        """转换为文本格式用于向量化"""
        sections = []
        
        sections.append(f"【核心观点】")
        for i, point in enumerate(self.key_takeaways, 1):
            sections.append(f"{i}. {point}")
        
        if self.performance_review:
            sections.append(f"\n【业绩点评】\n{self.performance_review}")
        
        if self.investment_highlights:
            sections.append(f"\n【投资亮点】")
            for i, highlight in enumerate(self.investment_highlights, 1):
                sections.append(f"{i}. {highlight}")
        
        if self.risk_warnings:
            sections.append(f"\n【风险提示】")
            for i, risk in enumerate(self.risk_warnings, 1):
                sections.append(f"{i}. {risk}")
        
        if self.valuation_analysis:
            sections.append(f"\n【估值分析】\n{self.valuation_analysis}")
        
        if self.investment_suggestion:
            sections.append(f"\n【投资建议】\n{self.investment_suggestion}")
        
        return "\n".join(sections)


class RuleBasedAnalyzer:
    """基于规则的分析器（无需LLM）"""
    
    def __init__(self):
        self.industry_benchmarks = {
            "银行": {
                "good_roe": 12,
                "good_npl_ratio": 1.5,  # 不良率
                "good_car": 12  # 资本充足率
            },
            "保险": {
                "good_roe": 10,
                "good_combined_ratio": 100
            },
            "default": {
                "good_roe": 10,
                "good_growth": 15
            }
        }
    
    def analyze(self, structured_data: Dict) -> ReportAnalysis:
        """
        基于结构化数据生成分析
        
        Args:
            structured_data: 结构化数据字典
            
        Returns:
            ReportAnalysis对象
        """
        analysis = ReportAnalysis()
        
        # 基本信息
        analysis.stock_code = structured_data.get("stock_code", "")
        analysis.stock_name = structured_data.get("stock_name", "")
        analysis.report_year = structured_data.get("report_year", "")
        
        # 提取财务数据
        financials = structured_data.get("financials", {})
        revenue = financials.get("total_revenue")
        profit = financials.get("net_profit")
        eps = financials.get("eps")
        roe = financials.get("roe")
        
        # 生成核心观点
        analysis.key_takeaways = self._generate_takeaways(structured_data, financials)
        
        # 业绩点评
        analysis.performance_review = self._generate_performance_review(structured_data, financials)
        
        # 投资亮点
        analysis.investment_highlights = self._generate_highlights(structured_data, financials)
        
        # 风险提示
        analysis.risk_warnings = self._generate_risks(structured_data)
        
        # 投资建议
        analysis.investment_suggestion = self._generate_suggestion(structured_data, financials)
        
        return analysis
    
    def _generate_takeaways(self, data: Dict, financials: Dict) -> List[str]:
        """生成核心观点"""
        takeaways = []
        
        stock_name = data.get("stock_name", "该公司")
        year = data.get("report_year", "本")
        
        # 营收情况
        revenue = financials.get("total_revenue")
        if revenue:
            if revenue > 100000000000:  # 1000亿
                takeaways.append(f"{stock_name}是行业龙头企业，营收规模超过千亿")
            elif revenue > 10000000000:  # 100亿
                takeaways.append(f"{stock_name}营收规模超过百亿，具备较强的市场竞争力")
        
        # 盈利能力
        profit = financials.get("net_profit")
        if profit and profit > 0:
            takeaways.append(f"{year}年实现净利润{self._format_number(profit)}，保持盈利状态")
        
        # ROE
        roe = financials.get("roe")
        if roe:
            if roe > 15:
                takeaways.append(f"净资产收益率(ROE)达{roe}%，盈利能力优秀")
            elif roe > 10:
                takeaways.append(f"净资产收益率(ROE)为{roe}%，盈利能力良好")
        
        # 分红
        dividend = data.get("dividend", {})
        dps = dividend.get("dividend_per_share")
        if dps and dps > 0:
            takeaways.append(f"公司实施分红方案，每股派息{dps}元，体现对股东的回报")
        
        # 如果观点太少，添加通用描述
        if len(takeaways) < 2:
            takeaways.append(f"{stock_name}持续经营，业务保持稳定发展")
        
        return takeaways[:5]  # 最多5条
    
    def _generate_performance_review(self, data: Dict, financials: Dict) -> str:
        """生成业绩点评"""
        stock_name = data.get("stock_name", "该公司")
        year = data.get("report_year", "本")
        
        reviews = []
        
        # 营收评价
        revenue = financials.get("total_revenue")
        if revenue:
            reviews.append(f"{year}年营业收入达到{self._format_number(revenue)}，")
        
        # 利润评价
        profit = financials.get("net_profit")
        if profit:
            if profit > 0:
                reviews.append(f"实现净利润{self._format_number(profit)}，")
            else:
                reviews.append("出现亏损，")
        
        # EPS评价
        eps = financials.get("eps")
        if eps:
            reviews.append(f"每股收益{eps}元。")
        
        # 业务摘要
        business = data.get("business_summary", "")
        if business:
            # 提取前100字
            summary = business[:100] + "..." if len(business) > 100 else business
            reviews.append(f"公司主营业务：{summary}")
        
        return "".join(reviews) if reviews else f"{stock_name}{year}年度经营情况正常。"
    
    def _generate_highlights(self, data: Dict, financials: Dict) -> List[str]:
        """生成投资亮点"""
        highlights = []
        
        # 财务亮点
        roe = financials.get("roe")
        if roe and roe > 12:
            highlights.append(f"ROE达{roe}%，盈利能力优于行业平均水平")
        
        eps = financials.get("eps")
        if eps and eps > 1:
            highlights.append(f"每股收益{eps}元，具备较强的盈利能力")
        
        # 分红亮点
        dividend = data.get("dividend", {})
        dps = dividend.get("dividend_per_share")
        if dps and dps > 0.5:
            highlights.append(f"每股分红{dps}元，股息回报可观")
        
        # 规模亮点
        revenue = financials.get("total_revenue")
        if revenue and revenue > 50000000000:  # 500亿
            highlights.append("营收规模领先，市场地位稳固")
        
        # 重大事项亮点
        events = data.get("major_events", [])
        if events:
            highlights.append(f"本年度完成{len(events)}项重大经营事项")
        
        return highlights[:4]
    
    def _generate_risks(self, data: Dict) -> List[str]:
        """生成风险提示"""
        risks = []
        
        # 从结构化数据获取风险
        risk_factors = data.get("risk_factors", [])
        risks.extend(risk_factors[:3])
        
        # 根据财务数据判断风险
        profit = data.get("financials", {}).get("net_profit")
        if profit is not None and profit < 0:
            risks.append("公司出现亏损，盈利能力存在压力")
        
        # 通用风险
        if len(risks) < 2:
            risks.extend([
                "宏观经济波动可能影响公司经营",
                "行业竞争加剧可能压缩利润空间"
            ])
        
        return risks[:4]
    
    def _generate_suggestion(self, data: Dict, financials: Dict) -> str:
        """生成投资建议"""
        stock_name = data.get("stock_name", "该公司")
        
        roe = financials.get("roe")
        profit = financials.get("net_profit")
        
        if roe and roe > 12 and profit and profit > 0:
            return f"{stock_name}基本面良好，盈利能力稳定，适合中长期关注。建议投资者结合市场环境和估值水平，审慎决策。"
        elif profit and profit > 0:
            return f"{stock_name}保持盈利，但需关注盈利能力提升空间。建议谨慎关注。"
        else:
            return f"{stock_name}经营面临挑战，建议投资者充分了解风险后谨慎决策。"
    
    def _format_number(self, num: float) -> str:
        """格式化数字"""
        if num >= 100000000:
            return f"{num/100000000:.2f}亿元"
        elif num >= 10000:
            return f"{num/10000:.2f}万元"
        else:
            return f"{num:.2f}元"


class LLMAnalyzer:
    """基于LLM的分析器（需要API key）"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.use_llm = api_key is not None
        
        # 如果没有API key，使用规则分析器作为fallback
        self.fallback = RuleBasedAnalyzer()
    
    def analyze(self, structured_data: Dict, full_text: str = "") -> ReportAnalysis:
        """
        使用LLM生成分析
        
        Args:
            structured_data: 结构化数据
            full_text: 年报全文（可选）
            
        Returns:
            ReportAnalysis对象
        """
        if not self.use_llm:
            print("未提供API key，使用规则分析器")
            return self.fallback.analyze(structured_data)
        
        # TODO: 实现LLM调用
        # 这里可以集成OpenAI、文心一言等API
        
        return self.fallback.analyze(structured_data)


def generate_analysis(
    structured_data: Dict,
    full_text: str = "",
    use_llm: bool = False,
    api_key: Optional[str] = None
) -> ReportAnalysis:
    """
    生成年报分析摘要
    
    Args:
        structured_data: 结构化数据
        full_text: 年报全文
        use_llm: 是否使用LLM
        api_key: LLM API key
        
    Returns:
        ReportAnalysis对象
    """
    if use_llm and api_key:
        analyzer = LLMAnalyzer(api_key=api_key)
    else:
        analyzer = RuleBasedAnalyzer()
    
    return analyzer.analyze(structured_data)


# 测试代码
if __name__ == "__main__":
    # 测试数据
    test_data = {
        "stock_code": "000001",
        "stock_name": "平安银行",
        "report_year": "2023",
        "financials": {
            "total_revenue": 164699000000,  # 1647亿
            "net_profit": 46455000000,  # 464亿
            "eps": 2.25,
            "roe": 13.5
        },
        "dividend": {
            "dividend_per_share": 0.719
        },
        "business_summary": "平安银行是中国内地首家公开上市的全国性股份制商业银行，主要从事商业银行业务。",
        "major_events": ["完成数字化转型", "发行绿色金融债券"],
        "risk_factors": ["信用风险上升", "利率市场化压力"]
    }
    
    # 使用规则分析器
    analyzer = RuleBasedAnalyzer()
    analysis = analyzer.analyze(test_data)
    
    print("=" * 60)
    print("年报分析摘要")
    print("=" * 60)
    print(analysis.to_text())
    print("\n" + "=" * 60)
    print("JSON格式：")
    print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
