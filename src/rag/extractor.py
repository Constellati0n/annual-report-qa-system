"""
年报结构化数据提取模块
从年报文本中提取关键信息，生成结构化数据
"""
import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class FinancialIndicators:
    """财务指标"""
    # 盈利能力
    total_revenue: Optional[float] = None  # 营业收入
    net_profit: Optional[float] = None  # 净利润
    gross_profit: Optional[float] = None  # 毛利润
    operating_profit: Optional[float] = None  # 营业利润
    eps: Optional[float] = None  # 基本每股收益
    diluted_eps: Optional[float] = None  # 稀释每股收益
    roe: Optional[float] = None  # 净资产收益率
    roa: Optional[float] = None  # 总资产收益率
    gross_margin: Optional[float] = None  # 毛利率
    net_margin: Optional[float] = None  # 净利率
    
    # 资产负债
    total_assets: Optional[float] = None  # 总资产
    total_liabilities: Optional[float] = None  # 总负债
    net_assets: Optional[float] = None  # 净资产
    asset_liability_ratio: Optional[float] = None  # 资产负债率
    current_ratio: Optional[float] = None  # 流动比率
    quick_ratio: Optional[float] = None  # 速动比率
    
    # 现金流
    operating_cash_flow: Optional[float] = None  # 经营活动现金流
    investing_cash_flow: Optional[float] = None  # 投资活动现金流
    financing_cash_flow: Optional[float] = None  # 筹资活动现金流
    
    # 成长性指标
    revenue_growth: Optional[float] = None  # 营收增长率
    profit_growth: Optional[float] = None  # 净利润增长率
    
    # 每股指标
    bps: Optional[float] = None  # 每股净资产
    cfps: Optional[float] = None  # 每股现金流


@dataclass
class CompanyInfo:
    """公司基本信息"""
    stock_code: str = ""
    stock_name: str = ""
    company_full_name: str = ""
    company_english_name: str = ""
    registered_address: str = ""
    office_address: str = ""
    legal_representative: str = ""
    secretary: str = ""  # 董事会秘书
    secretary_phone: str = ""
    secretary_email: str = ""
    industry: str = ""  # 所属行业
    main_business: str = ""  # 主营业务
    establishment_date: str = ""
    listing_date: str = ""


@dataclass
class ShareholderInfo:
    """股东信息"""
    total_shares: Optional[float] = None  # 总股本
    circulating_shares: Optional[float] = None  # 流通股本
    top10_shareholders: List[Dict] = None  # 前十大股东
    controlling_shareholder: str = ""  # 控股股东
    actual_controller: str = ""  # 实际控制人
    
    def __post_init__(self):
        if self.top10_shareholders is None:
            self.top10_shareholders = []


@dataclass
class MajorEvents:
    """重大事项"""
    dividend_plan: str = ""  # 分红方案
    major_investments: List[Dict] = None  # 重大投资
    major_litigation: List[str] = None  # 重大诉讼
    related_party_transactions: List[Dict] = None  # 关联交易
    guarantee_matters: List[Dict] = None  # 担保事项
    
    def __post_init__(self):
        if self.major_investments is None:
            self.major_investments = []
        if self.major_litigation is None:
            self.major_litigation = []
        if self.related_party_transactions is None:
            self.related_party_transactions = []
        if self.guarantee_matters is None:
            self.guarantee_matters = []


@dataclass
class BusinessAnalysis:
    """经营分析"""
    industry_overview: str = ""  # 行业概况
    company_position: str = ""  # 公司地位
    core_competitiveness: List[str] = None  # 核心竞争力
    business_risks: List[str] = None  # 经营风险
    development_strategy: str = ""  # 发展战略
    
    def __post_init__(self):
        if self.core_competitiveness is None:
            self.core_competitiveness = []
        if self.business_risks is None:
            self.business_risks = []


@dataclass
class StructuredAnnualReport:
    """结构化年报数据"""
    stock_code: str = ""
    stock_name: str = ""
    report_year: str = ""
    report_date: str = ""
    audit_opinion: str = ""  # 审计意见
    
    company_info: CompanyInfo = None
    financial_indicators: FinancialIndicators = None
    shareholder_info: ShareholderInfo = None
    major_events: MajorEvents = None
    business_analysis: BusinessAnalysis = None
    
    # 原始文本摘要
    executive_summary: str = ""  # 管理层讨论摘要
    key_highlights: List[str] = None  # 主要亮点
    risk_warnings: List[str] = None  # 风险提示
    
    def __post_init__(self):
        if self.company_info is None:
            self.company_info = CompanyInfo()
        if self.financial_indicators is None:
            self.financial_indicators = FinancialIndicators()
        if self.shareholder_info is None:
            self.shareholder_info = ShareholderInfo()
        if self.major_events is None:
            self.major_events = MajorEvents()
        if self.business_analysis is None:
            self.business_analysis = BusinessAnalysis()
        if self.key_highlights is None:
            self.key_highlights = []
        if self.risk_warnings is None:
            self.risk_warnings = []
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class AnnualReportExtractor:
    """年报数据提取器"""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译正则表达式模式"""
        return {
            # 财务指标模式
            'total_revenue': re.compile(r'营业总收入?[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'net_profit': re.compile(r'归属于[^\n]{0,20}净利润[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'operating_profit': re.compile(r'营业利润[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'total_assets': re.compile(r'资产总计[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'net_assets': re.compile(r'归属于[^\n]{0,20}所有者权益[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'eps': re.compile(r'基本每股收益[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'roe': re.compile(r'加权平均净资产收益率[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            'operating_cash_flow': re.compile(r'经营活动.*现金流量[\s\w]*?[：:]\s*([\d,\.]+)', re.IGNORECASE),
            
            # 公司信息模式
            'company_full_name': re.compile(r'公司全称[\s]*[：:]\s*([^\n]+)'),
            'registered_address': re.compile(r'注册地址[\s]*[：:]\s*([^\n]+)'),
            'legal_representative': re.compile(r'法定代表人[\s]*[：:]\s*([^\n]+)'),
            'secretary': re.compile(r'董事会秘书[\s]*[：:]\s*([^\n]+)'),
            'industry': re.compile(r'所属行业[\s]*[：:]\s*([^\n]+)'),
            
            # 重大事项模式
            'dividend_plan': re.compile(r'利润分配方案|分红方案[\s\S]{0,500}(?:每\d+股|派\s*\d+\.?\d*\s*元)'),
        }
    
    def extract(self, text: str, stock_code: str = "", stock_name: str = "", report_year: str = "") -> StructuredAnnualReport:
        """
        从年报文本提取结构化数据
        
        Args:
            text: 年报全文
            stock_code: 股票代码
            stock_name: 股票名称
            report_year: 报告年度
            
        Returns:
            结构化年报数据
        """
        report = StructuredAnnualReport(
            stock_code=stock_code,
            stock_name=stock_name,
            report_year=report_year
        )
        
        # 提取公司信息
        report.company_info = self._extract_company_info(text, stock_code, stock_name)
        
        # 提取财务指标
        report.financial_indicators = self._extract_financial_indicators(text)
        
        # 提取股东信息
        report.shareholder_info = self._extract_shareholder_info(text)
        
        # 提取重大事项
        report.major_events = self._extract_major_events(text)
        
        # 提取经营分析
        report.business_analysis = self._extract_business_analysis(text)
        
        # 提取摘要信息
        report.executive_summary = self._extract_executive_summary(text)
        report.key_highlights = self._extract_key_highlights(text)
        report.risk_warnings = self._extract_risk_warnings(text)
        
        return report
    
    def _extract_company_info(self, text: str, stock_code: str, stock_name: str) -> CompanyInfo:
        """提取公司信息"""
        info = CompanyInfo(stock_code=stock_code, stock_name=stock_name)
        
        patterns = self.patterns
        
        # 提取各项信息
        if match := patterns['company_full_name'].search(text):
            info.company_full_name = match.group(1).strip()
        
        if match := patterns['registered_address'].search(text):
            info.registered_address = match.group(1).strip()
        
        if match := patterns['legal_representative'].search(text):
            info.legal_representative = match.group(1).strip()
        
        if match := patterns['secretary'].search(text):
            info.secretary = match.group(1).strip()
        
        if match := patterns['industry'].search(text):
            info.industry = match.group(1).strip()
        
        # 提取主营业务（通常在第一节）
        business_match = re.search(r'主营业务[\s\S]{0,50}?[:：]([^\n]{10,500})', text)
        if business_match:
            info.main_business = business_match.group(1).strip()
        
        return info
    
    def _extract_financial_indicators(self, text: str) -> FinancialIndicators:
        """提取财务指标"""
        indicators = FinancialIndicators()
        patterns = self.patterns
        
        # 提取各项财务指标
        if match := patterns['total_revenue'].search(text):
            indicators.total_revenue = self._parse_number(match.group(1))
        
        if match := patterns['net_profit'].search(text):
            indicators.net_profit = self._parse_number(match.group(1))
        
        if match := patterns['operating_profit'].search(text):
            indicators.operating_profit = self._parse_number(match.group(1))
        
        if match := patterns['total_assets'].search(text):
            indicators.total_assets = self._parse_number(match.group(1))
        
        if match := patterns['net_assets'].search(text):
            indicators.net_assets = self._parse_number(match.group(1))
        
        if match := patterns['eps'].search(text):
            indicators.eps = self._parse_float(match.group(1))
        
        if match := patterns['roe'].search(text):
            indicators.roe = self._parse_float(match.group(1))
        
        if match := patterns['operating_cash_flow'].search(text):
            indicators.operating_cash_flow = self._parse_number(match.group(1))
        
        # 计算衍生指标
        if indicators.net_profit and indicators.total_revenue and indicators.total_revenue > 0:
            indicators.net_margin = indicators.net_profit / indicators.total_revenue
        
        if indicators.net_assets and indicators.total_assets and indicators.total_assets > 0:
            indicators.asset_liability_ratio = 1 - (indicators.net_assets / indicators.total_assets)
        
        return indicators
    
    def _extract_shareholder_info(self, text: str) -> ShareholderInfo:
        """提取股东信息"""
        info = ShareholderInfo()
        
        # 提取总股本
        share_match = re.search(r'总股本[\s\w]*?[：:]\s*([\d,\.]+)\s*股', text)
        if share_match:
            info.total_shares = self._parse_number(share_match.group(1))
        
        # 提取控股股东
        controller_match = re.search(r'控股股东[\s\w]*?[：:]\s*([^\n]{2,50})', text)
        if controller_match:
            info.controlling_shareholder = controller_match.group(1).strip()
        
        # 提取实际控制人
        actual_controller_match = re.search(r'实际控制人[\s\w]*?[：:]\s*([^\n]{2,50})', text)
        if actual_controller_match:
            info.actual_controller = actual_controller_match.group(1).strip()
        
        # 提取前十大股东
        info.top10_shareholders = self._extract_top10_shareholders(text)
        
        return info
    
    def _extract_top10_shareholders(self, text: str) -> List[Dict]:
        """提取前十大股东"""
        shareholders = []
        
        # 查找股东表格区域
        shareholder_section = re.search(
            r'前.*名股东.*持股情况[\s\S]{0,3000}?股东名称[\s\S]{0,5000}(?=§|章节|第[一二三四五六七八九十]+节)',
            text
        )
        
        if shareholder_section:
            section_text = shareholder_section.group(0)
            # 提取股东名称和持股比例
            pattern = r'([^\d\s]{2,30}?(?:公司|集团|基金|保险|社保|养老金|账户))\s+(\d+(?:,\d+)*)\s+(\d+\.?\d*)'
            matches = re.findall(pattern, section_text)
            
            for i, (name, shares, ratio) in enumerate(matches[:10]):
                shareholders.append({
                    'rank': i + 1,
                    'name': name.strip(),
                    'shares': self._parse_number(shares),
                    'ratio': float(ratio) if ratio else None
                })
        
        return shareholders
    
    def _extract_major_events(self, text: str) -> MajorEvents:
        """提取重大事项"""
        events = MajorEvents()
        
        # 提取分红方案
        if match := self.patterns['dividend_plan'].search(text):
            events.dividend_plan = match.group(0).strip()[:500]
        
        # 提取重大投资
        investment_matches = re.findall(
            r'(?:重大投资|投资项目|新建项目)[\s\S]{0,200}(?:投资总额|投资金额)[\s\w]*?[：:]\s*([\d,\.]+)[\s\w]*元',
            text
        )
        for match in investment_matches[:5]:
            events.major_investments.append({
                'description': match.strip()[:200],
                'amount': self._parse_number(match)
            })
        
        return events
    
    def _extract_business_analysis(self, text: str) -> BusinessAnalysis:
        """提取经营分析"""
        analysis = BusinessAnalysis()
        
        # 提取行业概况
        industry_match = re.search(
            r'(?:行业情况|行业分析|行业现状)[\s\S]{0,1000}(.*?)(?=§|公司|业务|第[一二三四五六七八九十]+节)',
            text
        )
        if industry_match:
            analysis.industry_overview = industry_match.group(1).strip()[:500]
        
        # 提取核心竞争力
        core_matches = re.findall(r'(?:核心竞争|竞争优势)[\s\S]{0,100}?(?:[：:]\s*)([^\n]{10,200})', text)
        analysis.core_competitiveness = [m.strip() for m in core_matches[:5]]
        
        # 提取经营风险
        risk_matches = re.findall(r'(?:风险|不确定性)[\s\S]{0,50}?(?:[：:]\s*)([^\n]{10,200})', text)
        analysis.business_risks = [m.strip() for m in risk_matches[:5]]
        
        return analysis
    
    def _extract_executive_summary(self, text: str) -> str:
        """提取管理层讨论摘要"""
        # 查找管理层讨论章节
        md_section = re.search(
            r'(?:管理层讨论与分析|经营情况讨论与分析)[\s\S]{0,2000}(.*?)(?=§|重要事项|股份变动)',
            text
        )
        if md_section:
            return md_section.group(1).strip()[:1000]
        return ""
    
    def _extract_key_highlights(self, text: str) -> List[str]:
        """提取主要亮点"""
        highlights = []
        
        # 查找增长指标
        growth_patterns = [
            r'(?:同比增长|较上年同期)[\s\w]*?[：:]\s*([\d\.]+)\s*%',
            r'(?:增长|增加)[\s\w]*?([\d\.]+)\s*%',
        ]
        
        for pattern in growth_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:
                try:
                    value = float(match)
                    if value > 10:  # 只保留显著增长
                        highlights.append(f"增长{value}%")
                except:
                    continue
        
        return highlights[:5]
    
    def _extract_risk_warnings(self, text: str) -> List[str]:
        """提取风险提示"""
        risks = []
        
        # 查找风险章节
        risk_section = re.search(
            r'(?:可能面对的风险|风险提示|风险因素)[\s\S]{0,2000}(.*?)(?=§|重要事项|股份变动)',
            text
        )
        
        if risk_section:
            section_text = risk_section.group(1)
            # 提取风险点
            risk_items = re.findall(r'(?:[（(]\d+[)）]|\d+[、\.])\s*([^\n]{10,100})', section_text)
            risks = [r.strip() for r in risk_items[:5]]
        
        return risks
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """解析数字（支持万、亿单位）"""
        if not num_str:
            return None
        
        try:
            num_str = num_str.replace(',', '').replace('，', '').strip()
            
            # 处理单位
            if '万亿' in num_str:
                return float(num_str.replace('万亿', '')) * 1e12
            elif '亿' in num_str:
                return float(num_str.replace('亿', '')) * 1e8
            elif '万' in num_str:
                return float(num_str.replace('万', '')) * 1e4
            else:
                return float(num_str)
        except:
            return None
    
    def _parse_float(self, num_str: str) -> Optional[float]:
        """解析浮点数"""
        try:
            return float(num_str.replace(',', ''))
        except:
            return None


# 测试代码
if __name__ == "__main__":
    # 测试提取器
    extractor = AnnualReportExtractor()
    
    # 测试文本
    test_text = """
    第一节 重要提示
    
    平安银行股份有限公司（以下简称"公司"）董事会保证本报告所载资料不存在任何虚假记载。
    
    第二节 公司简介
    
    公司全称：平安银行股份有限公司
    股票代码：000001
    注册地址：广东省深圳市罗湖区深南东路5047号
    法定代表人：谢永林
    董事会秘书：周强
    所属行业：货币金融服务
    主营业务：经有关监管机构批准的各项商业银行业务
    
    第三节 会计数据和财务指标
    
    营业总收入：1,646.99亿元
    归属于本行股东的净利润：464.55亿元
    基本每股收益：2.25元
    加权平均净资产收益率：10.89%
    资产总计：55,871.16亿元
    归属于本行股东权益：4,467.63亿元
    
    第四节 股东情况
    
    总股本：19,405,918,198股
    控股股东：中国平安保险（集团）股份有限公司
    实际控制人：无
    
    前10名股东持股情况：
    股东名称                    持股数量（股）    持股比例（%）
    中国平安保险（集团）        8,624,209,341     44.44
    香港中央结算有限公司        3,456,789,012     17.81
    
    第五节 重要事项
    
    利润分配方案：每10股派发现金股利人民币7.19元（含税）
    """
    
    report = extractor.extract(test_text, "000001", "平安银行", "2023")
    
    print("提取结果：")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
