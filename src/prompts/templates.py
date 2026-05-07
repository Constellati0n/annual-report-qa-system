"""
提示词模板管理器
提供年报分析专用的提示词模板
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class AnalysisType(Enum):
    """分析类型枚举"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    BUSINESS_REVIEW = "business_review"
    RISK_ASSESSMENT = "risk_assessment"
    INDUSTRY_COMPARISON = "industry_comparison"
    TREND_PREDICTION = "trend_prediction"
    SUMMARY = "summary"
    INVESTMENT_ADVICE = "investment_advice"
    GENERAL = "general"


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    description: str
    system_prompt: str
    user_template: str
    analysis_type: AnalysisType
    required_context: List[str]
    example_input: Optional[str] = None
    example_output: Optional[str] = None


class PromptManager:
    """提示词管理器"""
    
    # 基础系统提示词
    BASE_SYSTEM_PROMPT = """你是一位专业的企业年报分析专家，具备以下能力：

1. **财务分析能力**：精通财务报表分析，能够计算和解读各类财务指标
2. **行业洞察能力**：了解各行业发展趋势和竞争格局
3. **风险评估能力**：能够识别和评估投资风险
4. **数据解读能力**：善于从数据中发现规律和异常

**回答原则**：
- 基于提供的参考资料进行回答，确保数据准确
- 使用专业的财务术语和分析框架
- 结构化输出，逻辑清晰
- 如有不确定的信息，明确说明
- 引用具体的数据和事实支撑观点

**输出格式**：
- 使用Markdown格式
- 重要结论加粗显示
- 数据使用表格展示
- 分点论述，层次分明"""

    # 各类分析专用提示词
    TEMPLATES: Dict[AnalysisType, PromptTemplate] = {
        AnalysisType.FINANCIAL_ANALYSIS: PromptTemplate(
            name="财务分析",
            description="分析企业财务状况和财务指标",
            system_prompt=BASE_SYSTEM_PROMPT + """

**财务分析专用指引**：
1. 计算关键财务指标：ROE、ROA、毛利率、净利率、资产负债率等
2. 进行同比和环比分析
3. 与行业平均水平对比
4. 识别财务风险信号
5. 评估盈利质量和可持续性""",
            user_template="""请对以下公司进行财务分析：

**公司信息**：
{company_info}

**财务数据**：
{financial_data}

**参考资料**：
{context}

请从以下几个方面进行分析：
1. 盈利能力分析
2. 偿债能力分析
3. 运营效率分析
4. 成长能力分析
5. 现金流分析
6. 综合评价与建议""",
            analysis_type=AnalysisType.FINANCIAL_ANALYSIS,
            required_context=["financial_statements", "financial_ratios"],
            example_input="贵州茅台2023年年报",
            example_output="""## 贵州茅台2023年财务分析

### 一、盈利能力分析
**毛利率**：91.96%，同比提升0.11个百分点，显示公司强大的定价能力和成本控制能力。

**净利率**：52.49%，处于行业顶尖水平，反映公司高效的运营管理。

**ROE**：34.19%，连续多年保持在30%以上，为股东创造丰厚回报。

### 二、偿债能力分析
**资产负债率**：19.53%，财务结构稳健，偿债风险极低。

**流动比率**：4.85，短期偿债能力充足。

### 三、综合评价
贵州茅台展现出卓越的财务表现，盈利能力强、财务风险低、现金流充裕，是优质的投资标的。"""
        ),
        
        AnalysisType.BUSINESS_REVIEW: PromptTemplate(
            name="经营回顾",
            description="分析企业经营情况和业务发展",
            system_prompt=BASE_SYSTEM_PROMPT + """

**经营分析专用指引**：
1. 梳理主营业务发展情况
2. 分析各业务板块表现
3. 评估市场份额和竞争地位
4. 总结经营亮点和成就
5. 识别经营中存在的问题和挑战""",
            user_template="""请分析以下公司的经营情况：

**公司信息**：
{company_info}

**经营数据**：
{operational_data}

**参考资料**：
{context}

请从以下几个方面进行分析：
1. 主营业务发展情况
2. 各业务板块表现
3. 市场地位与竞争优势
4. 主要经营亮点
5. 面临的挑战与风险
6. 未来发展方向""",
            analysis_type=AnalysisType.BUSINESS_REVIEW,
            required_context=["business_segments", "market_data"],
            example_input="宁德时代2023年经营情况",
            example_output="""## 宁德时代2023年经营回顾

### 一、主营业务发展
动力电池业务营收同比增长XX%，全球市场份额持续领先。

### 二、业务板块表现
1. **动力电池**：营收XXX亿元，占比XX%
2. **储能电池**：营收XXX亿元，同比增长XX%
3. **电池材料**：营收XXX亿元

### 三、市场地位
全球动力电池装机量市占率XX%，连续X年全球第一。"""
        ),
        
        AnalysisType.RISK_ASSESSMENT: PromptTemplate(
            name="风险评估",
            description="评估投资风险和企业风险",
            system_prompt=BASE_SYSTEM_PROMPT + """

**风险评估专用指引**：
1. 识别市场风险因素
2. 分析经营风险
3. 评估财务风险
4. 关注政策和监管风险
5. 考虑技术和创新风险
6. 给出风险评级和投资建议""",
            user_template="""请评估投资以下公司的风险：

**公司信息**：
{company_info}

**风险相关信息**：
{risk_info}

**参考资料**：
{context}

请从以下几个方面进行评估：
1. 市场风险
2. 经营风险
3. 财务风险
4. 政策风险
5. 技术风险
6. 综合风险评级
7. 投资建议""",
            analysis_type=AnalysisType.RISK_ASSESSMENT,
            required_context=["risk_factors", "financial_data"],
            example_input="某新能源公司投资风险评估",
            example_output="""## 投资风险评估报告

### 一、市场风险
- 行业竞争加剧，价格战风险
- 原材料价格波动风险

### 二、经营风险
- 客户集中度较高
- 产能扩张风险

### 三、财务风险
- 资产负债率XX%，处于合理水平
- 现金流状况良好

### 四、综合风险评级
**中等风险** - 建议谨慎投资，关注行业竞争态势。"""
        ),
        
        AnalysisType.INDUSTRY_COMPARISON: PromptTemplate(
            name="行业对比",
            description="与同行业公司进行对比分析",
            system_prompt=BASE_SYSTEM_PROMPT + """

**行业对比专用指引**：
1. 选择可比公司进行对标
2. 对比财务指标
3. 分析市场地位差异
4. 比较竞争优势
5. 评估估值水平
6. 给出相对投资建议""",
            user_template="""请将以下公司与同行业进行对比分析：

**目标公司**：
{company_info}

**对比公司**：
{peer_companies}

**对比数据**：
{comparison_data}

**参考资料**：
{context}

请从以下几个方面进行对比：
1. 规模对比（营收、资产、市值）
2. 盈利能力对比
3. 成长性对比
4. 估值水平对比
5. 竞争优势对比
6. 行业地位分析
7. 投资建议""",
            analysis_type=AnalysisType.INDUSTRY_COMPARISON,
            required_context=["peer_data", "industry_benchmarks"],
            example_input="白酒行业公司对比",
            example_output="""## 白酒行业公司对比分析

### 一、规模对比
| 公司 | 营收(亿元) | 净利润(亿元) | 市值(亿元) |
|------|-----------|-------------|-----------|
| 贵州茅台 | 1505.60 | 747.34 | 21000 |
| 五粮液 | 832.72 | 302.11 | 5500 |
| 泸州老窖 | 302.33 | 132.45 | 2800 |

### 二、盈利能力对比
茅台毛利率91.96%，显著高于行业平均水平。"""
        ),
        
        AnalysisType.TREND_PREDICTION: PromptTemplate(
            name="趋势预测",
            description="预测企业未来发展趋势",
            system_prompt=BASE_SYSTEM_PROMPT + """

**趋势预测专用指引**：
1. 分析历史发展趋势
2. 考虑行业发展前景
3. 评估公司战略规划
4. 预测财务指标变化
5. 识别关键驱动因素
6. 给出概率性预测""",
            user_template="""请预测以下公司的未来发展趋势：

**公司信息**：
{company_info}

**历史数据**：
{historical_data}

**行业前景**：
{industry_outlook}

**参考资料**：
{context}

请从以下几个方面进行预测：
1. 营收增长预测
2. 盈利能力趋势
3. 市场份额变化
4. 业务发展重点
5. 潜在机遇与挑战
6. 综合展望""",
            analysis_type=AnalysisType.TREND_PREDICTION,
            required_context=["historical_data", "industry_forecast"],
            example_input="比亚迪未来发展趋势",
            example_output="""## 比亚迪发展趋势预测

### 一、营收增长预测
预计2024-2026年营收复合增长率XX%，主要驱动因素：
1. 新能源汽车市场持续扩大
2. 海外市场拓展
3. 电池外供业务增长

### 二、综合展望
**乐观情景**：市占率进一步提升，营收突破XXX亿元
**基准情景**：稳健增长，营收达到XXX亿元
**悲观情景**：竞争加剧，增速放缓至XX%"""
        ),
        
        AnalysisType.SUMMARY: PromptTemplate(
            name="摘要提取",
            description="提取年报关键信息摘要",
            system_prompt=BASE_SYSTEM_PROMPT + """

**摘要提取专用指引**：
1. 提取核心财务数据
2. 总结主要经营成果
3. 识别关键事件
4. 提炼风险提示
5. 概括未来规划
6. 保持简洁明了""",
            user_template="""请从以下年报内容中提取关键信息摘要：

**年报内容**：
{report_content}

**参考资料**：
{context}

请提取以下信息：
1. 公司概况
2. 核心财务数据
3. 主要经营成果
4. 重要事项
5. 风险提示
6. 未来展望
7. 关键数据表格""",
            analysis_type=AnalysisType.SUMMARY,
            required_context=["report_content"],
            example_input="某公司2023年年报",
            example_output="""## 年报关键信息摘要

### 核心财务数据
- 营业收入：XXX亿元（同比+X%）
- 净利润：XXX亿元（同比+X%）
- 总资产：XXX亿元
- ROE：XX%

### 主要经营成果
1. 主营业务稳健增长
2. 市场份额提升X个百分点
3. 新产品收入占比达XX%

### 重要事项
- 完成XX项目收购
- 推出XX新产品
- 获得XX资质认证"""
        ),
        
        AnalysisType.INVESTMENT_ADVICE: PromptTemplate(
            name="投资建议",
            description="提供投资建议和估值分析",
            system_prompt=BASE_SYSTEM_PROMPT + """

**投资建议专用指引**：
1. 进行估值分析（PE、PB、DCF等）
2. 评估投资价值
3. 分析催化剂和风险因素
4. 给出投资评级
5. 设定目标价区间
6. 提供投资策略建议""",
            user_template="""请对以下公司给出投资建议：

**公司信息**：
{company_info}

**估值数据**：
{valuation_data}

**参考资料**：
{context}

请从以下几个方面给出建议：
1. 估值分析
2. 投资价值评估
3. 催化剂分析
4. 风险因素
5. 投资评级
6. 目标价区间
7. 投资策略""",
            analysis_type=AnalysisType.INVESTMENT_ADVICE,
            required_context=["valuation_data", "market_data"],
            example_input="某科技公司投资建议",
            example_output="""## 投资建议报告

### 估值分析
当前PE(TTM)：XX倍，处于历史XX分位
行业平均PE：XX倍

### 投资评级
**买入** - 目标价XX元（上涨空间XX%）

### 核心逻辑
1. 行业景气度向上
2. 公司竞争优势明显
3. 估值处于合理区间

### 风险提示
- 宏观经济波动
- 行业竞争加剧
- 原材料价格上涨"""
        ),
        
        AnalysisType.GENERAL: PromptTemplate(
            name="通用问答",
            description="通用年报相关问题回答",
            system_prompt=BASE_SYSTEM_PROMPT,
            user_template="""请回答以下关于年报的问题：

**问题**：
{question}

**相关信息**：
{context}

请基于提供的参考资料，给出专业、准确的回答。""",
            analysis_type=AnalysisType.GENERAL,
            required_context=["general"],
            example_input="这家公司的主营业务是什么？",
            example_output="根据年报显示，该公司的主营业务包括..."
        )
    }
    
    def __init__(self):
        self.templates = self.TEMPLATES
    
    def get_template(self, analysis_type: AnalysisType) -> PromptTemplate:
        """
        获取指定类型的提示词模板
        
        Args:
            analysis_type: 分析类型
            
        Returns:
            提示词模板
        """
        return self.templates.get(analysis_type, self.templates[AnalysisType.GENERAL])
    
    def get_system_prompt(self, analysis_type: AnalysisType) -> str:
        """
        获取系统提示词
        
        Args:
            analysis_type: 分析类型
            
        Returns:
            系统提示词
        """
        template = self.get_template(analysis_type)
        return template.system_prompt
    
    def format_user_prompt(
        self,
        analysis_type: AnalysisType,
        context: str,
        **kwargs
    ) -> str:
        """
        格式化用户提示词
        
        Args:
            analysis_type: 分析类型
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            格式化后的用户提示词
        """
        template = self.get_template(analysis_type)
        
        # 构建参数
        params = {"context": context}
        params.update(kwargs)
        
        return template.user_template.format(**params)
    
    def build_full_prompt(
        self,
        analysis_type: AnalysisType,
        context: str,
        **kwargs
    ) -> tuple:
        """
        构建完整提示词
        
        Args:
            analysis_type: 分析类型
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            (system_prompt, user_prompt) 元组
        """
        system_prompt = self.get_system_prompt(analysis_type)
        user_prompt = self.format_user_prompt(analysis_type, context, **kwargs)
        
        return system_prompt, user_prompt
    
    def detect_analysis_type(self, question: str) -> AnalysisType:
        """
        根据问题自动检测分析类型
        
        Args:
            question: 用户问题
            
        Returns:
            分析类型
        """
        question_lower = question.lower()
        
        # 关键词匹配
        keywords = {
            AnalysisType.FINANCIAL_ANALYSIS: [
                "财务", "盈利", "收入", "利润", "资产", "负债", "现金流",
                "roe", "roa", "毛利率", "净利率", "资产负债率"
            ],
            AnalysisType.BUSINESS_REVIEW: [
                "经营", "业务", "发展", "市场", "销售", "产品", "客户"
            ],
            AnalysisType.RISK_ASSESSMENT: [
                "风险", "危险", "问题", "挑战", "不确定", "波动"
            ],
            AnalysisType.INDUSTRY_COMPARISON: [
                "对比", "比较", "同行", "行业", "竞争对手", "vs"
            ],
            AnalysisType.TREND_PREDICTION: [
                "预测", "趋势", "未来", "前景", "增长", "展望"
            ],
            AnalysisType.SUMMARY: [
                "摘要", "总结", "概况", "简介", "关键信息"
            ],
            AnalysisType.INVESTMENT_ADVICE: [
                "投资", "建议", "估值", "评级", "目标价", "买入", "卖出"
            ]
        }
        
        # 计算匹配度
        scores = {}
        for analysis_type, words in keywords.items():
            score = sum(1 for word in words if word in question_lower)
            if score > 0:
                scores[analysis_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return AnalysisType.GENERAL
    
    def list_templates(self) -> List[Dict]:
        """
        列出所有可用模板
        
        Returns:
            模板信息列表
        """
        return [
            {
                "type": template.analysis_type.value,
                "name": template.name,
                "description": template.description,
                "required_context": template.required_context
            }
            for template in self.templates.values()
        ]


# 全局提示词管理器实例
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """
    获取全局提示词管理器实例
    
    Returns:
        PromptManager实例
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
