"""
微调数据集构建器
用于构建年报分析任务的训练数据集
"""
import json
import random
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """训练样本"""
    instruction: str
    input: str
    output: str
    system: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "system": self.system,
            "metadata": self.metadata or {}
        }
    
    def to_chat_format(self) -> Dict:
        """转换为对话格式"""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        
        # 构建用户输入
        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})
        
        return {"messages": messages}


class AnnualReportQAGenerator:
    """年报问答对生成器"""
    
    # 年报分析任务类型
    TASK_TYPES = {
        "financial_analysis": {
            "name": "财务分析",
            "description": "分析财务报表数据，计算财务指标"
        },
        "business_review": {
            "name": "经营回顾",
            "description": "解读公司经营情况和业务发展"
        },
        "risk_assessment": {
            "name": "风险评估",
            "description": "识别和评估投资风险"
        },
        "industry_comparison": {
            "name": "行业对比",
            "description": "与同行业公司进行对比分析"
        },
        "trend_prediction": {
            "name": "趋势预测",
            "description": "预测公司未来发展趋势"
        },
        "summary_extraction": {
            "name": "摘要提取",
            "description": "提取年报关键信息摘要"
        }
    }
    
    # 财务指标模板
    FINANCIAL_METRICS = {
        "profitability": ["毛利率", "净利率", "ROE", "ROA", "ROIC"],
        "liquidity": ["流动比率", "速动比率", "现金比率"],
        "solvency": ["资产负债率", "权益乘数", "利息保障倍数"],
        "efficiency": ["总资产周转率", "存货周转率", "应收账款周转率"],
        "growth": ["营收增长率", "净利润增长率", "总资产增长率"]
    }
    
    def __init__(self):
        self.examples: List[TrainingExample] = []
    
    def generate_financial_analysis_qa(self, company_data: Dict) -> List[TrainingExample]:
        """
        生成财务分析问答对
        
        Args:
            company_data: 公司财务数据
            
        Returns:
            训练样本列表
        """
        examples = []
        
        # 模板1：直接询问财务指标
        templates = [
            {
                "instruction": "请分析{company_name}的盈利能力指标。",
                "input": "已知数据：\n营业收入：{revenue}亿元\n净利润：{net_profit}亿元\n总资产：{total_assets}亿元\n净资产：{net_assets}亿元",
                "output_template": """根据提供的数据，{company_name}的盈利能力分析如下：

1. **毛利率**：{gross_margin}%
   - 计算公式：(营业收入-营业成本)/营业收入
   - 分析：{gross_margin_analysis}

2. **净利率**：{net_margin}%
   - 计算公式：净利润/营业收入
   - 分析：{net_margin_analysis}

3. **ROE（净资产收益率）**：{roe}%
   - 计算公式：净利润/净资产
   - 分析：{roe_analysis}

4. **ROA（总资产收益率）**：{roa}%
   - 计算公式：净利润/总资产
   - 分析：{roa_analysis}

**综合评价**：{overall_analysis}"""
            },
            {
                "instruction": "请计算并分析{company_name}的偿债能力。",
                "input": "资产负债表数据：\n总资产：{total_assets}亿元\n总负债：{total_liabilities}亿元\n流动资产：{current_assets}亿元\n流动负债：{current_liabilities}亿元",
                "output_template": """{company_name}的偿债能力分析：

1. **资产负债率**：{debt_ratio}%
   - 计算公式：总负债/总资产
   - 分析：{debt_ratio_analysis}

2. **流动比率**：{current_ratio}
   - 计算公式：流动资产/流动负债
   - 分析：{current_ratio_analysis}

3. **速动比率**：{quick_ratio}
   - 计算公式：(流动资产-存货)/流动负债
   - 分析：{quick_ratio_analysis}

**风险提示**：{risk_assessment}"""
            }
        ]
        
        for template in templates:
            # 这里应该使用真实的财务数据填充模板
            # 简化示例，实际应用中需要更复杂的逻辑
            example = TrainingExample(
                instruction=template["instruction"].format(company_name=company_data.get("name", "某公司")),
                input=template["input"].format(**company_data),
                output="请根据实际数据生成分析结果",
                system="你是一位专业的财务分析师，擅长财务指标计算和分析。",
                metadata={"task_type": "financial_analysis", "company": company_data.get("name")}
            )
            examples.append(example)
        
        return examples
    
    def generate_business_review_qa(self, company_data: Dict) -> List[TrainingExample]:
        """生成经营回顾问答对"""
        examples = []
        
        templates = [
            {
                "instruction": "请总结{company_name}本年度的主要经营成果。",
                "input": "年报摘要：\n{report_summary}",
                "output_template": """{company_name}本年度主要经营成果：

**一、经营业绩**
1. 营业收入达到{revenue}亿元，同比增长{revenue_growth}%
2. 归属于上市公司股东的净利润为{net_profit}亿元，同比增长{profit_growth}%
3. 基本每股收益为{eps}元

**二、业务发展**
{business_development}

**三、主要亮点**
{key_highlights}

**四、面临挑战**
{challenges}"""
            },
            {
                "instruction": "分析{company_name}的核心竞争力。",
                "input": "公司介绍：\n{company_intro}\n\n业务板块：\n{business_segments}",
                "output_template": """{company_name}的核心竞争力分析：

**一、技术优势**
{technical_advantages}

**二、市场地位**
{market_position}

**三、品牌影响力**
{brand_strength}

**四、渠道优势**
{channel_advantages}

**五、管理团队**
{management_team}"""
            }
        ]
        
        for template in templates:
            example = TrainingExample(
                instruction=template["instruction"].format(company_name=company_data.get("name", "某公司")),
                input=template["input"].format(**company_data),
                output="请根据实际数据生成分析结果",
                system="你是一位资深的行业分析师，擅长企业经营分析。",
                metadata={"task_type": "business_review", "company": company_data.get("name")}
            )
            examples.append(example)
        
        return examples
    
    def generate_risk_assessment_qa(self, company_data: Dict) -> List[TrainingExample]:
        """生成风险评估问答对"""
        examples = []
        
        risk_categories = {
            "市场风险": ["市场需求变化", "竞争加剧", "价格波动"],
            "经营风险": ["供应链中断", "质量问题", "产能不足"],
            "财务风险": ["汇率波动", "利率变化", "流动性风险"],
            "政策风险": ["监管变化", "税收政策", "环保要求"],
            "技术风险": ["技术迭代", "知识产权", "人才流失"]
        }
        
        templates = [
            {
                "instruction": "请评估投资{company_name}的主要风险。",
                "input": "公司基本情况：\n{company_info}\n\n财务状况：\n{financial_status}",
                "output_template": """投资{company_name}的主要风险评估：

**一、市场风险**
{risk_market}

**二、经营风险**
{risk_operation}

**三、财务风险**
{risk_financial}

**四、政策风险**
{risk_policy}

**五、技术风险**
{risk_technology}

**综合风险评级**：{risk_rating}
**投资建议**：{investment_suggestion}"""
            }
        ]
        
        for template in templates:
            example = TrainingExample(
                instruction=template["instruction"].format(company_name=company_data.get("name", "某公司")),
                input=template["input"].format(**company_data),
                output="请根据实际数据生成风险评估",
                system="你是一位专业的投资分析师，擅长风险评估和投资建议。",
                metadata={"task_type": "risk_assessment", "company": company_data.get("name")}
            )
            examples.append(example)
        
        return examples
    
    def generate_comparison_qa(self, company_data: Dict, peer_data: List[Dict]) -> List[TrainingExample]:
        """生成对比分析问答对"""
        examples = []
        
        template = {
            "instruction": "请将{company_name}与同行业公司进行对比分析。",
            "input": "{company_name}数据：\n{company_metrics}\n\n同行业公司数据：\n{peer_metrics}",
            "output_template": """{company_name}与同行业公司对比分析：

**一、规模对比**
{size_comparison}

**二、盈利能力对比**
{profitability_comparison}

**三、成长性对比**
{growth_comparison}

**四、估值对比**
{valuation_comparison}

**五、行业地位**
{industry_position}

**结论**：{conclusion}"""
        }
        
        example = TrainingExample(
            instruction=template["instruction"].format(company_name=company_data.get("name", "某公司")),
            input=template["input"].format(
                company_name=company_data.get("name"),
                company_metrics=str(company_data),
                peer_metrics=str(peer_data)
            ),
            output="请根据实际数据生成对比分析",
            system="你是一位专业的行业研究员，擅长公司对比分析。",
            metadata={"task_type": "industry_comparison", "company": company_data.get("name")}
        )
        examples.append(example)
        
        return examples
    
    def generate_summary_qa(self, report_content: str) -> List[TrainingExample]:
        """生成摘要提取问答对"""
        examples = []
        
        templates = [
            {
                "instruction": "请从以下年报内容中提取关键信息摘要。",
                "input": report_content[:2000],  # 限制长度
                "output_template": ""**年报关键信息摘要**

**一、公司概况**
{company_overview}

**二、财务亮点**
{financial_highlights}

**三、经营成果**
{operational_results}

**四、未来展望**
{future_outlook}

**五、风险提示**
{risk_warnings}"""
            },
            {
                "instruction": "请提取以下文本中的财务数据。",
                "input": report_content[:1500],
                "output_template": ""**提取的财务数据**

营业收入：{revenue}
净利润：{net_profit}
总资产：{total_assets}
净资产：{net_assets}
每股收益：{eps}
毛利率：{gross_margin}
净利率：{net_margin}
ROE：{roe}"""
            }
        ]
        
        for template in templates:
            example = TrainingExample(
                instruction=template["instruction"],
                input=template["input"],
                output=template["output_template"],
                system="你是一位专业的信息提取专家，擅长从文本中提取结构化信息。",
                metadata={"task_type": "summary_extraction"}
            )
            examples.append(example)
        
        return examples
    
    def generate_dataset_from_templates(self, num_samples: int = 1000) -> List[TrainingExample]:
        """
        从模板生成数据集
        
        Args:
            num_samples: 生成样本数量
            
        Returns:
            训练样本列表
        """
        examples = []
        
        # 模拟公司数据
        mock_companies = [
            {"name": "贵州茅台", "stock_code": "600519", "industry": "白酒"},
            {"name": "中国平安", "stock_code": "601318", "industry": "保险"},
            {"name": "招商银行", "stock_code": "600036", "industry": "银行"},
            {"name": "宁德时代", "stock_code": "300750", "industry": "新能源"},
            {"name": "比亚迪", "stock_code": "002594", "industry": "汽车"},
        ]
        
        for i in range(num_samples):
            company = random.choice(mock_companies)
            task_type = random.choice(list(self.TASK_TYPES.keys()))
            
            if task_type == "financial_analysis":
                examples.extend(self.generate_financial_analysis_qa(company))
            elif task_type == "business_review":
                examples.extend(self.generate_business_review_qa(company))
            elif task_type == "risk_assessment":
                examples.extend(self.generate_risk_assessment_qa(company))
            elif task_type == "summary_extraction":
                examples.extend(self.generate_summary_qa("示例年报内容..."))
        
        return examples[:num_samples]


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, output_dir: str = "./data/train"):
        """
        初始化数据集构建器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qa_generator = AnnualReportQAGenerator()
    
    def build_training_dataset(
        self,
        num_samples: int = 1000,
        output_format: str = "jsonl",
        split_ratio: tuple = (0.8, 0.1, 0.1)
    ) -> Dict[str, Path]:
        """
        构建训练数据集
        
        Args:
            num_samples: 样本数量
            output_format: 输出格式（jsonl/json）
            split_ratio: 训练/验证/测试集比例
            
        Returns:
            数据集文件路径字典
        """
        logger.info(f"开始构建数据集，目标样本数: {num_samples}")
        
        # 生成样本
        examples = self.qa_generator.generate_dataset_from_templates(num_samples)
        
        # 打乱顺序
        random.shuffle(examples)
        
        # 划分数据集
        train_size = int(len(examples) * split_ratio[0])
        val_size = int(len(examples) * split_ratio[1])
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        # 保存数据集
        output_files = {}
        
        if output_format == "jsonl":
            output_files['train'] = self._save_jsonl(train_examples, "train.jsonl")
            output_files['validation'] = self._save_jsonl(val_examples, "validation.jsonl")
            output_files['test'] = self._save_jsonl(test_examples, "test.jsonl")
        else:
            output_files['train'] = self._save_json(train_examples, "train.json")
            output_files['validation'] = self._save_json(val_examples, "validation.json")
            output_files['test'] = self._save_json(test_examples, "test.json")
        
        # 保存数据集信息
        dataset_info = {
            "total_samples": len(examples),
            "train_samples": len(train_examples),
            "validation_samples": len(val_examples),
            "test_samples": len(test_examples),
            "task_types": list(AnnualReportQAGenerator.TASK_TYPES.keys()),
            "files": {k: str(v) for k, v in output_files.items()}
        }
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集构建完成！")
        logger.info(f"训练集: {len(train_examples)} 样本")
        logger.info(f"验证集: {len(val_examples)} 样本")
        logger.info(f"测试集: {len(test_examples)} 样本")
        
        return output_files
    
    def _save_jsonl(self, examples: List[TrainingExample], filename: str) -> Path:
        """保存为JSONL格式"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in examples:
                # 使用对话格式
                chat_data = example.to_chat_format()
                f.write(json.dumps(chat_data, ensure_ascii=False) + '\n')
        return filepath
    
    def _save_json(self, examples: List[TrainingExample], filename: str) -> Path:
        """保存为JSON格式"""
        filepath = self.output_dir / filename
        data = [ex.to_chat_format() for ex in examples]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """
        加载数据集
        
        Args:
            filepath: 文件路径
            
        Returns:
            数据列表
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.jsonl':
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建年报分析训练数据集")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="样本数量")
    parser.add_argument("--output-dir", type=str, default="./data/train",
                       help="输出目录")
    parser.add_argument("--format", type=str, default="jsonl",
                       choices=["jsonl", "json"],
                       help="输出格式")
    
    args = parser.parse_args()
    
    # 构建数据集
    builder = DatasetBuilder(output_dir=args.output_dir)
    output_files = builder.build_training_dataset(
        num_samples=args.num_samples,
        output_format=args.format
    )
    
    print("数据集构建完成！")
    print(f"训练集: {output_files['train']}")
    print(f"验证集: {output_files['validation']}")
    print(f"测试集: {output_files['test']}")


if __name__ == "__main__":
    main()
