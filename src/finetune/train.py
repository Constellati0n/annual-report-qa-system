"""
年报分析模型微调脚本
基于LoRA进行参数高效微调
"""
import os
import json
import torch
from pathlib import Path
from typing import List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


class AnnualReportTrainer:
    """年报分析模型训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = None,
        use_4bit: bool = True
    ):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            use_4bit: 是否使用4bit量化
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = str(project_root / "models" / "annual_report_assistant")
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print(f"\n加载模型: {self.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # 准备模型用于训练
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        
        print(f"模型加载完成")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def setup_lora(self):
        """配置LoRA"""
        print("\n配置LoRA...")
        
        lora_config = LoraConfig(
            r=64,  # LoRA秩
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"总参数量: {all_params / 1e6:.2f}M")
        print(f"训练比例: {100 * trainable_params / all_params:.2f}%")
    
    def prepare_dataset(self, data_file: str = None) -> Dataset:
        """
        准备训练数据集
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            数据集
        """
        if data_file is None:
            project_root = Path(__file__).parent.parent.parent
            data_file = str(project_root / "data" / "processed" / "structured" / "structured_reports.json")
        
        print(f"\n加载数据集: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            reports = json.load(f)
        
        # 构建训练样本
        training_data = []
        
        for report in reports:
            # 构建指令样本
            stock_name = report.get('stock_name', '')
            report_year = report.get('report_year', '')
            
            # 财务分析样本
            financial = report.get('financial_indicators', {})
            if financial.get('total_revenue') and financial.get('net_profit'):
                prompt = f"分析{stock_name}{report_year}年的财务状况。"
                response = self._build_financial_analysis(report)
                
                training_data.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response
                })
            
            # 股东分析样本
            shareholder = report.get('shareholder_info', {})
            if shareholder.get('controlling_shareholder'):
                prompt = f"分析{stock_name}的股权结构。"
                response = self._build_shareholder_analysis(report)
                
                training_data.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response
                })
            
            # 经营分析样本
            business = report.get('business_analysis', {})
            if business.get('industry_overview'):
                prompt = f"分析{stock_name}的经营情况和行业地位。"
                response = self._build_business_analysis(report)
                
                training_data.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response
                })
        
        print(f"构建了 {len(training_data)} 个训练样本")
        
        # 转换为Hugging Face Dataset
        dataset = Dataset.from_list(training_data)
        
        # 格式化数据
        def format_prompt(example):
            prompt = f"""<|im_start|>system
你是一个专业的企业年报分析助手。请基于提供的信息，对用户的问题进行详细、准确的分析。<|im_end|>
<|im_start|>user
{example['instruction']}{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
            
            return {'text': prompt}
        
        dataset = dataset.map(format_prompt)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=2048,
                padding='max_length'
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        
        return dataset
    
    def _build_financial_analysis(self, report: Dict) -> str:
        """构建财务分析文本"""
        financial = report.get('financial_indicators', {})
        
        analysis = f"""根据{report.get('stock_name', '')}{report.get('report_year', '')}年报财务数据：

**盈利能力**：
- 营业收入：{financial.get('total_revenue', 'N/A')}亿元
- 净利润：{financial.get('net_profit', 'N/A')}亿元
- 每股收益：{financial.get('eps', 'N/A')}元
- 净资产收益率(ROE)：{financial.get('roe', 'N/A')}%

**资产负债**：
- 总资产：{financial.get('total_assets', 'N/A')}亿元
- 净资产：{financial.get('net_assets', 'N/A')}亿元

**分析**：
该公司{report.get('report_year', '')}年整体财务状况{"良好" if financial.get('net_profit', 0) > 0 else "有待改善"}。
"""
        return analysis
    
    def _build_shareholder_analysis(self, report: Dict) -> str:
        """构建股东分析文本"""
        shareholder = report.get('shareholder_info', {})
        
        analysis = f"""根据{report.get('stock_name', '')}{report.get('report_year', '')}年报股东信息：

**股权结构**：
- 控股股东：{shareholder.get('controlling_shareholder', 'N/A')}
- 实际控制人：{shareholder.get('actual_controller', 'N/A')}
- 总股本：{shareholder.get('total_shares', 'N/A')}股

**分析**：
该公司股权结构{"集中" if shareholder.get('controlling_shareholder') else "分散"}，
由{shareholder.get('controlling_shareholder', '未知')}控股。
"""
        return analysis
    
    def _build_business_analysis(self, report: Dict) -> str:
        """构建经营分析文本"""
        business = report.get('business_analysis', {})
        
        analysis = f"""根据{report.get('stock_name', '')}{report.get('report_year', '')}年报经营分析：

**行业概况**：
{business.get('industry_overview', 'N/A')}

**核心竞争力**：
"""
        for i, core in enumerate(business.get('core_competitiveness', [])[:3], 1):
            analysis += f"{i}. {core}\n"
        
        analysis += f"""
**经营风险**：
"""
        for i, risk in enumerate(business.get('business_risks', [])[:3], 1):
            analysis += f"{i}. {risk}\n"
        
        return analysis
    
    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """
        训练模型
        
        Args:
            dataset: 训练数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        print("\n开始训练...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_8bit",
            learning_rate=learning_rate,
            warmup_ratio=0.03,
            weight_decay=0.001,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            bf16=False,
            group_by_length=True,
            report_to="none"
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        print(f"\n保存模型到: {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("训练完成！")
    
    def run(self, data_file: str = None):
        """运行完整训练流程"""
        print("=" * 60)
        print("年报分析模型微调")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model_and_tokenizer()
        
        # 2. 配置LoRA
        self.setup_lora()
        
        # 3. 准备数据
        dataset = self.prepare_dataset(data_file)
        
        # 4. 训练
        self.train(dataset)
        
        print("\n" + "=" * 60)
        print("微调完成！")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='微调年报分析模型')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='基础模型名称')
    parser.add_argument('--data', type=str, default=None,
                       help='训练数据文件')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = AnnualReportTrainer(
        model_name=args.model,
        output_dir=args.output
    )
    
    # 运行训练
    trainer.run(data_file=args.data)


if __name__ == "__main__":
    main()
