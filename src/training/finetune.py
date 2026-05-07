"""
模型微调脚本
用于微调 Qwen2.5-7B-Instruct 模型
"""
import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import load_dataset

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """微调配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./models/llm_finetuned"
    
    # 数据配置
    train_file: str = "./data/train/train.jsonl"
    validation_file: str = "./data/train/validation.jsonl"
    max_length: int = 2048
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 其他配置
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4


class AnnualReportTrainer:
    """年报分析模型训练器"""
    
    def __init__(self, config: FinetuneConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载模型: {self.config.model_name}")
        
        # 量化配置
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        # 为量化训练做准备
        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("模型加载完成")
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"总参数量: {total_params / 1e9:.2f}B")
        logger.info(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    def setup_lora(self):
        """配置LoRA"""
        logger.info("配置LoRA...")
        
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """加载数据集"""
        logger.info("加载数据集...")
        
        data_files = {}
        if Path(self.config.train_file).exists():
            data_files["train"] = self.config.train_file
        if Path(self.config.validation_file).exists():
            data_files["validation"] = self.config.validation_file
        
        if not data_files:
            raise ValueError("未找到训练数据文件")
        
        dataset = load_dataset("json", data_files=data_files)
        
        logger.info(f"训练集大小: {len(dataset.get('train', []))}")
        logger.info(f"验证集大小: {len(dataset.get('validation', []))}")
        
        return dataset
    
    def preprocess_function(self, examples):
        """预处理函数"""
        # 构建对话文本
        conversations = []
        for messages in examples["messages"]:
            conversation = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            conversations.append(conversation)
        
        # 编码
        model_inputs = self.tokenizer(
            conversations,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 设置labels（用于计算loss）
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        # 将pad_token的label设为-100（不计算loss）
        model_inputs["labels"][model_inputs["attention_mask"] == 0] = -100
        
        return model_inputs
    
    def train(self):
        """开始训练"""
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 配置LoRA
        self.setup_lora()
        
        # 加载数据集
        dataset = self.load_dataset()
        
        # 预处理数据集
        logger.info("预处理数据集...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # 保存和评估
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if "validation" in tokenized_dataset else "no",
            load_best_model_at_end=True if "validation" in tokenized_dataset else False,
            metric_for_best_model="eval_loss" if "validation" in tokenized_dataset else None,
            
            # 优化
            fp16=self.config.fp16,
            optim="paged_adamw_8bit" if self.config.load_in_4bit else "adamw_torch",
            lr_scheduler_type="cosine",
            
            # 日志
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["tensorboard"],
            
            # 其他
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 回调函数
        callbacks = []
        if "validation" in tokenized_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # 保存LoRA配置
        self.peft_config.save_pretrained(self.config.output_dir)
        
        logger.info(f"训练完成！模型已保存到: {self.config.output_dir}")
    
    def merge_and_save(self, output_path: Optional[str] = None):
        """
        合并LoRA权重并保存完整模型
        
        Args:
            output_path: 输出路径
        """
        output_path = output_path or f"{self.config.output_dir}_merged"
        
        logger.info("合并LoRA权重...")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, self.config.output_dir)
        
        # 合并权重
        model = model.merge_and_unload()
        
        # 保存
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"合并后的模型已保存到: {output_path}")


def load_config_from_yaml(yaml_path: str) -> FinetuneConfig:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    training_config = config_dict.get('training', {})
    lora_config = training_config.get('lora', {})
    
    return FinetuneConfig(
        model_name=config_dict.get('models', {}).get('llm', {}).get('base_model', 'Qwen/Qwen2.5-7B-Instruct'),
        output_dir=training_config.get('output_dir', './models/llm_finetuned'),
        num_epochs=training_config.get('num_epochs', 3),
        batch_size=training_config.get('batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2e-4),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=training_config.get('max_grad_norm', 0.3),
        lora_r=lora_config.get('r', 64),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0.1),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        save_steps=training_config.get('save_steps', 500),
        eval_steps=training_config.get('eval_steps', 500),
        logging_steps=training_config.get('logging_steps', 10),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_in_4bit=config_dict.get('models', {}).get('llm', {}).get('load_in_4bit', True)
    )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="微调年报分析模型")
    parser.add_argument("--config", type=str, default="./config/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--train-file", type=str, default=None,
                       help="训练数据文件路径")
    parser.add_argument("--validation-file", type=str, default=None,
                       help="验证数据文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--merge", action="store_true",
                       help="训练完成后合并LoRA权重")
    
    args = parser.parse_args()
    
    # 加载配置
    if Path(args.config).exists():
        config = load_config_from_yaml(args.config)
    else:
        config = FinetuneConfig()
    
    # 覆盖配置
    if args.train_file:
        config.train_file = args.train_file
    if args.validation_file:
        config.validation_file = args.validation_file
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 创建训练器
    trainer = AnnualReportTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 合并权重（可选）
    if args.merge:
        trainer.merge_and_save()


if __name__ == "__main__":
    main()
