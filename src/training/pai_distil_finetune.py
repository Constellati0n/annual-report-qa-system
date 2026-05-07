"""
PAI-DistilQwen2.5-7B-Instruct 模型微调脚本
针对阿里云PAI平台优化
支持SFT和DPO训练策略
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
class PAIConfig:
    """PAI训练配置"""
    # 模型配置
    model_name: str = "alibaba-pai/pai-distilqwen2.5-7b-instruct"
    output_dir: str = "./models/pai_distil_finetuned"
    
    # 训练策略: sft 或 dpo
    training_strategy: str = "sft"
    
    # 数据配置
    train_file: str = "./data/train/train.jsonl"
    validation_file: str = "./data/train/validation.jsonl"
    max_seq_length: int = 2048
    
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
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Chat Template配置
    apply_chat_template: bool = True
    custom_system_prompt: str = ""
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 其他配置
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 4


class PAIDistilTrainer:
    """PAI-DistilQwen2.5-7B-Instruct训练器"""
    
    def __init__(self, config: PAIConfig):
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
        
        logger.info(f"训练策略: {config.training_strategy}")
        logger.info(f"模型: {config.model_name}")
    
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
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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
        torch_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype) if self.config.bf16 else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        
        # 为量化训练做准备
        if self.config.load_in_4bit or self.config.load_in_8bit:
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
    
    def preprocess_sft(self, examples):
        """SFT数据预处理"""
        # 构建对话文本
        conversations = []
        
        for i in range(len(examples["instruction"])):
            # 获取system prompt
            if self.config.custom_system_prompt and "system_prompt" not in examples:
                system_prompt = self.config.custom_system_prompt
            else:
                system_prompt = examples.get("system_prompt", [""])[i] if "system_prompt" in examples else ""
            
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            
            # 构建消息列表
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": output})
            
            # 应用chat template
            if self.config.apply_chat_template:
                conversation = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # 手动构建
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
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 设置labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        model_inputs["labels"][model_inputs["attention_mask"] == 0] = -100
        
        return model_inputs
    
    def preprocess_dpo(self, examples):
        """DPO数据预处理"""
        # DPO格式: prompt, chosen, rejected
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        # 构建chosen和rejected的完整对话
        chosen_texts = []
        rejected_texts = []
        
        for i in range(len(prompts)):
            # Chosen对话
            chosen_messages = [
                {"role": "user", "content": prompts[i]},
                {"role": "assistant", "content": chosen[i]}
            ]
            chosen_text = self.tokenizer.apply_chat_template(
                chosen_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            chosen_texts.append(chosen_text)
            
            # Rejected对话
            rejected_messages = [
                {"role": "user", "content": prompts[i]},
                {"role": "assistant", "content": rejected[i]}
            ]
            rejected_text = self.tokenizer.apply_chat_template(
                rejected_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            rejected_texts.append(rejected_text)
        
        # 编码
        chosen_inputs = self.tokenizer(
            chosen_texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_inputs = self.tokenizer(
            rejected_texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
        }
    
    def train(self):
        """开始训练"""
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 配置LoRA
        self.setup_lora()
        
        # 加载数据集
        dataset = self.load_dataset()
        
        # 选择预处理方法
        if self.config.training_strategy == "sft":
            preprocess_function = self.preprocess_sft
            remove_columns = ["instruction", "output"]
            if "system_prompt" in dataset["train"].column_names:
                remove_columns.append("system_prompt")
        elif self.config.training_strategy == "dpo":
            preprocess_function = self.preprocess_dpo
            remove_columns = ["prompt", "chosen", "rejected"]
        else:
            raise ValueError(f"不支持的训练策略: {self.config.training_strategy}")
        
        # 预处理数据集
        logger.info("预处理数据集...")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns
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
            bf16=self.config.bf16,
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
            torch_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
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


def load_config_from_yaml(yaml_path: str) -> PAIConfig:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    lora_config = config_dict.get('lora', {})
    data_config = config_dict.get('data', {})
    
    return PAIConfig(
        model_name=model_config.get('name', 'alibaba-pai/pai-distilqwen2.5-7b-instruct'),
        output_dir=model_config.get('output_dir', './models/pai_distil_finetuned'),
        training_strategy=training_config.get('training_strategy', 'sft'),
        train_file=data_config.get('train_file', './data/train/train.jsonl'),
        validation_file=data_config.get('validation_file', './data/train/validation.jsonl'),
        max_seq_length=training_config.get('max_seq_length', 2048),
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
        load_in_4bit=model_config.get('load_in_4bit', True),
        load_in_8bit=model_config.get('load_in_8bit', False),
        bnb_4bit_compute_dtype=model_config.get('torch_dtype', 'bfloat16'),
        apply_chat_template=training_config.get('apply_chat_template', True),
        custom_system_prompt=training_config.get('custom_system_prompt', ''),
        save_steps=training_config.get('save_steps', 500),
        eval_steps=training_config.get('eval_steps', 500),
        logging_steps=training_config.get('logging_steps', 10),
        save_total_limit=training_config.get('save_total_limit', 3),
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
    )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PAI-DistilQwen2.5-7B-Instruct 微调")
    parser.add_argument("--config", type=str, default="./config/pai_distil_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--train-file", type=str, default=None,
                       help="训练数据文件路径")
    parser.add_argument("--validation-file", type=str, default=None,
                       help="验证数据文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--strategy", type=str, choices=["sft", "dpo"], default=None,
                       help="训练策略")
    parser.add_argument("--merge", action="store_true",
                       help="训练完成后合并LoRA权重")
    
    args = parser.parse_args()
    
    # 加载配置
    if Path(args.config).exists():
        config = load_config_from_yaml(args.config)
    else:
        config = PAIConfig()
    
    # 覆盖配置
    if args.train_file:
        config.train_file = args.train_file
    if args.validation_file:
        config.validation_file = args.validation_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.strategy:
        config.training_strategy = args.strategy
    
    # 创建训练器
    trainer = PAIDistilTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 合并权重（可选）
    if args.merge:
        trainer.merge_and_save()


if __name__ == "__main__":
    main()
