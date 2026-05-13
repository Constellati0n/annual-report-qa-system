"""
独立微调脚本 v3 - 72万条训练数据优化版
"""
import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    model_name: str = "/mnt/workspace/models/llm/qwen/Qwen3-8B"
    output_dir: str = "./models/llm_finetuned_v2"
    train_file: str = "./data/train/train_v2.json"
    validation_file: str = "./data/train/val_v2.json"
    max_length: int = 1536
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    save_steps: int = 5000
    eval_steps: int = 5000
    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 0


class AnnualReportTrainer:
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        torch.manual_seed(config.seed)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.config.model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        logger.info("Model loaded.")
    
    def setup_lora(self):
        logger.info("Setting up LoRA...")
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        logger.info("Loading dataset (streaming)...")
        dataset = load_dataset("json", data_files={
            "train": self.config.train_file,
            "validation": self.config.validation_file
        }, split={"train": "train[:90%]", "validation": "train[90%:]"})
        logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
        return dataset
    
    def preprocess_function(self, examples):
        texts = []
        for messages in examples["messages"]:
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            texts.append("\n".join(parts))
        
        tokenized = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            truncation=True
        )
        
        tokenized["labels"] = [
            [-100 if tid == self.tokenizer.pad_token_id else tid for tid in ids]
            for ids in tokenized["input_ids"]
        ]
        
        return tokenized
    
    def train(self):
        logger.info("Loading tokenizer for preprocessing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading raw dataset...")
        raw_dataset = load_dataset("json", data_files={
            "train": self.config.train_file,
            "validation": self.config.validation_file
        })
        logger.info(f"Raw - train: {len(raw_dataset['train'])}, val: {len(raw_dataset['validation'])}")
        
        logger.info("Tokenizing dataset (batch_size=500)...")
        tokenized = raw_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=500,
            remove_columns=raw_dataset["train"].column_names,
            desc="Tokenizing"
        )
        logger.info(f"Tokenized - train: {len(tokenized['train'])}, val: {len(tokenized['validation'])}")
        
        logger.info("Freeing raw dataset from memory...")
        del raw_dataset
        import gc
        gc.collect()
        
        logger.info("Loading model...")
        self.load_model_and_tokenizer()
        self.setup_lora()
        
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
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=self.config.fp16,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=[],
            seed=self.config.seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info("Starting training (726K samples, 1 epoch)...")
        trainer.train()
        
        logger.info("Saving model...")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Training done! Saved to {self.config.output_dir}")


def main():
    os.chdir("/mnt/workspace/annual_report_assistant")
    
    config = FinetuneConfig()
    
    logger.info(f"Dataset: {config.train_file}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Epochs: {config.num_epochs}, BS: {config.batch_size}x{config.gradient_accumulation_steps}")
    logger.info(f"Max length: {config.max_length}")
    logger.info(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    
    trainer = AnnualReportTrainer(config)
    trainer.train()
    
    logger.info("=== Fine-tuning Complete! ===")


if __name__ == "__main__":
    main()
