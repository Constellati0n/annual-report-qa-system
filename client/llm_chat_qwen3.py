#!/usr/bin/env python3
"""
Qwen3 LLM对话客户端
支持 Qwen3 特性：32K上下文、工具调用、思考模式控制
"""
import os
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class Qwen3ChatClient:
    """Qwen3 对话客户端 - 支持高级特性"""
    
    def __init__(
        self,
        model_path: str = None,
        base_model: str = "/mnt/workspace/models/llm/qwen/Qwen3-8B",
        device: str = None,
        load_in_4bit: bool = False,
        max_length: int = 32768,
        enable_thinking: bool = True,
        thinking_mode: str = "soft",  # "soft" 或 "hard"
        enable_tool_call: bool = True
    ):
        """
        初始化 Qwen3 客户端
        
        Args:
            model_path: 微调模型路径
            base_model: 基础模型路径
            device: 设备
            load_in_4bit: 是否4bit量化
            max_length: 最大上下文长度 (Qwen3 支持 32K)
            enable_thinking: 是否启用思考模式
            thinking_mode: 思考模式控制 (soft/hard)
            enable_tool_call: 是否启用工具调用
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.thinking_mode = thinking_mode
        self.enable_tool_call = enable_tool_call
        
        # 工具注册表
        self.tools: Dict[str, ToolDefinition] = {}
        
        print("=" * 60)
        print("初始化 Qwen3 对话客户端")
        print("=" * 60)
        print(f"设备: {device}")
        print(f"最大长度: {max_length}")
        print(f"思考模式: {thinking_mode} (启用: {enable_thinking})")
        print(f"工具调用: {enable_tool_call}")
        
        # 加载模型
        self._load_model(model_path, base_model)
        
        # 对话历史
        self.conversation_history = []
        
        print("=" * 60)
        print("Qwen3 模型加载完成！")
        print("=" * 60)
    
    def _load_model(self, model_path: str = None, base_model: str = None):
        """加载 Qwen3 模型"""
        peft_model_path = None
        base_model_path = None
        
        # 检查模型路径
        if model_path and os.path.exists(model_path):
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                peft_model_path = model_path
                print(f"检测到PEFT微调模型: {model_path}")
                
                try:
                    with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                        adapter_config = json.load(f)
                        config_base_model = adapter_config.get('base_model_name_or_path')
                    
                    if base_model and os.path.exists(base_model):
                        base_model_path = base_model
                    elif config_base_model and os.path.exists(config_base_model):
                        base_model_path = config_base_model
                    else:
                        raise FileNotFoundError("基础模型不存在")
                except Exception as e:
                    print(f"读取adapter配置失败: {e}")
                    raise
            else:
                base_model_path = model_path
                print(f"加载普通模型: {model_path}")
        elif base_model and os.path.exists(base_model):
            base_model_path = base_model
            print(f"加载基础模型: {base_model}")
        else:
            raise FileNotFoundError("未找到有效的模型路径")
        
        # 修复软链接
        if base_model_path and os.path.islink(base_model_path):
            real_path = os.path.realpath(base_model_path)
            if real_path != base_model_path:
                print(f"解析软链接: {base_model_path} -> {real_path}")
                base_model_path = real_path
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型 - Qwen3 使用 bfloat16
        print("加载 Qwen3 模型...")
        print(f"基础模型路径: {base_model_path}")
        
        # Qwen3 支持 32K 上下文
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        
        # 加载 PEFT adapter
        if peft_model_path:
            print(f"加载PEFT adapter: {peft_model_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, peft_model_path)
            print("✅ PEFT adapter 加载成功")
        
        self.model.eval()
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params / 1e9:.2f}B")
        print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    def register_tool(self, name: str, description: str, parameters: Dict, function: Callable):
        """注册工具"""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        print(f"✓ 注册工具: {name}")
    
    def get_tools_schema(self) -> List[Dict]:
        """获取工具 schema 用于模型调用"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def chat(
        self,
        message: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        enable_thinking: bool = None,  # 可覆盖默认设置
        tools: List[Dict] = None
    ) -> Any:
        """
        对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            stream: 是否流式输出
            enable_thinking: 是否启用思考模式
            tools: 工具定义列表
            
        Returns:
            若 stream=False 则返回字符串，若 stream=True 则返回生成器
        """
        # 使用默认设置或覆盖
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        
        if system_prompt is None:
            system_prompt = "你是一个专业的企业年报分析助手，擅长分析财务报表、解读经营情况、评估投资风险。"
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话
        for hist in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": str(hist["user"])})
            messages.append({"role": "assistant", "content": str(hist["assistant"])})
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        # 思考模式控制 - 软开关
        if thinking and self.thinking_mode == "soft":
            # 在消息中提示模型是否思考
            pass  # 由用户指令控制
        elif not thinking or self.thinking_mode == "hard":
            # 硬开关：添加不思考的指令
            messages[0]["content"] += "\n请直接回答，不需要展示思考过程。"
        
        # 应用对话模板
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or (self.get_tools_schema() if self.enable_tool_call else None)
        )
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.model.device)
        
        if stream:
            return self._generate_stream(inputs, max_new_tokens, temperature, top_p, message, prompt)
        else:
            return self._generate(inputs, max_new_tokens, temperature, top_p, message, prompt)
    
    def _generate(self, inputs, max_new_tokens, temperature, top_p, user_message, prompt) -> str:
        """生成回复"""
        with torch.no_grad():
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_token_id = self.tokenizer.eos_token_id or pad_token_id
            
            if temperature < 0.1:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 2048),
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 2048),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手回复
        if "<|im_start|>assistant" in response:
            assistant_response = response.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|")[0].strip()
            response = assistant_response
        else:
            response = response[len(prompt):].strip()
        
        # 清理响应
        response = self._clean_response(response)
        
        # 保存对话历史
        self.conversation_history.append({
            "user": user_message,
            "assistant": response
        })
        
        return response
    
    def _generate_stream(self, inputs, max_new_tokens, temperature, top_p, user_message, prompt):
        """流式生成"""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=min(max_new_tokens, 2048),
            do_sample=temperature >= 0.1,
            temperature=temperature if temperature >= 0.1 else None,
            top_p=top_p if temperature >= 0.1 else None,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        full_response = ""
        # 记录是否已经在打印（用于交互模式）
        is_interactive = sys.stdout.isatty()
        
        for text in streamer:
            if is_interactive:
                print(text, end="", flush=True)
            full_response += text
            yield text
        
        if is_interactive:
            print()
            
        thread.join()
        
        # 清理响应并保存历史
        cleaned_response = self._clean_response(full_response)
        
        self.conversation_history.append({
            "user": user_message,
            "assistant": cleaned_response
        })
    
    def _clean_response(self, response: str) -> str:
        """清理响应"""
        import re
        
        original_response = response
        
        # 检测系统提示词重复
        system_markers = [
            "你是一位专业的企业年报分析",
            "你是一个专业的企业年报分析",
            "<|im_start|>system"
        ]
        
        for marker in system_markers:
            if marker in response and response.count(marker) >= 2:
                if "<|im_start|>assistant" in response:
                    last_assistant = response.rfind("<|im_start|>assistant")
                    response = response[last_assistant + len("<|im_start|>assistant"):].strip()
                break
        
        # 移除用户标记
        user_markers = ["<|im_start|>user", "\nuser\n", "Human:", "用户："]
        for marker in user_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # 移除特殊token
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        # 清理无意义符号
        response = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]{3,}', ' ', response)
        response = re.sub(r'\s+', ' ', response)
        
        # 清理开头标记
        response = response.lstrip("# ").lstrip("## ").strip()
        
        return response.strip()
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "=" * 60)
        print("Qwen3 年报分析助手 - 交互式模式")
        print("=" * 60)
        print("命令：")
        print("  /think on|off  - 开启/关闭思考模式")
        print("  /clear         - 清空历史")
        print("  /quit          - 退出")
        print("=" * 60 + "\n")
        
        current_thinking = self.enable_thinking
        
        while True:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "/quit":
                print("再见！")
                break
            
            if user_input.lower() == "/clear":
                self.clear_history()
                continue
            
            if user_input.lower().startswith("/think "):
                mode = user_input.split()[1].lower()
                current_thinking = (mode == "on")
                print(f"思考模式: {'开启' if current_thinking else '关闭'}")
                continue
            
            # 生成回复
            print("\n助手: ", end="", flush=True)
            response = self.chat(user_input, stream=True, enable_thinking=current_thinking)
            # 迭代生成器以触发打印
            if hasattr(response, "__iter__"):
                for _ in response:
                    pass
            print("\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 对话客户端")
    parser.add_argument("--model-path", type=str, default=None,
                       help="微调模型路径")
    parser.add_argument("--base-model", type=str, default="/mnt/workspace/models/llm/qwen/Qwen3-8B",
                       help="基础模型路径")
    parser.add_argument("--max-length", type=int, default=32768,
                       help="最大上下文长度 (Qwen3 支持 32K)")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="使用4bit量化")
    parser.add_argument("--no-thinking", action="store_true",
                       help="禁用思考模式")
    parser.add_argument("--thinking-mode", type=str, default="soft",
                       choices=["soft", "hard"],
                       help="思考模式控制")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = Qwen3ChatClient(
        model_path=args.model_path,
        base_model=args.base_model,
        max_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=not args.no_thinking,
        thinking_mode=args.thinking_mode
    )
    
    # 启动交互式对话
    client.interactive_chat()


if __name__ == "__main__":
    main()
