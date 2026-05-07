#!/usr/bin/env python3
"""
简化版LLM服务 - 使用原生Transformers
不需要vLLM
"""

import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置
MODEL_PATH = "/mnt/workspace/models/finetuned"
PORT = int(os.getenv("LLM_PORT", 8000))
HOST = os.getenv("LLM_HOST", "0.0.0.0")

# 创建FastAPI应用
app = FastAPI(
    title="LLM Service",
    description="年报分析助手LLM服务",
    version="1.0.0"
)

# 全局模型和tokenizer
model = None
tokenizer = None

def load_model():
    """加载模型"""
    global model, tokenizer
    
    print("📥 加载LLM模型...")
    print(f"   路径: {MODEL_PATH}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（8-bit量化节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True
    )
    
    model.eval()
    
    print("✅ LLM模型加载完成")
    print(f"   设备: {model.device}")

# 请求模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen2.5-7b-finetuned"
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        sys.exit(1)
    
    load_model()

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "LLM Service",
        "model": "qwen2.5-7b-finetuned",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    聊天完成接口（兼容OpenAI格式）
    """
    try:
        # 构建对话历史
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 应用chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        # 解码
        response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # 提取助手回复
        # 找到最后一个assistant的内容
        if "assistant" in response_text:
            response_text = response_text.split("assistant")[-1].strip()
        
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 512):
    """简单生成接口"""
    try:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"generated_text": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """主函数"""
    print(f"=" * 60)
    print(f"🚀 启动LLM服务 (简化版)")
    print(f"=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"服务地址: http://{HOST}:{PORT}")
    print(f"=" * 60)
    
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
