#!/usr/bin/env python3
"""
Embedding模型服务 - 使用 BGE 模型（兼容旧版 transformers）
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 设置模型路径 - 使用 BGE 模型，兼容性好
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MODEL_PATH = "/mnt/workspace/models/embedding/bge-large-zh-v1.5"

# 创建FastAPI应用
app = FastAPI(
    title="Embedding Service",
    description="文本向量化服务 - BGE",
    version="1.0.0"
)

# 全局模型和tokenizer
model = None
tokenizer = None

def download_model():
    """下载模型"""
    global MODEL_PATH
    
    print("📥 下载Embedding模型...")
    
    try:
        # 尝试使用ModelScope
        from modelscope import snapshot_download
        MODEL_PATH = snapshot_download("BAAI/bge-large-zh-v1.5", 
                                       cache_dir="/mnt/workspace/models/embedding")
        print(f"✅ ModelScope下载完成: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ ModelScope失败: {e}")
        print("🔄 尝试HuggingFace...")
        MODEL_PATH = MODEL_NAME

def load_model():
    """加载模型"""
    global model, tokenizer
    
    print("📥 加载Embedding模型...")
    print(f"   模型: {MODEL_NAME}")
    
    from transformers import AutoTokenizer, AutoModel
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ 快速tokenizer加载失败: {e}")
        print("🔄 使用慢速tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )
    
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    print("✅ Embedding模型加载完成")
    print(f"   设备: {model.device}")
    print(f"   维度: 1024")

def encode_text(text: Union[str, List[str]]) -> np.ndarray:
    """编码文本"""
    if isinstance(text, str):
        text = [text]
    
    # 添加指令前缀（BGE模型推荐）
    instruction = "为这个句子生成表示以用于检索相关文章："
    text = [instruction + t for t in text]
    
    # 编码
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 [CLS] token 的嵌入
        embeddings = outputs.last_hidden_state[:, 0]
        # 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

# 请求/响应模型
class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    print("=" * 60)
    print("🚀 启动Embedding服务 (BGE)")
    print("=" * 60)
    print(f"地址: http://0.0.0.0:8001")
    print(f"接口: POST /embed")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        download_model()
    
    load_model()

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(model.device) if model else "unknown"
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    文本向量化接口
    
    - texts: 文本或文本列表
    - normalize: 是否归一化（默认True）
    """
    try:
        embeddings = encode_text(request.texts)
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1],
            count=embeddings.shape[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
