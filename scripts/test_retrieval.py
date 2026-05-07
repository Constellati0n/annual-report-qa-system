#!/usr/bin/env python3
"""
测试 RAG 检索功能
"""
import sys
sys.path.insert(0, '/mnt/workspace/annual_report_assistant')

import torch
from sentence_transformers import SentenceTransformer
import chromadb

def test_retrieval():
    print("测试 RAG 检索...")
    
    # 初始化模型
    model_path = "/mnt/workspace/models/embedding/BAAI/bge-large-zh-v1.5"
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 连接向量数据库
    client = chromadb.PersistentClient(path="./data/vector_db")
    collection = client.get_collection("annual_reports")
    
    # 测试查询
    query = "平安银行 净利润"
    print(f"\n查询: {query}")
    
    # 生成查询向量
    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    
    # 检索（不使用相似度过滤，看看原始分数）
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n检索结果:")
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        
        # ChromaDB 返回的是距离（越小越相似），转换为相似度分数
        similarity = 1 - distance
        
        print(f"\n  [{i+1}] ID: {doc_id}")
        print(f"      距离: {distance:.4f}")
        print(f"      相似度: {similarity:.4f}")
        print(f"      公司: {metadata.get('company_name', 'N/A')}")
        print(f"      年份: {metadata.get('year', 'N/A')}")
        print(f"      内容: {results['documents'][0][i][:100]}...")
    
    # 检查相似度分布
    print(f"\n相似度统计:")
    similarities = [1 - d for d in results['distances'][0]]
    print(f"  最高: {max(similarities):.4f}")
    print(f"  最低: {min(similarities):.4f}")
    print(f"  平均: {sum(similarities)/len(similarities):.4f}")
    
    # 检查有多少超过阈值 0.5
    above_threshold = sum(1 for s in similarities if s >= 0.5)
    print(f"\n超过阈值 0.5 的结果: {above_threshold}/{len(similarities)}")

if __name__ == "__main__":
    test_retrieval()
