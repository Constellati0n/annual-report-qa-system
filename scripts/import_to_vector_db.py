#!/usr/bin/env python3
"""
将处理好的 JSON 数据导入到向量数据库
"""
import sys
import os
import json
import argparse

# 添加项目路径
sys.path.insert(0, '/mnt/workspace/annual_report_assistant')

def import_documents(json_path: str):
    """导入文档到向量数据库"""
    print(f"导入文件: {json_path}")
    
    # 检查文件
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
        return False
    
    # 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"  加载了 {len(documents)} 个文档")
    
    # 导入到向量数据库
    try:
        import torch
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
        
        # 初始化嵌入模型
        print("\n初始化嵌入模型...")
        model_path = "/mnt/workspace/models/embedding/BAAI/bge-large-zh-v1.5"
        model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"  模型加载完成，维度: {model.get_sentence_embedding_dimension()}")
        
        # 初始化 ChromaDB
        print("初始化向量存储...")
        persist_dir = "./data/vector_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # 获取或创建集合
        collection = client.get_or_create_collection(
            name="annual_reports",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 批量添加文档
        print(f"\n添加文档到向量数据库...")
        batch_size = 50
        total = len(documents)
        
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            
            # 准备数据
            texts = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            ids = [doc["id"] for doc in batch]
            
            # 生成 embeddings
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            
            # 添加到向量存储
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"  进度: {min(i+batch_size, total)}/{total}")
        
        # 保存
        print("\n保存向量数据库...")
        # ChromaDB 自动保存
        
        print(f"\n✅ 导入完成！")
        print(f"  总文档数: {total}")
        
        # 检查知识库状态
        count = collection.count()
        print(f"  向量库文档数: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="导入 JSON 文档到向量数据库")
    parser.add_argument("json_path", default="data/processed_documents.json",
                       help="JSON 文件路径")
    
    args = parser.parse_args()
    
    import_documents(args.json_path)


if __name__ == "__main__":
    main()
