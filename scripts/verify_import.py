#!/usr/bin/env python3
"""
验证导入结果并修复持久化问题
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_and_fix():
    try:
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = "./data/vector_db"
        
        # 检查持久化目录
        print(f"检查持久化目录: {persist_dir}")
        if os.path.exists(persist_dir):
            files = os.listdir(persist_dir)
            print(f"  文件数: {len(files)}")
            for f in files[:5]:
                print(f"    - {f}")
        
        # 使用持久化客户端
        print("\n连接持久化数据库...")
        client = chromadb.PersistentClient(
            path=persist_dir
        )
        
        # 列出所有集合
        collections = client.list_collections()
        print(f"\n集合列表: {collections}")
        
        # 获取 annual_reports 集合
        try:
            collection = client.get_collection("annual_reports")
            count = collection.count()
            print(f"\n✅ annual_reports 集合存在")
            print(f"  文档数: {count}")
            
            if count > 0:
                # 测试查询
                print("\n测试查询...")
                results = collection.query(
                    query_texts=["净利润"],
                    n_results=3
                )
                print(f"  查询结果数: {len(results['ids'][0])}")
                
        except Exception as e:
            print(f"\n❌ 获取集合失败: {e}")
            print("\n尝试重新导入...")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    verify_and_fix()
