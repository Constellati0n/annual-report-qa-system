#!/usr/bin/env python3
"""
测试模型是否能正常生成
"""
import os
import sys
sys.path.insert(0, '/mnt/workspace/annual_report_assistant')

# 设置 CUDA 调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 60)
print("测试模型生成")
print("=" * 60)

try:
    from client.llm_chat import LLMChatClient
    
    print("\n1. 创建客户端...")
    client = LLMChatClient(
        model_path="/mnt/workspace/models/finetuned",
        load_in_4bit=False
    )
    
    print("\n2. 测试简单生成...")
    response = client.chat("你好", max_new_tokens=50)
    
    print(f"\n3. 响应: {response}")
    print("\n✅ 测试成功！")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
