#!/usr/bin/env python3
"""
模型路径检查和下载脚本
检查 Qwen3 系列模型是否存在，如不存在则提供下载建议
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelChecker:
    """模型检查器"""
    
    # 模型配置
    MODELS = {
        "llm": {
            "name": "Qwen3-8B",
            "path": "/mnt/workspace/models/llm/qwen/Qwen3-8B",
            "hf_repo": "Qwen/Qwen3-8B",
            "required_files": ["config.json", "tokenizer.json", "model.safetensors.index.json"],
            "description": "通义千问3-8B 基础语言模型"
        },
        "embedding": {
            "name": "Qwen3-Embedding-0.6B",
            "path": "/mnt/workspace/models/embedding/qwen/Qwen3-Embedding-0.6B",
            "hf_repo": "Qwen/Qwen3-Embedding-0.6B",
            "required_files": ["config.json", "tokenizer.json"],
            "description": "通义千问3-Embedding-0.6B 向量化模型"
        },
        "reranker": {
            "name": "Qwen3-Reranker-0.6B",
            "path": "/mnt/workspace/models/reranker/qwen/Qwen3-Reranker-0.6B",
            "hf_repo": "Qwen/Qwen3-Reranker-0.6B",
            "required_files": ["config.json", "tokenizer.json"],
            "description": "通义千问3-Reranker-0.6B 重排序模型"
        }
    }
    
    def __init__(self):
        self.results = {}
    
    def check_model(self, model_key: str) -> Tuple[bool, str]:
        """
        检查单个模型
        
        Returns:
            (exists, message)
        """
        model_info = self.MODELS[model_key]
        model_path = Path(model_info["path"])
        
        print(f"\n检查 {model_info['name']}...")
        print(f"  路径: {model_path}")
        
        if not model_path.exists():
            return False, f"模型目录不存在"
        
        # 检查必需文件
        missing_files = []
        for file_name in model_info["required_files"]:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            return False, f"缺少文件: {', '.join(missing_files)}"
        
        # 检查模型文件大小
        safetensors_files = list(model_path.glob("*.safetensors"))
        bin_files = list(model_path.glob("*.bin"))
        model_files = safetensors_files + bin_files
        
        if not model_files:
            return False, "未找到模型权重文件"
        
        total_size = sum(f.stat().st_size for f in model_files)
        size_gb = total_size / (1024**3)
        
        return True, f"✓ 正常 ({size_gb:.2f} GB)"
    
    def check_all_models(self) -> Dict[str, Tuple[bool, str]]:
        """检查所有模型"""
        print("=" * 60)
        print("Qwen3 系列模型检查")
        print("=" * 60)
        
        results = {}
        for key in self.MODELS:
            exists, message = self.check_model(key)
            results[key] = (exists, message)
            status = "✓" if exists else "✗"
            print(f"  状态: {status} {message}")
        
        return results
    
    def print_download_instructions(self, missing_models: List[str]):
        """打印下载说明"""
        print("\n" + "=" * 60)
        print("模型下载指南")
        print("=" * 60)
        
        for key in missing_models:
            model_info = self.MODELS[key]
            print(f"\n【{model_info['name']}】")
            print(f"  描述: {model_info['description']}")
            print(f"  HuggingFace: {model_info['hf_repo']}")
            print(f"  本地路径: {model_info['path']}")
            print(f"\n  下载命令:")
            print(f"  mkdir -p {os.path.dirname(model_info['path'])}")
            print(f"  huggingface-cli download {model_info['hf_repo']} --local-dir {model_info['path']}")
        
        print("\n" + "=" * 60)
        print("或使用 ModelScope 下载（国内推荐）:")
        print("=" * 60)
        
        for key in missing_models:
            model_info = self.MODELS[key]
            ms_repo = model_info['hf_repo'].replace("Qwen/", "qwen/")
            print(f"\n  # {model_info['name']}")
            print(f"  modelscope download --model {ms_repo} --local_dir {model_info['path']}")
    
    def generate_download_script(self, missing_models: List[str], output_path: str = "download_models.sh"):
        """生成下载脚本"""
        script_lines = [
            "#!/bin/bash",
            "# Qwen3 系列模型下载脚本",
            "# 生成时间: $(date)",
            "",
            "set -e",
            "",
            "echo '========================================'",
            "echo '下载 Qwen3 系列模型'",
            "echo '========================================'",
            "",
        ]
        
        for key in missing_models:
            model_info = self.MODELS[key]
            script_lines.extend([
                f"",
                f"# {model_info['name']} - {model_info['description']}",
                f"echo ''",
                f"echo '下载 {model_info['name']}...'",
                f"mkdir -p {os.path.dirname(model_info['path'])}",
                f"huggingface-cli download {model_info['hf_repo']} --local-dir {model_info['path']} --local-dir-use-symlinks False",
                f"echo '✓ {model_info['name']} 下载完成'",
            ])
        
        script_lines.extend([
            "",
            "echo ''",
            "echo '========================================'",
            "echo '所有模型下载完成!'",
            "echo '========================================'",
        ])
        
        script_content = "\n".join(script_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        os.chmod(output_path, 0o755)
        print(f"\n✓ 下载脚本已生成: {output_path}")
        print(f"  运行命令: bash {output_path}")
    
    def run(self):
        """运行检查"""
        results = self.check_all_models()
        
        # 统计结果
        missing = [k for k, (exists, _) in results.items() if not exists]
        existing = [k for k, (exists, _) in results.items() if exists]
        
        print("\n" + "=" * 60)
        print("检查结果汇总")
        print("=" * 60)
        print(f"  已存在模型: {len(existing)}/3")
        print(f"  缺失模型: {len(missing)}/3")
        
        if missing:
            print(f"\n  缺失: {', '.join(self.MODELS[k]['name'] for k in missing)}")
            self.print_download_instructions(missing)
            self.generate_download_script(missing)
            return 1
        else:
            print("\n✓ 所有模型已就绪!")
            return 0


def test_models():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("测试模型加载")
    print("=" * 60)
    
    try:
        # 测试 Embedding 模型
        print("\n1. 测试 Embedding 模型...")
        from src.rag.embedding import EmbeddingManager
        embedding = EmbeddingManager()
        test_texts = ["测试文本1", "测试文本2"]
        embeddings = embedding.encode(test_texts)
        print(f"  ✓ Embedding 维度: {embeddings.shape}")
        embedding.close()
        
        # 测试 LLM 模型
        print("\n2. 测试 LLM 模型...")
        from client.llm_chat import LLMChatClient
        client = LLMChatClient()
        response = client.chat("你好", max_new_tokens=50, temperature=0.3)
        print(f"  ✓ LLM 响应: {response[:50]}...")
        
        print("\n✓ 所有模型测试通过!")
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 模型检查工具")
    parser.add_argument("--test", action="store_true",
                       help="测试模型加载")
    
    args = parser.parse_args()
    
    # 运行检查
    checker = ModelChecker()
    exit_code = checker.run()
    
    # 如果检查通过且指定了测试，则运行测试
    if exit_code == 0 and args.test:
        exit_code = test_models()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
