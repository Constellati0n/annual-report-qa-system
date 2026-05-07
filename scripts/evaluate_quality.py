#!/usr/bin/env python3
"""
数据质量评估与改进工具
评估 chunk 切分和训练集质量，提供改进建议
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))


class DataQualityEvaluator:
    """数据质量评估器"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.recommendations = []
    
    def evaluate_chunks(self, chunks_file: str = "processed/knowledge_base/docs_raw_chunks.json") -> Dict:
        """评估 chunk 质量"""
        print("=" * 60)
        print("评估 Chunk 切分质量")
        print("=" * 60)
        
        chunks_path = self.data_dir / chunks_file
        if not chunks_path.exists():
            print(f"✗ Chunk 文件不存在: {chunks_path}")
            return {}
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"\n总 Chunk 数量: {len(chunks)}")
        
        # 1. 长度分布分析
        lengths = [len(chunk['content']) for chunk in chunks]
        avg_length = statistics.mean(lengths)
        median_length = statistics.median(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        print(f"\n长度分布:")
        print(f"  平均长度: {avg_length:.0f} 字符")
        print(f"  中位数: {median_length:.0f} 字符")
        print(f"  最小: {min_length} 字符")
        print(f"  最大: {max_length} 字符")
        
        # 2. 过短/过长 chunk 检测
        too_short = [c for c in chunks if len(c['content']) < 50]
        too_long = [c for c in chunks if len(c['content']) > 2000]
        
        if too_short:
            print(f"\n⚠️  过短 chunks (< 50字符): {len(too_short)} 个")
            self.issues.append(f"发现 {len(too_short)} 个过短 chunk")
            print(f"  示例: {too_short[0]['content'][:100]}...")
        
        if too_long:
            print(f"\n⚠️  过长 chunks (> 2000字符): {len(too_long)} 个")
            self.issues.append(f"发现 {len(too_long)} 个过长 chunk")
        
        # 3. 内容质量检查
        empty_or_whitespace = [c for c in chunks if not c['content'].strip()]
        if empty_or_whitespace:
            print(f"\n✗ 空内容 chunks: {len(empty_or_whitespace)} 个")
            self.issues.append(f"发现 {len(empty_or_whitespace)} 个空内容 chunk")
        
        # 4. 重复内容检测
        contents = [c['content'][:200] for c in chunks]  # 比较前200字符
        duplicates = [item for item, count in Counter(contents).items() if count > 1]
        if duplicates:
            print(f"\n⚠️  疑似重复 chunks: {len(duplicates)} 组")
            self.issues.append(f"发现 {len(duplicates)} 组重复内容")
        
        # 5. 元数据完整性检查
        incomplete_metadata = []
        for chunk in chunks:
            meta = chunk.get('metadata', {})
            if not meta.get('stock_code') or not meta.get('stock_name'):
                incomplete_metadata.append(chunk)
        
        if incomplete_metadata:
            print(f"\n⚠️  元数据不完整: {len(incomplete_metadata)} 个 chunks")
            print(f"  示例: {incomplete_metadata[0].get('metadata', {})}")
            self.issues.append(f"发现 {len(incomplete_metadata)} 个元数据不完整的 chunk")
        
        # 6. 章节分布分析
        sections = Counter([c.get('metadata', {}).get('section', 'unknown') for c in chunks])
        print(f"\n章节分布 (Top 10):")
        for section, count in sections.most_common(10):
            print(f"  {section}: {count} 个 chunks")
        
        return {
            "total_chunks": len(chunks),
            "avg_length": avg_length,
            "median_length": median_length,
            "too_short": len(too_short),
            "too_long": len(too_long),
            "duplicates": len(duplicates),
            "incomplete_metadata": len(incomplete_metadata),
            "section_distribution": dict(sections.most_common())
        }
    
    def evaluate_training_data(self, train_file: str = "training/train.json") -> Dict:
        """评估训练数据质量"""
        print("\n" + "=" * 60)
        print("评估训练数据质量")
        print("=" * 60)
        
        train_path = self.data_dir / train_file
        if not train_path.exists():
            print(f"✗ 训练文件不存在: {train_path}")
            return {}
        
        with open(train_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"\n总样本数: {len(training_data)}")
        
        # 1. 数据格式检查
        required_fields = ['instruction', 'input', 'output']
        format_issues = []
        for i, sample in enumerate(training_data):
            for field in required_fields:
                if field not in sample:
                    format_issues.append(f"样本 {i} 缺少字段: {field}")
        
        if format_issues:
            print(f"\n✗ 格式问题: {len(format_issues)} 个")
            for issue in format_issues[:5]:
                print(f"  - {issue}")
            self.issues.extend(format_issues[:10])
        else:
            print(f"\n✓ 所有样本格式正确")
        
        # 2. 内容质量检查
        empty_outputs = [s for s in training_data if not s.get('output', '').strip()]
        empty_instructions = [s for s in training_data if not s.get('instruction', '').strip()]
        
        if empty_outputs:
            print(f"\n✗ 空 output: {len(empty_outputs)} 个样本")
            self.issues.append(f"发现 {len(empty_outputs)} 个空 output")
        
        if empty_instructions:
            print(f"\n✗ 空 instruction: {len(empty_instructions)} 个样本")
            self.issues.append(f"发现 {len(empty_instructions)} 个空 instruction")
        
        # 3. 输出长度分析
        output_lengths = [len(s.get('output', '')) for s in training_data]
        if output_lengths:
            avg_output = statistics.mean(output_lengths)
            print(f"\n输出长度统计:")
            print(f"  平均: {avg_output:.0f} 字符")
            print(f"  中位数: {statistics.median(output_lengths):.0f} 字符")
            
            # 过短输出
            short_outputs = [s for s in training_data if len(s.get('output', '')) < 20]
            if short_outputs:
                print(f"\n⚠️  过短输出 (< 20字符): {len(short_outputs)} 个")
                self.issues.append(f"发现 {len(short_outputs)} 个过短 output")
        
        # 4. 类别分布
        categories = Counter([s.get('category', 'unknown') for s in training_data])
        print(f"\n类别分布:")
        for cat, count in categories.most_common():
            print(f"  {cat}: {count} 个")
        
        # 5. 数据多样性检查
        instructions = [s.get('instruction', '') for s in training_data]
        unique_instructions = set(instructions)
        print(f"\n指令多样性:")
        print(f"  唯一指令数: {len(unique_instructions)}")
        print(f"  重复指令: {len(instructions) - len(unique_instructions)}")
        
        if len(unique_instructions) < len(instructions) * 0.5:
            self.issues.append("指令多样性不足，存在大量重复模板")
        
        # 6. 数据质量问题示例
        print(f"\n质量问题示例:")
        problematic = [s for s in training_data 
                      if 'None' in str(s.get('output', '')) 
                      or '...' in str(s.get('output', ''))
                      or len(s.get('output', '')) < 10]
        
        if problematic:
            print(f"  发现 {len(problematic)} 个有问题的样本")
            for i, sample in enumerate(problematic[:3]):
                print(f"\n  问题样本 {i+1}:")
                print(f"    Instruction: {sample.get('instruction', '')[:50]}...")
                print(f"    Output: {sample.get('output', '')[:100]}...")
        
        return {
            "total_samples": len(training_data),
            "format_issues": len(format_issues),
            "empty_outputs": len(empty_outputs),
            "empty_instructions": len(empty_instructions),
            "short_outputs": len(short_outputs) if 'short_outputs' in locals() else 0,
            "category_distribution": dict(categories),
            "unique_instructions": len(unique_instructions),
            "problematic_samples": len(problematic)
        }
    
    def generate_recommendations(self):
        """生成改进建议"""
        print("\n" + "=" * 60)
        print("改进建议")
        print("=" * 60)
        
        recommendations = []
        
        # Chunk 切分建议
        recommendations.append({
            "category": "Chunk 切分",
            "issues": [
                "当前切分存在大量元数据不完整（缺少 stock_code, stock_name）",
                "部分 chunk 过短（< 50字符），可能是页眉页脚",
                "切分策略使用简单的固定长度，未充分利用年报结构"
            ],
            "solutions": [
                "1. 修复元数据提取逻辑，从文件名或内容中解析股票代码和名称",
                "2. 过滤掉过短的 chunks（< 100字符），这些通常是页眉页脚",
                "3. 使用 AnnualReportChunker 替代简单切分，按章节结构化切分",
                "4. 设置 chunk 长度范围: min=200, max=1500 字符",
                "5. 增加语义完整性检查，确保每个 chunk 包含完整信息"
            ]
        })
        
        # 训练数据建议
        recommendations.append({
            "category": "训练数据",
            "issues": [
                "训练样本中存在空值（None）和不完整数据",
                "输出内容过于简单，缺乏详细分析",
                "指令模板化严重，多样性不足",
                "缺少数据来源标注，难以追溯"
            ],
            "solutions": [
                "1. 清洗数据：移除包含 None 的样本，修复空值",
                "2. 增强输出：使用 LLM 生成更详细的分析回答",
                "3. 丰富指令：增加更多样化的问题类型（对比、趋势、预测等）",
                "4. 添加数据验证：确保所有数值都有单位（万元、亿元等）",
                "5. 增加负样本：添加一些无法回答的问题及其合理回复",
                "6. 平衡类别：确保各类型问题数量均衡"
            ]
        })
        
        # 数据处理流程建议
        recommendations.append({
            "category": "数据处理流程",
            "issues": [
                "PDF 提取文本时丢失格式信息",
                "表格数据转换为文本后结构混乱",
                "财务数据提取不完整"
            ],
            "solutions": [
                "1. 使用更专业的 PDF 解析工具（如 marker、 unstructured）",
                "2. 对表格数据单独处理，保留结构化信息",
                "3. 建立财务数据标准化提取流程",
                "4. 添加数据质量检查节点，过滤低质量数据",
                "5. 建立数据版本管理，便于追踪和回滚"
            ]
        })
        
        for rec in recommendations:
            print(f"\n【{rec['category']}】")
            print("\n存在的问题:")
            for issue in rec['issues']:
                print(f"  - {issue}")
            print("\n改进方案:")
            for solution in rec['solutions']:
                print(f"  {solution}")
        
        return recommendations
    
    def run(self):
        """运行完整评估"""
        print("\n" + "=" * 60)
        print("数据质量评估报告")
        print("=" * 60)
        
        # 评估 chunks
        chunk_stats = self.evaluate_chunks()
        
        # 评估训练数据
        train_stats = self.evaluate_training_data()
        
        # 生成建议
        recommendations = self.generate_recommendations()
        
        # 总结
        print("\n" + "=" * 60)
        print("评估总结")
        print("=" * 60)
        print(f"\n发现问题: {len(self.issues)} 个")
        if self.issues:
            print("\n主要问题:")
            for issue in self.issues[:10]:
                print(f"  - {issue}")
        
        # 保存报告
        report = {
            "chunk_stats": chunk_stats,
            "training_stats": train_stats,
            "issues": self.issues,
            "recommendations": recommendations
        }
        
        report_path = self.data_dir / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 详细报告已保存: {report_path}")
        
        return report


class DataCleaner:
    """数据清洗工具"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
    
    def clean_chunks(self, input_file: str = "processed/knowledge_base/docs_raw_chunks.json",
                     output_file: str = "processed/knowledge_base/docs_cleaned.json"):
        """清洗 chunks"""
        print("\n" + "=" * 60)
        print("清洗 Chunk 数据")
        print("=" * 60)
        
        input_path = self.data_dir / input_file
        output_path = self.data_dir / output_file
        
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        original_count = len(chunks)
        print(f"原始 chunks: {original_count}")
        
        # 清洗规则
        cleaned = []
        removed_reasons = Counter()
        
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            
            # 1. 移除过短的
            if len(content) < 100:
                removed_reasons['too_short'] += 1
                continue
            
            # 2. 移除过长的
            if len(content) > 2000:
                removed_reasons['too_long'] += 1
                # 可以在这里进行进一步切分
                continue
            
            # 3. 移除页眉页脚（包含大量点号）
            if content.count('.') > len(content) * 0.3:
                removed_reasons['likely_header_footer'] += 1
                continue
            
            # 4. 移除纯目录内容
            if '................................' in content and len(content) < 300:
                removed_reasons['table_of_contents'] += 1
                continue
            
            # 5. 清理内容
            content = self._clean_content(content)
            
            # 更新 chunk
            chunk['content'] = content
            cleaned.append(chunk)
        
        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
        print(f"\n清洗后 chunks: {len(cleaned)}")
        print(f"移除: {original_count - len(cleaned)}")
        print("\n移除原因:")
        for reason, count in removed_reasons.most_common():
            print(f"  {reason}: {count}")
        print(f"\n✓ 已保存: {output_path}")
        
        return cleaned
    
    def _clean_content(self, content: str) -> str:
        """清理内容"""
        # 移除多余的换行
        content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
        
        # 移除页码标记
        import re
        content = re.sub(r'--- Page \d+ ---', '', content)
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)  # 孤立数字（页码）
        
        # 移除多余的点号
        content = re.sub(r'\.{3,}', ' ', content)
        
        # 规范化空白
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def clean_training_data(self, input_file: str = "training/train.json",
                           output_file: str = "training/train_cleaned.json"):
        """清洗训练数据"""
        print("\n" + "=" * 60)
        print("清洗训练数据")
        print("=" * 60)
        
        input_path = self.data_dir / input_file
        output_path = self.data_dir / output_file
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)
        print(f"原始样本: {original_count}")
        
        cleaned = []
        removed_reasons = Counter()
        
        for sample in data:
            instruction = sample.get('instruction', '').strip()
            output = sample.get('output', '').strip()
            
            # 1. 移除空值
            if not instruction or not output:
                removed_reasons['empty_content'] += 1
                continue
            
            # 2. 移除包含 None 的
            if 'None' in output or 'None' in instruction:
                removed_reasons['contains_none'] += 1
                # 尝试修复
                output = output.replace('None', 'N/A')
                sample['output'] = output
            
            # 3. 移除过短的输出
            if len(output) < 10:
                removed_reasons['too_short_output'] += 1
                continue
            
            # 4. 清理输出
            output = self._clean_content(output)
            sample['output'] = output
            
            cleaned.append(sample)
        
        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
        print(f"\n清洗后样本: {len(cleaned)}")
        print(f"移除: {original_count - len(cleaned)}")
        print("\n移除原因:")
        for reason, count in removed_reasons.most_common():
            print(f"  {reason}: {count}")
        print(f"\n✓ 已保存: {output_path}")
        
        return cleaned


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据质量评估与清洗工具")
    parser.add_argument("--evaluate", action="store_true",
                       help="运行质量评估")
    parser.add_argument("--clean", action="store_true",
                       help="运行数据清洗")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="数据目录")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluator = DataQualityEvaluator(args.data_dir)
        evaluator.run()
    
    if args.clean:
        cleaner = DataCleaner(args.data_dir)
        cleaner.clean_chunks()
        cleaner.clean_training_data()
    
    if not args.evaluate and not args.clean:
        # 默认运行评估和清洗
        print("运行完整流程（评估 + 清洗）...")
        evaluator = DataQualityEvaluator(args.data_dir)
        evaluator.run()
        
        cleaner = DataCleaner(args.data_dir)
        cleaner.clean_chunks()
        cleaner.clean_training_data()


if __name__ == "__main__":
    main()
