"""
构建向量数据库流程
1. 加载解析后的年报数据
2. 切分文本为chunks
3. 生成向量嵌入
4. 存储到向量数据库
"""
import os
import json
import glob
from typing import List, Dict
from pathlib import Path

from chunker import chunk_documents, Chunk


class VectorDBBuilder:
    """向量数据库构建器"""
    
    def __init__(
        self,
        parsed_dir: str = None,
        chunk_dir: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        self.parsed_dir = parsed_dir or str(project_root / "data" / "processed" / "parsed")
        self.chunk_dir = chunk_dir or str(project_root / "data" / "processed" / "chunks")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 确保目录存在
        Path(self.chunk_dir).mkdir(parents=True, exist_ok=True)
    
    def load_parsed_reports(self) -> List[Dict]:
        """加载所有解析后的年报数据"""
        documents = []
        
        json_files = glob.glob(os.path.join(self.parsed_dir, "*_parsed.json"))
        print(f"找到 {len(json_files)} 个解析文件")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取文本和元数据
                text = data.get('full_text', '')
                basic_info = data.get('basic_info', {})
                financial_data = data.get('financial_data', {})
                
                metadata = {
                    'stock_code': basic_info.get('stock_code', ''),
                    'stock_name': basic_info.get('stock_name', ''),
                    'company_name': basic_info.get('company_name', ''),
                    'report_year': basic_info.get('report_year', ''),
                    'file_path': data.get('file_path', ''),
                    'pages': data.get('metadata', {}).get('pages', 0),
                    'total_revenue': financial_data.get('total_revenue'),
                    'net_profit': financial_data.get('net_profit'),
                    'total_assets': financial_data.get('total_assets'),
                    'eps': financial_data.get('eps')
                }
                
                if text:
                    documents.append({
                        'content': text,
                        'metadata': metadata
                    })
                    
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        return documents
    
    def create_chunks(self, documents: List[Dict]) -> List[Chunk]:
        """将文档切分为chunks"""
        print(f"\n开始切分 {len(documents)} 个文档...")
        
        chunks = chunk_documents(
            documents=documents,
            chunker_type="annual_report",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        print(f"切分完成，共生成 {len(chunks)} 个chunks")
        return chunks
    
    def save_chunks(self, chunks: List[Chunk]):
        """保存chunks到文件"""
        chunks_data = []
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'index': i
            }
            chunks_data.append(chunk_data)
        
        # 保存为JSON
        output_path = os.path.join(self.chunk_dir, "chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"Chunks已保存: {output_path}")
        
        # 生成统计信息
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            'section_distribution': {}
        }
        
        for chunk in chunks:
            section = chunk.metadata.get('section', 'unknown')
            stats['section_distribution'][section] = stats['section_distribution'].get(section, 0) + 1
        
        stats_path = os.path.join(self.chunk_dir, "chunk_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存: {stats_path}")
        
        return stats
    
    def build(self):
        """执行完整的构建流程"""
        print("=" * 60)
        print("构建向量数据库 - 文本切分阶段")
        print("=" * 60)
        
        # 1. 加载解析后的年报
        print("\n【步骤1】加载解析后的年报数据")
        documents = self.load_parsed_reports()
        
        if not documents:
            print("没有可用的文档")
            return
        
        # 2. 切分文本
        print("\n【步骤2】切分文本为chunks")
        chunks = self.create_chunks(documents)
        
        # 3. 保存chunks
        print("\n【步骤3】保存chunks")
        stats = self.save_chunks(chunks)
        
        print("\n" + "=" * 60)
        print("文本切分完成！")
        print("=" * 60)
        print(f"\n统计信息:")
        print(f"- 总chunks数: {stats['total_chunks']}")
        print(f"- 平均chunk大小: {stats['avg_chunk_size']:.0f} 字符")
        print(f"- 章节分布:")
        for section, count in sorted(stats['section_distribution'].items(), key=lambda x: -x[1])[:10]:
            print(f"  - {section}: {count}")
        
        return chunks


def main():
    """主函数"""
    builder = VectorDBBuilder(
        chunk_size=1000,      # 每个chunk约1000字符
        chunk_overlap=100     # 重叠100字符
    )
    
    chunks = builder.build()
    
    if chunks:
        print("\n示例chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Section: {chunk.metadata.get('section', 'unknown')}")
            print(f"  Content: {chunk.content[:150]}...")


if __name__ == "__main__":
    main()
