"""
文本Chunk切分模块
支持多种切分策略：按字符、按token、按语义、按Markdown结构
"""
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class Chunk:
    """文本块数据结构"""
    content: str
    metadata: Dict
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


class TextChunker:
    """文本切分器基类"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        切分文本
        
        Args:
            text: 原始文本
            metadata: 元数据
            
        Returns:
            Chunk列表
        """
        raise NotImplementedError
    
    def _merge_chunks(self, chunks: List[str], separator: str = "") -> List[str]:
        """合并小chunks达到目标大小"""
        merged = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) + len(separator) <= self.chunk_size:
                current_chunk = separator.join([current_chunk, chunk]) if current_chunk else chunk
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged


class RecursiveCharacterChunker(TextChunker):
    """
    递归字符切分器
    按优先级尝试不同的分隔符进行切分
    """
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """递归切分文本"""
        if metadata is None:
            metadata = {}
        
        chunks = self._recursive_split(text, self.separators)
        
        # 添加重叠
        chunks_with_overlap = self._add_overlap(chunks)
        
        # 创建Chunk对象
        return [
            Chunk(content=chunk, metadata=metadata.copy())
            for chunk in chunks_with_overlap
        ]
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分"""
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        if not separators:
            # 没有分隔符时直接切分
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 使用当前分隔符切分
        parts = text.split(separator)
        
        # 递归处理每个部分
        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining_separators))
        
        # 合并小chunks
        return self._merge_chunks(result, separator)
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠部分"""
        if not chunks or self.chunk_overlap <= 0:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 从前一个chunk末尾取重叠部分
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # 合并到当前chunk
            new_chunk = overlap_text + current_chunk
            result.append(new_chunk)
        
        return result


class MarkdownChunker(TextChunker):
    """
    Markdown文档切分器
    按Markdown结构（标题、段落、代码块等）切分
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        # Markdown标题正则
        self.header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
        # 代码块正则
        self.code_block_pattern = re.compile(r'```[\s\S]*?```')
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """按Markdown结构切分"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        current_section = {"title": "", "content": "", "level": 0}
        
        # 按行处理
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检测标题
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # 保存上一个section
                if current_section["content"]:
                    chunks.extend(self._create_chunks_from_section(current_section, metadata))
                
                # 开始新section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    "title": title,
                    "content": "",
                    "level": level
                }
            else:
                current_section["content"] += line + "\n"
            
            i += 1
        
        # 保存最后一个section
        if current_section["content"]:
            chunks.extend(self._create_chunks_from_section(current_section, metadata))
        
        return chunks
    
    def _create_chunks_from_section(self, section: Dict, base_metadata: Dict) -> List[Chunk]:
        """从section创建chunks"""
        content = section["content"].strip()
        if not content:
            return []
        
        metadata = {
            **base_metadata,
            "section_title": section["title"],
            "section_level": section["level"]
        }
        
        # 如果内容太长，进一步切分
        if len(content) > self.chunk_size:
            sub_chunker = RecursiveCharacterChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return sub_chunker.split_text(content, metadata)
        
        return [Chunk(content=content, metadata=metadata)]


class AnnualReportChunker(TextChunker):
    """
    年报专用切分器
    针对年报结构进行优化切分
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        
        # 年报章节关键词
        self.section_keywords = {
            "重要提示": ["重要提示", "董事会声明"],
            "公司简介": ["公司简介", "公司基本情况", "法定中文名称"],
            "财务数据": ["主要会计数据", "主要财务指标", "会计数据和财务指标"],
            "经营情况": ["管理层讨论", "经营情况讨论", "公司业务概要"],
            "重要事项": ["重要事项", "重大事项"],
            "股份变动": ["股份变动", "股本变动"],
            "股东情况": ["股东情况", "前十名股东"],
            "董事监事": ["董事、监事", "高级管理人员", "董监高"],
            "公司治理": ["公司治理", "股东大会", "董事会"],
            "财务报告": ["财务报告", "审计报告", "财务报表"],
            "资产负债表": ["资产负债表", "合并资产负债表"],
            "利润表": ["利润表", "合并利润表", "合并及公司利润表"],
            "现金流量": ["现金流量表", "合并现金流量表"]
        }
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """按年报结构切分"""
        if metadata is None:
            metadata = {}
        
        # 首先尝试识别章节
        sections = self._identify_sections(text)
        
        chunks = []
        for section_name, section_content in sections:
            section_metadata = {
                **metadata,
                "section": section_name
            }
            
            # 如果章节内容太长，进一步切分
            if len(section_content) > self.chunk_size:
                sub_chunker = RecursiveCharacterChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                section_chunks = sub_chunker.split_text(section_content, section_metadata)
                chunks.extend(section_chunks)
            else:
                chunks.append(Chunk(content=section_content, metadata=section_metadata))
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """识别年报章节"""
        sections = []
        
        # 构建章节分隔模式
        patterns = []
        for section_name, keywords in self.section_keywords.items():
            for keyword in keywords:
                # 匹配章节标题（支持多种格式）
                pattern = rf'(?:^|\n)\s*(?:第[一二三四五六七八九十]+节|第\d+节|[\d\.]+)?\s*{keyword}[\s]*[\n]'
                patterns.append((section_name, pattern, keyword))
        
        # 按关键词在文本中出现的顺序排序
        section_positions = []
        for section_name, pattern, keyword in patterns:
            for match in re.finditer(pattern, text):
                section_positions.append((match.start(), section_name, keyword))
        
        section_positions.sort(key=lambda x: x[0])
        
        # 提取章节内容
        if not section_positions:
            # 未识别到章节，整体作为一个chunk
            return [("全文", text)]
        
        for i, (pos, section_name, keyword) in enumerate(section_positions):
            start = pos
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)
            
            content = text[start:end].strip()
            if content:
                sections.append((section_name, content))
        
        return sections


class SemanticChunker(TextChunker):
    """
    语义切分器（简化版）
    基于句子边界和语义完整性切分
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        # 句子结束符
        self.sentence_endings = ['。', '！', '？', '.', '!', '?']
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """按语义切分"""
        if metadata is None:
            metadata = {}
        
        # 先切分成句子
        sentences = self._split_to_sentences(text)
        
        # 合并句子成chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 保存当前chunk
                if current_chunk:
                    chunk_text = ''.join(current_chunk)
                    chunks.append(Chunk(content=chunk_text, metadata=metadata.copy()))
                
                # 开始新chunk，保留部分重叠
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        
        # 保存最后一个chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append(Chunk(content=chunk_text, metadata=metadata.copy()))
        
        return chunks
    
    def _split_to_sentences(self, text: str) -> List[str]:
        """切分成句子"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in self.sentence_endings:
                sentences.append(current)
                current = ""
        
        if current:
            sentences.append(current)
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """获取重叠的句子"""
        overlap = []
        total_length = 0
        
        for sentence in reversed(sentences):
            if total_length + len(sentence) <= self.chunk_overlap:
                overlap.insert(0, sentence)
                total_length += len(sentence)
            else:
                break
        
        return overlap


def get_chunker(chunker_type: str = "recursive", **kwargs) -> TextChunker:
    """
    获取切分器实例
    
    Args:
        chunker_type: 切分器类型
        **kwargs: 配置参数
        
    Returns:
        TextChunker实例
    """
    chunkers = {
        "recursive": RecursiveCharacterChunker,
        "markdown": MarkdownChunker,
        "annual_report": AnnualReportChunker,
        "semantic": SemanticChunker
    }
    
    chunker_class = chunkers.get(chunker_type, RecursiveCharacterChunker)
    return chunker_class(**kwargs)


def chunk_documents(
    documents: List[Dict],
    chunker_type: str = "annual_report",
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Chunk]:
    """
    批量切分文档
    
    Args:
        documents: 文档列表 [{"content": "", "metadata": {}}]
        chunker_type: 切分器类型
        chunk_size: chunk大小
        chunk_overlap: 重叠大小
        
    Returns:
        Chunk列表
    """
    chunker = get_chunker(chunker_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_chunks = []
    for doc in documents:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        chunks = chunker.split_text(content, metadata)
        all_chunks.extend(chunks)
    
    return all_chunks


# 测试代码
if __name__ == "__main__":
    # 测试递归字符切分
    text = """这是第一段。这是第二段，包含多个句子。这是第三段。

这是新的一节。这里有一些内容。这里是更多内容。

最后一节的内容在这里。"""
    
    print("=== 递归字符切分 ===")
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.split_text(text, {"source": "test"})
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.content[:50]}...")
    
    # 测试年报切分器
    print("\n=== 年报切分器 ===")
    annual_text = """
第一节 重要提示
本公司董事会保证本报告所载资料不存在任何虚假记载。

第二节 公司简介
平安银行股份有限公司成立于1987年。

第三节 财务数据
2023年营业收入1000亿元。
"""
    
    annual_chunker = AnnualReportChunker(chunk_size=100, chunk_overlap=20)
    annual_chunks = annual_chunker.split_text(annual_text, {"stock": "000001"})
    for i, chunk in enumerate(annual_chunks):
        print(f"Chunk {i+1} [{chunk.metadata.get('section', 'unknown')}]: {chunk.content[:50]}...")
