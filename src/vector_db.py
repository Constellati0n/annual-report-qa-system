import os
import re
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

class VectorDB:
    def __init__(self):
        """初始化向量数据库"""
        # 初始化数据处理器
        self.data_processor = DataProcessor()
        
        # 内存存储
        self.collections = {}
    
    def create_collection(self, collection_name='financials'):
        """创建或获取集合"""
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                'documents': [],
                'metadatas': [],
                'ids': []
            }
        return self.collections[collection_name]
    
    def add_documents(self, ticker, collection_name='financials'):
        """向集合中添加文档"""
        # 加载处理后的数据
        data = self.data_processor.load_data(ticker)
        if not data:
            print(f"未找到{ ticker }的数据")
            return False
        
        # 创建或获取集合
        collection = self.create_collection(collection_name)
        
        # 生成嵌入文本
        texts = self.data_processor.generate_embedding_texts(data)
        
        # 准备文档ID和元数据
        ids = [f"{ticker}_{i}" for i in range(len(texts))]
        metadatas = []
        
        for i, item in enumerate(data):
            metadata = {
                'ticker': ticker,
                'type': item.get('type', ''),
                'metric': item.get('metric', ''),
                'value': item.get('value', ''),
            }
            if 'year' in item:
                metadata['year'] = item['year']
            metadatas.append(metadata)
        
        # 添加文档到集合
        collection['documents'].extend(texts)
        collection['metadatas'].extend(metadatas)
        collection['ids'].extend(ids)
        
        print(f"成功添加了{len(texts)}条文档到集合{collection_name}")
        return True
    
    def _calculate_similarity(self, text1, text2):
        """简单的文本相似度计算"""
        # 转换为小写
        text1 = text1.lower()
        text2 = text2.lower()
        
        # 提取数字
        numbers1 = re.findall(r'\d+', text1)
        numbers2 = re.findall(r'\d+', text2)
        
        # 提取关键词
        keywords1 = set(re.findall(r'\b\w+\b', text1))
        keywords2 = set(re.findall(r'\b\w+\b', text2))
        
        # 计算关键词相似度
        if keywords1 or keywords2:
            keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_similarity = 0
        
        # 计算数字相似度（如果有数字）
        number_similarity = 0
        if numbers1 and numbers2:
            # 检查是否有相同的数字
            common_numbers = set(numbers1) & set(numbers2)
            if common_numbers:
                number_similarity = 1
        
        # 综合相似度
        total_similarity = 0.7 * keyword_similarity + 0.3 * number_similarity
        return total_similarity
    
    def query(self, query_text, collection_name='financials', n_results=5, where=None):
        """查询向量数据库"""
        # 获取集合
        if collection_name not in self.collections:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }
        
        collection = self.collections[collection_name]
        
        # 计算相似度
        similarities = []
        for i, doc in enumerate(collection['documents']):
            similarity = self._calculate_similarity(query_text, doc)
            similarities.append((i, similarity))
        
        # 排序并获取前n个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in similarities[:n_results]]
        
        # 构建结果
        results = {
            'documents': [[collection['documents'][i] for i in top_indices]],
            'metadatas': [[collection['metadatas'][i] for i in top_indices]],
            'distances': [[1 - similarities[i][1] for i in range(len(top_indices))]],
            'ids': [[collection['ids'][i] for i in top_indices]]
        }
        
        return results
    
    def list_collections(self):
        """列出所有集合"""
        return list(self.collections.keys())
    
    def delete_collection(self, collection_name):
        """删除集合"""
        if collection_name in self.collections:
            del self.collections[collection_name]
            print(f"集合{collection_name}已删除")
    
    def get_collection_stats(self, collection_name='financials'):
        """获取集合统计信息"""
        if collection_name in self.collections:
            return len(self.collections[collection_name]['documents'])
        return 0

if __name__ == "__main__":
    # 测试向量数据库
    vdb = VectorDB()
    
    # 添加文档
    ticker = "AAPL"
    vdb.add_documents(ticker)
    
    # 测试查询
    query = "苹果公司2023年的总收入是多少？"
    results = vdb.query(query)
    
    print(f"查询结果 (top {len(results['documents'][0])}):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"\n结果 {i+1} (距离: {dist:.4f}):")
        print(f"文档: {doc}")
        print(f"元数据: {meta}")
