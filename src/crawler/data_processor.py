"""
数据清洗和结构化处理模块
处理爬取的年报数据，生成结构化数据集
"""
import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd


class DataProcessor:
    """年报数据处理器"""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    def process_single_report(self, parsed_data: Dict) -> Dict:
        """
        处理单份年报数据
        
        Args:
            parsed_data: PDF解析后的数据
            
        Returns:
            清洗后的结构化数据
        """
        processed = {
            "stock_code": parsed_data.get("basic_info", {}).get("stock_code", ""),
            "stock_name": parsed_data.get("basic_info", {}).get("stock_name", ""),
            "company_name": parsed_data.get("basic_info", {}).get("company_name", ""),
            "report_year": parsed_data.get("basic_info", {}).get("report_year", ""),
            "file_path": parsed_data.get("file_path", ""),
            "metadata": parsed_data.get("metadata", {}),
        }
        
        # 处理财务数据
        financial = parsed_data.get("financial_data", {})
        processed["financials"] = {
            "total_revenue": financial.get("total_revenue"),
            "net_profit": financial.get("net_profit"),
            "total_assets": financial.get("total_assets"),
            "net_assets": financial.get("net_assets"),
            "eps": financial.get("eps")
        }
        
        # 清洗文本内容
        full_text = parsed_data.get("full_text", "")
        processed["cleaned_text"] = self._clean_text(full_text)
        
        # 提取关键段落
        sections = parsed_data.get("sections", {})
        processed["key_sections"] = {
            k: self._clean_text(v) 
            for k, v in sections.items() 
            if v
        }
        
        # 生成摘要
        processed["summary"] = self._generate_summary(processed["cleaned_text"])
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """
        清洗文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除页码标记
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # 移除页眉页脚常见的数字和日期
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
        
        # 规范化标点
        text = text.replace('，', ',').replace('。', '.').replace('：', ':')
        
        return text.strip()
    
    def _generate_summary(self, text: str, max_length: int = 1000) -> str:
        """
        生成文本摘要
        
        Args:
            text: 原始文本
            max_length: 最大长度
            
        Returns:
            摘要文本
        """
        if not text:
            return ""
        
        # 提取前N个字符作为摘要
        summary = text[:max_length]
        
        # 尝试在句子边界截断
        last_period = summary.rfind('。')
        if last_period > max_length * 0.5:
            summary = summary[:last_period + 1]
        
        return summary
    
    def create_dataset(self, parsed_files: List[str]) -> pd.DataFrame:
        """
        从解析文件创建数据集
        
        Args:
            parsed_files: 解析后的JSON文件路径列表
            
        Returns:
            DataFrame格式的数据集
        """
        records = []
        
        for file_path in parsed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                
                processed = self.process_single_report(parsed_data)
                
                # 展平数据
                record = {
                    "stock_code": processed["stock_code"],
                    "stock_name": processed["stock_name"],
                    "company_name": processed["company_name"],
                    "report_year": processed["report_year"],
                    "file_path": processed["file_path"],
                    "pages": processed["metadata"].get("pages", 0),
                    "total_revenue": processed["financials"].get("total_revenue"),
                    "net_profit": processed["financials"].get("net_profit"),
                    "total_assets": processed["financials"].get("total_assets"),
                    "net_assets": processed["financials"].get("net_assets"),
                    "eps": processed["financials"].get("eps"),
                    "summary": processed["summary"],
                    "full_text": processed["cleaned_text"][:10000],  # 限制长度
                }
                
                records.append(record)
                
            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")
                continue
        
        df = pd.DataFrame(records)
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "annual_reports.csv"):
        """
        保存数据集
        
        Args:
            df: DataFrame数据集
            filename: 保存文件名
        """
        output_path = os.path.join(self.processed_dir, filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据集已保存: {output_path}")
        
        # 同时保存为JSON
        json_path = output_path.replace('.csv', '.json')
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"JSON格式已保存: {json_path}")
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        生成数据统计信息
        
        Args:
            df: DataFrame数据集
            
        Returns:
            统计信息字典
        """
        def convert_value(v):
            """转换numpy类型为Python原生类型"""
            if hasattr(v, 'item'):  # numpy类型
                return v.item()
            return v
        
        stats = {
            "total_reports": int(len(df)),
            "unique_companies": int(df["stock_code"].nunique()),
            "year_range": {
                "min": convert_value(df["report_year"].min()) if not df["report_year"].empty else None,
                "max": convert_value(df["report_year"].max()) if not df["report_year"].empty else None
            },
            "financial_stats": {}
        }
        
        # 财务数据统计
        for col in ["total_revenue", "net_profit", "total_assets", "eps"]:
            if col in df.columns:
                stats["financial_stats"][col] = {
                    "count": int(df[col].count()),
                    "mean": convert_value(df[col].mean()),
                    "median": convert_value(df[col].median()),
                    "min": convert_value(df[col].min()),
                    "max": convert_value(df[col].max())
                }
        
        # 年份分布
        year_dist = df["report_year"].value_counts().to_dict()
        stats["year_distribution"] = {str(k): int(v) for k, v in year_dist.items()}
        
        return stats
    
    def process_all_reports(self, parsed_dir: str = None):
        """处理所有已解析的年报"""
        # 查找所有解析文件
        if parsed_dir is None:
            parsed_dir = os.path.join(self.processed_dir, "parsed")
        
        parsed_files = []
        for root, dirs, files in os.walk(parsed_dir):
            for file in files:
                if file.endswith("_parsed.json"):
                    parsed_files.append(os.path.join(root, file))
        
        print(f"找到 {len(parsed_files)} 个解析文件")
        
        if not parsed_files:
            print("没有需要处理的文件")
            return
        
        # 创建数据集
        df = self.create_dataset(parsed_files)
        
        # 保存数据集
        self.save_dataset(df)
        
        # 生成统计信息
        stats = self.generate_statistics(df)
        
        # 保存统计信息
        stats_path = os.path.join(self.processed_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print("\n数据统计:")
        print(f"- 总报告数: {stats['total_reports']}")
        print(f"- 公司数量: {stats['unique_companies']}")
        print(f"- 年份范围: {stats['year_range']['min']} - {stats['year_range']['max']}")
        
        return df, stats


def main():
    """测试数据处理功能"""
    processor = DataProcessor()
    
    # 处理所有报告
    df, stats = processor.process_all_reports()
    
    if df is not None:
        print("\n数据集预览:")
        print(df.head())


if __name__ == "__main__":
    main()
