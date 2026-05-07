import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_stock_data(self, ticker, period='1y'):
        """获取股票基本数据"""
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    
    def get_financials(self, ticker, max_retries=3, retry_delay=2):
        """获取财务报表数据，包含错误处理和重试机制"""
        import time
        import requests
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # 获取损益表
                income_stmt = stock.financials
                
                # 获取资产负债表
                balance_sheet = stock.balance_sheet
                
                # 获取现金流量表
                cash_flow = stock.cashflow
                
                # 获取公司基本信息
                info = stock.info
                
                return {
                    'income_stmt': income_stmt,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'info': info
                }
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"请求过于频繁，正在重试... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    raise
            except Exception as e:
                print(f"获取数据时出错: {e}")
                raise
        
        # 如果所有重试都失败，返回空数据
        return {
            'income_stmt': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'info': {}
        }
    
    def process_financials(self, financials):
        """处理财务报表数据，转换为适合RAG的格式"""
        processed_data = []
        
        # 处理损益表
        if not financials['income_stmt'].empty:
            for col in financials['income_stmt'].columns:
                year = col.year
                for index, value in financials['income_stmt'][col].items():
                    if pd.notna(value):
                        processed_data.append({
                            'type': 'income_statement',
                            'year': year,
                            'metric': str(index),
                            'value': float(value),
                            'description': f"{year}年{index}为{value:.2f}"
                        })
        
        # 处理资产负债表
        if not financials['balance_sheet'].empty:
            for col in financials['balance_sheet'].columns:
                year = col.year
                for index, value in financials['balance_sheet'][col].items():
                    if pd.notna(value):
                        processed_data.append({
                            'type': 'balance_sheet',
                            'year': year,
                            'metric': str(index),
                            'value': float(value),
                            'description': f"{year}年{index}为{value:.2f}"
                        })
        
        # 处理现金流量表
        if not financials['cash_flow'].empty:
            for col in financials['cash_flow'].columns:
                year = col.year
                for index, value in financials['cash_flow'][col].items():
                    if pd.notna(value):
                        processed_data.append({
                            'type': 'cash_flow',
                            'year': year,
                            'metric': str(index),
                            'value': float(value),
                            'description': f"{year}年{index}为{value:.2f}"
                        })
        
        # 处理公司基本信息
        if financials['info']:
            info_items = [
                ('companyName', '公司名称'),
                ('sector', '行业'),
                ('industry', '产业'),
                ('country', '国家'),
                ('marketCap', '市值'),
                ('beta', '贝塔系数'),
                ('forwardPE', '预期市盈率'),
                ('trailingPE', ' trailing市盈率'),
                ('dividendYield', '股息收益率'),
                ('fiftyTwoWeekHigh', '52周最高价'),
                ('fiftyTwoWeekLow', '52周最低价')
            ]
            
            for key, desc in info_items:
                if key in financials['info'] and financials['info'][key] is not None:
                    processed_data.append({
                        'type': 'company_info',
                        'metric': desc,
                        'value': financials['info'][key],
                        'description': f"{desc}为{financials['info'][key]}"
                    })
        
        return processed_data
    
    def save_data(self, data, ticker):
        """保存处理后的数据"""
        file_path = os.path.join(self.data_dir, f"{ticker}_financials.json")
        df = pd.DataFrame(data)
        df.to_json(file_path, orient='records', force_ascii=False)
        return file_path
    
    def load_data(self, ticker):
        """加载处理后的数据"""
        file_path = os.path.join(self.data_dir, f"{ticker}_financials.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records')
            return df.to_dict('records')
        return []
    
    def generate_embedding_texts(self, data):
        """生成用于嵌入的文本"""
        texts = []
        for item in data:
            if 'description' in item:
                texts.append(item['description'])
            else:
                if 'year' in item:
                    text = f"{item['year']}年{item['type']}中的{item['metric']}为{item['value']}"
                else:
                    text = f"{item['type']}中的{item['metric']}为{item['value']}"
                texts.append(text)
        return texts

if __name__ == "__main__":
    # 测试数据处理和加载
    processor = DataProcessor()
    ticker = "AAPL"
    
    print(f"加载{ ticker }的模拟财务数据...")
    processed_data = processor.load_data(ticker)
    
    if processed_data:
        print(f"成功加载了{len(processed_data)}条数据")
        
        print("生成嵌入文本...")
        texts = processor.generate_embedding_texts(processed_data)
        print(f"生成了{len(texts)}条嵌入文本")
        print("前5条文本:")
        for text in texts[:5]:
            print(f"- {text}")
    else:
        print("未找到模拟数据，尝试从API获取...")
        financials = processor.get_financials(ticker)
        processed_data = processor.process_financials(financials)
        
        if processed_data:
            print(f"处理了{len(processed_data)}条数据")
            file_path = processor.save_data(processed_data, ticker)
            print(f"数据保存到: {file_path}")
        else:
            print("无法获取数据")
