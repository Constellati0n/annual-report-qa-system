"""
大规模年报爬虫
使用预定义股票代码范围进行批量爬取
"""
import os
import json
import time
import random
from typing import List, Dict, Set, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

from cninfo_crawler import CNInfoCrawler
from config import CrawlerConfig
from utils import random_delay

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mass_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MassCrawler:
    """大规模爬虫"""
    
    def __init__(
        self,
        max_workers: int = 5,
        request_delay: float = 1.0,
        max_retries: int = 3
    ):
        self.config = CrawlerConfig()
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.max_retries = max_retries
        
        # 进度文件路径
        self.progress_file = Path(self.config.DATA_DIR) / "mass_crawler_progress.json"
        
        # 已完成的股票集合
        self.completed_stocks: Set[str] = set()
        self.failed_stocks: Dict[str, str] = {}
        
        # 线程锁
        self.lock = Lock()
        
        # 统计信息
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "total_reports": 0
        }
        
        # 加载进度
        self._load_progress()
    
    def _load_progress(self):
        """加载爬取进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.completed_stocks = set(progress.get('completed', []))
                    self.failed_stocks = progress.get('failed', {})
                    self.stats = progress.get('stats', self.stats)
                logger.info(f"已加载进度: {len(self.completed_stocks)} 个完成, {len(self.failed_stocks)} 个失败")
            except Exception as e:
                logger.error(f"加载进度失败: {e}")
    
    def _save_progress(self):
        """保存爬取进度"""
        try:
            progress = {
                'completed': list(self.completed_stocks),
                'failed': self.failed_stocks,
                'stats': self.stats,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def generate_stock_codes(self, count: int = 5000) -> List[Dict]:
        """
        生成股票代码列表
        
        Args:
            count: 目标股票数量
            
        Returns:
            股票代码列表
        """
        stocks = []
        
        # 深交所主板 (000001-009999)
        for i in range(1, min(4000, count)):
            code = f"{i:06d}"
            if code.startswith('000') or code.startswith('001') or code.startswith('002') or code.startswith('003'):
                stocks.append({'code': code, 'name': '', 'exchange': 'sz'})
        
        # 上交所主板 (600000-609999)
        for i in range(600000, min(605000, 600000 + count)):
            code = str(i)
            stocks.append({'code': code, 'name': '', 'exchange': 'sh'})
        
        # 科创板 (688000-688999)
        for i in range(688000, min(688500, 688000 + count // 10)):
            code = str(i)
            stocks.append({'code': code, 'name': '', 'exchange': 'sh'})
        
        # 北交所 (430000-899999 中的部分)
        for i in range(830000, min(835000, 830000 + count // 20)):
            code = str(i)
            stocks.append({'code': code, 'name': '', 'exchange': 'bj'})
        
        # 随机打乱顺序，避免连续请求同一交易所
        random.shuffle(stocks)
        
        return stocks[:count]
    
    def crawl_single_stock(
        self,
        stock: Dict,
        max_reports: int = 5,
        years: Optional[List[int]] = None
    ) -> Dict:
        """爬取单只股票的年报"""
        stock_code = stock.get('code', '')
        stock_name = stock.get('name', '')
        
        # 检查是否已完成
        if stock_code in self.completed_stocks:
            return {'code': stock_code, 'status': 'skipped', 'count': 0}
        
        crawler = CNInfoCrawler()
        
        for attempt in range(self.max_retries):
            try:
                # 获取年报列表
                reports = crawler.get_annual_reports(stock_code, stock_name, years)
                
                if not reports:
                    with self.lock:
                        self.completed_stocks.add(stock_code)
                    return {'code': stock_code, 'status': 'no_reports', 'count': 0}
                
                # 限制报告数量
                if max_reports:
                    reports = reports[:max_reports]
                
                # 下载报告
                success_count = 0
                for report in reports:
                    if crawler.download_report(report):
                        success_count += 1
                    random_delay(0.5, 1.5)
                
                # 更新进度
                with self.lock:
                    self.completed_stocks.add(stock_code)
                    self.stats['completed'] += 1
                    self.stats['total_reports'] += success_count
                    if stock_code in self.failed_stocks:
                        del self.failed_stocks[stock_code]
                
                self._save_progress()
                
                return {
                    'code': stock_code,
                    'status': 'success',
                    'count': success_count,
                    'total': len(reports)
                }
                
            except Exception as e:
                logger.error(f"爬取 {stock_code} 失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    with self.lock:
                        self.failed_stocks[stock_code] = str(e)
                        self.stats['failed'] += 1
                    self._save_progress()
                    return {'code': stock_code, 'status': 'failed', 'error': str(e)}
        
        return {'code': stock_code, 'status': 'failed'}
    
    def mass_crawl(
        self,
        target_reports: int = 10000,
        max_reports_per_stock: int = 3,
        years: Optional[List[int]] = None
    ) -> Dict:
        """
        大规模爬取年报
        
        Args:
            target_reports: 目标报告数量
            max_reports_per_stock: 每只股票最大报告数
            years: 年份范围
            
        Returns:
            爬取统计
        """
        # 估算需要的股票数量
        estimated_stocks = int(target_reports / 2.5)  # 假设平均每只股票2.5份报告
        
        # 生成股票列表
        stocks = self.generate_stock_codes(estimated_stocks)
        
        # 过滤已完成的
        pending_stocks = [
            s for s in stocks
            if s.get('code') not in self.completed_stocks
        ]
        
        self.stats['total'] = len(pending_stocks)
        
        logger.info(f"开始大规模爬取: 目标 {target_reports} 份报告")
        logger.info(f"预计需要 {len(pending_stocks)} 只股票")
        logger.info(f"并发数: {self.max_workers}")
        
        results = []
        stop_crawling = False
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    self.crawl_single_stock,
                    stock,
                    max_reports_per_stock,
                    years
                ): stock
                for stock in pending_stocks
            }
            
            # 处理完成的任务
            for i, future in enumerate(as_completed(future_to_stock)):
                if stop_crawling:
                    break
                    
                stock = future_to_stock[future]
                stock_code = stock.get('code', '')
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 检查是否达到目标
                    if self.stats['total_reports'] >= target_reports:
                        logger.info(f"已达到目标报告数: {self.stats['total_reports']}")
                        stop_crawling = True
                    
                    # 打印进度
                    if (i + 1) % 50 == 0 or i == len(pending_stocks) - 1:
                        progress = (i + 1) / len(pending_stocks) * 100
                        logger.info(
                            f"进度: {i+1}/{len(pending_stocks)} ({progress:.1f}%) - "
                            f"完成: {self.stats['completed']}, 失败: {self.stats['failed']}, "
                            f"报告数: {self.stats['total_reports']}"
                        )
                
                except Exception as e:
                    logger.error(f"处理 {stock_code} 时出错: {e}")
                    with self.lock:
                        self.failed_stocks[stock_code] = str(e)
                        self.stats['failed'] += 1
        
        self._save_progress()
        
        return {
            'total_stocks': len(pending_stocks),
            'completed': self.stats['completed'],
            'failed': self.stats['failed'],
            'total_reports': self.stats['total_reports']
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='大规模爬取年报')
    parser.add_argument('--target', type=int, default=10000, help='目标报告数量')
    parser.add_argument('--max-reports', type=int, default=3, help='每只股票最大报告数')
    parser.add_argument('--workers', type=int, default=5, help='并发数')
    parser.add_argument('--start-year', type=int, default=2020, help='开始年份')
    parser.add_argument('--end-year', type=int, default=2024, help='结束年份')
    
    args = parser.parse_args()
    
    # 创建爬虫实例
    crawler = MassCrawler(
        max_workers=args.workers,
        request_delay=0.8
    )
    
    print(f"\n开始大规模爬取")
    print(f"目标报告数: {args.target}")
    print(f"年份范围: {args.start_year}-{args.end_year}")
    print(f"并发数: {args.workers}")
    print("=" * 60)
    
    # 开始爬取
    years = list(range(args.start_year, args.end_year + 1))
    result = crawler.mass_crawl(
        target_reports=args.target,
        max_reports_per_stock=args.max_reports,
        years=years
    )
    
    print("\n" + "=" * 60)
    print("爬取完成！")
    print("=" * 60)
    print(f"处理股票数: {result['total_stocks']}")
    print(f"成功: {result['completed']}")
    print(f"失败: {result['failed']}")
    print(f"总报告数: {result['total_reports']}")


if __name__ == "__main__":
    main()
