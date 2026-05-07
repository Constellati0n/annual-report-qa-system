"""
批量年报爬虫
支持大规模并发爬取、断点续传、进度监控
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

try:
    from cninfo_crawler import CNInfoCrawler
    from config import CrawlerConfig
    from utils import random_delay, ensure_dir
except ImportError:
    from .cninfo_crawler import CNInfoCrawler
    from .config import CrawlerConfig
    from .utils import random_delay, ensure_dir


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchCrawler:
    """批量爬虫"""
    
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
        self.progress_file = Path(self.config.DATA_DIR) / "crawler_progress.json"
        
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
    
    def get_stock_list(self, exchange: str = "sz", max_stocks: Optional[int] = None) -> List[Dict]:
        """
        获取股票列表
        
        Args:
            exchange: 交易所 (sz/sh/bj)
            max_stocks: 最大股票数量
            
        Returns:
            股票列表
        """
        crawler = CNInfoCrawler()
        stocks = crawler.get_stock_list(exchange)
        
        if max_stocks:
            stocks = stocks[:max_stocks]
        
        logger.info(f"从 {exchange} 交易所获取 {len(stocks)} 只股票")
        return stocks
    
    def get_all_stocks(self, max_per_exchange: Optional[int] = None) -> List[Dict]:
        """
        获取所有交易所的股票
        
        Args:
            max_per_exchange: 每个交易所最大数量
            
        Returns:
            股票列表
        """
        all_stocks = []
        
        for exchange in ["sz", "sh", "bj"]:
            stocks = self.get_stock_list(exchange, max_per_exchange)
            all_stocks.extend(stocks)
            time.sleep(2)  # 交易所间延迟
        
        logger.info(f"总共获取 {len(all_stocks)} 只股票")
        return all_stocks
    
    def crawl_single_stock(
        self,
        stock: Dict,
        max_reports: int = 5,
        years: Optional[List[int]] = None
    ) -> Dict:
        """
        爬取单只股票的年报
        
        Args:
            stock: 股票信息
            max_reports: 最大报告数
            years: 年份列表
            
        Returns:
            爬取结果
        """
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
                    random_delay(1, 2)
                
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
    
    def batch_crawl(
        self,
        stocks: List[Dict],
        max_reports: int = 5,
        years: Optional[List[int]] = None,
        resume: bool = True
    ) -> Dict:
        """
        批量爬取年报
        
        Args:
            stocks: 股票列表
            max_reports: 每只股票最大报告数
            years: 年份范围
            resume: 是否从上次进度继续
            
        Returns:
            爬取统计
        """
        if not resume:
            self.completed_stocks.clear()
            self.failed_stocks.clear()
        
        # 过滤已完成的股票
        pending_stocks = [
            s for s in stocks
            if s.get('code') not in self.completed_stocks
        ]
        
        self.stats['total'] = len(stocks)
        
        logger.info(f"开始批量爬取: 总共 {len(stocks)} 只, 待处理 {len(pending_stocks)} 只")
        logger.info(f"并发数: {self.max_workers}, 每只股票最多 {max_reports} 份报告")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    self.crawl_single_stock,
                    stock,
                    max_reports,
                    years
                ): stock
                for stock in pending_stocks
            }
            
            # 处理完成的任务
            for i, future in enumerate(as_completed(future_to_stock)):
                stock = future_to_stock[future]
                stock_code = stock.get('code', '')
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 打印进度
                    if (i + 1) % 10 == 0 or i == len(pending_stocks) - 1:
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
            'total': len(stocks),
            'completed': self.stats['completed'],
            'failed': self.stats['failed'],
            'skipped': len(stocks) - len(pending_stocks),
            'total_reports': self.stats['total_reports']
        }
    
    def retry_failed(self, max_reports: int = 5) -> Dict:
        """重试失败的股票"""
        if not self.failed_stocks:
            logger.info("没有失败的股票需要重试")
            return {'retried': 0}
        
        failed_codes = list(self.failed_stocks.keys())
        logger.info(f"重试 {len(failed_codes)} 只失败的股票")
        
        # 构建股票列表
        stocks = [{'code': code, 'name': ''} for code in failed_codes]
        
        # 清空失败列表
        self.failed_stocks.clear()
        
        return self.batch_crawl(stocks, max_reports=max_reports)


def main():
    """主函数 - 批量爬取年报"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量爬取年报')
    parser.add_argument('--count', type=int, default=100, help='爬取股票数量')
    parser.add_argument('--max-reports', type=int, default=3, help='每只股票最大报告数')
    parser.add_argument('--workers', type=int, default=5, help='并发数')
    parser.add_argument('--exchange', type=str, default='all', 
                       choices=['sz', 'sh', 'bj', 'all'], help='交易所')
    parser.add_argument('--retry-failed', action='store_true', help='重试失败的股票')
    parser.add_argument('--start-year', type=int, default=2020, help='开始年份')
    parser.add_argument('--end-year', type=int, default=2024, help='结束年份')
    
    args = parser.parse_args()
    
    # 创建爬虫实例
    batch_crawler = BatchCrawler(
        max_workers=args.workers,
        request_delay=1.0
    )
    
    # 重试失败的股票
    if args.retry_failed:
        result = batch_crawler.retry_failed(max_reports=args.max_reports)
        print(f"\n重试完成: {result}")
        return
    
    # 获取股票列表
    if args.exchange == 'all':
        # 每个交易所平均分配
        per_exchange = args.count // 3 + 1
        stocks = batch_crawler.get_all_stocks(max_per_exchange=per_exchange)
        stocks = stocks[:args.count]
    else:
        stocks = batch_crawler.get_stock_list(args.exchange, args.count)
    
    if not stocks:
        print("没有获取到股票列表")
        return
    
    print(f"\n准备爬取 {len(stocks)} 只股票的年报")
    print(f"年份范围: {args.start_year}-{args.end_year}")
    print(f"每只股票最多 {args.max_reports} 份报告")
    print(f"并发数: {args.workers}")
    print("=" * 60)
    
    # 开始批量爬取
    years = list(range(args.start_year, args.end_year + 1))
    result = batch_crawler.batch_crawl(
        stocks=stocks,
        max_reports=args.max_reports,
        years=years,
        resume=True
    )
    
    print("\n" + "=" * 60)
    print("爬取完成！")
    print("=" * 60)
    print(f"总股票数: {result['total']}")
    print(f"成功: {result['completed']}")
    print(f"失败: {result['failed']}")
    print(f"跳过: {result['skipped']}")
    print(f"总报告数: {result['total_reports']}")


if __name__ == "__main__":
    main()
