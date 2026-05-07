"""
巨潮资讯网爬虫
主要数据源：http://www.cninfo.com.cn
"""
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlencode

import requests

try:
    from .config import CrawlerConfig, CNINFO_CONFIG
    from .utils import create_session, download_file, random_delay, sanitize_filename, ensure_dir
except ImportError:
    from config import CrawlerConfig, CNINFO_CONFIG
    from utils import create_session, download_file, random_delay, sanitize_filename, ensure_dir


class CNInfoCrawler:
    """巨潮资讯网年报爬虫"""
    
    def __init__(self):
        self.config = CrawlerConfig()
        self.session = create_session()
        self.base_url = CNINFO_CONFIG["base_url"]
        self.announcement_api = CNINFO_CONFIG["announcement_api"]
        self.stock_list_api = CNINFO_CONFIG["stock_list_api"]
        
        # 确保数据目录存在
        ensure_dir(self.config.RAW_DIR)
        
        # 缓存股票orgId映射
        self._org_id_cache = {}
        
    def get_stock_org_id(self, stock_code: str) -> str:
        """
        获取股票的orgId
        
        Args:
            stock_code: 股票代码
            
        Returns:
            orgId
        """
        if stock_code in self._org_id_cache:
            return self._org_id_cache[stock_code]
        
        url = "http://www.cninfo.com.cn/new/information/topSearch/query"
        params = {
            "keyWord": stock_code,
            "maxNum": 10
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        try:
            response = self.session.post(url, data=params, headers=headers, timeout=30)
            data = response.json()
            
            for item in data:
                if item.get("code") == stock_code:
                    org_id = item.get("orgId", "")
                    self._org_id_cache[stock_code] = org_id
                    return org_id
                    
        except Exception as e:
            print(f"获取orgId失败 {stock_code}: {e}")
        
        return ""
    
    def get_stock_list(self, exchange: str = "sz") -> List[Dict]:
        """
        获取股票列表
        
        Args:
            exchange: 交易所代码 (sz/sh/bj)
            
        Returns:
            股票列表
        """
        # 使用搜索API获取股票列表
        # 通过搜索常见字符来获取股票
        url = "http://www.cninfo.com.cn/new/information/topSearch/query"
        
        # 搜索关键词列表（用于获取不同股票）
        search_keywords = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ]
        
        all_stocks = []
        seen_codes = set()
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        for keyword in search_keywords:
            try:
                params = {
                    "keyWord": keyword,
                    "maxNum": 30
                }
                
                response = self.session.post(url, data=params, headers=headers, timeout=10)
                data = response.json()
                
                if isinstance(data, list):
                    for item in data:
                        code = item.get("code", "")
                        org_id = item.get("orgId", "")
                        
                        # 根据orgId判断交易所 (gssz=深圳, gssh=上海, gsbj=北京)
                        if exchange == "sz" and not org_id.startswith("gssz"):
                            continue
                        if exchange == "sh" and not org_id.startswith("gssh"):
                            continue
                        if exchange == "bj" and not org_id.startswith("gsbj"):
                            continue
                        
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            all_stocks.append({
                                "code": code,
                                "name": item.get("name", ""),
                                "orgId": org_id
                            })
                
                # 避免请求过快
                time.sleep(0.3)
                
            except Exception as e:
                print(f"搜索关键词 '{keyword}' 失败: {e}")
                continue
        
        return all_stocks
    
    def search_announcements(
        self,
        stock_code: str,
        stock_name: str = "",
        start_date: str = "",
        end_date: str = "",
        page_num: int = 1,
        page_size: int = 30
    ) -> Dict:
        """
        搜索公告列表
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            page_num: 页码
            page_size: 每页数量
            
        Returns:
            公告列表数据
        """
        # 巨潮资讯网公告查询API
        url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        
        # 获取orgId
        org_id = self.get_stock_org_id(stock_code)
        if not org_id:
            print(f"无法获取 {stock_code} 的orgId")
            return {}
        
        # 构建查询条件
        if stock_code.startswith("6"):
            column = "sse"
        elif stock_code.startswith(("0", "3")):
            column = "szse"
        else:
            column = "bjse"
        
        # 构建股票代码（code,orgId格式）
        stock_code_full = f"{stock_code},{org_id}"
        
        params = {
            "stock": stock_code_full,
            "tabName": "fulltext",
            "pageSize": page_size,
            "pageNum": page_num,
            "column": column,
            "category": "category_ndbg_szsh",  # 年报分类
            "seDate": f"{start_date}~{end_date}" if start_date and end_date else "",
            "isHLtitle": "true"
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "http://www.cninfo.com.cn/new/information/topSearch/query"
        }
        
        try:
            response = self.session.post(
                url, 
                data=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"搜索公告失败 {stock_code}: {e}")
            return {}
    
    def get_annual_reports(
        self,
        stock_code: str,
        stock_name: str = "",
        years: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        获取指定股票的年报列表
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            years: 年份列表，默认使用配置中的年份范围
            
        Returns:
            年报信息列表
        """
        if years is None:
            years = list(range(self.config.START_YEAR, self.config.END_YEAR + 1))
        
        reports = []
        start_date = f"{min(years)}-01-01"
        end_date = f"{max(years)}-12-31"
        
        page_num = 1
        while True:
            data = self.search_announcements(
                stock_code=stock_code,
                stock_name=stock_name,
                start_date=start_date,
                end_date=end_date,
                page_num=page_num,
                page_size=30
            )
            
            announcements = data.get("announcements", [])
            if not announcements:
                break
            
            for item in announcements:
                title = item.get("announcementTitle", "")
                
                # 筛选年报（排除摘要、修订版等）
                if self._is_annual_report(title):
                    report_info = {
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "title": title,
                        "announcement_id": item.get("announcementId"),
                        "announcement_time": item.get("announcementTime"),
                        "adjunct_url": item.get("adjunctUrl"),
                        "pdf_url": f"http://static.cninfo.com.cn/{item.get('adjunctUrl')}"
                    }
                    reports.append(report_info)
            
            # 检查是否还有下一页
            total_records = data.get("totalRecords", 0)
            if page_num * 30 >= total_records:
                break
                
            page_num += 1
            random_delay(0.5, 1.5)
        
        return reports
    
    def _is_annual_report(self, title: str) -> bool:
        """
        判断是否为年报（排除摘要、修订版、补充公告等）
        
        Args:
            title: 公告标题
            
        Returns:
            是否为年报
        """
        title = title.lower()
        
        # 必须包含"年度报告"
        if "年度报告" not in title:
            return False
        
        # 排除非完整年报
        exclude_keywords = [
            "摘要",
            "修订",
            "更正",
            "补充",
            "提示性",
            "停牌",
            "复牌",
            "延期",
            "变更"
        ]
        
        for keyword in exclude_keywords:
            if keyword in title:
                return False
        
        return True
    
    def download_report(
        self,
        report_info: Dict,
        save_dir: Optional[str] = None
    ) -> bool:
        """
        下载单份年报
        
        Args:
            report_info: 年报信息字典
            save_dir: 保存目录
            
        Returns:
            下载是否成功
        """
        if save_dir is None:
            save_dir = os.path.join(
                self.config.RAW_DIR,
                report_info["stock_code"]
            )
        
        ensure_dir(save_dir)
        
        # 构建文件名
        filename = sanitize_filename(
            f"{report_info['stock_code']}_{report_info['stock_name']}_{report_info['title']}.pdf"
        )
        save_path = os.path.join(save_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            print(f"文件已存在: {filename}")
            return True
        
        # 下载PDF
        pdf_url = report_info.get("pdf_url")
        if not pdf_url:
            print(f"PDF链接不存在: {report_info['title']}")
            return False
        
        print(f"正在下载: {filename}")
        success = download_file(pdf_url, save_path, self.session)
        
        if success:
            print(f"下载成功: {filename}")
            # 保存元数据
            self._save_metadata(report_info, save_dir)
        else:
            print(f"下载失败: {filename}")
        
        random_delay(1, 2)
        return success
    
    def _save_metadata(self, report_info: Dict, save_dir: str):
        """保存年报元数据"""
        metadata_path = os.path.join(save_dir, "metadata.json")
        
        metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        # 检查是否已存在
        exists = any(
            m.get("announcement_id") == report_info.get("announcement_id")
            for m in metadata
        )
        
        if not exists:
            metadata.append(report_info)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def crawl_stock_reports(
        self,
        stock_code: str,
        stock_name: str = "",
        max_reports: Optional[int] = None
    ) -> int:
        """
        爬取指定股票的所有年报
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_reports: 最大下载数量
            
        Returns:
            成功下载的数量
        """
        print(f"\n开始爬取 {stock_code} {stock_name} 的年报...")
        
        # 获取年报列表
        reports = self.get_annual_reports(stock_code, stock_name)
        print(f"找到 {len(reports)} 份年报")
        
        if max_reports:
            reports = reports[:max_reports]
        
        # 下载年报
        success_count = 0
        for report in reports:
            if self.download_report(report):
                success_count += 1
        
        print(f"成功下载 {success_count}/{len(reports)} 份年报")
        return success_count
    
    def crawl_multiple_stocks(
        self,
        stock_list: List[Dict],
        max_reports_per_stock: Optional[int] = None
    ) -> Dict:
        """
        批量爬取多只股票的年报
        
        Args:
            stock_list: 股票列表 [{"code": "", "name": ""}]
            max_reports_per_stock: 每只股票最大下载数量
            
        Returns:
            爬取统计信息
        """
        stats = {
            "total_stocks": len(stock_list),
            "success_stocks": 0,
            "total_reports": 0,
            "success_reports": 0,
            "failed_stocks": []
        }
        
        for i, stock in enumerate(stock_list):
            print(f"\n[{i+1}/{len(stock_list)}] 处理股票: {stock.get('code')} {stock.get('name')}")
            
            try:
                count = self.crawl_stock_reports(
                    stock_code=stock.get("code"),
                    stock_name=stock.get("name"),
                    max_reports=max_reports_per_stock
                )
                
                if count > 0:
                    stats["success_stocks"] += 1
                    stats["success_reports"] += count
                    
            except Exception as e:
                print(f"处理失败: {e}")
                stats["failed_stocks"].append(stock.get("code"))
            
            # 股票间延迟
            random_delay(2, 4)
        
        return stats


if __name__ == "__main__":
    # 测试代码
    crawler = CNInfoCrawler()
    
    # 测试获取年报列表
    reports = crawler.get_annual_reports("000001", "平安银行")
    print(f"\n找到 {len(reports)} 份年报")
    
    for report in reports[:3]:
        print(f"- {report['title']} ({report['announcement_time']})")
