import os
from dataclasses import dataclass
from typing import List


@dataclass
class CrawlerConfig:
    """爬虫配置类"""
    
    # 数据存储路径
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # 下载配置
    DOWNLOAD_TIMEOUT = 60
    MAX_RETRIES = 3
    CONCURRENT_DOWNLOADS = 5
    
    # 请求配置
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    REQUEST_DELAY = 1  # 请求间隔(秒)
    
    # 年报筛选配置
    START_YEAR = 2020
    END_YEAR = 2024
    
    # 支持的交易所
    EXCHANGES = ["sz", "sh", "bj"]  # 深交所、上交所、北交所
    
    # 报告类型
    REPORT_TYPES = {
        "annual": "年报",
        "semi_annual": "半年报", 
        "quarterly": "季报"
    }


# 巨潮资讯网API配置
CNINFO_CONFIG = {
    "base_url": "http://www.cninfo.com.cn/new/information",
    "stock_list_api": "http://www.cninfo.com.cn/new/information/getStockList",
    "announcement_api": "http://www.cninfo.com.cn/new/information/getAnnouncementList",
    "download_url": "http://static.cninfo.com.cn/"
}

# 上交所配置
SSE_CONFIG = {
    "base_url": "http://query.sse.com.cn",
    "company_list_api": "http://query.sse.com.cn/commonQuery.do",
    "disclosure_api": "http://query.sse.com.cn/commonQuery.do"
}

# 深交所配置
SZSE_CONFIG = {
    "base_url": "http://www.szse.cn/api",
    "company_list_api": "http://www.szse.cn/api/market/ssjjhq/getTimeData",
    "disclosure_api": "http://www.szse.cn/api/disc/announcement/getAnnouncementList"
}
