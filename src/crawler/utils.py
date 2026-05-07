import os
import time
import random
import hashlib
from pathlib import Path
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session(max_retries: int = 3, timeout: int = 60) -> requests.Session:
    """创建带重试机制的HTTP会话"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    return session


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_file_md5(filepath: str) -> str:
    """计算文件MD5"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def random_delay(min_seconds: float = 0.5, max_seconds: float = 2.0):
    """随机延迟，避免请求过快"""
    time.sleep(random.uniform(min_seconds, max_seconds))


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:200]  # 限制长度


def download_file(
    url: str, 
    save_path: str, 
    session: Optional[requests.Session] = None,
    timeout: int = 60
) -> bool:
    """
    下载文件到指定路径
    
    Args:
        url: 文件URL
        save_path: 保存路径
        session: HTTP会话
        timeout: 超时时间
        
    Returns:
        下载是否成功
    """
    if session is None:
        session = create_session()
    
    try:
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        ensure_dir(os.path.dirname(save_path))
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def format_stock_code(code: str, exchange: str) -> str:
    """
    格式化股票代码
    
    Args:
        code: 股票代码
        exchange: 交易所 (sz/sh/bj)
        
    Returns:
        格式化后的代码
    """
    code = code.strip()
    
    if exchange == "sz":
        if code.startswith("0") or code.startswith("3"):
            return f"{code}.SZ"
        elif code.startswith("00"):
            return f"{code}.SZ"
    elif exchange == "sh":
        if code.startswith("6"):
            return f"{code}.SH"
        elif code.startswith("68"):
            return f"{code}.SH"
    elif exchange == "bj":
        if code.startswith("8") or code.startswith("4"):
            return f"{code}.BJ"
    
    return code
