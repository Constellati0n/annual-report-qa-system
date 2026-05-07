#!/usr/bin/env python3
"""
前端Web服务器
提供静态文件服务和API代理
"""

import os
import sys
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.request
import urllib.error

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent

class APIHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        # API代理请求
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        # 默认返回index.html
        if self.path == '/':
            self.path = '/index.html'
        
        return super().do_GET()
    
    def do_POST(self):
        """处理POST请求"""
        # API代理请求
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        self.send_error(404)
    
    def proxy_to_backend(self):
        """代理请求到后端API"""
        try:
            # 后端API地址
            backend_url = f"http://localhost:8000{self.path[4:]}"  # 去掉/api前缀
            
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            
            # 创建请求
            req = urllib.request.Request(
                backend_url,
                data=body,
                headers={
                    'Content-Type': self.headers.get('Content-Type', 'application/json')
                },
                method=self.command
            )
            
            # 发送请求（增加超时时间到300秒 - 5分钟）
            with urllib.request.urlopen(req, timeout=300) as response:
                # 返回响应
                self.send_response(response.status)
                for header, value in response.headers.items():
                    if header.lower() not in ['transfer-encoding', 'content-length']:
                        self.send_header(header, value)
                
                # 如果是流式响应，不发送 Content-Length
                is_streaming = response.headers.get('Content-Type') == 'text/event-stream'
                if not is_streaming:
                    self.send_header('Content-Length', response.headers.get('Content-Length', '0'))
                
                self.end_headers()
                
                # 流式转发内容
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                
        except urllib.error.HTTPError as e:
            self.send_error(e.code, e.reason)
        except Exception as e:
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"{args[0]}")


def run_server(port=8080):
    """运行Web服务器"""
    try:
        server_address = ('0.0.0.0', port)  # 明确绑定到0.0.0.0
        httpd = HTTPServer(server_address, APIHandler)
        
        logger.info("=" * 60)
        logger.info("🌐 年报分析助手前端服务已启动")
        logger.info("=" * 60)
        logger.info(f"访问地址: http://0.0.0.0:{port}")
        logger.info(f"静态文件目录: {WEB_DIR}")
        logger.info(f"API代理: /api/* -> http://localhost:8000/*")
        logger.info("=" * 60)
        logger.info("按 Ctrl+C 停止服务")
        logger.info("=" * 60)
        
        httpd.serve_forever()
    except OSError as e:
        logger.error(f"启动失败: {e}")
        if e.errno == 98:  # Address already in use
            logger.error(f"端口 {port} 已被占用")
        elif e.errno == 13:  # Permission denied
            logger.error(f"没有权限绑定端口 {port}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    # 设置无缓冲输出（必须在最开始）
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    
    parser = argparse.ArgumentParser(description='年报分析助手Web服务器')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机')
    args = parser.parse_args()
    
    # 立即输出启动信息
    print(f"[{__file__}] 正在启动前端服务...", flush=True)
    print(f"[{__file__}] 参数: host={args.host}, port={args.port}", flush=True)
    
    run_server(port=args.port)
