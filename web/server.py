"""
前端静态页面服务器
提供 SPA 页面和 API 代理功能，基于 FastAPI + httpx 异步实现
"""
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", 80))

app = FastAPI(title="年报分析助手 - 前端服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_http_client: httpx.AsyncClient = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _http_client


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


WEB_DIR = Path(__file__).parent


@app.get("/")
async def serve_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
async def serve_favicon():
    favicon_path = WEB_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404)


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def api_proxy(request: Request, path: str):
    target_url = f"{API_BASE_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    client = get_http_client()

    method = request.method
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    body = None
    if method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    try:
        response = await client.request(
            method=method,
            url=target_url,
            headers=headers,
            content=body,
            follow_redirects=True,
        )

        if "text/event-stream" in response.headers.get("content-type", ""):
            async def sse_proxy():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                sse_proxy(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        return StreamingResponse(
            response.aiter_bytes(),
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="后端服务不可用")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="后端服务超时")


def main():
    logger.info(f"启动前端服务: http://{WEB_HOST}:{WEB_PORT}")
    logger.info(f"代理后端 API: {API_BASE_URL}")
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")


if __name__ == "__main__":
    main()