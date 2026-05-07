# 年报分析助手 - Qwen3 版本

基于 Qwen3-8B 大语言模型的企业年报智能分析系统，支持 RAG 检索增强生成、32K 长上下文、工具调用等高级特性。

## 系统架构

```
用户浏览器
    ↓
前端服务 (web/server.py) :8080
    ↓ (API代理 /api/*)
后端服务 (api/main.py) :8000
    ↓
RAG检索 (ChromaDB + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B)
    ↓
LLM生成 (Qwen3-8B)
```

## 技术栈

- **大语言模型**: Qwen3-8B (32K 上下文)
- **Embedding模型**: Qwen3-Embedding-0.6B
- **Reranker模型**: Qwen3-Reranker-0.6B
- **向量数据库**: ChromaDB
- **后端框架**: FastAPI
- **前端**: HTML + JavaScript
- **部署环境**: 阿里云 PAI DSW (NVIDIA A10, 23GB显存)

## 模型配置

### 模型路径配置

```yaml
# config/config.yaml
models:
  llm:
    path: "/mnt/workspace/models/llm/qwen/Qwen3-8B"
    max_length: 32768  # 32K 上下文
    enable_thinking: true  # 思考模式
  
  embedding:
    path: "/mnt/workspace/models/embedding/qwen/Qwen3-Embedding-0.6B"
  
  reranker:
    path: "/mnt/workspace/models/reranker/qwen/Qwen3-Reranker-0.6B"
```

### 模型下载

```bash
# 使用 HuggingFace CLI
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /mnt/workspace/models/llm/qwen/Qwen3-8B

huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir /mnt/workspace/models/embedding/qwen/Qwen3-Embedding-0.6B

huggingface-cli download Qwen/Qwen3-Reranker-0.6B \
  --local-dir /mnt/workspace/models/reranker/qwen/Qwen3-Reranker-0.6B

# 或使用 ModelScope（国内推荐）
modelscope download --model qwen/Qwen3-8B \
  --local_dir /mnt/workspace/models/llm/qwen/Qwen3-8B
```

## 快速部署

### 1. 环境准备

```bash
# 登录阿里云 PAI DSW
ssh root@<dsw-instance-id>

# 创建工作目录
mkdir -p /mnt/workspace/annual_report_assistant
cd /mnt/workspace/annual_report_assistant
```

### 2. 上传项目文件

```bash
# 解压项目压缩包
tar -xzf annual_report_assistant_qwen3.tar.gz

# 或手动上传后解压
unzip annual_report_assistant_qwen3.zip
```

### 3. 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 4. 检查模型

```bash
# 检查模型是否存在
python3 scripts/check_models.py

# 如果模型不存在，会生成下载脚本
bash download_models.sh
```

### 5. 构建向量数据库

```bash
# 处理年报PDF并构建向量库
python3 scripts/build_vector_db.py \
  --data-dir ./data/raw \
  --vector-db-path ./data/vector_db \
  --chunk-size 512 \
  --chunk-overlap 128
```

### 6. 启动服务

```bash
# 一键启动所有服务
bash start_all_services.sh

# 或分别启动
# 启动后端API
python3 -m api.main &

# 启动前端服务
python3 web/server.py --port 8080 &
```

### 7. 验证服务

```bash
# 检查服务状态
bash monitor_services.sh

# 测试API
curl http://localhost:8000/health

# 测试模型
python3 scripts/check_models.py --test
```

## 使用指南

### Web 界面

浏览器访问: `http://<服务器IP>:8080`

### API 调用

```bash
# 分析年报
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "分析美的集团2023年财务状况",
    "company": "美的集团",
    "year": "2023",
    "enable_thinking": true,
    "max_tokens": 2048,
    "temperature": 0.7
  }'

# 财务分析
curl -X POST http://localhost:8000/analyze/financial \
  -H "Content-Type