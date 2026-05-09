# LoRA 权重 - 年报智能问答微调模型

基于 Qwen3-8B 微调的 LoRA 适配器权重，专精于中国上市公司年报的财务分析、经营回顾、风险评估等任务。

## 模型信息

| 项目 | 值 |
|------|-----|
| 基座模型 | Qwen3-8B |
| 微调方法 | LoRA |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| Dropout | 0.1 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 训练数据 | 约 9,000 条真实年报问答对 |
| Epochs | 3 |
| 验证 Loss | 0.078 |

## 第一步：合并分卷

将当前目录下所有分卷下载到同一文件夹后，执行合并解压。

### Windows

```powershell
cmd /c "copy /b adapter_model.tar.gz.part_* adapter_model.tar.gz"
tar -xzf adapter_model.tar.gz
```

### Linux / macOS

```bash
cat adapter_model.tar.gz.part_a* > adapter_model.tar.gz
tar -xzf adapter_model.tar.gz
```

### 解压后目录结构

```
models/
  llm_finetuned/
    adapter_model.safetensors    # LoRA 权重 (666 MB)
```

## 第二步：合并权重文件

将本目录下的 `adapter_config.json`、`tokenizer_config.json`、`chat_template.jinja`、`README.md` 复制到 `models/llm_finetuned/` 中，与 `adapter_model.safetensors` 放在一起。

你可以在仓库根目录找到这些文件：
- `models/llm_finetuned/adapter_config.json`
- `models/llm_finetuned/tokenizer_config.json`
- `models/llm_finetuned/chat_template.jinja`
- `models/llm_finetuned/README.md`

最终目录结构：

```
models/
  llm_finetuned/
    adapter_model.safetensors    # 从分卷解压得到
    adapter_config.json          # 从仓库复制
    tokenizer_config.json        # 从仓库复制
    chat_template.jinja          # 从仓库复制
    README.md                    # 从仓库复制
```

## 第三步：安装依赖

```bash
pip install transformers>=4.57.0 peft>=0.14.0 torch>=2.0.0
```

## 第四步：加载使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型（需要先从 ModelScope/HuggingFace 下载 Qwen3-8B）
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",          # 或本地路径如 ./Qwen3-8B
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(
    base_model,
    "./models/llm_finetuned"  # 适配器目录
)

model.eval()
```

## 第五步：推理示例

```python
tokenizer = AutoTokenizer.from_pretrained("./models/llm_finetuned", trust_remote_code=True)

messages = [
    {"role": "system", "content": "你是一位专业的财务分析师，擅长分析上市公司年报数据。"},
    {"role": "user", "content": "请分析贵州茅台2023年的财务状况，包括盈利能力、偿债能力和成长能力。"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## 基座模型下载

Qwen3-8B 基座模型可从以下渠道获取：

**ModelScope**（国内推荐）:
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

**HuggingFace**:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B
```
