# LoRA 权重合并说明

## 下载后合并

将本目录下所有分卷下载到同一文件夹后执行合并：

### Windows (PowerShell)
```powershell
Get-Content adapter_model.tar.gz.part_a* | Set-Content adapter_model.tar.gz   
# 解压
tar -xzf adapter_model.tar.gz
```

不，上面用的是二进制文件，PowerShell 的 Get-Content 不适合。正确方式：

### Windows (PowerShell)
```powershell
cmd /c "copy /b adapter_model.tar.gz.part_* adapter_model.tar.gz"
tar -xzf adapter_model.tar.gz
```

### Linux / macOS
```bash
cat adapter_model.tar.gz.part_a* > adapter_model.tar.gz
tar -xzf adapter_model.tar.gz
```

## 解压后得到
```
models/
  llm_finetuned/
    adapter_model.safetensors     # LoRA 权重 (666MB)
```

## 使用方法
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",  # 或本地路径
    device_map="auto",
    torch_dtype="float16"
)
model = PeftModel.from_pretrained(base_model, "./models/llm_finetuned")
```
