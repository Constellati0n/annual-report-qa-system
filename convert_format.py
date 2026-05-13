import json
import random
from pathlib import Path

src = Path(r'c:\selfway\homework\ai\data\train_all\checkpoint_2000.jsonl')
out_dir = Path(r'c:\selfway\homework\ai\data\upload_train')
out_dir.mkdir(exist_ok=True)

chat_data = []
with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        messages = []
        if item.get('system_prompt'):
            messages.append({"role": "system", "content": item['system_prompt']})
        user_content = item['instruction']
        if item.get('input'):
            user_content += f"\n\n{item['input']}"
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": item['output']})
        chat_data.append({"messages": messages})

random.seed(42)
random.shuffle(chat_data)

n = len(chat_data)
train_n = int(n * 0.9)
val_n = int(n * 0.05)

train_data = chat_data[:train_n]
val_data = chat_data[train_n:train_n + val_n]
test_data = chat_data[train_n + val_n:]

for name, data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
    path = out_dir / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{name}: {len(data)} samples -> {path}")

print(f"\nTotal: {n}")
