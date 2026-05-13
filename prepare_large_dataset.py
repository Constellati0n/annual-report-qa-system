"""
流式处理 806,777 条训练数据，拆分为 train/val/test (90/5/5)
避免全量加载到内存
"""
import random
import json
import os
from pathlib import Path

src = Path(r'c:\selfway\homework\ai\data\processed\training\train_dataset_merged_chatml.jsonl')
out_dir = Path(r'c:\selfway\homework\ai\data\train_v2')
out_dir.mkdir(parents=True, exist_ok=True)

seed = 42
random.seed(seed)

n_val = int(806777 * 0.05)
n_test = int(806777 * 0.05)
n_train = 806777 - n_val - n_test

val_indices = set(random.sample(range(806777), n_val))
remaining = [i for i in range(806777) if i not in val_indices]
test_indices = set(random.sample(remaining, n_test))

print(f'Train: {n_train}')
print(f'Val:   {n_val}')
print(f'Test:  {n_test}')

train_f = open(out_dir / 'train.json', 'w', encoding='utf-8')
val_f = open(out_dir / 'val.json', 'w', encoding='utf-8')
test_f = open(out_dir / 'test.json', 'w', encoding='utf-8')

for fp in [train_f, val_f, test_f]:
    fp.write('[\n')

train_count = val_count = test_count = 0
is_first = {train_f: True, val_f: True, test_f: True}

with open(src, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except:
            continue

        if i in val_indices:
            target = val_f
            val_count += 1
        elif i in test_indices:
            target = test_f
            test_count += 1
        else:
            target = train_f
            train_count += 1

        if not is_first[target]:
            target.write(',\n')
        else:
            is_first[target] = False
        json.dump(item, target, ensure_ascii=False)

        if (i + 1) % 100000 == 0:
            print(f'Progress: {i+1}/806777')

for fp in [train_f, val_f, test_f]:
    fp.write('\n]\n')
    fp.close()

print(f'Done! Train={train_count}, Val={val_count}, Test={test_count}')
print(f'Output: {out_dir}')
