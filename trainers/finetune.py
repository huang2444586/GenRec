from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import json
import random
import os
from transformers import TrainingArguments, Trainer
import json
import pandas as pd
import torch
from datasets import Dataset
import swanlab

SEED = 666
random.seed(SEED)
DATASET_PATH = os.path.join('.', 'data', 'beauty', 'dataset.jsonl')

dataset = MsDataset.load(
   dataset_name=DATASET_PATH,
   data_files=DATASET_PATH,
)

data_list = list(dataset)
random.shuffle(data_list)

split_idx = int(len(data_list) * 0.9)

train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"The dataset has been split successfully.")
print(f"Train Set Size：{len(train_data)}")
print(f"Val Set Size：{len(val_data)}")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2-0.5B", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2-0.5B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2-0.5B", device_map="auto", torch_dtype=torch.bfloat16)

os.environ["SWANLAB_PROJECT"]="qwen2-sft-genrec"


