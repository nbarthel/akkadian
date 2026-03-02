"""Verify Windows environment setup."""
import torch
import transformers
import datasets
import sacrebleu
import pandas as pd

print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets:     {datasets.__version__}")
print(f"sacrebleu:    {sacrebleu.__version__}")
print(f"pandas:       {pd.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")

# Verify data loads
df = pd.read_parquet("data/processed/train.parquet")
print(f"Train data:   {len(df)} rows")
gold = df[df["quality"] == "gold"]
print(f"Gold subset:  {len(gold)} rows")

comp = pd.read_parquet("data/processed/val_competition.parquet")
print(f"Competition:  {len(comp)} rows")

# Quick model load test
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
print(f"Model params: {model.num_parameters():,}")
print("Setup OK!")
