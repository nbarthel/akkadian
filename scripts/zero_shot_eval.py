"""Zero-shot evaluation of HuggingFace Akkadian models on competition val.
Run with: .venv-dml/Scripts/python.exe scripts/zero_shot_eval.py
Uses DirectML for GPU acceleration on Windows.
"""
import sys
import time
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF

# DirectML device
device = torch_directml.device()
print(f"Device: {device} ({torch_directml.device_name(0)})")

# Data
comp_df = pd.read_parquet('data/processed/val_competition.parquet')
refs = comp_df['translation'].tolist()
trans = comp_df['transliteration'].tolist()
print(f"Competition val: {len(comp_df)} samples")
print(f"Avg source length: {comp_df['transliteration'].str.len().mean():.0f} chars")

PREFIX = "translate Akkadian to English: "
MAX_SOURCE = 768
MAX_TARGET = 512
NUM_BEAMS = 5


def score(preds, refs):
    b = BLEU().corpus_score(preds, [refs]).score
    c = CHRF(word_order=2).corpus_score(preds, [refs]).score
    g = math.sqrt(max(b, 0) * max(c, 0))
    return b, c, g


def generate(model, tokenizer, texts, prefix, batch_size=2):
    model.eval()
    all_preds = []
    inputs = [prefix + str(t) for t in texts]
    n_batches = math.ceil(len(inputs) / batch_size)
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        enc = tokenizer(
            batch, max_length=MAX_SOURCE, truncation=True,
            padding=True, return_tensors='pt'
        )
        # Move to DirectML device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_length=MAX_TARGET,
                num_beams=NUM_BEAMS,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)
        all_preds.extend(decoded)
        bn = i // batch_size + 1
        sys.stdout.write(f"\r  Batch {bn}/{n_batches}")
        sys.stdout.flush()
    print()
    return all_preds


models = [
    ("notninja/byt5-base-akkadian", 2),
    ("Hippopoto0/akkadianT5", 4),
]

results = []

for model_name, bs in models:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    try:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded in {time.time()-t0:.1f}s ({n_params:.0f}M params)")

        for prefix, label in [(PREFIX, "with prefix"), ("", "no prefix")]:
            print(f"\n  {label}:")
            t0 = time.time()
            preds = generate(model, tokenizer, trans, prefix, batch_size=bs)
            elapsed = time.time() - t0
            b, c, g = score(preds, refs)
            print(f"  BLEU={b:.2f}  chrF++={c:.2f}  geo_mean={g:.4f}  ({elapsed:.1f}s)")
            results.append((model_name, label, b, c, g, preds))

        # Show examples from best config for this model
        model_results = [r for r in results if r[0] == model_name]
        best = max(model_results, key=lambda x: x[4])
        print(f"\n  Best: {best[1]} (geo_mean={best[4]:.4f})")
        print(f"\n  --- Sample Predictions ---")
        for j in range(min(3, len(best[5]))):
            print(f"  [{j}] Src: {trans[j][:100]}")
            print(f"      Out: {best[5][j][:200]}")
            print(f"      Ref: {refs[j][:200]}")
            print()

        del model
        import gc; gc.collect()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'Model':<40} {'Config':<15} {'BLEU':>8} {'chrF++':>8} {'geo_mean':>10}")
print("-" * 85)
for name, cfg, b, c, g, _ in sorted(results, key=lambda x: -x[4]):
    print(f"{name:<40} {cfg:<15} {b:>8.2f} {c:>8.2f} {g:>10.4f}")
print("-" * 85)
print(f"Settings: MAX_SOURCE={MAX_SOURCE}, MAX_TARGET={MAX_TARGET}, num_beams={NUM_BEAMS}")
print(f"Device: DirectML ({torch_directml.device_name(0)})")
