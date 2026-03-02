"""Evaluate ONNX-exported ByT5 on competition val via ONNX Runtime + DirectML.

Usage:
    .venv-dml/Scripts/python.exe scripts/onnx_eval.py
    .venv-dml/Scripts/python.exe scripts/onnx_eval.py --model-dir models/byt5-base-akkadian-onnx
    .venv-dml/Scripts/python.exe scripts/onnx_eval.py --batch-size 4 --num-beams 5

Requires: pip install optimum[onnxruntime] onnxruntime-directml
"""

import argparse
import io
import math
import sys
import time
import warnings

# Fix Windows console encoding for Akkadian diacritics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "byt5-base-akkadian-onnx"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PREFIX = "translate Akkadian to English: "
MAX_SOURCE = 768
MAX_TARGET = 512


def score(preds, refs):
    b = BLEU().corpus_score(preds, [refs]).score
    c = CHRF(word_order=2).corpus_score(preds, [refs]).score
    g = math.sqrt(max(b, 0) * max(c, 0))
    return b, c, g


def main():
    parser = argparse.ArgumentParser(description="ONNX ByT5 eval on competition val")
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR),
        help=f"Path to ONNX model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-source", type=int, default=MAX_SOURCE)
    parser.add_argument("--max-target", type=int, default=MAX_TARGET)
    parser.add_argument("--no-prefix", action="store_true", help="Skip the translation prefix")
    parser.add_argument(
        "--provider", default="dml",
        choices=["dml", "cpu"],
        help="Execution provider: dml (DirectML GPU) or cpu (default: dml)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print(f"Run export first: python scripts/export_onnx.py")
        sys.exit(1)

    # Load data
    comp_df = pd.read_parquet(DATA_DIR / "val_competition.parquet")
    refs = comp_df["translation"].tolist()
    trans = comp_df["transliteration"].tolist()
    print(f"Competition val: {len(comp_df)} samples")
    print(f"Avg source length: {comp_df['transliteration'].str.len().mean():.0f} chars")

    # Load ONNX model
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: optimum not installed. Run:")
        print("  pip install optimum[onnxruntime] onnxruntime-directml")
        sys.exit(1)

    if args.provider == "dml":
        provider = "DmlExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    print(f"\nLoading ONNX model from {model_dir}")
    print(f"Provider: {provider}")
    t0 = time.time()

    # Pass separate encoder/decoder file names (optimum defaults to merged decoder)
    model = ORTModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        provider=provider,
        encoder_file_name="encoder_model.onnx",
        decoder_file_name="decoder_model.onnx",
        decoder_with_past_file_name="decoder_with_past_model.onnx",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Generate translations
    prefix = "" if args.no_prefix else PREFIX
    inputs = [prefix + str(t) for t in trans]
    n_batches = math.ceil(len(inputs) / args.batch_size)

    print(f"\nGenerating ({n_batches} batches, beam={args.num_beams})...")
    all_preds = []
    t0 = time.time()

    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i : i + args.batch_size]
        enc = tokenizer(
            batch,
            max_length=args.max_source,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=args.max_target,
            num_beams=args.num_beams,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_preds.extend(decoded)
        bn = i // args.batch_size + 1
        sys.stdout.write(f"\r  Batch {bn}/{n_batches}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed / len(trans):.2f}s/sample)")

    # Score
    b, c, g = score(all_preds, refs)
    print(f"\n{'='*60}")
    print(f"  BLEU     = {b:.2f}")
    print(f"  chrF++   = {c:.2f}")
    print(f"  geo_mean = {g:.4f}")
    print(f"{'='*60}")

    # Sample predictions
    print(f"\n--- Sample Predictions ---")
    for j in range(min(5, len(all_preds))):
        print(f"\n[{j}] Src: {trans[j][:120]}...")
        print(f"    Out: {all_preds[j][:250]}")
        print(f"    Ref: {refs[j][:250]}")

    print(f"\nSettings: max_source={args.max_source}, max_target={args.max_target}, "
          f"beams={args.num_beams}, batch={args.batch_size}, provider={provider}")


if __name__ == "__main__":
    main()
