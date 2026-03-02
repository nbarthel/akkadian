"""Export ByT5 encoder-decoder model to ONNX via HuggingFace Optimum.

Usage:
    .venv-dml/Scripts/python.exe scripts/export_onnx.py
    .venv-dml/Scripts/python.exe scripts/export_onnx.py --model notninja/byt5-base-akkadian
    .venv-dml/Scripts/python.exe scripts/export_onnx.py --model outputs/byt5-curriculum/phase2_best

Requires: pip install optimum[onnxruntime] onnxruntime-directml
Output:  models/<model-slug>-onnx/ (encoder_model.onnx, decoder_model.onnx, etc.)
"""

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def slug(model_name: str) -> str:
    """Convert model name/path to a filesystem-safe slug."""
    # For HF hub names like 'notninja/byt5-base-akkadian'
    name = model_name.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def main():
    parser = argparse.ArgumentParser(description="Export ByT5 to ONNX")
    parser.add_argument(
        "--model", default="notninja/byt5-base-akkadian",
        help="HuggingFace model ID or local path (default: notninja/byt5-base-akkadian)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: models/<slug>-onnx)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or str(PROJECT_ROOT / "models" / f"{slug(args.model)}-onnx")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {args.model}")
    print(f"Output: {output_path}")
    print(f"Opset:  {args.opset}")

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
    except ImportError:
        print("ERROR: optimum not installed. Run:")
        print("  pip install optimum[onnxruntime] onnxruntime-directml")
        sys.exit(1)

    print("\nExporting to ONNX (this may take a few minutes)...")
    t0 = time.time()

    model = ORTModelForSeq2SeqLM.from_pretrained(
        args.model,
        export=True,
    )
    model.save_pretrained(str(output_path))

    # Also save the tokenizer alongside the ONNX model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(str(output_path))

    elapsed = time.time() - t0
    print(f"Export complete in {elapsed:.1f}s")

    # List output files
    onnx_files = list(output_path.glob("*.onnx"))
    print(f"\nONNX files ({len(onnx_files)}):")
    for f in sorted(onnx_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    print(f"\nReady for eval:")
    print(f"  python scripts/onnx_eval.py --model-dir {output_path}")


if __name__ == "__main__":
    main()
