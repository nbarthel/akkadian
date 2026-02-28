# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle competition entry for the **Deep Past Challenge** — translating Old Assyrian cuneiform transliterations to English. Competition runs until March 23, 2026. Submission is via Kaggle Notebooks (code competition).

**Evaluation metric:** geometric mean of BLEU and chrF++ → `√(BLEU × chrF++)`. Both metrics come from `sacrebleu`.

## Commands

### Training (legacy baseline)
```bash
python src/train_baseline.py \
  --data-dir data/raw \
  --output-dir outputs/t5-akkadian \
  --submission-dir submissions \
  --model-name google/byt5-small \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-length 512
```

### Curriculum training
```bash
# Phase 1: General Akkadian (all gold data, ~126K samples)
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 1 \
  --model-name google/byt5-small \
  --epochs 5 --batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 --warmup-ratio 0.1 \
  --max-source-length 1024 --max-target-length 1024

# Phase 2: Old Assyrian specialization (~15K samples)
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 2 \
  --resume-from outputs/byt5-curriculum/phase1_best \
  --epochs 10 --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 --warmup-ratio 0.05 \
  --max-source-length 1024 --max-target-length 1024

# Phase 3: Lexicon augmentation (optional)
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 3 \
  --resume-from outputs/byt5-curriculum/phase2_best \
  --epochs 2 --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 --warmup-ratio 0.05 \
  --max-source-length 1024 --max-target-length 1024
```

### Data download
```bash
kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw
```

### Kaggle notebook submission
```bash
# Push notebook to Kaggle for submission
kaggle kernels push -p notebooks/
```
Notebook metadata lives in `notebooks/kernel-metadata.json` (id: `nicbarthelemy1/deep-past-byt5-baseline`).

### Dataset assembly
```bash
python -m src.data.assemble                                    # Full pipeline (all sources)
python -m src.data.assemble --sources kaggle,oare_sentences    # Local sources only (no network)
python -m src.data.assemble --force-refresh                    # Re-download external sources
```

## Architecture

### Training pipeline (`src/train_baseline.py`)
Single-file pipeline using HuggingFace `Seq2SeqTrainer` with curriculum learning support:

**Legacy mode** (`--phase 0`, default): reads `data/raw/train.csv` (1561 samples), 90/10 split.

**Curriculum mode** (`--phase 1/2/3`): reads assembled parquet from `data/processed/`.
- Phase 1: all gold-quality pairs (~126K) — general Akkadian
- Phase 2: Old Assyrian dialect only (~15K) — competition domain specialization
- Phase 3: Phase 2 + lexicon entries — vocabulary expansion
- `--resume-from` loads model from previous phase's best checkpoint

Key functions:
- `load_phase_data()` — filters parquet by phase (gold/dialect/lexicon)
- `preprocess_function()` — tokenizes with separate `max_source_length`/`max_target_length`
- `score_predictions()` — reusable BLEU/chrF++/geo_mean scorer with prefix support
- `FullValCallback` — logs full validation metrics alongside competition val
- `create_compute_metrics()` — HuggingFace Trainer metrics callback

Dual validation: `val_competition.parquet` (88 rows, Old Assyrian) for checkpoint selection, `val.parquet` (8K rows) for monitoring. Best checkpoint selection uses `geo_mean` metric.

### Data pipeline (`src/data/`)
Modular source adapters in `src/data/sources/`. Each exports `load(**kwargs) -> DataFrame` returning rows in a unified schema (transliteration, translation, source, dialect, genre, quality, has_translation, multimodal_ref). Registry in `src/data/sources/__init__.py` auto-discovers adapters.

`src/data/assemble.py` orchestrates: load all sources -> normalize Unicode -> deduplicate (priority: kaggle > oare > oracc > hf > lexicon) -> split 90/5/5 -> write Parquet to `data/processed/`.

Sources: kaggle (1.5K), oare_sentences (8.4K), hf_phucthaiv02 (83K), hf_cipher (50K), lexicon (21K), oracc (when available). Total: ~161K pairs.

### Model
**ByT5-small** (`google/byt5-small`) — byte-level T5 that tokenizes at the byte level, avoiding vocabulary mismatch issues with Akkadian transliteration characters. Encoder-decoder, 6 layers each, d_model=512.

### Notebooks
- `notebooks/baseline_byt5.ipynb` — local development notebook
- `notebooks/kaggle_baseline_byt5.ipynb` — the actual Kaggle submission notebook (GPU + internet enabled)
- `notebooks/01_baseline.ipynb` — initial exploration

### Data
- `data/raw/train.csv` — transliteration/translation pairs (1561 rows)
- `data/raw/test.csv` — transliterations to predict (4 rows)
- `data/raw/eBL_Dictionary.csv` — Akkadian word definitions (19K entries)
- `data/raw/OA_Lexicon_eBL.csv` — Old Assyrian lexicon (39K entries)
- `data/raw/published_texts.csv` — cuneiform tablet metadata (8K texts)
- `data/raw/Sentences_Oare_FirstWord_LinNum.csv` — sentence-level data (9.8K rows)

### Outputs
- `outputs/t5-akkadian/checkpoint-*/` — model checkpoints with weights, optimizer state, tokenizer
- `submissions/` — CSV files formatted as `id,translation` for Kaggle submission

## GPU Setup (AMD Strix Halo / WSL2)

Local GPU: **AMD Radeon 8060S** (gfx1151 / RDNA 3.5), 15.8 GB VRAM, 40 CUs.

```bash
# Activate GPU environment before training
source scripts/gpu_env.sh
```

Required components:
- **ROCm 7.2** system install (`/opt/rocm`)
- **librocdxg v1.1.0** — DXG passthrough for WSL2 (installed as `libhsakmt.so.1`)
- **PyTorch nightly ROCm 7.2** — `pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2`
- **System HSA runtime** replacing bundled one in `_rocm_sdk_core/lib/`
- **HSA image shim** (`~/.local/lib/libhsa_image_shim.so`) — 4 stub functions for v2 image APIs

Key env vars: `HSA_ENABLE_DXG_DETECTION=1`, `HSA_ENABLE_SDMA=0`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

## Key Dependencies
- `transformers` + `datasets` (HuggingFace) — model training and data handling
- `sacrebleu` — BLEU and chrF++ scoring
- `torch` (ROCm/CUDA) — GPU backend
- `scikit-learn` — train/val splitting
- `pandas` — data I/O

## Competition Strategy

See the two reference documents in `docs/` for full details:
- **`docs/akkadian-llm-guide.md`** — model selection analysis, recommended data pipeline (5 phases), complete training data sources (ORACC, CDLI, HF datasets, cross-lingual resources), and research paper survey
- **`docs/akkadian-decoding-resources.md`** — curated catalog of datasets, models, corpora, tools, and papers for Akkadian NLP

### Approach
1. **Model ladder:** Start with `google/byt5-small` for fast iteration, then scale to `byt5-base` (580M) or `byt5-large` (1.2B) once the data pipeline is solid. ByT5's byte-level tokenization avoids BPE mismatch on Akkadian diacritics and Sumerograms.
2. **Fine-tuning with hyperx-ie:** Use the conceptor-algebra engine from `~/ai/memonix/hyperx/hyperx-ie` for behavioral steering during fine-tuning. hyperx-ie composes conceptors (soft projection matrices) via Boolean algebra (AND/OR/NOT) to steer model activations toward desired translation behaviors.
3. **Data assembly first:** Before serious training, build a complete normalized training set per the pipeline in `docs/akkadian-llm-guide.md` Part II §I:
   - Phase 1: Merge ORACC + HF datasets + competition data → deduplicate and normalize transliteration conventions
   - Phase 2: Augment vocabulary from ORACC glossaries, FactGrid lexemes, BabyFST inflections
   - Phase 3: Optional cross-lingual pretraining on Arabic/Hebrew → English
   - Phase 4: Multi-script augmentation with Sumerian → English pairs
   - Phase 5: Evaluate with BLEU, chrF++, geometric mean

## Domain Notes
- Training data is extremely small (1561 samples) — data augmentation and auxiliary data matter
- The auxiliary files (dictionary, lexicon, sentences) are underutilized in the baseline
- Transliterations use specialized notation: `{d}`, `{ki}` for determinatives; `[...]` for damaged/missing text; `_` for word boundaries
- Old Assyrian is a dialect of Akkadian focused on trade/commerce records
