# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle competition entry for the **Deep Past Challenge** — translating Old Assyrian cuneiform transliterations to English. Competition runs until March 23, 2026. Submission is via Kaggle Notebooks (code competition).

**Evaluation metric:** geometric mean of BLEU and chrF++ → `√(BLEU × chrF++)`. Both metrics come from `sacrebleu`.

## Commands

### Training
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
Single-file pipeline using HuggingFace `Seq2SeqTrainer`:
1. `load_data()` — reads `train.csv` (1561 samples) and `test.csv` (4 samples) from `data/raw/`
2. 90/10 train/val split (`random_state=42`)
3. `preprocess_function()` — tokenizes with prefix `"translate Akkadian to English: "`, max 512 tokens
4. Training with gradient accumulation (4 steps), FP16 if CUDA, warmup 10%, weight decay 0.01
5. `compute_metrics()` — calculates BLEU, chrF, and `geo_mean` (the competition metric)
6. Beam search inference (5 beams, no 3-gram repeat) → writes `submissions/*.csv`

Best checkpoint selection uses `geo_mean` metric. Keeps top 2 checkpoints.

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
