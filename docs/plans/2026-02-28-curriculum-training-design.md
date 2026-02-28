# Curriculum Training Design for ByT5 Akkadian Translation

*2026-02-28*

## Overview

Upgrade `src/train_baseline.py` to support curriculum learning on the assembled 161K-pair dataset, targeting the Deep Past Kaggle competition (Old Assyrian → English, scored by √(BLEU × chrF++)).

**Hardware:** AMD Strix Halo, 96GB VRAM (128GB shared).

**Model ladder:** ByT5-small (300M) for pipeline validation → ByT5-large (1.2B) for final training.

## Training Phases

### Phase 1 — General Akkadian (all gold data)

- **Data:** 125,849 gold-quality sentence pairs (all dialects) from `train.parquet WHERE quality = 'gold'`
- **Purpose:** Learn general Akkadian→English translation patterns, vocabulary, grammar
- **Epochs:** 5
- **Learning rate:** 5e-5
- **Prefix:** `"translate Akkadian to English: "`

### Phase 2 — Old Assyrian Specialization

- **Data:** ~14,613 Old Assyrian pairs from `train.parquet WHERE dialect = 'old_assyrian'`
- **Purpose:** Specialize on competition domain (trade/commerce records)
- **Epochs:** 10
- **Learning rate:** 1e-5 (lower to avoid catastrophic forgetting)
- **Resume from:** Phase 1 best checkpoint

### Phase 3 — Lexicon Augmentation (optional)

- **Data:** Phase 2 data + 19,517 lexicon entries (word→definition pairs)
- **Purpose:** Expand vocabulary coverage for rare/unseen words
- **Epochs:** 2
- **Learning rate:** 5e-6
- **Resume from:** Phase 2 best checkpoint
- **Risk:** Word-level pairs differ structurally from sentences. Evaluate and keep/drop based on competition val score.

## Data Loading

Load from assembled parquet files in `data/processed/`:

```
Phase 1: train.parquet WHERE quality = 'gold'
Phase 2: train.parquet WHERE dialect = 'old_assyrian'
Phase 3: train.parquet WHERE dialect = 'old_assyrian' UNION lexicon entries
```

No re-assembly needed — filter in-memory from pre-built parquet.

## Evaluation

Two evaluation sets loaded simultaneously:

| Dataset | Source | Purpose |
|---------|--------|---------|
| `val_competition.parquet` (88 rows) | Kaggle Old Assyrian | Checkpoint selection (`metric_for_best_model`) |
| `val.parquet` (8,076 rows) | Full mixed | Monitoring for stability |

Metrics logged per epoch:
- `competition_bleu`, `competition_chrf`, `competition_geo_mean`
- `full_bleu`, `full_chrf`, `full_geo_mean`

Best model selected by `competition_geo_mean`.

## ByT5 Byte-Level Settings

ByT5 tokenizes at the byte level. Akkadian diacritics (š, ṣ, ṭ, ā, ē) use 2 bytes each in UTF-8. A 50-word transliteration (~250 chars) becomes ~350 byte tokens.

- **Max source length:** 1024 bytes (covers 99%+ of transliterations)
- **Max target length:** 1024 bytes

## Hyperparameters

### ByT5-small (pipeline validation)

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Learning rate | 5e-5 | 1e-5 | 5e-6 |
| Epochs | 5 | 10 | 2 |
| Batch size | 16 | 8 | 8 |
| Grad accumulation | 2 | 4 | 4 |
| Effective batch | 32 | 32 | 32 |
| Warmup ratio | 0.1 | 0.05 | 0.05 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Max source length | 1024 | 1024 | 512 |
| Max target length | 1024 | 1024 | 256 |

### ByT5-large (final training)

Same effective batch sizes. Reduce per-device batch size (4-8) and increase gradient accumulation to compensate. 96GB VRAM handles this comfortably.

### Generation (inference)

- Beam search: 5 beams
- No repeat ngram: 3
- Length penalty: 1.0
- Early stopping: True

## CLI Interface

```bash
# Phase 1: General Akkadian
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 1 \
  --model-name google/byt5-small \
  --epochs 5 --batch-size 16 --learning-rate 5e-5

# Phase 2: Old Assyrian specialization
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 2 \
  --resume-from outputs/byt5-curriculum/phase1_best \
  --epochs 10 --batch-size 8 --learning-rate 1e-5

# Phase 3: Lexicon augmentation (optional)
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 3 \
  --resume-from outputs/byt5-curriculum/phase2_best \
  --epochs 2 --batch-size 8 --learning-rate 5e-6
```

### New CLI flags

- `--phase {0,1,2,3}` — Training phase (0 = legacy behavior on data/raw/train.csv)
- `--resume-from <path>` — Resume from checkpoint directory
- `--max-source-length <int>` — Max byte tokens for source (default 1024)
- `--max-target-length <int>` — Max byte tokens for target (default 1024)

### Backward compatibility

`--phase 0` (or omitted) loads `data/raw/train.csv` with the original 90/10 split, matching the baseline behavior.

## Script Changes to `src/train_baseline.py`

1. **Data loading:** Replace `load_data()` with parquet-based loading + phase-specific filtering
2. **Dual evaluation:** `compute_metrics` evaluates both `val_competition` and `val` sets
3. **Checkpoint naming:** `phase{N}_best/` directories under `--output-dir`
4. **New flags:** `--phase`, `--resume-from`, `--max-source-length`, `--max-target-length`
5. **Submission:** Reads `data/raw/test.csv` for competition submission output
6. **Separate source/target max lengths:** ByT5 byte sequences need independent control

## Checkpoint Layout

```
outputs/byt5-curriculum/
├── phase1_best/          # Best Phase 1 checkpoint
├── phase1_checkpoints/   # All Phase 1 epoch checkpoints
├── phase2_best/          # Best Phase 2 checkpoint
├── phase2_checkpoints/
├── phase3_best/
├── phase3_checkpoints/
└── final/                # Final model for submission
```
