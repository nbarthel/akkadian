# Curriculum Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade `src/train_baseline.py` to support 3-phase curriculum learning on the assembled 161K-pair dataset, with dual validation (competition + full) and phase-aware checkpointing.

**Architecture:** Extend the existing single-file training script with new data loading (parquet-based, phase-filtered), separate source/target max lengths for ByT5 byte sequences, dual assessment using val_competition as primary, and `--phase`/`--resume-from` CLI flags for curriculum phasing.

**Tech Stack:** transformers (Seq2SeqTrainer), datasets, pandas, sacrebleu, torch (ROCm)

## Key Data Facts

- `data/processed/train.parquet`: 145,366 rows, columns: transliteration, translation, source, dialect, genre, quality, has_translation, multimodal_ref
- `data/processed/val.parquet`: 8,076 rows (same schema)
- `data/processed/val_competition.parquet`: 88 rows (kaggle-source Old Assyrian only)
- `data/raw/test.csv`: 4 rows, columns: id, text_id, line_start, line_end, transliteration
- Phase 1 filter: `quality == 'gold'` -> 125,849 rows
- Phase 2 filter: `dialect == 'old_assyrian'` -> 14,613 rows (8,900 gold + 5,713 lexicon)
- Phase 3 filter: Phase 2 data + remaining lexicon entries
- Phase 0 (legacy): reads `data/raw/train.csv` with 90/10 split

---

### Task 1: Phase-based data loading function

**Files:**
- Modify: `src/train_baseline.py` (add `load_phase_data` function)
- Create: `tests/test_train_data_loading.py`

**Step 1: Write the failing tests**

Create `tests/test_train_data_loading.py`:

```python
"""Tests for phase-based data loading in train_baseline."""
import pandas as pd
import pytest
from pathlib import Path

from src.train_baseline import load_phase_data


@pytest.fixture
def processed_dir(tmp_path):
    """Create minimal parquet files mimicking data/processed/."""
    rows = [
        {"transliteration": "a-na", "translation": "to", "source": "kaggle_train",
         "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
         "has_translation": True, "multimodal_ref": None},
        {"transliteration": "sha", "translation": "of", "source": "hf_phucthaiv02_translation",
         "dialect": "unknown", "genre": "unknown", "quality": "gold",
         "has_translation": True, "multimodal_ref": None},
        {"transliteration": "sharrum", "translation": "king", "source": "ebl_dictionary",
         "dialect": "unknown", "genre": "unknown", "quality": "lexicon",
         "has_translation": True, "multimodal_ref": None},
        {"transliteration": "i-di-in", "translation": "he gave", "source": "oare_sentences",
         "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
         "has_translation": True, "multimodal_ref": None},
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "train.parquet", index=False)

    val_rows = [rows[0], rows[1]]
    pd.DataFrame(val_rows).to_parquet(tmp_path / "val.parquet", index=False)

    comp_rows = [rows[0]]
    pd.DataFrame(comp_rows).to_parquet(tmp_path / "val_competition.parquet", index=False)

    return tmp_path


def test_phase1_loads_gold_only(processed_dir):
    train_df, val_df, comp_df = load_phase_data(processed_dir, phase=1)
    assert len(train_df) == 3  # 3 gold rows
    assert (train_df["quality"] == "gold").all()
    assert len(comp_df) == 1


def test_phase2_loads_old_assyrian(processed_dir):
    train_df, val_df, comp_df = load_phase_data(processed_dir, phase=2)
    assert len(train_df) == 2  # old_assyrian rows only
    assert (train_df["dialect"] == "old_assyrian").all()


def test_phase3_loads_old_assyrian_plus_lexicon(processed_dir):
    train_df, val_df, comp_df = load_phase_data(processed_dir, phase=3)
    # old_assyrian (2) + remaining lexicon (1) = 3
    assert len(train_df) == 3


def test_phase0_is_none(processed_dir):
    """Phase 0 returns None -- caller uses legacy load_data()."""
    result = load_phase_data(processed_dir, phase=0)
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_data_loading.py -v`
Expected: FAIL with ImportError (load_phase_data doesn't exist)

**Step 3: Implement load_phase_data**

Add to `src/train_baseline.py` after the existing `load_data` function:

```python
def load_phase_data(data_dir: Path, phase: int):
    """Load parquet data filtered by curriculum phase.

    Returns (train_df, val_df, val_competition_df) or None for phase 0.
    """
    if phase == 0:
        return None

    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    comp_df = pd.read_parquet(data_dir / 'val_competition.parquet')

    if phase == 1:
        train_df = train_df[train_df['quality'] == 'gold'].reset_index(drop=True)
    elif phase == 2:
        train_df = train_df[train_df['dialect'] == 'old_assyrian'].reset_index(drop=True)
    elif phase == 3:
        oa = train_df[train_df['dialect'] == 'old_assyrian']
        lex = train_df[(train_df['quality'] == 'lexicon') & (train_df['dialect'] != 'old_assyrian')]
        train_df = pd.concat([oa, lex], ignore_index=True)

    return train_df, val_df, comp_df
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_data_loading.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add tests/test_train_data_loading.py src/train_baseline.py
git commit -m "feat: add phase-based data loading for curriculum training"
```

---

### Task 2: Separate source/target tokenization

**Files:**
- Modify: `src/train_baseline.py` (update `preprocess_function`)
- Modify: `tests/test_train_data_loading.py` (add tokenization tests)

**Step 1: Write the failing test**

Add to `tests/test_train_data_loading.py`:

```python
from src.train_baseline import preprocess_function


def test_preprocess_separate_lengths():
    """Source and target can have different max lengths."""
    examples = {
        "transliteration": ["a-na bi-tim"],
        "translation": ["to the house"],
    }
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    prefix = "translate Akkadian to English: "

    result = preprocess_function(
        examples, tokenizer, prefix,
        max_source_length=64, max_target_length=32,
    )
    assert len(result["input_ids"][0]) == 64
    assert len(result["labels"][0]) == 32
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_data_loading.py::test_preprocess_separate_lengths -v`
Expected: FAIL (preprocess_function doesn't accept max_source_length/max_target_length)

**Step 3: Update preprocess_function**

Replace the existing `preprocess_function` in `src/train_baseline.py`:

```python
def preprocess_function(examples, tokenizer, prefix, max_length=512,
                        max_source_length=None, max_target_length=None):
    """Tokenize inputs and targets with separate max lengths."""
    src_len = max_source_length or max_length
    tgt_len = max_target_length or max_length

    inputs = [prefix + text for text in examples['transliteration']]
    targets = examples['translation']

    model_inputs = tokenizer(
        inputs,
        max_length=src_len,
        truncation=True,
        padding='max_length'
    )

    labels = tokenizer(
        targets,
        max_length=tgt_len,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs
```

**Step 4: Run tests to verify all pass**

Run: `pytest tests/test_train_data_loading.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/train_baseline.py tests/test_train_data_loading.py
git commit -m "feat: support separate source/target max lengths for ByT5"
```

---

### Task 3: Scoring helper function

**Files:**
- Modify: `src/train_baseline.py` (add `score_predictions`)
- Modify: `tests/test_train_data_loading.py` (add scoring tests)

**Step 1: Write the failing test**

Add to `tests/test_train_data_loading.py`:

```python
from src.train_baseline import score_predictions


def test_score_predictions():
    """score_predictions computes bleu, chrf, geo_mean."""
    preds = ["the king gave silver"]
    refs = ["the king gave silver"]
    metrics = score_predictions(preds, refs, prefix="competition")
    assert "competition_bleu" in metrics
    assert "competition_chrf" in metrics
    assert "competition_geo_mean" in metrics
    assert metrics["competition_bleu"] > 0
    assert metrics["competition_geo_mean"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_data_loading.py::test_score_predictions -v`
Expected: FAIL (score_predictions doesn't exist)

**Step 3: Implement score_predictions**

Add to `src/train_baseline.py`:

```python
def score_predictions(predictions, references, prefix=""):
    """Compute BLEU, chrF++, and geo_mean on a list of prediction/reference strings."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    bleu_score = bleu.corpus_score(predictions, [references]).score
    chrf_score = chrf.corpus_score(predictions, [references]).score
    geo_mean = np.sqrt(max(bleu_score, 0) * max(chrf_score, 0))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}bleu": bleu_score,
        f"{p}chrf": chrf_score,
        f"{p}geo_mean": geo_mean,
    }
```

**Step 4: Run tests to verify all pass**

Run: `pytest tests/test_train_data_loading.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add src/train_baseline.py tests/test_train_data_loading.py
git commit -m "feat: add score_predictions for dual scoring support"
```

---

### Task 4: CLI arguments and training loop integration

**Files:**
- Modify: `src/train_baseline.py` (update `main()` and `train()`)

This task integrates all pieces into the training loop. The training loop relies on HuggingFace Trainer internals which are impractical to unit test; validation happens via the smoke test in Task 5.

**Step 1: Update argparse in main()**

Add these arguments to the parser in `main()`:

```python
parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2, 3],
                    help='Curriculum phase: 0=legacy, 1=all gold, 2=old assyrian, 3=+lexicon')
parser.add_argument('--resume-from', type=str, default=None,
                    help='Resume from checkpoint directory')
parser.add_argument('--max-source-length', type=int, default=1024,
                    help='Max source sequence length in bytes (ByT5)')
parser.add_argument('--max-target-length', type=int, default=1024,
                    help='Max target sequence length in bytes (ByT5)')
parser.add_argument('--warmup-ratio', type=float, default=0.1,
                    help='Warmup ratio')
```

**Step 2: Rewrite train() function**

Replace the `train()` function with phase-aware logic. Key changes:

1. Data loading: If `phase > 0`, use `load_phase_data()` instead of `load_data()`
2. Model loading: If `--resume-from`, load model/tokenizer from that path
3. Tokenization: Pass `max_source_length` and `max_target_length` separately
4. Primary assessment dataset: Use `val_competition` for checkpoint selection when `phase > 0`
5. Custom callback: After each assessment, also score on full val set and log
6. Output dirs: Save to `phase{N}_checkpoints/` and `phase{N}_best/`
7. Submission: Always read `data/raw/test.csv` for competition submission

See design doc `docs/plans/2026-02-28-curriculum-training-design.md` for full train() implementation.

The callback class `FullValCallback(TrainerCallback)` runs prediction on the full val set after each assessment step and logs full_val_bleu, full_val_chrf, full_val_geo_mean via `score_predictions()`.

**Step 3: Verify existing tests still pass**

Run: `pytest tests/test_train_data_loading.py -v`
Expected: 6 PASSED

**Step 4: Commit**

```bash
git add src/train_baseline.py
git commit -m "feat: integrate curriculum phases into training loop"
```

---

### Task 5: Smoke test the training pipeline

**Files:**
- Modify: `tests/test_train_data_loading.py` (add integration test)

**Step 1: Write smoke test**

Add to `tests/test_train_data_loading.py`:

```python
@pytest.mark.integration
def test_phase1_smoke(processed_dir, tmp_path):
    """Smoke test: Phase 1 training runs without errors on tiny data."""
    import sys
    from unittest.mock import patch

    output_dir = tmp_path / "output"
    sub_dir = tmp_path / "submissions"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({
        "id": [0], "text_id": ["abc"], "line_start": [1],
        "line_end": [2], "transliteration": ["a-na bi-tim"]
    }).to_csv(raw_dir / "test.csv", index=False)

    args = [
        "train_baseline.py",
        "--data-dir", str(processed_dir),
        "--output-dir", str(output_dir),
        "--submission-dir", str(sub_dir),
        "--model-name", "google/byt5-small",
        "--phase", "1",
        "--epochs", "1",
        "--batch-size", "2",
        "--max-source-length", "32",
        "--max-target-length", "32",
        "--learning-rate", "5e-5",
    ]
    with patch.object(sys, "argv", args):
        from src.train_baseline import main
        main()

    assert (output_dir / "phase1_best").exists()
```

Note: The smoke test needs `data/raw/test.csv` to exist at the hardcoded path in the script. The test creates a fake one in tmp_path, but we need to patch the raw_dir path. Either adjust the test to mock the path or add a `--raw-dir` flag. The simplest fix: update the script to derive raw_dir from the parent of data_dir (e.g., `data_dir.parent / 'raw'` when phase > 0).

**Step 2: Run smoke test**

Run: `pytest tests/test_train_data_loading.py::test_phase1_smoke -v -s -m integration`
Expected: PASS (trains 1 epoch on 3 samples, ~30-60s)

**Step 3: Commit**

```bash
git add tests/test_train_data_loading.py
git commit -m "test: add smoke test for curriculum training pipeline"
```

---

### Task 6: Run Phase 1 training (ByT5-small)

**Files:** None (execution only)

**Step 1: Run Phase 1**

```bash
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 1 \
  --model-name google/byt5-small \
  --epochs 5 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1 \
  --max-source-length 1024 \
  --max-target-length 1024
```

Expected: Trains on ~126K gold samples for 5 epochs. Saves best model to `outputs/byt5-curriculum/phase1_best/`.

**Step 2: Verify output**

```bash
ls outputs/byt5-curriculum/phase1_best/
cat submissions/byt5_phase1.csv
```

**Step 3: Record Phase 1 metrics**

Note the final `competition_geo_mean` -- this is the baseline for Phase 2 comparison.

---

### Task 7: Run Phase 2 training (Old Assyrian specialization)

**Files:** None (execution only)

**Step 1: Run Phase 2**

```bash
python src/train_baseline.py \
  --data-dir data/processed \
  --output-dir outputs/byt5-curriculum \
  --phase 2 \
  --resume-from outputs/byt5-curriculum/phase1_best \
  --epochs 10 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.05 \
  --max-source-length 1024 \
  --max-target-length 1024
```

Expected: Fine-tunes on ~15K Old Assyrian samples. Competition geo_mean should improve over Phase 1.

---

### Task 8: Update CLAUDE.md with curriculum training commands

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add curriculum training commands under the Commands section, alongside the existing baseline command.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add curriculum training commands to CLAUDE.md"
```
