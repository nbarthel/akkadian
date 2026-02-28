"""Tests for phase-based data loading in train_baseline."""
import pandas as pd
import pytest
from pathlib import Path

from src.train_baseline import load_phase_data, score_predictions


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


def test_preprocess_separate_lengths():
    """Source and target can have different max lengths."""
    from src.train_baseline import preprocess_function
    from transformers import AutoTokenizer

    examples = {
        "transliteration": ["a-na bi-tim"],
        "translation": ["to the house"],
    }
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    prefix = "translate Akkadian to English: "

    result = preprocess_function(
        examples, tokenizer, prefix,
        max_source_length=64, max_target_length=32,
    )
    assert len(result["input_ids"][0]) == 64
    assert len(result["labels"][0]) == 32


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


@pytest.mark.integration
def test_phase1_smoke(processed_dir, tmp_path):
    """Smoke test: Phase 1 training runs without errors on tiny data."""
    import sys
    from unittest.mock import patch

    output_dir = tmp_path / "output"
    sub_dir = tmp_path / "submissions"

    # Create fake raw dir with test.csv (train() reads it for submission)
    raw_dir = processed_dir.parent / "raw"
    if not raw_dir.exists():
        # processed_dir is tmp_path-based, so place raw as sibling
        raw_dir = processed_dir / ".." / "raw"
        raw_dir = raw_dir.resolve()
        raw_dir.mkdir(parents=True, exist_ok=True)
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
    assert (sub_dir / "byt5_phase1.csv").exists()
