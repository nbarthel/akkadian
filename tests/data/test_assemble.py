# tests/data/test_assemble.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.assemble import deduplicate, split_dataset, run_pipeline
from src.data.schema import SCHEMA_COLUMNS


def _make_df(rows):
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        df = _make_df([
            {"transliteration": "a-na", "translation": "to", "source": "kaggle_train",
             "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
            {"transliteration": "a-na", "translation": "to", "source": "hf_phucthaiv02_translation",
             "dialect": "unknown", "genre": "unknown", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
        ])
        result = deduplicate(df)
        assert len(result) == 1

    def test_keeps_higher_priority_source(self):
        df = _make_df([
            {"transliteration": "a-na", "translation": "to", "source": "hf_phucthaiv02_translation",
             "dialect": "unknown", "genre": "unknown", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
            {"transliteration": "a-na", "translation": "to", "source": "kaggle_train",
             "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
        ])
        result = deduplicate(df)
        assert result.iloc[0]["source"] == "kaggle_train"

    def test_keeps_different_translations(self):
        df = _make_df([
            {"transliteration": "a-na", "translation": "to", "source": "kaggle_train",
             "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
            {"transliteration": "a-na", "translation": "for", "source": "hf_phucthaiv02_translation",
             "dialect": "unknown", "genre": "unknown", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""},
        ])
        result = deduplicate(df)
        assert len(result) == 2


class TestSplitDataset:
    def test_split_proportions(self):
        rows = [
            {"transliteration": f"word-{i}", "translation": f"def-{i}", "source": "test",
             "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""}
            for i in range(100)
        ]
        df = _make_df(rows)
        train, val, test = split_dataset(df)
        assert len(train) == 90
        assert len(val) == 5
        assert len(test) == 5

    def test_no_overlap_between_splits(self):
        rows = [
            {"transliteration": f"word-{i}", "translation": f"def-{i}", "source": "test",
             "dialect": "old_assyrian", "genre": "trade", "quality": "gold",
             "has_translation": True, "multimodal_ref": ""}
            for i in range(100)
        ]
        df = _make_df(rows)
        train, val, test = split_dataset(df)
        train_set = set(train["transliteration"])
        val_set = set(val["transliteration"])
        test_set = set(test["transliteration"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
