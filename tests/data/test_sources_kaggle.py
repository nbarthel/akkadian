# tests/data/test_sources_kaggle.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.sources.kaggle import load
from src.data.schema import SCHEMA_COLUMNS, validate_dataframe


def test_load_returns_valid_schema(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        'oare_id,transliteration,translation\n'
        'abc-123,a-na šu-ut,he said\n'
        'def-456,KIŠIB {d}UTU,Seal of Šamaš\n'
    )
    df = load(data_dir=tmp_path)
    validate_dataframe(df)
    assert len(df) == 2
    assert set(df.columns) == set(SCHEMA_COLUMNS)


def test_load_tags_correctly(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        'oare_id,transliteration,translation\n'
        'abc-123,a-na,to\n'
    )
    df = load(data_dir=tmp_path)
    row = df.iloc[0]
    assert row["source"] == "kaggle_train"
    assert row["dialect"] == "old_assyrian"
    assert row["genre"] == "trade"
    assert row["quality"] == "gold"
    assert row["has_translation"] is True


def test_load_normalizes_transliteration(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        'oare_id,transliteration,translation\n'
        'abc-123,sza-ru-um,king\n'
    )
    df = load(data_dir=tmp_path)
    assert df.iloc[0]["transliteration"] == "ša-ru-um"
