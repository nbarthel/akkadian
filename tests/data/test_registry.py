# tests/data/test_registry.py
import pytest
from src.data.sources import list_sources, load_source


def test_list_sources_includes_kaggle():
    sources = list_sources()
    assert "kaggle" in sources


def test_list_sources_includes_all_adapters():
    sources = list_sources()
    expected = {"kaggle", "oare_sentences", "hf_phucthaiv02", "hf_cipher", "lexicon"}
    assert expected.issubset(set(sources))


def test_load_source_by_name(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        'oare_id,transliteration,translation\n'
        'abc,a-na,to\n'
    )
    df = load_source("kaggle", data_dir=tmp_path)
    assert len(df) == 1
