import pytest
import pandas as pd
from src.data.schema import SCHEMA_COLUMNS, validate_dataframe


def test_validate_accepts_valid_dataframe():
    df = pd.DataFrame({
        "transliteration": ["a-na"],
        "translation": ["to"],
        "source": ["test"],
        "dialect": ["old_assyrian"],
        "genre": ["trade"],
        "quality": ["gold"],
        "has_translation": [True],
        "multimodal_ref": [""],
    })
    result = validate_dataframe(df)
    assert len(result) == 1


def test_validate_rejects_missing_column():
    df = pd.DataFrame({"transliteration": ["a-na"]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataframe(df)


def test_validate_rejects_invalid_quality():
    df = pd.DataFrame({
        "transliteration": ["a-na"],
        "translation": ["to"],
        "source": ["test"],
        "dialect": ["old_assyrian"],
        "genre": ["trade"],
        "quality": ["invalid_tier"],
        "has_translation": [True],
        "multimodal_ref": [""],
    })
    with pytest.raises(ValueError, match="quality"):
        validate_dataframe(df)
