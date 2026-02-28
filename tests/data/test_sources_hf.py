# tests/data/test_sources_hf.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data.schema import validate_dataframe


class TestPhucthaiv02:
    def test_load_returns_valid_schema(self):
        from src.data.sources.hf_phucthaiv02 import load

        mock_ds_train = [
            {"transliteration": "a-na", "translation": "to", "pdf_name": "AKT.pdf", "page": 1},
            {"transliteration": "um-ma", "translation": "thus", "pdf_name": "AKT.pdf", "page": 2},
        ]
        mock_ds_align = [
            {"transliteration": "šu-ut", "translation": "he"},
        ]

        with patch("src.data.sources.hf_phucthaiv02.load_dataset") as mock_load:
            mock_load.side_effect = [mock_ds_train, mock_ds_align]
            df = load()

        validate_dataframe(df)
        assert len(df) == 3
        assert df.iloc[0]["source"] == "hf_phucthaiv02_translation"

    def test_load_skips_empty_transliterations(self):
        from src.data.sources.hf_phucthaiv02 import load

        mock_ds = [
            {"transliteration": "", "translation": "empty", "pdf_name": None, "page": None},
            {"transliteration": "a-na", "translation": "to", "pdf_name": None, "page": None},
        ]
        with patch("src.data.sources.hf_phucthaiv02.load_dataset") as mock_load:
            mock_load.side_effect = [mock_ds, []]
            df = load()

        assert len(df) == 1


class TestCipherLing:
    def test_load_extracts_nested_fields(self):
        from src.data.sources.hf_cipher import load

        mock_ds = [
            {
                "translation": {
                    "ak": "cuneiform...",
                    "en": "the king said",
                    "tr": "LUGAL iq-bi",
                }
            },
        ]
        with patch("src.data.sources.hf_cipher.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            df = load()

        validate_dataframe(df)
        assert len(df) == 1
        assert df.iloc[0]["transliteration"] == "LUGAL iq-bi"
        assert df.iloc[0]["translation"] == "the king said"

    def test_load_skips_entries_without_transliteration(self):
        from src.data.sources.hf_cipher import load

        mock_ds = [
            {"translation": {"ak": "cuneiform", "en": "the king", "tr": ""}},
            {"translation": {"ak": "cuneiform", "en": "he said", "tr": "iq-bi"}},
        ]
        with patch("src.data.sources.hf_cipher.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            df = load()

        assert len(df) == 1
