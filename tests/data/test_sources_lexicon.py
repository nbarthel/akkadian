# tests/data/test_sources_lexicon.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.sources.lexicon import load
from src.data.schema import validate_dataframe


def test_load_ebl_dictionary(tmp_path):
    dict_csv = tmp_path / "eBL_Dictionary.csv"
    dict_csv.write_text(
        'word,definition,derived_from\n'
        'šarrum,"king; ruler",\n'
        'awīlum,"man; gentleman",cf. awīlu\n'
    )
    lex_csv = tmp_path / "OA_Lexicon_eBL.csv"
    lex_csv.write_text('type,form,norm,lexeme,eBL,I_IV,A_D,Female(f),Alt_lex\n')

    df = load(data_dir=tmp_path)
    validate_dataframe(df)
    assert len(df) == 2
    assert df.iloc[0]["quality"] == "lexicon"
    assert df.iloc[0]["transliteration"] == "šarrum"
    assert "king" in df.iloc[0]["translation"]


def test_load_oa_lexicon(tmp_path):
    dict_csv = tmp_path / "eBL_Dictionary.csv"
    dict_csv.write_text('word,definition,derived_from\n')

    lex_csv = tmp_path / "OA_Lexicon_eBL.csv"
    lex_csv.write_text(
        'type,form,norm,lexeme,eBL,I_IV,A_D,Female(f),Alt_lex\n'
        'word,um-ma,umma,umma,https://ebl.lmu.de/dictionary?word=umma,,,,\n'
    )
    df = load(data_dir=tmp_path)
    assert len(df) >= 1


def test_load_skips_empty_definitions(tmp_path):
    dict_csv = tmp_path / "eBL_Dictionary.csv"
    dict_csv.write_text(
        'word,definition,derived_from\n'
        'šarrum,"king",\n'
        'empty,,\n'
    )
    lex_csv = tmp_path / "OA_Lexicon_eBL.csv"
    lex_csv.write_text('type,form,norm,lexeme,eBL,I_IV,A_D,Female(f),Alt_lex\n')

    df = load(data_dir=tmp_path)
    assert len(df) == 1
