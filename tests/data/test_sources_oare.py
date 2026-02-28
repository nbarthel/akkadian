# tests/data/test_sources_oare.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.sources.oare_sentences import load, _segment_sentences
from src.data.schema import validate_dataframe


def _write_test_files(tmp_path):
    sentences = tmp_path / "Sentences_Oare_FirstWord_LinNum.csv"
    sentences.write_text(
        'display_name,text_uuid,sentence_uuid,sentence_obj_in_text,translation,'
        'first_word_transcription,first_word_spelling,first_word_number,'
        'first_word_obj_in_text,line_number,side,column\n'
        'Test Text,text-001,sent-001,1,He said to him.,umma,um-ma,1,1,1.0,1,1\n'
        'Test Text,text-001,sent-002,2,Give me silver.,iddinam,i-dí-nam,4,4,2.0,1,1\n'
    )
    published = tmp_path / "published_texts.csv"
    published.write_text(
        'oare_id,online transcript,cdli_id,aliases,label,publication_catalog,'
        'description,genre_label,inventory_position,online_catalog,note,'
        'interlinear_commentary,online_information,excavation_no,oatp_key,'
        'eBL_id,AICC_translation,transliteration_orig,transliteration\n'
        'text-001,,,,,,,letter,,,,,,,,,,,'
        'um-ma šu-ut-ma a-na i-dí-nam KÙ.BABBAR\n'
    )
    return tmp_path


def test_segment_sentences():
    words = ["um-ma", "šu-ut-ma", "a-na", "i-dí-nam", "KÙ.BABBAR"]
    word_starts = [1, 4]
    segments = _segment_sentences(words, word_starts)
    assert segments == ["um-ma šu-ut-ma a-na", "i-dí-nam KÙ.BABBAR"]


def test_load_returns_valid_schema(tmp_path):
    _write_test_files(tmp_path)
    df = load(data_dir=tmp_path)
    validate_dataframe(df)
    assert len(df) == 2


def test_load_extracts_sentence_transliterations(tmp_path):
    _write_test_files(tmp_path)
    df = load(data_dir=tmp_path)
    assert df.iloc[0]["transliteration"] == "um-ma šu-ut-ma a-na"
    assert df.iloc[1]["transliteration"] == "i-dí-nam KÙ.BABBAR"


def test_load_skips_sentences_without_published_text(tmp_path):
    sentences = tmp_path / "Sentences_Oare_FirstWord_LinNum.csv"
    sentences.write_text(
        'display_name,text_uuid,sentence_uuid,sentence_obj_in_text,translation,'
        'first_word_transcription,first_word_spelling,first_word_number,'
        'first_word_obj_in_text,line_number,side,column\n'
        'Orphan,no-match,sent-999,1,No match.,foo,foo,1,1,1.0,1,1\n'
    )
    published = tmp_path / "published_texts.csv"
    published.write_text(
        'oare_id,online transcript,cdli_id,aliases,label,publication_catalog,'
        'description,genre_label,inventory_position,online_catalog,note,'
        'interlinear_commentary,online_information,excavation_no,oatp_key,'
        'eBL_id,AICC_translation,transliteration_orig,transliteration\n'
    )
    df = load(data_dir=tmp_path)
    assert len(df) == 0
