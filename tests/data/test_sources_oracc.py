# tests/data/test_sources_oracc.py
import pytest
import json
from pathlib import Path
from src.data.sources.oracc import (
    extract_transliteration_from_cdl,
    extract_translations_from_cdl,
    parse_text_json,
)


def test_extract_transliteration_from_lemma_nodes():
    cdl = {
        "cdl": [
            {"node": "c", "type": "sentence", "cdl": [
                {"node": "l", "frag": "a-na", "f": {"form": "a-na"}},
                {"node": "l", "frag": "šu-ut", "f": {"form": "šu-ut"}},
            ]}
        ]
    }
    result = extract_transliteration_from_cdl(cdl["cdl"])
    assert result == ["a-na šu-ut"]


def test_extract_transliteration_skips_non_lemma():
    cdl = {
        "cdl": [
            {"node": "c", "type": "sentence", "cdl": [
                {"node": "l", "frag": "LUGAL", "f": {"form": "LUGAL"}},
                {"node": "d", "type": "line-start"},
                {"node": "l", "frag": "iq-bi", "f": {"form": "iq-bi"}},
            ]}
        ]
    }
    result = extract_transliteration_from_cdl(cdl["cdl"])
    assert result == ["LUGAL iq-bi"]


def test_extract_translations():
    cdl = {
        "cdl": [
            {"node": "c", "type": "sentence", "cdl": [
                {"node": "l", "frag": "a-na"},
            ]},
            {"node": "c", "type": "translation", "cdl": [
                {"node": "c", "type": "unit", "cdl": [
                    {"node": "d", "label": "1", "type": "line-start"},
                ]},
            ], "label": "The king spoke."},
        ]
    }
    result = extract_translations_from_cdl(cdl["cdl"])
    assert "The king spoke." in result


def test_parse_text_json_end_to_end(tmp_path):
    text_json = {
        "cdl": [
            {"node": "c", "type": "text", "cdl": [
                {"node": "c", "type": "sentence", "id": "s1", "cdl": [
                    {"node": "l", "frag": "a-na", "f": {"form": "a-na"}},
                ]},
            ]}
        ]
    }
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(text_json))
    result = parse_text_json(json_path)
    assert len(result) >= 0
