# Dataset Assembly Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular, reproducible pipeline that assembles ~80K+ Akkadian transliteration-English parallel pairs from competition data, HuggingFace datasets, local auxiliary files, and ORACC, with Unicode normalization, deduplication, and dialect/genre/quality tagging.

**Architecture:** Source-per-module pipeline. Each data source has an adapter in `src/data/sources/` producing a standardized DataFrame. A central `assemble.py` orchestrates: load all → normalize → deduplicate → split → write Parquet. Raw downloads cached in `data/external/`.

**Tech Stack:** Python 3, pandas, pyarrow (Parquet), HuggingFace `datasets`, `requests` (ORACC), `unicodedata`, `pytest`

**Key data facts discovered during research:**
- `phucthaiv02/akkadian-translation` train split: 72,653 pairs (`transliteration`, `translation`, `pdf_name`, `page`). Its validation split IS the 1,561 kaggle train.csv rows — dedup required.
- `phucthaiv02/akkadian_english_sentences_alignment_2`: 10,677 pairs (`transliteration`, `translation`)
- `cipher-ling/akkadian`: 50,478 entries with NESTED dict `translation: {ak: cuneiform, en: english, tr: transliteration}` — need to extract `tr`→`en` pairs
- `veezbo/akkadian_english_corpus`: 9,720 entries of ENGLISH TEXT ONLY (no transliterations) — usable only for back-translation, skip for now
- `mik3ml/akkadian`: gated, requires manual HF access request
- OARE sentences (9,782 rows): have translations + first_word_spelling but NOT full transliterations. Published_texts has full text transliterations. Can reconstruct sentence-level transliterations by joining on `text_uuid`=`oare_id` and segmenting by `first_word_number` positions. Word positions VERIFIED to align correctly.
- Published_texts AICC_translation column contains URLs, not actual translations.

---

### Task 1: Schema and project structure

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/schema.py`
- Create: `src/data/sources/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/data/test_schema.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/nbarthel/ai/kaggle/akkadian && python -m pytest tests/data/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.schema'`

**Step 3: Write minimal implementation**

```python
# src/data/__init__.py
# (empty)

# src/data/sources/__init__.py
# (empty)

# tests/__init__.py
# (empty)

# tests/data/__init__.py
# (empty)
```

```python
# src/data/schema.py
"""Unified schema for the Akkadian dataset assembly pipeline."""
import pandas as pd

SCHEMA_COLUMNS = [
    "transliteration",
    "translation",
    "source",
    "dialect",
    "genre",
    "quality",
    "has_translation",
    "multimodal_ref",
]

VALID_DIALECTS = {"old_assyrian", "neo_assyrian", "old_babylonian", "middle_babylonian", "unknown"}
VALID_GENRES = {"trade", "royal_inscription", "literary", "legal", "letter", "administrative", "unknown"}
VALID_QUALITIES = {"gold", "silver", "lexicon"}


def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


def make_row(
    transliteration: str,
    translation: str,
    source: str,
    dialect: str = "unknown",
    genre: str = "unknown",
    quality: str = "gold",
    has_translation: bool = True,
    multimodal_ref: str = "",
) -> dict:
    """Create a single row dict conforming to the schema."""
    return {
        "transliteration": transliteration,
        "translation": translation,
        "source": source,
        "dialect": dialect,
        "genre": genre,
        "quality": quality,
        "has_translation": has_translation,
        "multimodal_ref": multimodal_ref,
    }


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the unified schema.

    Raises ValueError if columns are missing or values are invalid.
    Returns the DataFrame unchanged if valid.
    """
    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    invalid_dialect = set(df["dialect"].unique()) - VALID_DIALECTS
    if invalid_dialect:
        raise ValueError(f"Invalid dialect values: {invalid_dialect}")

    invalid_genre = set(df["genre"].unique()) - VALID_GENRES
    if invalid_genre:
        raise ValueError(f"Invalid genre values: {invalid_genre}")

    invalid_quality = set(df["quality"].unique()) - VALID_QUALITIES
    if invalid_quality:
        raise ValueError(f"Invalid quality values: {invalid_quality}")

    return df
```

**Step 4: Run test to verify it passes**

Run: `cd /home/nbarthel/ai/kaggle/akkadian && python -m pytest tests/data/test_schema.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/data/__init__.py src/data/schema.py src/data/sources/__init__.py tests/__init__.py tests/data/__init__.py tests/data/test_schema.py
git commit -m "feat(data): add unified schema with validation"
```

---

### Task 2: Unicode normalizer

**Files:**
- Create: `src/data/normalize.py`
- Create: `tests/data/test_normalize.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_normalize.py
import pytest
from src.data.normalize import normalize_transliteration


class TestAsciiToDiacritics:
    def test_sz_to_shin(self):
        assert normalize_transliteration("sza-ru-um") == "ša-ru-um"

    def test_s_comma_to_sade(self):
        assert normalize_transliteration("s,a-lim") == "ṣa-lim"

    def test_t_comma_to_tet(self):
        assert normalize_transliteration("t,up-pu") == "ṭup-pu"

    def test_preserves_existing_diacritics(self):
        assert normalize_transliteration("ša-ru-um") == "ša-ru-um"


class TestSubscriptDigits:
    def test_ascii_digit_to_subscript(self):
        assert normalize_transliteration("du3") == "du₃"

    def test_preserves_existing_subscript(self):
        assert normalize_transliteration("du₃") == "du₃"

    def test_bi4_to_subscript(self):
        assert normalize_transliteration("bi4-ma") == "bi₄-ma"

    def test_no_subscript_for_standalone_numbers(self):
        # Numbers not attached to syllables should stay as-is
        assert normalize_transliteration("10 ma-na") == "10 ma-na"

    def test_no_subscript_for_sign_index_1(self):
        # Index 1 is the default reading — never written as subscript
        assert normalize_transliteration("du1") == "du1"


class TestSumerograms:
    def test_determinatives_lowercase_braced(self):
        assert normalize_transliteration("{D}UTU") == "{d}UTU"

    def test_preserves_lowercase_determinatives(self):
        assert normalize_transliteration("{d}UTU") == "{d}UTU"

    def test_logograms_stay_uppercase(self):
        assert normalize_transliteration("LUGAL") == "LUGAL"


class TestUnicodeNormalization:
    def test_nfc_normalization(self):
        # Decomposed š (s + combining caron) → precomposed š
        decomposed = "s\u030c"  # s + combining caron
        result = normalize_transliteration(decomposed)
        assert result == "š"

    def test_whitespace_normalization(self):
        assert normalize_transliteration("a-na  bi4-ma") == "a-na bi₄-ma"


class TestRealWorldExamples:
    def test_competition_sample(self):
        """Should not change already-normalized competition data."""
        text = "KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-(d)IM"
        assert normalize_transliteration(text) == text

    def test_mixed_conventions(self):
        text = "sza-ru-um LUGAL du3 {D}UTU"
        expected = "ša-ru-um LUGAL du₃ {d}UTU"
        assert normalize_transliteration(text) == expected
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/data/normalize.py
"""Unicode normalization for Akkadian transliterations."""
import re
import unicodedata

# ASCII to Unicode diacritic mappings
_ASCII_REPLACEMENTS = [
    ("sz", "š"),
    ("SZ", "Š"),
    ("s,", "ṣ"),
    ("S,", "Ṣ"),
    ("t,", "ṭ"),
    ("T,", "Ṭ"),
]

# Subscript digit mapping (2-9 only; 1 is default reading, never subscripted)
_SUBSCRIPT_DIGITS = str.maketrans("23456789", "₂₃₄₅₆₇₈₉")

# Pattern: a syllable (lowercase letters) immediately followed by a digit 2-9
# Must NOT match standalone numbers like "10" or "0.33333"
_SUBSCRIPT_PATTERN = re.compile(r"(?<=[a-zšṣṭḫāēīū])([2-9])(?=\b|[-\s.,;:])")

# Pattern: uppercase determinative braces like {D} or {KI}
_DETERMINATIVE_PATTERN = re.compile(r"\{([A-Z]+)\}")

# Known determinatives (lowercase when braced)
_DETERMINATIVES = {"d", "ki", "f", "m", "lu", "na4", "giš", "tug2", "kuš", "urudu"}


def _replace_ascii_diacritics(text: str) -> str:
    for ascii_form, unicode_form in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_form, unicode_form)
    return text


def _subscript_digits(text: str) -> str:
    def _sub(match: re.Match) -> str:
        return match.group(1).translate(_SUBSCRIPT_DIGITS)
    return _SUBSCRIPT_PATTERN.sub(_sub, text)


def _normalize_determinatives(text: str) -> str:
    def _det_lower(match: re.Match) -> str:
        content = match.group(1).lower()
        return "{" + content + "}"
    return _DETERMINATIVE_PATTERN.sub(_det_lower, text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_transliteration(text: str) -> str:
    """Normalize an Akkadian transliteration to Unicode with diacritics.

    Applies in order:
    1. ASCII to Unicode diacritics (sz→š, s,→ṣ, t,→ṭ)
    2. Subscript digits on syllables (du3→du₃, but not standalone numbers)
    3. Determinative braces to lowercase ({D}→{d})
    4. NFC Unicode normalization
    5. Whitespace normalization
    """
    text = _replace_ascii_diacritics(text)
    text = _subscript_digits(text)
    text = _normalize_determinatives(text)
    text = unicodedata.normalize("NFC", text)
    text = _normalize_whitespace(text)
    return text
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_normalize.py -v`
Expected: all PASSED

**Step 5: Commit**

```bash
git add src/data/normalize.py tests/data/test_normalize.py
git commit -m "feat(data): add Unicode normalizer for Akkadian transliterations"
```

---

### Task 3: Kaggle source adapter

**Files:**
- Create: `src/data/sources/kaggle.py`
- Create: `tests/data/test_sources_kaggle.py`

**Step 1: Write the failing test**

```python
# tests/data/test_sources_kaggle.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.sources.kaggle import load
from src.data.schema import SCHEMA_COLUMNS, validate_dataframe


def test_load_returns_valid_schema(tmp_path):
    # Create minimal train.csv
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_sources_kaggle.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/data/sources/kaggle.py
"""Source adapter for Kaggle competition train.csv."""
import pandas as pd
from pathlib import Path
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(data_dir: Path = Path("data/raw"), **kwargs) -> pd.DataFrame:
    """Load competition training data and return unified schema DataFrame."""
    train_path = data_dir / "train.csv"
    raw = pd.read_csv(train_path)

    rows = []
    for _, r in raw.iterrows():
        rows.append(make_row(
            transliteration=normalize_transliteration(str(r["transliteration"])),
            translation=str(r["translation"]),
            source="kaggle_train",
            dialect="old_assyrian",
            genre="trade",
            quality="gold",
            has_translation=True,
        ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_sources_kaggle.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/kaggle.py tests/data/test_sources_kaggle.py
git commit -m "feat(data): add Kaggle competition source adapter"
```

---

### Task 4: OARE sentences source adapter

This is the trickiest adapter. OARE sentences have translations but only `first_word_spelling`. Full transliterations come from `published_texts.csv` joined on `text_uuid`=`oare_id`. We segment the full text transliteration into sentences using `first_word_number` boundaries.

**Files:**
- Create: `src/data/sources/oare_sentences.py`
- Create: `tests/data/test_sources_oare.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_sources_oare.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.sources.oare_sentences import load, _segment_sentences
from src.data.schema import validate_dataframe


def _write_test_files(tmp_path):
    """Create minimal OARE + published_texts test data."""
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
    # Sentence 1 starts at word 1, sentence 2 starts at word 4
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
    # Empty published_texts
    published = tmp_path / "published_texts.csv"
    published.write_text(
        'oare_id,online transcript,cdli_id,aliases,label,publication_catalog,'
        'description,genre_label,inventory_position,online_catalog,note,'
        'interlinear_commentary,online_information,excavation_no,oatp_key,'
        'eBL_id,AICC_translation,transliteration_orig,transliteration\n'
    )
    df = load(data_dir=tmp_path)
    assert len(df) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_sources_oare.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/data/sources/oare_sentences.py
"""Source adapter for OARE sentences joined with published_texts transliterations."""
import pandas as pd
from pathlib import Path
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def _segment_sentences(words: list[str], word_starts: list[int]) -> list[str]:
    """Split a word list into sentence segments using 1-based start positions.

    word_starts: sorted list of 1-based word indices where each sentence begins.
    Returns list of space-joined sentence strings.
    """
    segments = []
    for i, start in enumerate(word_starts):
        begin = start - 1  # convert to 0-based
        if i + 1 < len(word_starts):
            end = word_starts[i + 1] - 1
        else:
            end = len(words)
        segment = " ".join(words[begin:end])
        segments.append(segment)
    return segments


def load(data_dir: Path = Path("data/raw"), **kwargs) -> pd.DataFrame:
    """Load OARE sentences with transliterations reconstructed from published_texts."""
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
    published_path = data_dir / "published_texts.csv"

    oare = pd.read_csv(sentences_path)
    published = pd.read_csv(published_path)

    # Build text_uuid -> full transliteration lookup
    translit_lookup = {}
    for _, pt in published.iterrows():
        oare_id = str(pt["oare_id"])
        translit = pt.get("transliteration", "")
        if pd.notna(translit) and str(translit).strip():
            translit_lookup[oare_id] = str(translit).strip()

    # Group OARE sentences by text_uuid
    rows = []
    for text_uuid, group in oare.groupby("text_uuid"):
        text_uuid = str(text_uuid)
        if text_uuid not in translit_lookup:
            continue

        full_translit = translit_lookup[text_uuid]
        words = full_translit.split()

        # Sort sentences by first_word_number
        group = group.sort_values("first_word_number")
        word_starts = group["first_word_number"].astype(int).tolist()
        translations = group["translation"].tolist()

        segments = _segment_sentences(words, word_starts)

        for segment, translation in zip(segments, translations):
            if not segment.strip() or pd.isna(translation) or not str(translation).strip():
                continue
            rows.append(make_row(
                transliteration=normalize_transliteration(segment),
                translation=str(translation),
                source="oare_sentences",
                dialect="old_assyrian",
                genre="trade",
                quality="gold",
                has_translation=True,
            ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_sources_oare.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/oare_sentences.py tests/data/test_sources_oare.py
git commit -m "feat(data): add OARE sentences adapter with transliteration reconstruction"
```

---

### Task 5: HuggingFace source adapters

Three adapters for the HF datasets. phucthaiv02 is the largest (72K + 10K pairs). cipher-ling has a nested dict requiring extraction. veezbo is English-only — skip for now.

**Files:**
- Create: `src/data/sources/hf_phucthaiv02.py`
- Create: `src/data/sources/hf_cipher.py`
- Create: `tests/data/test_sources_hf.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_sources_hf.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data.schema import validate_dataframe


class TestPhucthaiv02:
    def test_load_returns_valid_schema(self):
        from src.data.sources.hf_phucthaiv02 import load

        # Mock HF datasets to avoid network calls
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_sources_hf.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/data/sources/hf_phucthaiv02.py
"""Source adapter for phucthaiv02 HuggingFace Akkadian datasets."""
import pandas as pd
from datasets import load_dataset
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(cache_dir: str = "data/external/hf_phucthaiv02", **kwargs) -> pd.DataFrame:
    """Load phucthaiv02/akkadian-translation (train) + akkadian_english_sentences_alignment_2."""
    rows = []

    # Dataset 1: akkadian-translation (72K train split)
    ds1 = load_dataset("phucthaiv02/akkadian-translation", split="train", trust_remote_code=True)
    for item in ds1:
        translit = str(item.get("transliteration", "")).strip()
        translation = str(item.get("translation", "")).strip()
        if not translit or not translation:
            continue
        rows.append(make_row(
            transliteration=normalize_transliteration(translit),
            translation=translation,
            source="hf_phucthaiv02_translation",
            dialect="unknown",
            genre="unknown",
            quality="gold",
            has_translation=True,
        ))

    # Dataset 2: akkadian_english_sentences_alignment_2 (10K)
    ds2 = load_dataset(
        "phucthaiv02/akkadian_english_sentences_alignment_2",
        split="train",
        trust_remote_code=True,
    )
    for item in ds2:
        translit = str(item.get("transliteration", "")).strip()
        translation = str(item.get("translation", "")).strip()
        if not translit or not translation:
            continue
        rows.append(make_row(
            transliteration=normalize_transliteration(translit),
            translation=translation,
            source="hf_phucthaiv02_alignment",
            dialect="unknown",
            genre="unknown",
            quality="gold",
            has_translation=True,
        ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
```

```python
# src/data/sources/hf_cipher.py
"""Source adapter for cipher-ling/akkadian HuggingFace dataset."""
import pandas as pd
from datasets import load_dataset
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(cache_dir: str = "data/external/hf_cipher", **kwargs) -> pd.DataFrame:
    """Load cipher-ling/akkadian. Extracts transliteration from nested translation.tr field."""
    rows = []

    ds = load_dataset("cipher-ling/akkadian", split="train", trust_remote_code=True)
    for item in ds:
        trans_dict = item.get("translation", {})
        if not isinstance(trans_dict, dict):
            continue
        translit = str(trans_dict.get("tr", "")).strip()
        english = str(trans_dict.get("en", "")).strip()
        if not translit or not english:
            continue
        rows.append(make_row(
            transliteration=normalize_transliteration(translit),
            translation=english,
            source="hf_cipher_ling",
            dialect="unknown",
            genre="unknown",
            quality="gold",
            has_translation=True,
        ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_sources_hf.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/hf_phucthaiv02.py src/data/sources/hf_cipher.py tests/data/test_sources_hf.py
git commit -m "feat(data): add HuggingFace source adapters (phucthaiv02 + cipher-ling)"
```

---

### Task 6: Lexicon source adapter

Generates word→definition pairs from the local eBL Dictionary and OA Lexicon files. Tagged as `quality=lexicon` for lower training weight.

**Files:**
- Create: `src/data/sources/lexicon.py`
- Create: `tests/data/test_sources_lexicon.py`

**Step 1: Write the failing tests**

```python
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
    # Empty lexicon
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_sources_lexicon.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/data/sources/lexicon.py
"""Source adapter for eBL Dictionary and OA Lexicon files."""
import pandas as pd
from pathlib import Path
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(data_dir: Path = Path("data/raw"), **kwargs) -> pd.DataFrame:
    """Load word→definition pairs from eBL Dictionary and OA Lexicon."""
    rows = []

    # eBL Dictionary: word → definition
    dict_path = data_dir / "eBL_Dictionary.csv"
    if dict_path.exists():
        ebl = pd.read_csv(dict_path)
        for _, r in ebl.iterrows():
            word = str(r.get("word", "")).strip()
            definition = str(r.get("definition", "")).strip()
            if not word or not definition or definition == "nan":
                continue
            # Clean definition: remove quotes and parenthetical grammar notes
            definition = definition.strip('"')
            rows.append(make_row(
                transliteration=normalize_transliteration(word),
                translation=definition,
                source="ebl_dictionary",
                dialect="unknown",
                genre="unknown",
                quality="lexicon",
                has_translation=True,
            ))

    # OA Lexicon: form → norm (transliteration form mapping)
    lex_path = data_dir / "OA_Lexicon_eBL.csv"
    if lex_path.exists():
        lex = pd.read_csv(lex_path)
        # Deduplicate: group by lexeme, take unique form→lexeme mappings
        seen_lexemes = set()
        for _, r in lex.iterrows():
            form = str(r.get("form", "")).strip()
            lexeme = str(r.get("lexeme", "")).strip()
            if not form or not lexeme or lexeme == "nan" or lexeme in seen_lexemes:
                continue
            seen_lexemes.add(lexeme)
            rows.append(make_row(
                transliteration=normalize_transliteration(form),
                translation=lexeme,  # normalized form serves as "translation"
                source="oa_lexicon",
                dialect="old_assyrian",
                genre="unknown",
                quality="lexicon",
                has_translation=True,
            ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_sources_lexicon.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/lexicon.py tests/data/test_sources_lexicon.py
git commit -m "feat(data): add lexicon source adapter (eBL Dictionary + OA Lexicon)"
```

---

### Task 7: Source registry

Auto-discovers all adapter modules in `src/data/sources/` and provides a unified `load_all()` function.

**Files:**
- Modify: `src/data/sources/__init__.py`
- Create: `tests/data/test_registry.py`

**Step 1: Write the failing test**

```python
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
    # Create minimal test data
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        'oare_id,transliteration,translation\n'
        'abc,a-na,to\n'
    )
    df = load_source("kaggle", data_dir=tmp_path)
    assert len(df) == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_registry.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

```python
# src/data/sources/__init__.py
"""Source registry — discovers and loads all source adapters."""
import importlib
from pathlib import Path
import pandas as pd

_SOURCES_DIR = Path(__file__).parent
_EXCLUDE = {"__init__"}


def list_sources() -> list[str]:
    """Return names of all available source adapter modules."""
    sources = []
    for f in sorted(_SOURCES_DIR.glob("*.py")):
        name = f.stem
        if name not in _EXCLUDE:
            sources.append(name)
    return sources


def load_source(name: str, **kwargs) -> pd.DataFrame:
    """Load a single source by module name, passing kwargs to its load() function."""
    module = importlib.import_module(f"src.data.sources.{name}")
    return module.load(**kwargs)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_registry.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/__init__.py tests/data/test_registry.py
git commit -m "feat(data): add source registry with auto-discovery"
```

---

### Task 8: Assembler — orchestrator with dedup, split, and output

The main pipeline script. Loads all sources, deduplicates with priority, splits stratified, and writes Parquet + CSV outputs.

**Files:**
- Create: `src/data/assemble.py`
- Create: `tests/data/test_assemble.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_assemble.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/data/assemble.py
"""Dataset assembly pipeline — orchestrates source loading, dedup, split, and output."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schema import SCHEMA_COLUMNS, validate_dataframe
from src.data.sources import list_sources, load_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Source priority for deduplication (lower = higher priority)
SOURCE_PRIORITY = {
    "kaggle_train": 0,
    "oare_sentences": 1,
    "oracc": 2,
    "hf_phucthaiv02_translation": 3,
    "hf_phucthaiv02_alignment": 4,
    "hf_cipher_ling": 5,
    "ebl_dictionary": 10,
    "oa_lexicon": 11,
}


def _source_priority(source: str) -> int:
    return SOURCE_PRIORITY.get(source, 99)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (transliteration, translation) pairs, keeping highest-priority source."""
    df = df.copy()
    df["_priority"] = df["source"].map(_source_priority)
    df = df.sort_values("_priority")
    df = df.drop_duplicates(subset=["transliteration", "translation"], keep="first")
    df = df.drop(columns=["_priority"])
    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test with stratification by source where possible."""
    # First split: train vs (val+test)
    holdout_frac = val_frac + test_frac
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, random_state=random_state,
    )
    # Second split: val vs test (50/50 of holdout)
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_frac / holdout_frac, random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_pipeline(
    sources: list[str] | None = None,
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> dict:
    """Run the full assembly pipeline.

    Returns a stats dict with counts per source and split sizes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sources
    available = list_sources()
    if sources:
        selected = [s for s in sources if s in available]
    else:
        selected = available

    log.info(f"Loading {len(selected)} sources: {selected}")

    # Load all sources
    frames = []
    source_stats = {}
    for name in selected:
        log.info(f"  Loading {name}...")
        try:
            df = load_source(name, data_dir=data_dir)
            validate_dataframe(df)
            source_stats[name] = len(df)
            log.info(f"    -> {len(df)} rows")
            frames.append(df)
        except Exception as e:
            log.warning(f"    -> FAILED: {e}")
            source_stats[name] = f"error: {e}"

    if not frames:
        log.error("No data loaded from any source.")
        return {"error": "no data"}

    # Combine
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined: {len(combined)} rows")

    # Deduplicate
    deduped = deduplicate(combined)
    log.info(f"After dedup: {len(deduped)} rows (removed {len(combined) - len(deduped)})")

    # Filter to parallel pairs only for training splits
    parallel = deduped[deduped["has_translation"]].copy()
    log.info(f"Parallel pairs (has_translation=True): {len(parallel)}")

    # Split
    train_df, val_df, test_df = split_dataset(parallel)
    log.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Competition-specific validation set (kaggle-source Old Assyrian only)
    val_competition = val_df[
        (val_df["source"] == "kaggle_train") & (val_df["dialect"] == "old_assyrian")
    ].reset_index(drop=True)

    # Write outputs
    deduped.to_parquet(output_dir / "all_data.parquet", index=False)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    if len(val_competition):
        val_competition.to_parquet(output_dir / "val_competition.parquet", index=False)

    # Backward-compatible CSV for train_baseline.py
    compat = train_df[["transliteration", "translation"]].copy()
    compat.to_csv(output_dir / "train_compat.csv", index=False)

    # Stats
    stats = {
        "total_combined": len(combined),
        "total_deduped": len(deduped),
        "total_parallel": len(parallel),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "val_competition_size": len(val_competition),
        "source_counts": source_stats,
        "dialect_distribution": deduped["dialect"].value_counts().to_dict(),
        "quality_distribution": deduped["quality"].value_counts().to_dict(),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Outputs written to {output_dir}/")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Assemble Akkadian training dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated list of sources (default: all)")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    run_pipeline(
        sources=sources,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_assemble.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/data/assemble.py tests/data/test_assemble.py
git commit -m "feat(data): add assembler with dedup, split, and Parquet output"
```

---

### Task 9: ORACC scraper adapter

Scrapes ORACC sub-project JSON bundles for transliteration↔translation pairs. This is the largest potential gold-standard source (~50K pairs). The CDL tree parsing is the most complex part.

**Files:**
- Create: `src/data/sources/oracc.py`
- Create: `tests/data/test_sources_oracc.py`

**Step 1: Write the failing tests**

```python
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
    """CDL tree with simple lemma nodes."""
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
                {"node": "d", "type": "line-start"},  # structural node
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
    # Translation extraction from label attributes
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
    assert len(result) >= 0  # May be 0 if no translations found
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_sources_oracc.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/data/sources/oracc.py
"""Source adapter for ORACC JSON API — scrapes cuneiform transliteration/translation pairs."""
import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration

log = logging.getLogger(__name__)

# ORACC sub-projects with significant Akkadian content
ORACC_PROJECTS = [
    ("saao/saa01", "neo_assyrian", "letter"),
    ("saao/saa02", "neo_assyrian", "letter"),
    ("saao/saa05", "neo_assyrian", "letter"),
    ("saao/saa08", "neo_assyrian", "letter"),
    ("saao/saa10", "neo_assyrian", "letter"),
    ("saao/saa13", "neo_assyrian", "letter"),
    ("saao/saa15", "neo_assyrian", "letter"),
    ("saao/saa16", "neo_assyrian", "letter"),
    ("saao/saa17", "neo_assyrian", "letter"),
    ("saao/saa18", "neo_assyrian", "letter"),
    ("saao/saa19", "neo_assyrian", "letter"),
    ("rinap/rinap1", "neo_assyrian", "royal_inscription"),
    ("rinap/rinap3", "neo_assyrian", "royal_inscription"),
    ("rinap/rinap4", "neo_assyrian", "royal_inscription"),
    ("riao", "neo_assyrian", "royal_inscription"),
    ("ribo/ribo2", "old_babylonian", "royal_inscription"),
    ("cams/gkab", "neo_assyrian", "literary"),
]

ORACC_BASE_URL = "http://oracc.museum.upenn.edu"


def extract_transliteration_from_cdl(cdl_nodes: list) -> list[str]:
    """Recursively walk CDL tree and extract transliteration lines (one per sentence)."""
    sentences = []
    _walk_cdl_for_transliteration(cdl_nodes, sentences, current_words=[])
    return sentences


def _walk_cdl_for_transliteration(
    nodes: list, sentences: list[str], current_words: list[str]
):
    for node in nodes:
        ntype = node.get("node")
        if ntype == "l":
            # Lemma node — extract form
            form = node.get("frag", "")
            if not form:
                f_dict = node.get("f", {})
                form = f_dict.get("form", "")
            if form:
                current_words.append(form)
        elif ntype == "c":
            chunk_type = node.get("type", "")
            children = node.get("cdl", [])
            if chunk_type == "sentence":
                # Start a new sentence
                inner_words = []
                _walk_cdl_for_transliteration(children, sentences, inner_words)
                if inner_words:
                    sentences.append(" ".join(inner_words))
            elif chunk_type == "translation":
                pass  # Skip translation chunks in transliteration pass
            else:
                _walk_cdl_for_transliteration(children, sentences, current_words)
        elif ntype == "d":
            pass  # Structural/discontinuity node — skip


def extract_translations_from_cdl(cdl_nodes: list) -> list[str]:
    """Extract English translation strings from CDL tree."""
    translations = []
    _walk_cdl_for_translations(cdl_nodes, translations)
    return translations


def _walk_cdl_for_translations(nodes: list, translations: list[str]):
    for node in nodes:
        ntype = node.get("node")
        if ntype == "c":
            chunk_type = node.get("type", "")
            label = node.get("label", "")
            if chunk_type == "translation" and label:
                translations.append(label)
            children = node.get("cdl", [])
            if children:
                _walk_cdl_for_translations(children, translations)


def parse_text_json(json_path: Path) -> list[dict]:
    """Parse a single ORACC text JSON file into (transliteration, translation) pairs."""
    with open(json_path) as f:
        data = json.load(f)

    cdl = data.get("cdl", [])
    translits = extract_transliteration_from_cdl(cdl)
    translations = extract_translations_from_cdl(cdl)

    pairs = []
    # Pair up by position — ORACC aligns sentences and translations in order
    for translit, translation in zip(translits, translations):
        if translit.strip() and translation.strip():
            pairs.append({"transliteration": translit, "translation": translation})

    return pairs


def _download_project(project: str, cache_dir: Path) -> Path | None:
    """Download and extract ORACC project JSON zip. Returns extracted dir or None."""
    project_cache = cache_dir / project.replace("/", "_")
    if project_cache.exists() and any(project_cache.glob("*.json")):
        return project_cache

    url = f"{ORACC_BASE_URL}/{project}/json/{project.split('/')[-1]}.zip"
    log.info(f"    Downloading {url}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"    Failed to download {project}: {e}")
        return None

    project_cache.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(project_cache)
    except zipfile.BadZipFile:
        log.warning(f"    Bad zip file for {project}")
        return None

    return project_cache


def load(
    cache_dir: str | Path = Path("data/external/oracc"),
    data_dir: Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load transliteration-translation pairs from ORACC sub-projects."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for project, dialect, genre in ORACC_PROJECTS:
        project_dir = _download_project(project, cache_dir)
        if project_dir is None:
            continue

        # Find text JSON files (not corpus.json or catalogue.json)
        json_files = list(project_dir.rglob("*.json"))
        text_files = [
            f for f in json_files
            if f.name not in ("corpus.json", "catalogue.json", "index.json")
            and "corpusjson" not in str(f)
        ]

        for text_file in text_files:
            try:
                pairs = parse_text_json(text_file)
                for pair in pairs:
                    rows.append(make_row(
                        transliteration=normalize_transliteration(pair["transliteration"]),
                        translation=pair["translation"],
                        source=f"oracc_{project.replace('/', '_')}",
                        dialect=dialect,
                        genre=genre,
                        quality="gold",
                        has_translation=True,
                    ))
            except Exception as e:
                log.debug(f"    Error parsing {text_file}: {e}")

    log.info(f"  ORACC total: {len(rows)} pairs")
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_sources_oracc.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/data/sources/oracc.py tests/data/test_sources_oracc.py
git commit -m "feat(data): add ORACC JSON scraper with CDL tree parsing"
```

---

### Task 10: Integration test — full pipeline end-to-end

Run the pipeline on the actual local data (kaggle + oare_sentences + lexicon) to verify everything works together before adding network-dependent sources.

**Files:**
- Create: `tests/data/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/data/test_integration.py
"""Integration test — runs pipeline on local data sources only (no network)."""
import pytest
from pathlib import Path
from src.data.assemble import run_pipeline


@pytest.mark.integration
def test_pipeline_local_sources(tmp_path):
    """Run pipeline with kaggle + oare_sentences + lexicon (all local files)."""
    data_dir = Path("data/raw")
    if not (data_dir / "train.csv").exists():
        pytest.skip("Competition data not available")

    stats = run_pipeline(
        sources=["kaggle", "oare_sentences", "lexicon"],
        data_dir=data_dir,
        output_dir=tmp_path,
    )

    assert stats["train_size"] > 0
    assert stats["val_size"] > 0
    assert stats["test_size"] > 0
    assert (tmp_path / "all_data.parquet").exists()
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "train_compat.csv").exists()
    assert (tmp_path / "stats.json").exists()

    # Should have more data than just kaggle alone
    assert stats["total_combined"] > 1561
    print(f"\nPipeline stats: {stats}")
```

**Step 2: Run integration test**

Run: `python -m pytest tests/data/test_integration.py -v -m integration -s`
Expected: PASSED, with printed stats showing combined data > 1561 rows

**Step 3: Commit**

```bash
git add tests/data/test_integration.py
git commit -m "test: add integration test for local data pipeline"
```

---

### Task 11: Run full pipeline and verify outputs

Run the complete pipeline (including HF downloads and ORACC scraping) and inspect the results.

**Step 1: Run the full pipeline**

Run: `cd /home/nbarthel/ai/kaggle/akkadian && python src/data/assemble.py --output-dir data/processed 2>&1 | tee pipeline_output.log`

Expected: Logs showing each source loading, dedup count, split sizes. Should produce 50K+ total rows.

**Step 2: Verify output files exist**

Run: `ls -la data/processed/`

Expected: `all_data.parquet`, `train.parquet`, `val.parquet`, `test.parquet`, `val_competition.parquet`, `train_compat.csv`, `stats.json`

**Step 3: Inspect stats**

Run: `python -c "import json; print(json.dumps(json.load(open('data/processed/stats.json')), indent=2))"`

Expected: Stats showing source breakdown, dialect distribution, quality tiers

**Step 4: Quick smoke test — train baseline on new data**

Run: `python src/train_baseline.py --data-dir data/processed --epochs 1 --batch-size 4`

This uses `train_compat.csv` from the assembled dataset. Verify it loads and starts training without errors. Kill after first epoch.

**Step 5: Commit pipeline output log and stats**

```bash
git add data/processed/stats.json pipeline_output.log
git commit -m "data: assemble full training dataset — stats and pipeline log"
```

---

### Task 12: Exploration notebook

Create the analysis notebook documenting the assembled dataset.

**Files:**
- Create: `notebooks/02_dataset_assembly.ipynb`

**Step 1: Create notebook with analysis cells**

The notebook should import from `src.data.assemble` and `src.data.sources`, load the assembled data from `data/processed/`, and create:

1. Source inventory table (rows per source)
2. Normalization before/after examples
3. Overlap heatmap between sources
4. Dialect/genre distribution bar charts
5. Quality tier breakdown
6. Final dataset statistics
7. Multimodal catalog (if any)

Use `matplotlib` for charts. Load data via `pd.read_parquet("data/processed/all_data.parquet")`.

**Step 2: Run notebook to verify**

Run: `jupyter nbconvert --execute notebooks/02_dataset_assembly.ipynb --to html --output-dir outputs/`

**Step 3: Commit**

```bash
git add notebooks/02_dataset_assembly.ipynb
git commit -m "docs: add dataset assembly exploration notebook"
```

---

### Task 13: Update CLAUDE.md with new pipeline commands

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add pipeline commands and data flow docs to CLAUDE.md**

Add to the Commands section:

```markdown
### Dataset assembly
```bash
python src/data/assemble.py                                    # Full pipeline (all sources)
python src/data/assemble.py --sources kaggle,oare_sentences    # Local sources only (no network)
python src/data/assemble.py --force-refresh                    # Re-download external sources
```
```

Add to the Architecture section a note about the data pipeline:

```markdown
### Data pipeline (`src/data/`)
Modular source adapters in `src/data/sources/`. Each exports `load(**kwargs) -> DataFrame`.
`src/data/assemble.py` orchestrates: load all → normalize (Unicode) → deduplicate (priority-based) → split 90/5/5 → write Parquet to `data/processed/`.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with dataset assembly pipeline"
```
