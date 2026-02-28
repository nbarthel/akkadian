"""Source adapter for phucthaiv02 HuggingFace Akkadian datasets."""
import pandas as pd
from datasets import load_dataset
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(cache_dir: str = "data/external/hf_phucthaiv02", **kwargs) -> pd.DataFrame:
    """Load phucthaiv02/akkadian-translation (train) + akkadian_english_sentences_alignment_2."""
    rows = []

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
