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
