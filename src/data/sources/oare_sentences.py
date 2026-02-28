"""Source adapter for OARE sentences joined with published_texts transliterations."""
import pandas as pd
from pathlib import Path
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def _segment_sentences(words: list[str], word_starts: list[int]) -> list[str]:
    """Split a word list into sentence segments using 1-based start positions."""
    segments = []
    for i, start in enumerate(word_starts):
        begin = start - 1
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

    translit_lookup = {}
    for _, pt in published.iterrows():
        oare_id = str(pt["oare_id"])
        translit = pt.get("transliteration", "")
        if pd.notna(translit) and str(translit).strip():
            translit_lookup[oare_id] = str(translit).strip()

    rows = []
    for text_uuid, group in oare.groupby("text_uuid"):
        text_uuid = str(text_uuid)
        if text_uuid not in translit_lookup:
            continue

        full_translit = translit_lookup[text_uuid]
        words = full_translit.split()

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
