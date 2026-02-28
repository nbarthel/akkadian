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
