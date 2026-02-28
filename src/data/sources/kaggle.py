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

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df["has_translation"] = df["has_translation"].astype(object)
    return df
