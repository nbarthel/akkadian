"""Source adapter for eBL Dictionary and OA Lexicon files."""
import pandas as pd
from pathlib import Path
from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration


def load(data_dir: Path = Path("data/raw"), **kwargs) -> pd.DataFrame:
    """Load word->definition pairs from eBL Dictionary and OA Lexicon."""
    rows = []

    dict_path = data_dir / "eBL_Dictionary.csv"
    if dict_path.exists():
        ebl = pd.read_csv(dict_path)
        for _, r in ebl.iterrows():
            word = str(r.get("word", "")).strip()
            definition = str(r.get("definition", "")).strip()
            if not word or not definition or definition == "nan":
                continue
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

    lex_path = data_dir / "OA_Lexicon_eBL.csv"
    if lex_path.exists():
        lex = pd.read_csv(lex_path)
        seen_lexemes = set()
        for _, r in lex.iterrows():
            form = str(r.get("form", "")).strip()
            lexeme = str(r.get("lexeme", "")).strip()
            if not form or not lexeme or lexeme == "nan" or lexeme in seen_lexemes:
                continue
            seen_lexemes.add(lexeme)
            rows.append(make_row(
                transliteration=normalize_transliteration(form),
                translation=lexeme,
                source="oa_lexicon",
                dialect="old_assyrian",
                genre="unknown",
                quality="lexicon",
                has_translation=True,
            ))

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
