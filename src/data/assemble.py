"""Dataset assembly pipeline — orchestrates source loading, dedup, split, and output."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test."""
    holdout_frac = val_frac + test_frac
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, random_state=random_state,
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_frac / holdout_frac, random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_pipeline(
    sources: Optional[List[str]] = None,
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> Dict:
    """Run the full assembly pipeline. Returns stats dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    available = list_sources()
    if sources:
        selected = [s for s in sources if s in available]
    else:
        selected = available

    log.info(f"Loading {len(selected)} sources: {selected}")

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

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined: {len(combined)} rows")

    deduped = deduplicate(combined)
    log.info(f"After dedup: {len(deduped)} rows (removed {len(combined) - len(deduped)})")

    parallel = deduped[deduped["has_translation"]].copy()
    log.info(f"Parallel pairs (has_translation=True): {len(parallel)}")

    train_df, val_df, test_df = split_dataset(parallel)
    log.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    val_competition = val_df[
        (val_df["source"] == "kaggle_train") & (val_df["dialect"] == "old_assyrian")
    ].reset_index(drop=True)

    deduped.to_parquet(output_dir / "all_data.parquet", index=False)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    if len(val_competition):
        val_competition.to_parquet(output_dir / "val_competition.parquet", index=False)

    compat = train_df[["transliteration", "translation"]].copy()
    compat.to_csv(output_dir / "train_compat.csv", index=False)

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
