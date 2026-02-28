# tests/data/test_integration.py
"""Integration test -- runs pipeline on local data sources only (no network)."""
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
