"""Source registry — discovers and loads all source adapters."""
import importlib
from pathlib import Path
from typing import List

import pandas as pd

_SOURCES_DIR = Path(__file__).parent
_EXCLUDE = {"__init__"}


def list_sources() -> List[str]:
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
