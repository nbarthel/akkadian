"""Source adapter for ORACC JSON API — scrapes cuneiform transliteration/translation pairs."""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import requests

from src.data.schema import make_row, SCHEMA_COLUMNS
from src.data.normalize import normalize_transliteration

log = logging.getLogger(__name__)

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


def extract_transliteration_from_cdl(cdl_nodes: list) -> List[str]:
    """Recursively walk CDL tree and extract transliteration lines (one per sentence)."""
    sentences: List[str] = []
    _walk_cdl_for_transliteration(cdl_nodes, sentences, current_words=[])
    return sentences


def _walk_cdl_for_transliteration(
    nodes: list, sentences: List[str], current_words: List[str]
) -> None:
    for node in nodes:
        ntype = node.get("node")
        if ntype == "l":
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
                inner_words: List[str] = []
                _walk_cdl_for_transliteration(children, sentences, inner_words)
                if inner_words:
                    sentences.append(" ".join(inner_words))
            elif chunk_type == "translation":
                pass
            else:
                _walk_cdl_for_transliteration(children, sentences, current_words)
        elif ntype == "d":
            pass


def extract_translations_from_cdl(cdl_nodes: list) -> List[str]:
    """Extract English translation strings from CDL tree."""
    translations: List[str] = []
    _walk_cdl_for_translations(cdl_nodes, translations)
    return translations


def _walk_cdl_for_translations(nodes: list, translations: List[str]) -> None:
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


def parse_text_json(json_path: Path) -> List[dict]:
    """Parse a single ORACC text JSON file into (transliteration, translation) pairs."""
    with open(json_path) as f:
        data = json.load(f)

    cdl = data.get("cdl", [])
    translits = extract_transliteration_from_cdl(cdl)
    translations = extract_translations_from_cdl(cdl)

    pairs = []
    for translit, translation in zip(translits, translations):
        if translit.strip() and translation.strip():
            pairs.append({"transliteration": translit, "translation": translation})

    return pairs


def _download_project(project: str, cache_dir: Path) -> Optional[Path]:
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
    cache_dir: Union[str, Path] = Path("data/external/oracc"),
    data_dir: Optional[Path] = None,
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
