# Dataset Assembly Pipeline Design

*2026-02-28 — Comprehensive Akkadian dataset for ByT5 fine-tuning*

## Objective

Assemble a deduplicated, Unicode-normalized training dataset from all available Akkadian parallel corpora to maximize performance on the Deep Past Challenge (Kaggle). The competition evaluates transliteration-to-English translation using `sqrt(BLEU * chrF++)` on Old Assyrian trade records.

Current baseline trains on 1,561 pairs. Target: 50K–100K+ pairs with dialect/genre/quality metadata for weighted training.

## Deliverables

1. `src/data/assemble.py` — reproducible pipeline script
2. `notebooks/02_dataset_assembly.ipynb` — exploration and validation notebook
3. `data/processed/*.parquet` — assembled dataset splits
4. `data/processed/train_compat.csv` — backward-compatible CSV for `train_baseline.py`

---

## 1. Unified Schema

Every source adapter produces rows conforming to this schema:

| Column | Type | Description |
|--------|------|-------------|
| `transliteration` | str | Unicode-normalized Akkadian transliteration |
| `translation` | str | English translation (empty for monolingual data) |
| `source` | str | Adapter origin, e.g. `kaggle_train`, `oracc_saa`, `hf_phucthaiv02` |
| `dialect` | str | `old_assyrian`, `neo_assyrian`, `old_babylonian`, `unknown` |
| `genre` | str | `trade`, `royal_inscription`, `literary`, `legal`, `letter`, `unknown` |
| `quality` | str | `gold` (human-verified), `silver` (machine/unverified), `lexicon` (dictionary-derived) |
| `has_translation` | bool | True for parallel pairs, False for monolingual transliterations |
| `multimodal_ref` | str | Path/URL to associated image if from multimodal dataset, empty otherwise |

---

## 2. Unicode Normalization

All transliterations normalized to Unicode with diacritics. Normalizer lives in `src/data/normalize.py`, shared by all adapters.

**Rules:**

1. ASCII to Unicode diacritics: `sz` -> `š`, `s,` -> `ṣ`, `t,` -> `ṭ`
2. Subscript digit normalization: `du3` / `du₃` -> consistent `du₃`
3. Sumerogram casing: determinatives `{d}`, `{ki}`, `{f}` lowercase-braced; logograms `LUGAL`, `DINGIR` uppercase
4. NFC Unicode normalization (precomposed characters)
5. Whitespace and hyphenation standardization

---

## 3. Source Adapters

### File layout

```
src/data/
├── normalize.py          # Unicode normalizer
├── assemble.py           # Orchestrator: load -> normalize -> dedup -> split -> write
├── schema.py             # Schema definition + validation
└── sources/
    ├── __init__.py        # Source registry (auto-discovers adapters)
    ├── kaggle.py          # Competition train.csv
    ├── oare_sentences.py  # Sentences_Oare CSV
    ├── hf_phucthaiv02.py  # Two HF datasets
    ├── hf_cipher.py       # cipher-ling/akkadian
    ├── hf_veezbo.py       # veezbo corpus
    ├── oracc.py           # ORACC JSON API scraper
    ├── hf_mik3ml.py       # Large gated dataset
    ├── lexicon.py         # eBL Dictionary + OA Lexicon -> synthetic pairs
    └── hf_multimodal.py   # Catalog-only for image datasets
```

### Tier 1 — Gold parallel data

| Adapter | Source | Est. Pairs | Method |
|---------|--------|-----------|--------|
| `kaggle.py` | `data/raw/train.csv` | 1,561 | Direct CSV read. Tag `old_assyrian` / `trade` / `gold` |
| `oare_sentences.py` | `data/raw/Sentences_Oare_FirstWord_LinNum.csv` | ~9,782 | Parse translation column, reconstruct transliterations. Tag `old_assyrian` / `gold` |
| `hf_phucthaiv02.py` | `phucthaiv02/akkadian-translation` + `akkadian_english_sentences_alignment_2` | 10K–100K | HF datasets API. Dialect/genre from metadata where available |
| `hf_cipher.py` | `cipher-ling/akkadian` | 10K–100K | HF download. Tag `quality=gold` |
| `hf_veezbo.py` | `veezbo/akkadian_english_corpus` | 1K–10K | HF download. MIT license |

### Tier 2 — Silver/augmentation data

| Adapter | Source | Est. Pairs | Method |
|---------|--------|-----------|--------|
| `oracc.py` | ORACC JSON API | ~50K | Scrape sub-projects (SAA, RINAP, RIAO, RIBo, CASPo, CCPo, CAMS/GKAB, SAAo, BLMS). Tag by sub-project for dialect/genre. `quality=gold` |
| `hf_mik3ml.py` | `mik3ml/akkadian` | 100K–1M | Gated access. Tag `quality=silver` until verified |
| `lexicon.py` | `data/raw/eBL_Dictionary.csv` + `OA_Lexicon_eBL.csv` | synthetic | Word-to-definition pairs. `quality=lexicon` |

### Tier 3 — Cataloged for later (multimodal)

| Adapter | Source | Action |
|---------|--------|--------|
| `hf_multimodal.py` | `hrabalm/mtm24-akkadian-v3`, `markzeits/fll-cuneiform`, `boatbomber/CuneiformPhotosMSII` | Download metadata only, extract text pairs where available, populate `multimodal_ref`. Flag for future OCR pipeline |

### Not included initially

Cross-lingual (Phase 3/4 from guide): Arabic ATHAR, Hebrew OPUS, Sumerian ETCSL. Separate adapters added later if training plateaus.

---

## 4. ORACC Scraper

ORACC is the single largest gold-standard source (~50K pairs). Projects publish JSON bundles at:

```
http://oracc.museum.upenn.edu/{project}/json/{project}.zip
```

**Target sub-projects:** SAA, RINAP, RIAO, RIBo, CASPo, CCPo, CAMS/GKAB, SAAo, BLMS

**Extraction logic:**

1. Download JSON zip per project -> cache in `data/external/oracc/{project}/`
2. Parse `corpus.json` for text metadata (period, genre)
3. For each text, walk the CDL (Cuneiform Description Language) tree:
   - Extract transliteration lines from `"type": "l"` (lemma) nodes
   - Extract English translations from `"tr"` nodes
   - Align at line or sentence level
4. Tag dialect from period metadata (Old Assyrian, Neo-Assyrian, etc.)
5. Tag genre from project identity (SAA=letters, RINAP=royal_inscription, etc.)

**CDL tree parsing:** The nested JSON structure interleaves lemma nodes with structural nodes (`"type": "c"` chunks, `"type": "d"` discontinuities). The adapter recursively walks the tree to reconstruct full line transliterations from leaf nodes.

**Fallback:** If JSON API is unreliable, fall back to ATF files from ORACC GitHub mirrors (simpler parsing, less metadata).

---

## 5. Pipeline Orchestrator

`assemble.py` flow:

```
1. Discover all source adapters from registry
2. For each source:
   a. Check data/external/{source}/ cache — skip download if present
   b. Call adapter.load() -> raw DataFrame
   c. Apply normalize.py to transliteration column
   d. Validate against schema.py
3. Concatenate all DataFrames
4. Deduplicate on normalized (transliteration, translation) pairs
   Priority: kaggle > oare_sentences > oracc > hf_* > lexicon
5. Log stats: total rows, per-source counts, dialect/genre/quality distribution
6. Split 90/5/5 train/val/test, stratified by source + dialect
7. Write outputs to data/processed/
```

**CLI interface:**

```bash
python src/data/assemble.py                    # Run full pipeline with caching
python src/data/assemble.py --force-refresh    # Re-download all sources
python src/data/assemble.py --sources kaggle,oare_sentences  # Subset of sources
```

**Outputs:**

```
data/processed/
├── all_data.parquet           # Full merged dataset with all metadata
├── train.parquet              # 90% training split
├── val.parquet                # 5% validation split (all sources)
├── test.parquet               # 5% test split
├── val_competition.parquet    # Competition-relevant val (kaggle-source Old Assyrian only)
├── train_compat.csv           # Backward-compatible CSV for train_baseline.py
├── stats.json                 # Dataset statistics
└── multimodal_catalog.json    # Flagged image datasets for future OCR
```

**Caching:** Raw downloads stored in `data/external/{source_name}/`. Adapters check cache first. Only re-process normalization/dedup on each run.

---

## 6. Exploration Notebook

`notebooks/02_dataset_assembly.ipynb` sections:

1. **Source inventory** — Run each adapter, show sample rows, count pairs per source
2. **Normalization audit** — Before/after examples for each rule, edge cases
3. **Overlap analysis** — Heatmap of pairwise overlap between sources (validates dedup)
4. **Dialect/genre distribution** — Bar charts of final dataset composition
5. **Quality tier breakdown** — Gold vs. silver vs. lexicon proportions
6. **Final dataset stats** — Total pairs, split sizes, vocabulary coverage, sequence lengths
7. **Multimodal catalog** — Table of flagged image datasets for future work

The notebook calls `assemble.py` functions directly — analysis on top of the pipeline, not a parallel implementation.

---

## 7. Training Integration

**Backward compatibility:** Point `train_baseline.py` at the new data with `--data-dir data/processed` using `train_compat.csv`. Immediate comparison: 1.5K vs. 50K+ pairs, same model and hyperparameters.

**Weighted training (for hyperx-ie):** The `quality`, `dialect`, and `genre` columns enable:

- Sample weighting: upweight `old_assyrian` + `trade` + `gold` (competition domain match)
- Curriculum learning: train on all data first, then fine-tune on Old Assyrian only
- Quality-aware loss: lower weight for `silver` and `lexicon` to prevent learning from machine translation errors

**Dual validation:** The `val_competition.parquet` split (kaggle-source Old Assyrian only) gives the competition-relevant eval signal alongside the broad validation set.
