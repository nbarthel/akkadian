# Best Base LLM for Decoding Ancient Akkadian
## Model Selection Analysis & Complete Training Data Resource Guide

*Compiled February 2026*

---

# PART I: RECOMMENDED BASE MODEL

## Executive Recommendation: ByT5 (Byte-level T5)

After surveying all published benchmarks, the active Kaggle competition, the RANLP 2025 comparative study (Jones & Mitkov), the UC Berkeley CuneiTranslate project, the ACL 2025 Ancient Language Processing workshop, and the original Gutherz/Gordin PNAS Nexus pipeline, the evidence converges on a clear winner and a strong runner-up.

### Primary: Google ByT5 (byte-level Text-to-Text Transfer Transformer)

**Why ByT5 is the best fit for Akkadian:**

ByT5 operates at the byte level rather than the subword/token level. This is a decisive advantage for Akkadian because:

1. **No tokenizer mismatch.** Standard BPE/SentencePiece tokenizers are trained on modern languages and shred Akkadian transliterations (e.g., `ša-ru-um`, `LUGAL`, `ù-a-ar`) into meaningless fragments. ByT5 sidesteps this entirely—every byte is a token, so the model processes the actual characters of the transliteration without any pretrained tokenizer bias.

2. **Proven on cuneiform.** Lu et al. (2025, ACL Workshop on Ancient Language Processing) benchmarked ByT5 against mT5 specifically for cuneiform lemmatization and found **ByT5 outperformed mT5**, achieving 80.55% accuracy on raw lemmas and 82.59% on generalized lemmas. This is the most directly relevant head-to-head comparison available.

3. **Superior on low-resource and morphologically rich languages.** Edman et al. (2024, TACL) performed an extensive comparison of ByT5 vs. mT5 across dozens of languages for NMT. They found ByT5 excels precisely where fine-tuning data is limited, and it handles orthographically similar words and rare words better—both critical for Akkadian, which has extensive dialectal variation across 3,000 years and sparse training data.

4. **Handles the diacritical / Unicode problem gracefully.** Akkadian transliterations use accented characters (š, ṣ, ṭ, ā, ē, ī, ū), subscript numerals for sign disambiguation (du₃, ba₂), and mixed-case Sumerograms (LUGAL, DINGIR). Subword tokenizers trained on English/Latin web text butcher these. ByT5 treats them as raw bytes.

5. **Right-sized for the data.** ByT5-base is ~580M parameters. This is large enough to learn complex morphological patterns but small enough to fine-tune on ~50K sentence pairs without catastrophic overfitting, unlike 7B+ decoder-only models that need vastly more data.

**Recommended variant:** `google/byt5-base` (580M) for the best data-efficiency-to-capacity ratio. Scale to `byt5-large` (1.2B) if compute allows and you have >100K training pairs.

### Strong Runner-Up: Meta NLLB-200 (No Language Left Behind)

**Why NLLB deserves consideration:**

1. **Pretrained on 200 languages including Arabic, Hebrew, and Amharic**—all Semitic languages sharing deep structural features with Akkadian (root-pattern morphology, verb conjugation systems, triconsonantal roots). The Jones & Mitkov (2025) study found strong evidence that "fine-tuning machine translation models from related languages" is critical. NLLB's Semitic language exposure gives it implicit priors on Akkadian morphology that ByT5 lacks.

2. **The UC Berkeley CuneiTranslate project (2024)** tested T5, mT5, Helsinki opus-mt-ar-en (Arabic→English), and NLLB side-by-side on Akkadian translation from ORACC data. NLLB's architecture with its Arabic pretraining showed particular promise for capturing Semitic morphological patterns.

3. **Efficient fine-tuning.** NLLB-200 distilled (600M) is comparable in size to ByT5-base and responds well to LoRA adaptation, keeping compute costs manageable.

**Key limitation:** NLLB uses SentencePiece tokenization, so you'll need to either extend the tokenizer vocabulary with Akkadian-specific tokens or accept some tokenization noise.

### Models to Avoid / Use Cautiously

| Model | Issue |
|-------|-------|
| **Decoder-only LLMs (Mistral-7B, LLaMA-2/3, Qwen)** | The RANLP 2025 study found all LLMs (Mistral, Qwen) scored within a narrow 4.6% F1 band of each other but were **cost-inefficient** compared to smaller seq2seq models. They require LoRA/QLoRA and still struggle with the low-resource regime. Mistral slightly outperformed Qwen on BLEU for short Akkadian sentences, but both fell behind MarianMT variants fine-tuned from related languages. |
| **Vanilla T5 (English-only)** | T5 has no multilingual pretraining. It works (Krueger's AICC model proves this), but you're training from scratch on language structure. ByT5 or mT5 are strictly better starting points. |
| **MarianMT (generic)** | MarianMT from English performs poorly. However, **MarianMT fine-tuned from Spanish or Arabic** (i.e., a Semitic/rich-morphology starting point) performed competitively in the RANLP 2025 study. This confirms the transfer-from-related-languages hypothesis. |
| **GPT-4 / Claude / Gemini (API-only)** | Cannot be fine-tuned on custom data. Useful for RAG-augmented in-context translation (Shu et al. 2024 showed promising BERTScore results), but not for building a dedicated Akkadian translation system. |

### Recommended Architecture Strategy

The optimal approach is a **two-stage pipeline**, mirroring the original Gutherz/Gordin architecture:

```
Stage 1: Cuneiform Unicode → Transliteration (C2T)
  - Model: ByT5-base (character-level OCR-like task)
  - This is the simpler task: 97% accuracy achieved by Gordin et al. (2020)

Stage 2: Transliteration → English (T2E)  
  - Model: ByT5-base OR NLLB-200 (distilled)
  - Fine-tuned on ORACC parallel sentence pairs
  - Augmented with FactGrid/Wikidata lexeme data via RAG
```

For a **single-model approach**, ByT5-base fine-tuned end-to-end on transliteration→English is the best choice.

---

# PART II: COMPLETE TRAINING DATA SOURCES

## A. PRIMARY AKKADIAN PARALLEL CORPORA

These are sources of aligned Akkadian (transliteration) ↔ English sentence pairs, which form the core training data for any translation model.

### A1. ORACC (Open Richly Annotated Cuneiform Corpus)
- **URL:** https://oracc.museum.upenn.edu/
- **Size:** ~50,000+ translated sentence pairs (Akkadian portion)
- **Format:** ATF (ASCII Transliteration Format) with lemmatization, POS tags, glossaries, and line-by-line English translations
- **License:** CC BY-SA 3.0
- **Quality:** Gold standard. Expert-annotated by Assyriologists. The primary training data used in the landmark Gutherz/Gordin PNAS Nexus paper.
- **Key sub-projects for Akkadian:**
  - **SAA (State Archives of Assyria)** — Neo-Assyrian royal letters, treaties, divination queries, court poetry (SAA 1–21). The best-annotated Akkadian subcorpus.
  - **RINAP** (Royal Inscriptions of the Neo-Assyrian Period) — Sargon II, Sennacherib, Esarhaddon, Ashurbanipal
  - **RIAO** (Royal Inscriptions of Assyria Online) — 10th–9th century BCE inscriptions
  - **RIBo** (Royal Inscriptions of Babylonia) — Babylonian royal texts
  - **CASPo** (Corpus of Ancient Sumerian and Babylonian Prayers online) — includes Akkadian prayer sub-projects
  - **CCPo** (Cuneiform Commentaries Project) — scholarly commentaries 8th–2nd century BCE
  - **CAMS/GKAB** — astronomical and astrological texts
  - **BLMS** (Bilingual Literary Manuscripts in Sumerian) — Sumerian–Akkadian bilinguals
  - **ARMEP** — multi-project search across Neo-Assyrian texts
  - **SAAo** (State Archives of Assyria Online) — digitized SAA volumes
  - **DCCLT** (Digital Corpus of Cuneiform Lexical Texts) — lexical lists including bilingual Sumerian–Akkadian vocabularies
- **Notes:** Glossaries within each sub-project contain word-level definitions usable for dictionary augmentation or RAG.

### A2. CDLI (Cuneiform Digital Library Initiative)
- **URL:** https://cdli.mpiwg-berlin.mpg.de/
- **Size:** 350,000+ cuneiform objects cataloged; transliterations for a significant subset; translations for a smaller subset
- **Format:** C-ATF (Canonical ATF), catalog metadata, photographs, line art
- **License:** Open access
- **Quality:** Primarily a catalog/archive. Transliterations vary in completeness. Translations are sparser than ORACC.
- **Notes:** Critical for the **untranslated** Akkadian corpus—hundreds of thousands of transliterated texts without English, which can be used for:
  - Monolingual language modeling / pretraining
  - Self-supervised objectives (masked language modeling, denoising)
  - Back-translation data augmentation

### A3. AICC (AI Cuneiform Corpus)
- **URL:** https://praeclarum.org/2023/06/09/cuneiform.html
- **Size:** 130,000 AI-translated texts (Sumerian + Akkadian → English)
- **Format:** Text
- **Quality:** Machine-generated translations from Krueger's T5 model. NOT human-verified. Useful for pretraining or weak supervision, but should be treated as silver-standard data and not mixed into gold-standard training splits without filtering.
- **HF model:** [praeclarum/cuneiform](https://hf.co/praeclarum/cuneiform)

### A4. Deep Past Challenge (Kaggle, Active Competition)
- **URL:** https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
- **Size:** Old Assyrian business records with parallel translations
- **Timeline:** Dec 16, 2025 – Mar 23, 2026 ($50,000 prize pool)
- **Significance:** First large-scale competitive benchmark for Akkadian MT. Scoring uses geometric mean of BLEU × chrF++. Competition data becomes a post-competition benchmark.
- **Sponsor:** Deep Past Initiative (deeppast.org), funded by XTX Markets

### A5. Hugging Face Datasets

| Dataset | Size | Content | Link |
|---------|------|---------|------|
| phucthaiv02/akkadian-translation | 10K–100K | Akkadian–English parallel pairs | [Link](https://hf.co/datasets/phucthaiv02/akkadian-translation) |
| phucthaiv02/akkadian_english_sentences_alignment_2 | 10K–100K | Sentence-aligned pairs (v2, Jan 2026) | [Link](https://hf.co/datasets/phucthaiv02/akkadian_english_sentences_alignment_2) |
| cipher-ling/akkadian | 10K–100K | Akkadian text dataset | [Link](https://hf.co/datasets/cipher-ling/akkadian) |
| veezbo/akkadian_english_corpus | 1K–10K | Cleaned Akkadian–English, MIT license | [Link](https://hf.co/datasets/veezbo/akkadian_english_corpus) |
| mik3ml/akkadian | 100K–1M | Large-scale, CC-BY-4.0, gated | [Link](https://hf.co/datasets/mik3ml/akkadian) |
| phucthaiv02/akkadian-publish-texts | 1K–10K | Published Akkadian texts in CSV | [Link](https://hf.co/datasets/phucthaiv02/akkadian-publish-texts) |

---

## B. LINGUISTIC ANNOTATION & MORPHOLOGICAL RESOURCES

These provide the structured linguistic knowledge (POS, morphology, syntax, lemmatization) that enriches translation training.

### B1. Universal Dependencies Treebanks for Akkadian

| Treebank | Sentences | Tokens | Period/Genre | Link |
|----------|-----------|--------|--------------|------|
| **UD Akkadian-RIAO** | 1,845 | 22,277 | Neo-Assyrian royal inscriptions (10th–9th c. BCE) | [GitHub](https://github.com/UniversalDependencies/UD_Akkadian-RIAO) |
| **UD Akkadian-PISANDUB** | 101 | 1,852 | Babylonian royal inscriptions | [UD page](https://universaldependencies.org/treebanks/akk_pisandub/) |
| **UD Akkadian-MCONG** | (in development) | — | Middle Babylonian Kongress texts | [UD page](https://universaldependencies.org/akk/) |

- **Annotation:** Full morphosyntactic: UPOS, features (case, number, gender, tense, mood, person), lemmas, dependency relations
- **Morphological system:** 5 cases (nominative, accusative, genitive, locative, terminative), 3 numbers (sg, pl, dual), 2 genders + common, verbal stems (G, D, Š, N + Gt, Dt, Št, Nt)

### B2. Akkadian-language-models Repository (Ong & Gordin)
- **URL:** https://github.com/megamattc/Akkadian-language-models
- **Content:** CoNLL-U annotated files from SAA volumes (letters, treaties, divination, court poetry) with full morphosyntactic annotation. Covers SAA 1–21 at varying completion percentages.
- **Method:** Iterative human-in-the-loop bootstrap annotation using Spacy + manual correction
- **Paper:** Ong & Gordin (2024), "Linguistic Annotation of Cuneiform Texts using Treebanks and Deep Learning"

### B3. BabyFST — Finite-State Morphological Analyzer for Akkadian
- **Paper:** Sahala, Silfverberg, Arppe & Lindén (LREC 2020)
- **Lexicon:** 255 verbal roots, 1,918 nouns, 235 adjectives, 625 names, 40 prepositions
- **Framework:** HFST / Foma finite-state compiler
- **Use for training:** Can generate all valid inflected forms for known lemmas → synthetic data augmentation for morphological coverage

### B4. BabyLemmatizer 2.0
- **Paper:** Sahala & Lindén (2023)
- **Description:** State-of-the-art neural lemmatizer for Akkadian, successor to BabyFST for disambiguation tasks. Can be used as a preprocessing step to normalize training data.

### B5. ORACC Glossaries
- **Access:** Available within each ORACC sub-project
- **Content:** Complete word lists with lemmatization, POS, citation form, guide word, and sense information for every word attested in the sub-project
- **Use for training:** Extract as bilingual dictionary for RAG augmentation or vocabulary seeding. The CuneiTranslate project demonstrated that exploding lemmas to their multiple senses improves translation performance.

---

## C. LEXICAL & DICTIONARY RESOURCES

### C1. FactGrid Cuneiform / Wikidata Lexemes
- **URL:** https://database.factgrid.de/ + https://zenodo.org/records/10819306
- **Content:** The largest Linked Open Data database of Sumerian and Akkadian lexemes mapped to all English senses. Includes Hittite. CSV format with lexeme IDs, labels, POS, and sense mappings.
- **Languages covered:** Akkadian, Sumerian, Hittite, Elamite, Arabic, Hebrew, English, and others
- **License:** CC0
- **Use for training:** Augment training vocabulary; create synthetic bilingual pairs; build RAG dictionaries

### C2. ePSD2 (electronic Pennsylvania Sumerian Dictionary)
- **URL:** http://oracc.museum.upenn.edu/epsd2/
- **Content:** Sumerian glossary, corpora, catalogue, sign list, and secondary literature index
- **Relevance:** Many Akkadian texts contain Sumerograms. ePSD2 maps Sumerian logograms to their Akkadian readings—essential for handling logographic elements in Akkadian text.

### C3. CAD (Chicago Assyrian Dictionary)
- **Status:** 21-volume complete dictionary (A–Z), now freely available digitally
- **Content:** Comprehensive Akkadian dictionary with attested forms, translations, and textual citations
- **Use:** Gold-standard lexical resource. Could be digitized/parsed for dictionary-based augmentation.

### C4. CDA (Concise Dictionary of Akkadian)
- **Authors:** Black, George & Postgate
- **Content:** ~8,000 entries, more accessible than CAD
- **Use:** Lightweight dictionary for quick lemma→English lookup in RAG systems

### C5. Ancient Mesopotamian Lexical Lists
- **Content:** Ancient bilingual Sumerian–Akkadian vocabularies (e.g., Ea = nâqu, Ura = ḫubullu, An = Anum). These are the original "dictionaries" created by Babylonian scholars themselves.
- **Digitized in:** ORACC/DCCLT project
- **Use:** Historically grounded word-pair training data; unique domain-specific vocabulary

---

## D. RELATED LANGUAGE RESOURCES (CROSS-LINGUAL TRANSFER)

Akkadian is an East Semitic language. Leveraging data from related Semitic languages can provide implicit morphological priors during pretraining or multi-task fine-tuning.

### D1. Arabic (Modern Standard + Classical)

| Resource | Description | Link |
|----------|-------------|------|
| **ATHAR dataset** | 66K Classical Arabic → English translation pairs | [HF](https://hf.co/datasets/mohamed-khalil/ATHAR) |
| **UN Parallel Corpus** | Arabic–English parallel UN documents | via OPUS |
| **Arabic UD Treebanks** | PADT, PUD, NYUAD — full morphosyntactic annotation | [UD](https://universaldependencies.org/) |
| **Helsinki opus-mt-ar-en** | Pretrained Arabic→English NMT model | [HF](https://hf.co/Helsinki-NLP/opus-mt-ar-en) |

**Why Arabic helps:** Arabic shares triconsonantal root morphology with Akkadian (e.g., Akkadian *šarrum* "king" ← root Š-R-R; Arabic *malik* "king" ← root M-L-K follow the same templatic pattern). Fine-tuning from Arabic→English primes the model for root-pattern morphology.

### D2. Hebrew (Biblical + Modern)

| Resource | Description | Link |
|----------|-------------|------|
| **UD Ancient Hebrew PTNK** | Biblical Hebrew treebank from BHS with ETCBC morphology | [UD](https://universaldependencies.org/) |
| **UD Hebrew-HTB/IAHLTwiki** | Modern Hebrew treebanks | [UD](https://universaldependencies.org/) |
| **OPUS parallel corpora** | Hebrew–English parallel texts | [OPUS](https://opus.nlpl.eu/) |

**Why Hebrew helps:** Biblical Hebrew is the closest well-resourced language to Akkadian. Both are ancient Semitic languages with case systems, construct states (smixut/status constructus), and shared vocabulary (Akk. *šulmu* ≈ Heb. *shalom*).

### D3. Aramaic

| Resource | Description |
|----------|-------------|
| **Deep Aramaic** (Aioanei et al. 2023) | 250,000 synthetic training images for Old Aramaic letter classification. Methodology transferable to cuneiform sign generation. |
| **UD Assyrian (Uppsala)** | Small treebank for Modern Standard Assyrian (Neo-Aramaic). Same script family heritage. |

### D4. Ethiopic / Ge'ez
- **Resources:** Amharic NER datasets, Tigrinya corpora
- **Relevance:** South Semitic languages preserving archaic features. Limited transfer value but useful for broadening Semitic language coverage in multilingual pretraining.

### D5. Ugaritic
- **Content:** ~1,500 known texts in a cuneiform alphabetic script, closely related to early Northwest Semitic
- **Digitized in:** ORACC and specialized corpora
- **Relevance:** Bridge language between cuneiform writing traditions and alphabetic Semitic

---

## E. ANALOGOUS ANCIENT LANGUAGES (CUNEIFORM FAMILY)

These are non-Semitic languages written in cuneiform that share the writing system and can provide multi-task learning signal.

### E1. Sumerian

| Resource | Description | Link |
|----------|-------------|------|
| **ETCSL** | ~400 Sumerian literary compositions with English translations | [Oxford](https://etcsl.orinst.ox.ac.uk/) |
| **ETCSRI** | Sumerian royal inscriptions | via ORACC |
| **SzuTao/Cuneiform** | 100K–1M Sumerian cuneiform symbols dataset | [HF](https://hf.co/datasets/SzuTao/Cuneiform) |
| **BDTNS** | Database of Neo-Sumerian Texts (72,000+ texts, ~2.2% translated) | [bdtns.filol.csic.es](http://bdtns.filol.csic.es/) |
| **Sumerian Lexicon v3.0** | Comprehensive Sumerian–English dictionary | [sumerian.org](https://www.sumerian.org/) |

**Why Sumerian helps:** Not linguistically related to Akkadian, but shares the same cuneiform writing system. Bilingual Sumerian–Akkadian texts are abundant. Training a model on both languages simultaneously (as Krueger's T5 does, and as Thalesian/cuneiformBase-400m does) improves generalization across sign-reading ambiguity.

### E2. Hittite

| Resource | Description |
|----------|-------------|
| **Thalesian/cuneiformBase-400m** | UMT5 model covering Hittite among other cuneiform languages | [HF](https://hf.co/Thalesian/cuneiformBase-400m) |
| **HPM (Hittite Parsed Morphology)** | Hittite texts with morphological analysis |
| **FactGrid Hittite lexemes** | Included in the Wikidata Lemmatization Dataset |

**Why Hittite helps:** Indo-European language written in cuneiform. Shares sign inventory with Akkadian. Multi-task training improves sign disambiguation.

### E3. Elamite
- Included in Thalesian/cuneiformBase-400m and FactGrid
- Limited parallel data, but useful for broadening cuneiform sign coverage

### E4. Linear B (Mycenaean Greek)
- Included in Thalesian/cuneiformBase-400m
- Different script system but similar decipherment challenges; useful for transfer learning on ancient language structure

---

## F. CUNEIFORM SIGN / OCR DATASETS

For the tablet→transliteration pipeline (Stage 1):

| Dataset | Content | Link |
|---------|---------|------|
| **HeiCuBeDa** | ~500 annotated cuneiform tablets (3D + photos) | via GigaMesh |
| **MaiCuBeDa** | Additional annotated tablets | via GigaMesh |
| **boatbomber/CuneiformPhotosMSII** | 10K–100K paired photo↔MSII curvature images | [HF](https://hf.co/datasets/boatbomber/CuneiformPhotosMSII) |
| **markzeits/fll-cuneiform** | 1K–10K cuneiform images with text labels | [HF](https://hf.co/datasets/markzeits/fll-cuneiform) |
| **markzeits/cuneiform_real_words** | 1K–10K real cuneiform word images | [HF](https://hf.co/datasets/markzeits/cuneiform_real_words) |
| **hrabalm/mtm24-akkadian-v3** | 100K–1M multimodal (image + text) | [HF](https://hf.co/datasets/hrabalm/mtm24-akkadian-v3) |

---

## G. RESEARCH PAPERS & BENCHMARKS

### Landmark Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| Translating Akkadian to English with NMT | Gutherz, Gordin, Sáenz, Levy, Berant | 2023 | First Akkadian NMT, BLEU4 36.52/37.47 |
| Reading Akkadian Cuneiform Using NLP | Gordin et al. | 2020 | 97% accuracy cuneiform→transliteration |
| Restoration of Fragmentary Babylonian Texts Using RNNs | Fetaya, Lifshitz, Aaron, Gordin | 2020 | RNN gap-filling in damaged tablets |
| Filling the Gaps in Ancient Akkadian Texts (MLM) | Lazar et al. | 2021 | BERT masked language modeling for Akkadian |
| Akkadian Treebank for Neo-Assyrian Royal Inscriptions | Luukko, Sahala, Hardwick, Lindén | 2020 | First UD treebank for Akkadian |
| BabyFST: Finite-State Model of Ancient Babylonian | Sahala, Silfverberg, Arppe, Lindén | 2020 | First computational morphological analyzer |
| Linguistic Annotation Using Treebanks and Deep Learning | Ong & Gordin | 2024 | Iterative human-in-the-loop annotation pipeline |
| Lemmatization of Cuneiform Using ByT5 | Lu, Huang, Xu, Feng, Xu | 2025 | ByT5 > mT5 for cuneiform lemmatization |
| Evaluating Transformers for Akkadian (RANLP 2025) | Jones & Mitkov | 2025 | 6-model comparison: LLMs cost-inefficient vs seq2seq; transfer from related languages is critical |
| Translating Akkadian with Transfer Learning (ICAART 2025) | Nehme, Azar, Kutsalo, Possik | 2025 | Transfer learning from Romance/Semitic languages |
| CNN Cuneiform Sign Detection | Stötzner, Homburg, Mara | 2023 | RepPoints detector on 3D tablet renderings |
| Are Character-level Translations Worth the Wait? | Edman, Sarti, Toral, van Noord, Bisazza | 2024 | Comprehensive ByT5 vs mT5 for NMT |
| CuneiTranslate (UC Berkeley) | Berkeley iSchool capstone | 2024 | Compared T5, mT5, opus-mt-ar-en, NLLB on Akkadian |
| Machine Translation from One Book (MTOB) | Tanzer et al. (Google) | 2024 | Benchmark for LLM translation from reference materials only |

### Active Research Groups

| Group | Institution | Focus |
|-------|------------|-------|
| **Digital Pasts Lab** | Ariel University (Israel) | Gordin lab — NMT, treebanks, Babylonian Inscription project |
| **ANEE / FIN-CLARIN** | University of Helsinki | Luukko, Sahala — treebanks, morphological analyzers |
| **DANES** (Digital Ancient Near Eastern Studies) | International consortium | Community coordination for digital Assyriology |
| **FactGrid Cuneiform** | UC Berkeley / Erfurt | Linked Open Data dictionaries |
| **Deep Past Initiative** | Yale / Non-profit | Kaggle competition, democratizing cuneiform access |

---

## H. TOOLS & CODE REPOSITORIES

| Tool | Description | Link |
|------|-------------|------|
| **Akkademia** | Gutherz/Gordin NMT pipeline (C2E + T2E) | [GitHub](https://github.com/gaigutherz/Akkademia) |
| **Akkadian-language-models** | Ong/Gordin Spacy models + annotated CoNLL-U data | [GitHub](https://github.com/megamattc/Akkadian-language-models) |
| **GigaMesh** | 3D mesh processing for cuneiform tablets (MSII renderings) | [gigamesh.eu](https://gigamesh.eu) |
| **BabyFST** | Finite-state morphological analyzer (HFST/Foma) | via Helsinki NLP |
| **BabyLemmatizer 2.0** | Neural lemmatizer for Akkadian | via Helsinki NLP |
| **Cuneify** | Transliteration → Unicode cuneiform glyph converter | via Akkademia repo |
| **L2 (ORACC)** | Dictionary-based lemmatizer used across all ORACC projects | via ORACC |
| **eAkkadian** | Online Akkadian language course + digital tools guide | [digitalpasts.github.io/eAkkadian](https://digitalpasts.github.io/eAkkadian/) |

---

## I. RECOMMENDED DATA PIPELINE

Given the model recommendation (ByT5-base) and the resources above, here is the optimal data assembly strategy:

### Phase 1: Core Training Data (~50K–100K pairs)
1. Scrape all ORACC sub-projects for transliteration↔English sentence pairs
2. Merge with phucthaiv02/akkadian-translation and cipher-ling/akkadian HF datasets
3. Deduplicate and normalize transliteration conventions (ASCII ↔ Unicode diacritics)
4. Split 90/5/5 (train/val/test), stratified by text genre and period

### Phase 2: Vocabulary Augmentation
5. Extract ORACC glossaries + FactGrid/Wikidata lexemes → bilingual dictionary
6. Generate synthetic sentence pairs from BabyFST inflected forms + dictionary definitions
7. Optionally include AICC silver-standard translations with lower training weight

### Phase 3: Cross-Lingual Pretraining (Optional but Recommended)
8. Multi-task fine-tune on Arabic→English (ATHAR dataset) + Hebrew→English (OPUS) before Akkadian fine-tuning
9. This primes the model's encoder for Semitic morphological patterns

### Phase 4: Multi-Script Augmentation (Optional)
10. Include Sumerian→English pairs from ETCSL for shared cuneiform sign coverage
11. Include bilingual Sumerian–Akkadian texts from ORACC/BLMS for internal consistency

### Phase 5: Evaluation
12. Test on held-out ORACC data + Deep Past Challenge test set
13. Metrics: BLEU4, chrF++, BERTScore (geometric mean of BLEU × chrF++ for competition compatibility)
14. Human evaluation by Assyriologists for genre-specific accuracy

---

*This document synthesizes findings from: PNAS Nexus (2023), RANLP 2025, ACL ALP Workshop 2025, ICAART 2025, TACL 2024, UC Berkeley CuneiTranslate (2024), the Kaggle Deep Past Challenge, Universal Dependencies, ORACC, CDLI, FactGrid, and the Hugging Face Hub.*
