# Decoding Akkadian: A Resource Dataset for AI-Assisted Translation

> A curated collection of datasets, models, corpora, papers, and tools for applying modern NLP and machine learning to the ancient Akkadian language and cuneiform script.

---

## 1. Foundational Digital Corpora

These are the upstream scholarly sources from which virtually all ML datasets are derived.

| Corpus | Description | URL |
|--------|-------------|-----|
| **ORACC** (Open Richly Annotated Cuneiform Corpus) | The richest annotated cuneiform corpus; includes transliterations, translations, lemmatizations, and POS tags across dozens of sub-projects. CC BY-SA 3.0. | [oracc.museum.upenn.edu](https://oracc.museum.upenn.edu/) |
| **CDLI** (Cuneiform Digital Library Initiative) | The foundational global catalog — 350,000+ cuneiform objects with metadata, transliterations, and images. Updated nightly. | [cdli.mpiwg-berlin.mpg.de](https://cdli.mpiwg-berlin.mpg.de/) |
| **AICC** (AI Cuneiform Corpus) | 130,000 AI-translated cuneiform texts (Sumerian + Akkadian → English) built from CDLI/ORACC by Frank Krueger using a custom T5 model. The largest translated cuneiform corpus online. | [praeclarum.org/2023/06/09/cuneiform.html](https://praeclarum.org/2023/06/09/cuneiform.html) |
| **eBL** (electronic Babylonian Literature) | Fragmentarium for literary texts — annotate, transliterate, and reconstruct Mesopotamian literary works. Extensive philological editions. | [www.ebl.lmu.de](https://www.ebl.lmu.de/) |
| **ETCSL** (Electronic Text Corpus of Sumerian Literature) | ~400 Sumerian literary compositions with translations. Useful for bilingual Sumerian–Akkadian training data. | [etcsl.orinst.ox.ac.uk](https://etcsl.orinst.ox.ac.uk/) |
| **ARMEP** (Ancient Records of Middle Eastern Polities) | Multi-project search across ORACC, focused on first millennium BCE political texts. | [oracc.museum.upenn.edu/armep](https://oracc.museum.upenn.edu/armep/) |
| **SAA** (State Archives of Assyria) | Scholarly editions of Neo-Assyrian royal letters, treaties, divination queries, and more. Multiple volumes digitized on ORACC. | via ORACC sub-projects |

---

## 2. Hugging Face Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **phucthaiv02/akkadian-translation** | 10K–100K | Akkadian–English parallel translation pairs in Parquet format | [HF Link](https://hf.co/datasets/phucthaiv02/akkadian-translation) |
| **phucthaiv02/akkadian_english_sentences_alignment_2** | 10K–100K | Sentence-aligned Akkadian–English pairs (v2, Jan 2026) | [HF Link](https://hf.co/datasets/phucthaiv02/akkadian_english_sentences_alignment_2) |
| **cipher-ling/akkadian** | 10K–100K | Akkadian text dataset (149 downloads, 5 likes) | [HF Link](https://hf.co/datasets/cipher-ling/akkadian) |
| **veezbo/akkadian_english_corpus** | 1K–10K | Cleaned Akkadian–English corpus for text generation / LLM fine-tuning. MIT license. | [HF Link](https://hf.co/datasets/veezbo/akkadian_english_corpus) |
| **hrabalm/mtm24-akkadian-v3** | 100K–1M | Multimodal (image + text) Akkadian dataset, likely cuneiform sign images with transliterations | [HF Link](https://hf.co/datasets/hrabalm/mtm24-akkadian-v3) |
| **hrabalm/mtm24-akkadian-v1** | 10K–100K | Earlier version of the above multimodal dataset | [HF Link](https://hf.co/datasets/hrabalm/mtm24-akkadian-v1) |
| **mik3ml/akkadian** | 100K–1M | Large-scale Akkadian dataset. CC-BY-4.0. Gated access. | [HF Link](https://hf.co/datasets/mik3ml/akkadian) |
| **phucthaiv02/akkadian-publish-texts** | 1K–10K | Published Akkadian texts in CSV format | [HF Link](https://hf.co/datasets/phucthaiv02/akkadian-publish-texts) |
| **SzuTao/Cuneiform** | 100K–1M | Sumerian cuneiform symbols dataset in CSV | [HF Link](https://hf.co/datasets/SzuTao/Cuneiform) |
| **markzeits/fll-cuneiform** | 1K–10K | Cuneiform image–text dataset for OCR tasks | [HF Link](https://hf.co/datasets/markzeits/fll-cuneiform) |
| **markzeits/cuneiform_real_words** | 1K–10K | Real-word cuneiform images with labels | [HF Link](https://hf.co/datasets/markzeits/cuneiform_real_words) |
| **boatbomber/CuneiformPhotosMSII** | 10K–100K | Paired photo-realistic renders ↔ MSII curvature visualizations of cuneiform tablets. Apache 2.0. | [HF Link](https://hf.co/datasets/boatbomber/CuneiformPhotosMSII) |

---

## 3. Hugging Face Models

### Translation Models

| Model | Architecture | Description | Link |
|-------|-------------|-------------|------|
| **Thalesian/cuneiformBase-400m** | UMT5 (400M) | Multi-language cuneiform model — Akkadian, Sumerian, Hittite, Linear B, Elamite → English. Translation + transliteration. *Feb 2026, 246 downloads.* | [HF Link](https://hf.co/Thalesian/cuneiformBase-400m) |
| **praeclarum/cuneiform** | T5 | The original AICC model for Akkadian/Sumerian → English. MIT license. 315 downloads. | [HF Link](https://hf.co/praeclarum/cuneiform) |
| **ragunath-ravi/mamba-akkadian-translator** | Mamba 2.8B + LoRA | State-space model fine-tuned for Akkadian → English. Apache 2.0. | [HF Link](https://hf.co/ragunath-ravi/mamba-akkadian-translator) |
| **frankmorales2020/akkadian-to-english-translator** | mBART | Akkadian–English translation using multilingual BART. | [HF Link](https://hf.co/frankmorales2020/akkadian-to-english-translator) |
| **notninja/byt5-base-akkadian** | ByT5 | Byte-level T5 for Akkadian (handles rare characters well). 184 downloads. | [HF Link](https://hf.co/notninja/byt5-base-akkadian) |
| **nexaaii/akkadian-t5-base-v1** | T5 | Akkadian T5 model with GGUF quantized variant available. | [HF Link](https://hf.co/nexaaii/akkadian-t5-base-v1) |
| **Hippopoto0/akkadianT5** | T5 | MIT-licensed Akkadian T5 translation model. | [HF Link](https://hf.co/Hippopoto0/akkadianT5) |
| **Hippopoto0/akkadian-marianMT** | MarianMT | Lightweight translation model for Akkadian. MIT license. | [HF Link](https://hf.co/Hippopoto0/akkadian-marianMT) |
| **Hippopoto0/akkadianQwen** | Qwen2 | Akkadian fine-tune on Qwen architecture. MIT license. | [HF Link](https://hf.co/Hippopoto0/akkadianQwen) |

### LLM Fine-tunes

| Model | Base | Description | Link |
|-------|------|-------------|------|
| **veezbo/LLama-2-7b-chat-hf-akkadian** | LLaMA-2 7B Chat | PEFT/LoRA fine-tune on Akkadian corpus for conversational Akkadian generation. | [HF Link](https://hf.co/veezbo/LLama-2-7b-chat-hf-akkadian) |
| **veezbo/LLama-2-7b-hf-akkadian** | LLaMA-2 7B | Base model variant of the above. | [HF Link](https://hf.co/veezbo/LLama-2-7b-hf-akkadian) |
| **Hippopoto0/akkadianMistral** | Mistral 7B v0.3 | 4-bit quantized LoRA fine-tune for Akkadian tasks. | [HF Link](https://hf.co/Hippopoto0/akkadianMistral) |

### NLLB / Multilingual Adapters

| Model | Architecture | Description | Link |
|-------|-------------|-------------|------|
| **phucthaiv02/akkadian-nllb** | M2M-100 (NLLB) | NLLB adapted for Akkadian feature extraction. | [HF Link](https://hf.co/phucthaiv02/akkadian-nllb) |
| **phucthaiv02/akkadian-nllb-2** | M2M-100 (NLLB) | Text-to-text variant of NLLB for Akkadian. | [HF Link](https://hf.co/phucthaiv02/akkadian-nllb-2) |
| **NAQarabash/nllb-akkadian-transliteration-lora** | NLLB + LoRA | Transliteration-focused LoRA adapter. | [HF Link](https://hf.co/NAQarabash/nllb-akkadian-transliteration-lora) |

### OCR / Vision

| Model | Description | Link |
|-------|-------------|------|
| **markzeits/fll-deepseek-ocr-cuneiform** | DeepSeek-based OCR for cuneiform sign recognition. | [HF Link](https://hf.co/markzeits/fll-deepseek-ocr-cuneiform) |
| **marie-saccucci/cuneiform-segmentation-unet** | U-Net for cuneiform tablet segmentation. | [HF Link](https://hf.co/marie-saccucci/cuneiform-segmentation-unet) |

---

## 4. Key Research Papers

### Landmark: Akkadian Neural Machine Translation

**"Translating Akkadian to English with Neural Machine Translation"**
*Gutherz, Gordin, Sáenz, Levy & Berant — PNAS Nexus, May 2023*

The foundational paper for Akkadian NMT. Trained on ~50,000 ORACC sentence pairs. Achieved BLEU4 scores of 36.52 (cuneiform → English) and 37.47 (transliteration → English). Introduced both C2E (cuneiform-to-English) and T2E (transliteration-to-English) pipelines. Code available as "Akkademia" on GitHub.

- Paper: [doi.org/10.1093/pnasnexus/pgad096](https://doi.org/10.1093/pnasnexus/pgad096)
- GitHub: [github.com/gaigutherz/Akkademia](https://github.com/gaigutherz/Akkademia)

### Earlier Work by the Same Group

**"Reading Akkadian Cuneiform Using Natural Language Processing"**
*Gordin et al. — PLoS ONE, 2020*

Achieved 97% accuracy on cuneiform-to-transliteration using NLP methods. The precursor to the full translation pipeline.

- Paper: [doi.org/10.1371/journal.pone.0240511](https://doi.org/10.1371/journal.pone.0240511)

### Text Restoration

**"Restoration of Fragmentary Babylonian Texts Using Recurrent Neural Networks"**
*Fetaya, Lifshitz, Aaron & Gordin — PNAS, 2020*

Used RNNs to predict missing text in damaged clay tablets.

- Paper: [doi.org/10.1073/pnas.2003794117](https://doi.org/10.1073/pnas.2003794117)

**"Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach"**
*Lazar et al. — EMNLP 2021*

Applied BERT-style masked language modeling to reconstruct damaged Akkadian passages.

- Paper: [aclanthology.org/2021.emnlp-main.368](https://aclanthology.org/2021.emnlp-main.368)

### Morphological / Linguistic Annotation

**"Linguistic Annotation of Cuneiform Texts Using Treebanks and Deep Learning"**
*Ong & Gordin — Digital Scholarship in the Humanities, 2024*

Full morphosyntactic annotation pipeline for Akkadian using iterative human-in-the-loop treebank development.

- Paper: [doi.org/10.1093/llc/fqae002](https://doi.org/10.1093/llc/fqae002)
- Code & Data: [github.com/megamattc/Akkadian-language-models](https://github.com/megamattc/Akkadian-language-models)

**"BabyFST: A Finite-State Based Computational Model of Ancient Babylonian"**
*Sahala, Silfverberg, Arppe & Lindén — LREC 2020*

Finite-state transducer for Babylonian morphological analysis.

### Cuneiform Sign Detection / OCR

**"CNN Based Cuneiform Sign Detection Learned from Annotated 3D Renderings"**
*Stötzner, Homburg & Mara — 2023*

RepPoints-based detector for cuneiform sign localization using the HeiCuBeDa and MaiCuBeDa 3D tablet datasets (~500 annotated tablets).

- Paper: [hf.co/papers/2308.11277](https://hf.co/papers/2308.11277)

### Related Ancient Language Work

**"Deep Aramaic: Synthetic Data Paradigm for Machine Learning in Epigraphy"**
*Aioanei et al. — 2023*

Generated 250,000 synthetic training images for Old Aramaic letter classification — methodology directly transferable to cuneiform.

- Paper: [hf.co/papers/2310.07310](https://hf.co/papers/2310.07310)

**"Exploring Large Language Models for Classical Philology"**
*Riemenschneider & Frank — 2023*

RoBERTa and T5 models for Ancient Greek — provides a strong template for adapting modern architectures to dead languages.

- Paper: [hf.co/papers/2305.13698](https://hf.co/papers/2305.13698)

**"Larth: Dataset and Machine Translation for Etruscan"**
*Vico & Spanakis — 2023*

2,891 Etruscan → English pairs — methodologically comparable low-resource ancient language MT effort.

- Paper: [hf.co/papers/2310.05688](https://hf.co/papers/2310.05688)

---

## 5. GitHub Repositories & Tools

| Repository | Description | Link |
|-----------|-------------|------|
| **Akkademia** | Source code for the Gutherz/Gordin NMT pipeline (C2E and T2E). Includes data preprocessing, training, and evaluation scripts. | [github.com/gaigutherz/Akkademia](https://github.com/gaigutherz/Akkademia) |
| **Akkadian-language-models** | Tools and annotated data for ML on Akkadian — includes CoNLL-U treebank files from SAA volumes with morphosyntactic annotations. | [github.com/megamattc/Akkadian-language-models](https://github.com/megamattc/Akkadian-language-models) |
| **GigaMesh** | Open-source framework for 3D mesh processing of cuneiform tablets. Produces the MSII curvature renderings used in OCR research. | [gigamesh.eu](https://gigamesh.eu) |
| **Cuneify** | Tool for converting transliterations to Unicode cuneiform glyphs (used in the PNAS Nexus paper). | Referenced in Akkademia repo |
| **ORACC ATF Tools** | The ATF (ASCII Transliteration Format) checker, lemmatizer, and related tooling for working with cuneiform digital editions. | [oracc.museum.upenn.edu/doc/help/editinginatf](https://oracc.museum.upenn.edu/doc/help/editinginatf/) |

---

## 6. Online Learning & Reference

| Resource | Description | URL |
|----------|-------------|-----|
| **eAkkadian** | Digital Pasts Lab's online Akkadian language course, including a guide to digital Assyriology projects. | [digitalpasts.github.io/eAkkadian](https://digitalpasts.github.io/eAkkadian/) |
| **ORACC Akkadian Linguistic Annotation** | Official guide to lemmatization, POS tagging, and morphological annotation conventions for Akkadian in ORACC. | [oracc.museum.upenn.edu/doc/help/languages/akkadian](https://oracc.museum.upenn.edu/doc/help/languages/akkadian/) |
| **CDLI ATF Primer** | How to read and write cuneiform text files in the standard digital interchange format. | [oracc.museum.upenn.edu/doc/help/editinginatf/cdliatf](https://oracc.museum.upenn.edu/doc/help/editinginatf/cdliatf/) |
| **ML4AL Workshop** (ACL 2024) | Annual workshop on Machine Learning for Ancient Languages — papers on cuneiform transliteration, sign detection, and more. | [aclanthology.org/venues/ml4al](https://aclanthology.org/venues/ml4al/) |

---

## 7. Key Challenges & Opportunities

**Why Akkadian is hard for ML:**

- Extreme low-resource scenario (~50K translated sentence pairs vs. billions for modern languages)
- 3,000+ years of diachronic variation — sign forms, dialects (Old Babylonian, Neo-Assyrian, etc.), and genres shift dramatically
- Cuneiform is inherently logographic + syllabic, with extensive polyphony (one sign, multiple readings) and homophony (one sound, multiple signs)
- Clay tablets are frequently damaged, requiring gap-filling alongside translation
- Training data is heavily skewed toward administrative/legal genres; literary texts remain underrepresented

**Active frontiers:**

- Multimodal pipelines: 3D tablet scans → sign detection → transliteration → translation (end-to-end)
- Human-in-the-loop annotation acceleration (the Ong & Gordin iterative treebank approach)
- Cross-lingual transfer from modern Semitic languages (Arabic, Hebrew) as related linguistic priors
- Synthetic data generation for cuneiform sign recognition
- LLM fine-tuning with LoRA/QLoRA for efficient Akkadian adaptation on consumer hardware

---

*Compiled February 2026. Sources: Hugging Face Hub, ORACC, CDLI, PNAS Nexus, ACL Anthology, GitHub.*
