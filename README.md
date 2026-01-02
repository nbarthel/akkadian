# Deep Past Challenge: Translate Akkadian to English

A Kaggle competition to build AI models that translate 4,000-year-old Old Assyrian business records into English.

## Competition Overview

| Detail | Value |
|--------|-------|
| **Organizer** | [Deep Past Initiative](https://www.deeppast.org/) |
| **Platform** | [Kaggle](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) |
| **Timeline** | December 16, 2025 - March 23, 2026 |
| **Prize Pool** | $50,000 |
| **Submission** | Notebooks |

## Challenge Description

The Deep Past Challenge is a machine learning and language translation competition focused on unlocking the 4,000-year-old trade records of Assyrian vendors. Cuneiform script represents "the ultimate computational frontier" - there is no harder script for a computer to decipher.

### Dataset
- 22,000+ ancient cuneiform tablets
- ~50% remain untranslated
- Focus on Old Assyrian business/trade records

### Evaluation Metric
Submissions are scored using the **geometric mean of BLEU and chrF++**:
- **BLEU**: Measures word/phrase overlap with reference translations
- **chrF++**: Measures character-level similarity
- **Final Score**: √(BLEU × chrF++)

This dual metric forces models to balance both word-level accuracy and character-level fidelity.

## Repository Structure

```
akkadian/
├── data/
│   ├── raw/           # Original competition data
│   ├── processed/     # Cleaned/transformed data
│   └── external/      # External datasets
├── notebooks/         # Jupyter notebooks for analysis & submission
├── models/            # Saved model checkpoints
├── src/
│   ├── data/          # Data loading & preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model architectures
│   └── visualization/ # Plotting utilities
└── submissions/       # Generated submission files
```

## Getting Started

1. Download competition data from Kaggle:
   ```bash
   kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw
   ```

2. Explore the baseline notebook in `notebooks/`

3. Submit via Kaggle Notebooks (Code Competition)

## Resources

- [Competition Page](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
- [Deep Past Initiative](https://www.deeppast.org/)
- [Discussion Forum](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion)

## License

Competition data is subject to Kaggle competition rules and Deep Past Initiative terms.
