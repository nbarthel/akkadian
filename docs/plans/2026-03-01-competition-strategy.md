# Deep Past Competition Strategy

**Competition:** Deep Past Initiative — Machine Translation (Akkadian → English)
**Deadline:** March 23, 2026 (22 days remaining as of March 1)
**Metric:** √(BLEU × chrF++) via sacrebleu
**Test set:** 4 tablet-scale transliterations (long inputs, 300–900 chars)

---

## Current State

### What's Done
- **Data pipeline:** 161K assembled pairs in `data/processed/` (train: 145K, val: 8K, competition val: 95)
- **Training code:** `src/train_baseline.py` supports 3-phase curriculum learning with dual validation
- **Kaggle setup:** Dataset uploaded (`nicbarthelemy1/akkadian-assembled-161k`), curriculum notebook pushed and running

### What's NOT Done
- No completed training run — no baseline score on the leaderboard
- No evaluation of existing HuggingFace models on our competition val proxy
- No submission to Kaggle yet

### Data Breakdown (train.parquet)

| Source | Rows | Quality |
|--------|------|---------|
| hf_phucthaiv02_translation | 65,438 | gold |
| hf_cipher_ling | 41,912 | gold |
| ebl_dictionary | 13,833 | lexicon |
| hf_phucthaiv02_alignment | 9,566 | gold |
| oare_sentences | 7,476 | gold |
| oa_lexicon | 5,735 | lexicon |
| kaggle_train | 1,395 | gold |

**Critical mismatch:** Training data averages 65 chars/sample, but competition val averages 435 chars (tablet-scale). `MAX_SOURCE_LENGTH` must be ≥768 bytes to avoid truncating test inputs.

---

## Available Models on HuggingFace

| Model | Arch | Size | Notes |
|-------|------|------|-------|
| `notninja/byt5-base-akkadian` | ByT5-base | 580M | Already fine-tuned on Akkadian — potential warm start for Phase 2 |
| `Thalesian/cuneiformBase-400m` | UMT5 | 400M | Multi-language cuneiform (Feb 2026). Evaluate zero-shot |
| `phucthaiv02/akkadian-nllb-2` | NLLB/M2M | ~600M | NLLB adapted for Akkadian. Potential KD teacher |
| `Hippopoto0/akkadianT5` | T5 | ~220M | MIT license, lightweight |
| `Hippopoto0/akkadian-marianMT` | MarianMT | ~74M | Lightweight, fast inference |

**Key insight:** `notninja/byt5-base-akkadian` can skip Phase 1 entirely — it IS a general-Akkadian fine-tuned ByT5-base. Jump straight to Phase 2 (Old Assyrian specialization), saving 10+ GPU hours.

---

## Model Strategy: Three Tiers

### Tier 1: Baseline (Week 1 — get a score on the board)

1. **Zero-shot eval** existing HF models on `val_competition.parquet` to establish a floor
2. **Phase 2 only** from `notninja/byt5-base-akkadian` → 10 epochs on 15K Old Assyrian samples (~2-3 hours on T4)
3. **Fix MAX_SOURCE_LENGTH** to 768+ bytes for tablet-scale inputs
4. **Submit** — get on the leaderboard

### Tier 2: Scaled Training (Week 2 — improve the score)

5. **ByT5-base curriculum** if `notninja` warm-start underperforms: Phase 1 (3 epochs on 126K gold, ~4-5h) → Phase 2 (10 epochs on 15K OA, ~2-3h)
6. **Sequence-level distillation from NLLB** (see below)
7. **Request access** to `mik3ml/akkadian` gated dataset (potentially 100K+ new pairs)
8. **Tune decoding:** length_penalty (0.6–1.4), beam size (5–10), no_repeat_ngram_size (try 0 vs 3)

### Tier 3: Advanced Techniques (Week 3 — squeeze the metric)

9. **Ensemble:** Average predictions from fine-tuned ByT5 + `Thalesian/cuneiformBase-400m`
10. **Dictionary-augmented inference:** Prepend relevant eBL Dictionary entries to test inputs
11. **Full KD training** from NLLB if sequence-level distillation showed promise
12. **Final submission:** Best single model or ensemble with tuned decoding

---

## NLLB Knowledge Distillation Assessment

### The Question
Is cross-architecture distillation from NLLB-200 (token-level teacher) into ByT5-large (byte-level student) worth the implementation cost?

### Why It Could Help
- NLLB's Semitic language priors (Arabic, Hebrew, Amharic) share morphological patterns with Akkadian (triconsonantal roots, case endings, bound morphemes)
- Soft targets capture inter-word relationships that hard labels miss
- `phucthaiv02/akkadian-nllb-2` already exists as an Akkadian-adapted teacher — no need to fine-tune NLLB from scratch
- ByT5's byte-level tokenization handles cuneiform notation better, but benefits from NLLB's linguistic knowledge

### Why It's Risky
- **Cross-architecture alignment is hard:** Token-level and byte-level vocabularies don't align. Can't compare logits directly. Must use sequence-level or hidden-state alignment.
- **Memory pressure:** NLLB 1.3B + ByT5-large 1.2B = ~5GB+ in fp16 just for weights, plus optimizer states and activations. Very tight on T4 16GB.
- **Implementation time:** Custom `DistillationTrainer` with cross-architecture loss = 2-3 days to implement and debug.
- **No baseline yet:** Optimizing an advanced technique before proving the basic pipeline works is premature.
- **4 test samples:** High variance means marginal model improvements may not register.

### Recommendation: Sequence-Level Distillation (Practical Middle Ground)

Instead of full KD training with custom loss, use **sequence-level distillation**:

1. Run `phucthaiv02/akkadian-nllb-2` on all training transliterations → generate English translations
2. Add NLLB-generated translations as additional training targets (with lower weight or as a separate augmentation pass)
3. Train ByT5 on both original + NLLB-generated data

**Benefits:**
- Gets ~80% of KD benefit with ~20% of the complexity
- No custom trainer needed — standard fine-tuning on augmented data
- No memory issues — run NLLB inference separately, then train ByT5 alone
- Can be done in a few hours of Kaggle GPU time

**When to escalate to full KD:**
- Only after Tier 1 baseline is on the leaderboard
- Only if sequence-level distillation shows clear improvement on competition val
- Only if ByT5-large fits in T4 memory at reasonable batch size (test first)

### Full KD Implementation (If Pursued)

```python
class DistillationTrainer(Seq2SeqTrainer):
    """Cross-architecture distillation: NLLB teacher → ByT5 student."""

    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha  # weight: alpha * KD_loss + (1-alpha) * CE_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward pass (ByT5)
        student_outputs = model(**inputs)
        ce_loss = student_outputs.loss

        # Teacher forward pass (NLLB) — need to re-tokenize inputs
        # for NLLB's tokenizer, which is the cross-architecture challenge
        with torch.no_grad():
            teacher_outputs = self.teacher(**teacher_inputs)

        # Sequence-level KL divergence on generated sequences
        # (since vocab sizes differ, align at sequence probability level)
        kd_loss = self._sequence_kl_divergence(
            student_outputs, teacher_outputs, self.temperature
        )

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return (loss, student_outputs) if return_outputs else loss
```

**Key challenge:** The `teacher_inputs` must be re-tokenized with NLLB's SentencePiece tokenizer, and probability alignment must happen at the sequence level (not per-position) since vocabularies differ.

---

## Kaggle Constraints

| Resource | Limit |
|----------|-------|
| T4 GPU VRAM | 16 GB |
| System RAM | ~30 GB |
| Weekly GPU quota | 30 hours |
| Submission runtime | 9 hours max |
| Disk space | ~20 GB working |

**Implication:** Phase 1 (5 epochs × 126K samples) may exceed 9 hours on T4. Use `notninja/byt5-base-akkadian` warm start to skip Phase 1, or reduce to 2-3 epochs.

---

## High-Leverage Quick Wins

1. **Warm-start from `notninja/byt5-base-akkadian`** — skip Phase 1 entirely, save 10+ GPU hours
2. **Fix MAX_SOURCE_LENGTH to 768+** — competition inputs are tablet-scale, current 512 truncates them
3. **Zero-shot eval existing models** — 5 minutes of inference, establishes a score floor
4. **Dictionary-augmented inference** — for 4 test inputs, prepend relevant eBL Dictionary entries to the source transliteration. High-leverage for very little effort.
5. **Tune length_penalty** — tablet translations are long; default length_penalty=1.0 may produce outputs that are too short
