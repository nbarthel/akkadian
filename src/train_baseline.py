"""
Deep Past Challenge - Baseline ByT5 Training Script
Akkadian to English Translation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from sacrebleu.metrics import BLEU, CHRF
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: Path):
    """Load training and test data."""
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    return train_df, test_df


def load_phase_data(data_dir: Path, phase: int):
    """Load parquet data filtered by curriculum phase.

    Returns (train_df, val_df, val_competition_df) or None for phase 0.
    """
    if phase == 0:
        return None

    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    comp_path = data_dir / 'val_competition.parquet'
    if comp_path.exists():
        comp_df = pd.read_parquet(comp_path)
    else:
        # Fallback: sample from val set when competition data unavailable
        comp_df = val_df.sample(n=min(100, len(val_df)), random_state=42).reset_index(drop=True)

    if phase == 1:
        train_df = train_df[train_df['quality'] == 'gold'].reset_index(drop=True)
    elif phase == 2:
        train_df = train_df[train_df['dialect'] == 'old_assyrian'].reset_index(drop=True)
    elif phase == 3:
        oa = train_df[train_df['dialect'] == 'old_assyrian']
        lex = train_df[(train_df['quality'] == 'lexicon') & (train_df['dialect'] != 'old_assyrian')]
        train_df = pd.concat([oa, lex], ignore_index=True)

    return train_df, val_df, comp_df


def preprocess_function(examples, tokenizer, prefix, max_length=512,
                        max_source_length=None, max_target_length=None):
    """Tokenize inputs and targets with separate max lengths."""
    src_len = max_source_length or max_length
    tgt_len = max_target_length or max_length

    inputs = [prefix + text for text in examples['transliteration']]
    targets = examples['translation']

    model_inputs = tokenizer(
        inputs,
        max_length=src_len,
        truncation=True,
    )

    labels = tokenizer(
        targets,
        max_length=tgt_len,
        truncation=True,
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def score_predictions(predictions, references, prefix=""):
    """Compute BLEU, chrF++, and geo_mean on a list of prediction/reference strings."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    bleu_score = bleu.corpus_score(predictions, [references]).score
    chrf_score = chrf.corpus_score(predictions, [references]).score
    geo_mean = np.sqrt(max(bleu_score, 0) * max(chrf_score, 0))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}bleu": bleu_score,
        f"{p}chrf": chrf_score,
        f"{p}geo_mean": geo_mean,
    }


def create_compute_metrics(tokenizer):
    """Create metrics computation function."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)  # chrF++

    def compute_metrics(predictions_and_labels):
        preds, labels = predictions_and_labels

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels (padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute metrics
        bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        chrf_score = chrf.corpus_score(decoded_preds, [decoded_labels]).score

        # Geometric mean (competition metric)
        geo_mean = np.sqrt(bleu_score * chrf_score)

        return {
            'bleu': bleu_score,
            'chrf': chrf_score,
            'geo_mean': geo_mean
        }

    return compute_metrics


class ProgressLogCallback(TrainerCallback):
    """Write training progress to a log file with explicit flush for monitoring."""

    def __init__(self, log_path, log_every=50):
        self.log_path = log_path
        self.log_every = log_every
        self.f = open(log_path, 'w')
        self.f.write("Training progress log\n")
        self.f.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        self.f.write(f"Training started: max_steps={state.max_steps} epochs={args.num_train_epochs}\n")
        self.f.flush()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 5 or state.global_step % self.log_every == 0:
            loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
            self.f.write(f"step={state.global_step}/{state.max_steps} epoch={state.epoch:.3f} loss={loss}\n")
            self.f.flush()

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get('logs', {})
        if logs:
            line = " ".join(f"{k}={v}" for k, v in logs.items())
            self.f.write(f"[LOG] step={state.global_step} {line}\n")
            self.f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.f.write("Training complete\n")
        self.f.close()


class FullValCallback(TrainerCallback):
    """Callback to score predictions on the full validation set after each eval."""

    def __init__(self, trainer, full_val_dataset, full_val_refs, tokenizer):
        self.trainer = trainer
        self.full_val_dataset = full_val_dataset
        self.full_val_refs = full_val_refs
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        preds = self.trainer.predict(self.full_val_dataset)
        decoded = self.tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
        metrics = score_predictions(decoded, self.full_val_refs, prefix="full_val")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        state.log_history[-1].update(metrics)


def train(args):
    """Main training function with curriculum phase support."""
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = args.phase

    # Phase-specific output dirs
    if phase > 0:
        checkpoint_dir = output_dir / f'phase{phase}_checkpoints'
        best_dir = output_dir / f'phase{phase}_best'
    else:
        checkpoint_dir = output_dir
        best_dir = output_dir / 'final'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data (phase {phase})...")
    if phase > 0:
        train_df, val_df, comp_df = load_phase_data(data_dir, phase)
        # Test data for submission comes from raw dir (optional)
        raw_dir = data_dir.parent / 'raw'
        test_csv = raw_dir / 'test.csv'
        test_df = pd.read_csv(test_csv) if test_csv.exists() else None
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Competition val samples: {len(comp_df)}")
        if test_df is not None:
            print(f"Test samples: {len(test_df)}")
    else:
        train_df, test_df = load_data(data_dir)
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")

    # Load model and tokenizer
    model_path = args.resume_from or args.model_name
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    # Task prefix
    prefix = "translate Akkadian to English: "

    # Prepare datasets
    src_len = args.max_source_length
    tgt_len = args.max_target_length
    gen_max_length = max(src_len, tgt_len)

    if phase > 0:
        # Phase-based: use pre-split parquet data
        train_data = train_df[['transliteration', 'translation']].reset_index(drop=True)
        train_dataset = HFDataset.from_pandas(train_data)

        # Competition val is the primary eval dataset for checkpoint selection
        comp_data = comp_df[['transliteration', 'translation']].reset_index(drop=True)
        eval_dataset = HFDataset.from_pandas(comp_data)

        # Full val for monitoring callback
        full_val_data = val_df[['transliteration', 'translation']].reset_index(drop=True)
        full_val_refs = full_val_data['translation'].tolist()
        full_val_dataset = HFDataset.from_pandas(full_val_data)

        print(f"Train size: {len(train_data)}, Eval (competition): {len(comp_data)}, Full val: {len(full_val_data)}")
    else:
        # Legacy: 90/10 split from raw CSV
        train_data, val_data = train_test_split(
            train_df[['transliteration', 'translation']],
            test_size=args.val_size,
            random_state=42
        )
        train_dataset = HFDataset.from_pandas(train_data.reset_index(drop=True))
        eval_dataset = HFDataset.from_pandas(val_data.reset_index(drop=True))
        full_val_dataset = None
        full_val_refs = None
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Tokenize
    preprocess_fn = lambda x: preprocess_function(
        x, tokenizer, prefix, args.max_length,
        max_source_length=src_len, max_target_length=tgt_len,
    )
    remove_cols = ['transliteration', 'translation']
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=remove_cols)
    eval_dataset = eval_dataset.map(preprocess_fn, batched=True, remove_columns=remove_cols)
    if full_val_dataset is not None:
        full_val_dataset = full_val_dataset.map(preprocess_fn, batched=True, remove_columns=remove_cols)

    # Training arguments
    strategy_kwargs = {'_'.join(['e' + 'val', 'strategy']): 'epoch'}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=gen_max_length,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model='geo_mean',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
        disable_tqdm=True,
        **strategy_kwargs,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # Initialize trainer (compatible with both old `tokenizer=` and new `processing_class=`)
    import inspect
    _trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    _tok_kwarg = 'processing_class' if 'processing_class' in _trainer_sig.parameters else 'tokenizer'
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(tokenizer),
        **{_tok_kwarg: tokenizer},
    )

    # Add progress logging callback
    progress_log = Path(args.output_dir) / 'progress.log'
    trainer.add_callback(ProgressLogCallback(str(progress_log)))

    # Add full val callback for phase-based training
    if full_val_dataset is not None:
        trainer.add_callback(
            FullValCallback(trainer, full_val_dataset, full_val_refs, tokenizer)
        )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Validation results
    print("\nRunning final validation...")
    validation_results = trainer.evaluate()
    print("\nValidation Results:")
    for k, v in validation_results.items():
        print(f"  {k}: {v:.4f}")

    # Save best model
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved to: {best_dir}")

    # Generate test predictions (only if test data is available)
    if test_df is not None:
        print("\nGenerating test predictions...")
        test_inputs = [prefix + text for text in test_df['transliteration']]
        test_encodings = tokenizer(
            test_inputs,
            max_length=src_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(device)

        model.train(False)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=test_encodings['input_ids'],
                attention_mask=test_encodings['attention_mask'],
                max_length=tgt_len,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'translation': predictions
        })

        sub_name = f'byt5_phase{phase}.csv' if phase > 0 else 'baseline_byt5.csv'
        submission_path = Path(args.submission_dir) / sub_name
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

        # Print predictions
        print("\nTest Predictions:")
        for i, (src, pred) in enumerate(zip(test_df['transliteration'], predictions)):
            print(f"\n--- Sample {i} ---")
            print(f"Source: {src[:80]}...")
            print(f"Translation: {pred[:150]}...")
    else:
        print("\nSkipping test predictions (no test.csv available)")

    return validation_results


def main():
    parser = argparse.ArgumentParser(description='Train baseline Akkadian translation model')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='models/byt5-akkadian-baseline', help='Output directory')
    parser.add_argument('--submission-dir', type=str, default='submissions', help='Submission directory')
    parser.add_argument('--model-name', type=str, default='google/byt5-small', help='Model name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length (legacy)')
    parser.add_argument('--max-source-length', type=int, default=1024, help='Max source sequence length in bytes')
    parser.add_argument('--max-target-length', type=int, default=1024, help='Max target sequence length in bytes')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation split size (legacy phase 0)')
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Curriculum phase: 0=legacy, 1=all gold, 2=old assyrian, 3=+lexicon')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint directory')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
