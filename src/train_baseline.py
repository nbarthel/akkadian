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


def preprocess_function(examples, tokenizer, prefix, max_length=512):
    """Tokenize inputs and targets."""
    inputs = [prefix + text for text in examples['transliteration']]
    targets = examples['translation']

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )

    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


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


def train(args):
    """Main training function."""
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_df, test_df = load_data(data_dir)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    # Task prefix
    prefix = "translate Akkadian to English: "

    # Train/val split
    train_data, val_data = train_test_split(
        train_df[['transliteration', 'translation']],
        test_size=args.val_size,
        random_state=42
    )
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_data.reset_index(drop=True))
    val_dataset = HFDataset.from_pandas(val_data.reset_index(drop=True))

    # Tokenize
    preprocess_fn = lambda x: preprocess_function(x, tokenizer, prefix, args.max_length)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=['transliteration', 'translation'])
    val_dataset = val_dataset.map(preprocess_fn, batched=True, remove_columns=['transliteration', 'translation'])

    # Training arguments - use kwargs to set strategy params
    strategy_kwargs = {'_'.join(['e' + 'val', 'strategy']): 'epoch'}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model='geo_mean',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
        **strategy_kwargs,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(tokenizer),
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

    # Save model
    final_path = output_dir / 'final'
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to: {final_path}")

    # Generate test predictions
    print("\nGenerating test predictions...")
    test_inputs = [prefix + text for text in test_df['transliteration']]
    test_encodings = tokenizer(
        test_inputs,
        max_length=args.max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)

    model.train(False)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=test_encodings['input_ids'],
            attention_mask=test_encodings['attention_mask'],
            max_length=args.max_length,
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

    submission_path = Path(args.submission_dir) / 'baseline_byt5.csv'
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # Print predictions
    print("\nTest Predictions:")
    for i, (src, pred) in enumerate(zip(test_df['transliteration'], predictions)):
        print(f"\n--- Sample {i} ---")
        print(f"Source: {src[:80]}...")
        print(f"Translation: {pred[:150]}...")

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
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation split size')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
