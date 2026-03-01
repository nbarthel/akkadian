"""Test training step speed at different dataset sizes."""
import torch, time, pandas as pd, inspect, sys, os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback)
from datasets import Dataset

LOG = '/tmp/step_speed_results.txt'
f = open(LOG, 'w')

def log(msg):
    print(msg, flush=True)
    f.write(msg + '\n')
    f.flush()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Device: {device}')

tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('google/byt5-small').to(device)
log(f'Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')

df = pd.read_parquet('data/processed/train.parquet')
gold = df[df['quality'] == 'gold'][['transliteration', 'translation']].reset_index(drop=True)
log(f'Total gold rows: {len(gold)}')

prefix = 'translate Akkadian to English: '

def preprocess(examples):
    inputs = tokenizer([prefix + t for t in examples['transliteration']], max_length=512, truncation=True)
    labels = tokenizer(examples['translation'], max_length=512, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

for n in [500, 5000, 50000, 125787]:
    actual_n = min(n, len(gold))
    subset = gold.head(actual_n)
    ds = Dataset.from_pandas(subset)

    t0 = time.time()
    ds = ds.map(preprocess, batched=True, remove_columns=['transliteration', 'translation'])
    tok_time = time.time() - t0
    log(f'\nn={actual_n}: tokenized in {tok_time:.1f}s')

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=f'/tmp/test_n{actual_n}',
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=3,
        logging_steps=1,
        bf16=True,
        report_to='none',
        disable_tqdm=True,
    )

    class StepLogger(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.t = time.time()
            log(f'  train_begin (max_steps={state.max_steps})')
        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
            log(f'  step {state.global_step}: {now - self.t:.2f}s loss={loss}')
            self.t = now

    sig = inspect.signature(Seq2SeqTrainer.__init__)
    kw = dict(model=model, args=args, train_dataset=ds, data_collator=collator)
    kw['processing_class' if 'processing_class' in sig.parameters else 'tokenizer'] = tokenizer
    trainer = Seq2SeqTrainer(**kw)
    trainer.add_callback(StepLogger())

    log(f'n={actual_n}: starting 3 steps...')
    t0 = time.time()
    result = trainer.train()
    total = time.time() - t0
    log(f'n={actual_n}: done in {total:.1f}s ({total/3:.2f}s/step) loss={result.training_loss:.4f}')

log('\nAll done!')
f.close()
