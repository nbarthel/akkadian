"""Quick training test to verify BF16 + dynamic padding works."""
import sys, torch, pandas as pd, time, inspect, os
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

LOG = open('/home/nbarthel/projects/akkadian/logs/test_training.log', 'w')
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')
    LOG.flush()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Device: {device}')

tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('google/byt5-small').to(device)
log(f'Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')

df = pd.read_parquet('data/processed/train.parquet')
gold = df[df['quality']=='gold'].head(500)[['transliteration','translation']].reset_index(drop=True)
ds = Dataset.from_pandas(gold)

prefix = 'translate Akkadian to English: '
def preprocess(examples):
    inputs = tokenizer([prefix + t for t in examples['transliteration']], max_length=512, truncation=True)
    labels = tokenizer(examples['translation'], max_length=512, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

ds = ds.map(preprocess, batched=True, remove_columns=['transliteration','translation'])
log(f'Dataset: {len(ds)} samples, sample input len={len(ds[0]["input_ids"])}')

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

args = Seq2SeqTrainingArguments(
    output_dir='/tmp/test_bf16_training',
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    max_steps=20,
    logging_steps=1,
    bf16=True,
    report_to='none',
    disable_tqdm=True,
)

sig = inspect.signature(Seq2SeqTrainer.__init__)
kw = dict(model=model, args=args, train_dataset=ds, data_collator=collator)
kw['processing_class' if 'processing_class' in sig.parameters else 'tokenizer'] = tokenizer

from transformers import TrainerCallback
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log(f"  step={state.global_step} " + " ".join(f"{k}={v}" for k,v in logs.items()))

trainer = Seq2SeqTrainer(**kw)
trainer.add_callback(LogCallback())

log('Starting training (20 steps, bf16, bs=16, dynamic padding)...')
t0 = time.time()
result = trainer.train()
t1 = time.time()
log(f'Done in {t1-t0:.1f}s ({(t1-t0)/20:.2f}s/step)')
log(f'Final loss: {result.training_loss:.4f}')
LOG.close()
