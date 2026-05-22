from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

class HallucinationDS(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples; self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        e = encode_sample(self.samples[idx], self.tok, self.max_length)
        return {"input_ids": e.input_ids, "attention_mask": e.attention_mask, "labels": e.labels}

train_ds = HallucinationDS(train_samples, tokenizer, MAX_LENGTH)
val_ds   = HallucinationDS(val_samples,   tokenizer, MAX_LENGTH)

def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id or 0
    def pad(seq, val): return seq + [val] * (ml - len(seq))
    return {
        "input_ids":      torch.tensor([pad(b["input_ids"], pad_id) for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0)  for b in batch], dtype=torch.long),
        "labels":         torch.tensor([pad(b["labels"], -100)        for b in batch], dtype=torch.long),
    }

import inspect
ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)

if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    try:
        model.config.use_cache = False
    except Exception:
        pass

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    eval_steps=EVAL_EVERY_STEPS,
    save_steps=EVAL_EVERY_STEPS,
    save_total_limit=2,
    logging_steps=50,
    bf16=torch.cuda.is_available(),
    seed=SEED,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
# Pick whichever name this transformers version recognizes for eval/save strategy.
if "eval_strategy" in ta_params:
    args_kwargs["eval_strategy"] = "steps"
elif "evaluation_strategy" in ta_params:
    args_kwargs["evaluation_strategy"] = "steps"
args_kwargs["save_strategy"] = "steps"

args = TrainingArguments(**args_kwargs)

trainer_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate)
trainer_params = set(inspect.signature(Trainer.__init__).parameters)
# transformers ≥4.46 renamed `tokenizer` to `processing_class`.
if "processing_class" in trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)
trainer.train()
