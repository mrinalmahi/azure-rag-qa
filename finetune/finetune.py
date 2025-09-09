import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "bioasq_decoder_tiny")  
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tinyllama_bioasq_lora")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

NUM_EPOCHS = 3
MAX_STEPS  = None
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR         = 2e-4
WARMUP     = 0.03
LOG_STEPS  = 50
EVAL_STEPS = 200
SAVE_STEPS = 200
MAX_LEN    = 1024   

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

ds = load_from_disk(DATA_DIR)
train_ds = ds["train"]
val_ds   = ds["validation"]

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

try:
    from transformers import BitsAndBytesConfig
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_cfg,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    print("✅ Loaded base model in 8-bit for LoRA training.")
except Exception:
    print("⚠️ bitsandbytes not available; loading full-precision model.")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

model = get_peft_model(model, LORA)

for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

if hasattr(model, "config"):
    model.config.use_cache = False

model.print_trainable_parameters()

@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]

        labels    = [f["labels"] for f in features] if "labels" in features[0] else [ids[:] for ids in input_ids]

        # Pad inputs
        batch_inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].long()

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        batch_labels[batch_labels == pad_id] = -100

        batch_inputs["labels"] = batch_labels
        return batch_inputs

collator = DataCollatorForCausalLM(tokenizer=tok, max_length=MAX_LEN)

args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  num_train_epochs=NUM_EPOCHS if not MAX_STEPS else 1,
  max_steps=MAX_STEPS if MAX_STEPS else -1,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=GRAD_ACCUM,
  learning_rate=LR,
  warmup_ratio=WARMUP,
  logging_steps=LOG_STEPS,
  eval_strategy="steps",
  eval_steps=EVAL_STEPS,
  save_strategy="steps",
  save_steps=SAVE_STEPS,
  save_total_limit=2,
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  greater_is_better=False,            
  fp16=True,
  report_to="none",
  dataloader_num_workers=0,
  save_safetensors=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,            
    data_collator=collator,   
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train()
    best = trainer.state.best_model_checkpoint
    print("BEST:", trainer.state.best_model_checkpoint)
    trainer.save_model(OUTPUT_DIR)  
    tok.save_pretrained(OUTPUT_DIR)
    if best:
            with open(os.path.join(OUTPUT_DIR, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
                f.write(best)
    print(f" Finished. Artifacts saved to: {OUTPUT_DIR}")
