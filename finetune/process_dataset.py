from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict
import os

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYS = "You are a biomedical Q&A assistant. Answer concisely and only from the given context."
USR_TMPL = (
    "Answer the question using only the context.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

def build_example(ex: Dict, tok: AutoTokenizer, max_len: int = 1024):
    """Convert one (question, context, answer) row into decoder-only tensors."""
    question = (ex.get("question") or "").strip()
    context  = (ex.get("context")  or "").strip()
    answer   = (ex.get("answer")   or "").strip()
    if not answer:
        answer = "I don't know."

    if len(context) > 2500:
        context = context[:2500]

    prompt_text = (
        f"<|system|>\n{SYS}\n"
        f"<|user|>\n" + USR_TMPL.format(context=context, question=question) + "\n"
        f"<|assistant|>\n"
    )

    # Tokenize prompt and answer separately so we can mask prompt in labels.
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tok(answer, add_special_tokens=False).input_ids

    if len(answer_ids) >= max_len:
        kept_prompt_ids = []
        kept_answer_ids = answer_ids[-max_len:]
    else:
        space_for_prompt = max_len - len(answer_ids)
        kept_prompt_ids = prompt_ids[-space_for_prompt:] if len(prompt_ids) > space_for_prompt else prompt_ids
        kept_answer_ids = answer_ids

    input_ids = kept_prompt_ids + kept_answer_ids
    attention_mask = [1] * len(input_ids)

    # Labels: mask prompt with -100, learn only on the answer positions.
    labels = [-100] * len(kept_prompt_ids) + kept_answer_ids

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    # Resolve project root so we save under <project>/data/...
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "data", "bioasq_decoder_tiny")
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)

    # 1) Load a clean, Parquet-backed BioASQ Task B revision.
    ds = load_dataset("BastienHot/BioASQ-Task-B-Revised")
    # Splits: ds["train"], ds["test"]; each row has question/context/answer.

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Many chat models lack a pad token; use EOS as pad so batching works later.
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # 3) Map rows -> model-ready tensors
    def mapper(ex): return build_example(ex, tok, max_len=1024)
    original_cols = ds["train"].column_names
    proc = ds.map(mapper, remove_columns=original_cols, desc="Tokenizing & masking", num_proc=1)

    tiny = DatasetDict({
        "train": proc["train"].select(range(4000)),      # ~4k examples
        "validation": proc["test"].select(range(500)),   # quick val
    })
    tiny.save_to_disk(out_dir)
    print(f"✅ Saved processed dataset to: {out_dir}")

    # 5) Quick sanity check
    sample = tiny["train"][0]
    print({k: len(sample[k]) for k in sample})  # lengths
    print("labels head (should be -100s):", sample["labels"][:20])
    print("labels tail (should be token IDs):", sample["labels"][-20:])

    # Decode a bit so you can *see* prompt vs answer
    head_dec = tok.decode(sample["input_ids"][:200], skip_special_tokens=False)
    tail_dec = tok.decode(sample["input_ids"][-200:], skip_special_tokens=False)
    print("\n--- Decoded head (prompt region) ---\n", head_dec)
    print("\n--- Decoded tail (contains answer) ---\n", tail_dec)

if __name__ == "__main__":
    main()
