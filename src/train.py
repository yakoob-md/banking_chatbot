"""
train.py – Fine-tune Gemma 2B on the banking support dataset using QLoRA.

Usage:
    python src/train.py

Requires ~3.5 GB VRAM (RTX 2050 compatible with 4-bit quantization).
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

from src.preprocess import load_raw_dataset, build_hf_dataset, tokenize_dataset
from src.utils import (
    DATA_PATH,
    ADAPTER_DIR,
    load_base_model_for_training,
)


# --------------------------------------------------------------------------- #
# Training hyper-parameters (tuned for 4 GB VRAM)
# --------------------------------------------------------------------------- #
# TRAINING_ARGS = TrainingArguments(
#     output_dir=ADAPTER_DIR,
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=8,   # effective batch = 8
#     warmup_steps=50,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=50,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     report_to="none",                # disable wandb / tensorboard
#     optim="paged_adamw_8bit",        # memory-efficient optimiser
#     dataloader_pin_memory=False,
#     save_total_limit=2,
#     group_by_length=True,
# )

TRAINING_ARGS = TrainingArguments(
    output_dir=ADAPTER_DIR,

    # 🔥 ADD THIS LINE
    max_steps=200,

    # 🔥 CHANGE THIS (optional but recommended)
    num_train_epochs=1,   # or remove completely

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",   # 🔥 change from "epoch"
    eval_steps=50,                 # evaluate every 50 steps
    save_strategy="steps",         # 🔥 change from "epoch"
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",
    dataloader_pin_memory=False,
    save_total_limit=2,
    group_by_length=True,
)


def main():
    print("=" * 60)
    print("  Banking Chatbot – QLoRA Fine-Tuning (Gemma 2B)")
    print("=" * 60)

    # 1. Load & preprocess data
    raw_data = load_raw_dataset(DATA_PATH)
    dataset_dict = build_hf_dataset(raw_data, val_split=0.05)

    # 2. Load model & tokenizer
    model, tokenizer = load_base_model_for_training()

    # 3. Tokenize
    tokenized = tokenize_dataset(dataset_dict, tokenizer)

    # 4. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    print("\n[train] Starting training …")
    trainer.train()

    # 7. Save adapter + tokenizer
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n[train] Adapter saved to: {ADAPTER_DIR}")

    # 8. Quick eval
    eval_results = trainer.evaluate()
    print(f"\n[train] Evaluation results: {eval_results}")
    print("\n[train] Training complete!")


if __name__ == "__main__":
    main()
