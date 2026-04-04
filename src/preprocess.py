"""
preprocess.py – Dataset loading and tokenization for Gemma 2B fine-tuning.
"""

import json
import os
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizer


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = "### Instruction:\n{input}\n### Response:\n{output}"
PROMPT_TEMPLATE_INFERENCE = "### Instruction:\n{input}\n### Response:\n"

MAX_LENGTH = 512   # keep well within 4 GB VRAM budget


def load_raw_dataset(dataset_path: str) -> List[Dict]:
    """Load raw JSON dataset from disk."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[preprocess] Loaded {len(data)} samples from {dataset_path}")
    return data


def format_sample(sample: Dict) -> Dict:
    """Convert a raw {input, output} dict to instruction format."""
    text = PROMPT_TEMPLATE.format(
        input=sample["input"].strip(),
        output=sample["output"].strip(),
    )
    return {"text": text}


def build_hf_dataset(raw_data: List[Dict], val_split: float = 0.05) -> Dict[str, Dataset]:
    """
    Format all samples, create a HuggingFace Dataset, and split into
    train / validation sets.
    """
    formatted = [format_sample(s) for s in raw_data]
    hf_dataset = Dataset.from_list(formatted)

    split = hf_dataset.train_test_split(test_size=val_split, seed=42)
    print(
        f"[preprocess] Train: {len(split['train'])} | "
        f"Val: {len(split['test'])} samples"
    )
    return {"train": split["train"], "val": split["test"]}


def tokenize_dataset(
    dataset_dict: Dict[str, Dataset],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Dataset]:
    """
    Tokenize the formatted text column.
    Returns datasets with 'input_ids', 'attention_mask', and 'labels'.
    """

    def _tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        # For causal LM, labels == input_ids; mask padding with -100
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in tokenized["input_ids"]
        ]
        tokenized["labels"] = labels
        return tokenized

    tokenized = {}
    for split_name, ds in dataset_dict.items():
        tokenized[split_name] = ds.map(
            _tokenize,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split_name}",
        )
        tokenized[split_name].set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    return tokenized


if __name__ == "__main__":
    # Quick sanity check
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")
    raw = load_raw_dataset(DATA_PATH)
    splits = build_hf_dataset(raw)
    print("Sample formatted text:")
    print(splits["train"][0])
