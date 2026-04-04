"""
utils.py – Shared helpers: model loading, LoRA config, path constants.
"""

import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "dataset.json")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
ADAPTER_DIR  = os.path.join(MODELS_DIR, "banking_chatbot_adapter")

# --------------------------------------------------------------------------- #
# Model identifier
# --------------------------------------------------------------------------- #
BASE_MODEL_ID = "google/gemma-2b-it"   # instruction-tuned variant of Gemma 2B


# --------------------------------------------------------------------------- #
# Quantization config (4-bit QLoRA)
# --------------------------------------------------------------------------- #
def get_bnb_config() -> BitsAndBytesConfig:
    # bitsandbytes (4-bit/8-bit) requires CUDA. Fallback to None if not available.
    if not torch.cuda.is_available():
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# --------------------------------------------------------------------------- #
# LoRA configuration
# --------------------------------------------------------------------------- #
def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],   # Gemma attention projections
    )


# --------------------------------------------------------------------------- #
# Load base model + tokenizer (for training)
# --------------------------------------------------------------------------- #
def load_base_model_for_training():
    """Load Gemma 2B in 4-bit and wrap with LoRA adapters."""
    print(f"[utils] Loading base model: {BASE_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    # Gemma uses eos as pad; set explicitly to avoid warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    # Use float32 on CPU for better compatibility; float16 on GPU
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # model = prepare_model_for_kbit_training(model)

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Load fine-tuned model (for inference)
# --------------------------------------------------------------------------- #
def load_finetuned_model(adapter_dir: str = ADAPTER_DIR):
    """Load the base model and merge the saved LoRA adapter."""
    from peft import PeftModel

    print(f"[utils] Loading fine-tuned adapter from: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load base model (without quantization if on CPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(), # Returns None if no CUDA
        device_map=device_map,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer
