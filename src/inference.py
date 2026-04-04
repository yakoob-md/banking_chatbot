"""
inference.py – Generate responses from the fine-tuned banking chatbot.

Usage (standalone):
    python src/inference.py

Or import generate_response() for use in app.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.preprocess import PROMPT_TEMPLATE_INFERENCE
from src.utils import ADAPTER_DIR, load_finetuned_model

# --------------------------------------------------------------------------- #
# Generation settings
# --------------------------------------------------------------------------- #
MAX_NEW_TOKENS = 120   # Reduced from 256 for faster responses on 4GB VRAM
TEMPERATURE    = 0.7
TOP_P          = 0.9
REPETITION_PENALTY = 1.15

# Module-level cache so Flask doesn't reload on every request
_model = None
_tokenizer = None


def _load_model_once():
    """Load model into module-level cache (called once per process)."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_finetuned_model(ADAPTER_DIR)
    return _model, _tokenizer


def _unload_finetuned():
    """Evict the cached fine-tuned model from VRAM to free space for the base model."""
    global _model, _tokenizer
    import gc
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def generate_base_response(user_input: str) -> str:
    """
    Generate a response from the unmodified base Gemma model (no LoRA adapter).
    Evicts the cached fine-tuned model first to free VRAM, loads base model,
    generates, then frees it so fine-tuned can be reloaded next call.
    """
    import gc
    from src.utils import BASE_MODEL_ID, get_bnb_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Step 1: Free the cached fine-tuned model so we have VRAM headroom
    _unload_finetuned()

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.eval()

    prompt = PROMPT_TEMPLATE_INFERENCE.format(input=user_input.strip())
    inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(base_model.device)

    with torch.no_grad():
        out = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    response = base_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Step 2: Free the base model so fine-tuned can load fresh
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return response



def generate_response(user_input: str) -> str:
    """
    Given a raw user message, return the chatbot's response string.
    """
    model, tokenizer = _load_model_once()

    prompt = PROMPT_TEMPLATE_INFERENCE.format(input=user_input.strip())

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=300,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# --------------------------------------------------------------------------- #
# Comparison helper (base vs fine-tuned)
# --------------------------------------------------------------------------- #
def compare_base_vs_finetuned(user_input: str):
    """
    Qualitative comparison: print base model output vs fine-tuned output.
    Useful for evaluation and academic demonstration.
    """
    from src.utils import BASE_MODEL_ID, get_bnb_config
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("\n" + "=" * 60)
    print("  BASE MODEL RESPONSE (no fine-tuning)")
    print("=" * 60)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.eval()

    prompt = PROMPT_TEMPLATE_INFERENCE.format(input=user_input.strip())
    inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=base_tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    base_response = base_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"Input   : {user_input}")
    print(f"Response: {base_response}")

    # Free up GPU memory so the finetuned model can load
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  FINE-TUNED MODEL RESPONSE")
    print("=" * 60)
    ft_response = generate_response(user_input)
    print(f"Input   : {user_input}")
    print(f"Response: {ft_response}")
    print("=" * 60 + "\n")

    return base_response, ft_response


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Banking Chatbot – Inference")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        response = generate_response(user_input)
        print(f"Bot: {response}\n")
