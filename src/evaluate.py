import os
import sys
import json
import random
import torch
import gc
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Ensure project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import BASE_MODEL_ID, ADAPTER_DIR, DATA_PATH, get_bnb_config
from src.preprocess import PROMPT_TEMPLATE_INFERENCE

# Evaluation parameters
NUM_SAMPLES = 1  # Fast check
MAX_NEW_TOKENS = 100

def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def load_base():
    print(f"[eval] Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map=device_map,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def load_ft():
    print(f"[eval] Loading fine-tuned model from: {ADAPTER_DIR}")
    from src.utils import load_finetuned_model
    return load_finetuned_model(ADAPTER_DIR)

def generate(model, tokenizer, query):
    prompt = PROMPT_TEMPLATE_INFERENCE.format(input=query.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def calculate_metrics(generated, reference):
    # BLEU
    smooth = SmoothingFunction().method1
    gen_tokens = generated.lower().split()
    ref_tokens = [reference.lower().split()]
    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth) * 100
    
    # Simple semantic overlap (Accuracy proxy)
    ref_set = set(reference.lower().split())
    gen_set = set(generated.lower().split())
    # Exclude common stop words for better accuracy measure
    stopwords = {'a', 'the', 'is', 'at', 'which', 'on', 'and', 'to', 'your', 'my', 'I'}
    ref_keywords = ref_set - stopwords
    matches = len(ref_keywords.intersection(gen_set))
    accuracy = (matches / len(ref_keywords)) * 100 if ref_keywords else 0
    
    return bleu, accuracy

def main():
    print("=" * 60)
    print("  BANKBOT AI METRIC CALCULATION")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    test_samples = random.sample(full_data, NUM_SAMPLES)
    
    results = {"base": {"bleu": [], "acc": []}, "ft": {"bleu": [], "acc": []}}
    
    # Evaluate Base
    model, tokenizer = load_base()
    print(f"\n[eval] Evaluating BASE model on {NUM_SAMPLES} samples...")
    for item in test_samples:
        gen = generate(model, tokenizer, item['input'])
        bleu, acc = calculate_metrics(gen, item['output'])
        results["base"]["bleu"].append(bleu)
        results["base"]["acc"].append(acc)
    
    unload_model(model, tokenizer)
    
    # Evaluate FT
    model, tokenizer = load_ft()
    print(f"\n[eval] Evaluating FINE-TUNED model on {NUM_SAMPLES} samples...")
    for item in test_samples:
        gen = generate(model, tokenizer, item['input'])
        bleu, acc = calculate_metrics(gen, item['output'])
        results["ft"]["bleu"].append(bleu)
        results["ft"]["acc"].append(acc)
    
    unload_model(model, tokenizer)
    
    # Final Averages
    base_bleu = sum(results["base"]["bleu"]) / NUM_SAMPLES
    base_acc = sum(results["base"]["acc"]) / NUM_SAMPLES
    ft_bleu = sum(results["ft"]["bleu"]) / NUM_SAMPLES
    ft_acc = sum(results["ft"]["acc"]) / NUM_SAMPLES
    
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"BASE      - BLEU: {base_bleu:.2f}, Accuracy Proxy: {base_acc:.2f}")
    print(f"FINE-TUNED - BLEU: {ft_bleu:.2f}, Accuracy Proxy: {ft_acc:.2f}")
    print("=" * 60)
    
    # Write to file
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "actual_metrics.json"), 'w') as f:
        json.dump({
            "base_bleu": round(base_bleu, 2),
            "base_acc": round(base_acc, 2),
            "ft_bleu": round(ft_bleu, 2),
            "ft_acc": round(ft_acc, 2)
        }, f, indent=4)

if __name__ == "__main__":
    main()
