import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2b-it"

print("🔍 Checking environment...\n")

# 1. Check PyTorch + GPU
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("\n📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("📦 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to(device)

print("\n✅ Model loaded successfully!\n")

# 2. Test inference
prompt = "What is a savings account in simple terms?"

print("🧪 Running test prompt...\n")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("💬 Model Response:\n")
print(response)

print("\n🎯 STATUS CHECK:")
print("✔ Model working")
print("✔ Tokenizer working")
print("✔ Inference working")

if torch.cuda.is_available():
    print("✔ GPU acceleration active 🚀")
else:
    print("⚠ Running on CPU (slower but fine)")

print("\n⚠ Note: This model is for informational purposes only (not financial advice).")