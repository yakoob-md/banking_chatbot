
# 🏦 Banking Customer Support Chatbot

Fine-tuned **Gemma 2B** on a 2 500-sample banking support dataset using **QLoRA**
(4-bit quantisation + LoRA adapters via PEFT), deployable on a 4 GB VRAM GPU
(RTX 2050 or equivalent).

---
hu

## 📁 Project Structure h

```
banking_chatbot/
│── data/
│   └── dataset.json          ← 2 500 Q&A samples (input / output)
│
│── src/
│   ├── preprocess.py         ← Dataset loading, formatting, tokenisation
│   ├── utils.py              ← Model loading helpers, path constants
│   ├── train.py              ← QLoRA fine-tuning script
│   └── inference.py          ← Response generation + base-vs-FT comparison
│
│── app/
│   ├── app.py                ← Flask web server
│   └── templates/
│       └── index.html        ← Chat UI
│
│── models/
│   └── banking_chatbot_adapter/   ← Saved LoRA adapter (created after training)
│
│── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- CUDA-capable GPU (≥ 4 GB VRAM) with CUDA 12.1 drivers
- Git + pip

### 2. Clone / unzip the project
```bash
cd banking_chatbot
```

### 3. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **CUDA wheel** – if the above installs a CPU-only PyTorch:
> ```bash
> pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
>     -f https://download.pytorch.org/whl/torch_stable.html
> ```

### 5. HuggingFace login (required for Gemma weights)
```bash
pip install huggingface_hub
huggingface-cli login   
```
You must also accept Gemma's licence at  
https://huggingface.co/google/gemma-2b-it

---

## 🚀 Training

```bash
python src/train.py
```

What happens:
1. Loads `data/dataset.json` (2 500 samples)
2. Formats as `### Instruction:\n{input}\n### Response:\n{output}`
3. Tokenises to max length 512
4. Loads `google/gemma-2b-it` in **4-bit NF4** quantisation
5. Wraps with **LoRA** (r=8, alpha=16, target: q_proj + v_proj)
6. Trains for **3 epochs** with batch=1 + gradient_accumulation=8
7. Saves adapter to `models/banking_chatbot_adapter/`

Expected training time on RTX 2050: ~2–4 hours for 3 epochs.

---

## 💬 Running the Chat UI

```bash
python app/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Features:
- Chat-style layout with typing indicator
- Handles multi-turn messages
- Warns users never to share PINs / OTPs

---

## 🔬 Evaluation – Base vs Fine-tuned

```python
from src.inference import compare_base_vs_finetuned

compare_base_vs_finetuned("I lost my debit card, what should I do?")
```

Example output:

```
=============================================
  BASE MODEL RESPONSE (no fine-tuning)
=============================================
Input   : I lost my debit card, what should I do?
Response: A lost debit card is unfortunate. You might want to consider ...
          [generic / off-topic]

=============================================
  FINE-TUNED MODEL RESPONSE
=============================================
Input   : I lost my debit card, what should I do?
Response: 1. Please follow standard banking procedure for this issue.
          2. Contact customer care or use mobile banking.
          3. Keep transaction/reference details ready.
          4. Do not share OTP, PIN, or passwords.
=============================================
```

---

## 📊 Dataset

| Field   | Details |
|---------|---------|
| Samples | 2 500   |
| Format  | `{"input": "...", "output": "..."}` |
| Topics  | Account balance, fraud, RBI rules, chargebacks, international payments, card block, OTP safety |
| Special | Emotion-aware inputs, multi-step responses, edge-case handling |

---

## 🛠️ Key Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-2b-it` |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| LoRA rank `r` | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | `q_proj`, `v_proj` |
| Batch size | 1 (effective 8 with grad accum) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max seq length | 512 |

---

## ⚠️ Notes

- Do **not** share OTPs, PINs, or passwords with anyone—including this chatbot.
- The model generates advisory responses; always verify with your bank.
- On first load, model download (~5 GB) may take several minutes.
