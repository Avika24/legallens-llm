# ⚖️ Standalone Legal Transformer LLM (From Scratch)

A complete, self-contained **Decoder-Only Transformer Language Model** built strictly from scratch in **PyTorch**. 
This folder contains **only the native LLM core**: no frontend, no web servers, and zero cloud API keys.

---

## 📂 Project Directory Structure

```text
legal-llm/
├── model.py           # 6-layer Decoder-Only Transformer architecture (PyTorch)
├── tokenizer.py       # Custom legal Regex based word level tokenizer (5,787 tokens)
├── engine.py          # Autoregressive generation & live token streaming engine
├── trainer.py         # PyTorch AdamW training loop on Apple Silicon GPU (MPS) / CPU
├── run.py             # Interactive CLI runner (Prompting, Risk, Rewrite, Retrain)
├── test_llm.py        # 5 automated unit tests verifying architecture & math
├── requirements.txt   # Minimal dependencies (torch, numpy)
└── data/
    ├── vocab.json                     # Serialized legal vocabulary
    ├── legal_transformer_weights.pt   # Pretrained PyTorch checkpoint (1.47M params)
    ├── constitution_corpus.json       # Indian Constitution text reference
    └── legal_qa_corpus.json           # Indian Supreme Court QA dataset
```

---

## 🚀 How to Open and Run in VS Code

### 1. Open this folder in VS Code
Run in your Mac Terminal:
```bash
code ~/LegalLens-LLM
```
*(Or open VS Code, click **File → Open Folder...**, and select `~/LegalLens-LLM`)*

### 2. Activate the Python Virtual Environment
In the VS Code terminal:
```bash
# Use the virtual environment with PyTorch pre-installed:
source venv/bin/activate
```

### 3. Run the Interactive LLM CLI
```bash
python3 run.py
```
This launches the interactive terminal menu where you can:
- **1:** Enter custom prompts for real-time autoregressive token generation.
- **2:** Run clause risk analysis (`<clause> ... <risk>`).
- **3:** Run clause rewriting (`<clause> ... <rewrite>`).
- **4:** Inspect model architecture and parameter count.
- **5:** Retrain the model on your Mac's Apple Silicon Metal GPU (`mps`).

---

## 📐 Transformer Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only | Causal autoregressive language model |
| **Total Parameters** | **12,958,848** (~12.96M) | Full trainable parameter count |
| **Transformer Blocks** | **6 Layers** | Stacked Pre-LayerNorm blocks |
| **Attention Heads** | **6 Heads** | Multi-Head Self-Attention ($d_k = 64$) |
| **Embedding Dim ($d_{model}$)** | **384** | High-dimensional vector representation size |
| **FFN Dimension ($d_{ff}$)** | **1536** | Position-wise feed-forward ($4 \times d_{model}$) with GeLU |
| **Context Length** | **256 tokens** | Max input/output sequence length |
| **Vocabulary Size** | **5,787 tokens** | Tailored for Indian statutes, contracts, and case law |
| **Weight Tying** | Enabled | $W_{\text{lm\_head}} = W_{\text{token\_emb}}$ (saves 2,222,208 parameters) |
| **Normalization** | Pre-LN | LayerNorm before MHA and FFN (modern GPT/LLaMA style) |
| **Hardware Target** | Apple Silicon (`mps`) | Auto-falls back to CUDA or CPU |

---

## 🔬 Mathematical Formulation

### 1. Multi-Head Causal Self-Attention
For input representations $X \in \mathbb{R}^{B \times T \times d_{model}}$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $M_{i, j} = 0$ for $j \le i$, and $M_{i, j} = -\infty$ for $j > i$ (causal lower-triangular mask).

### 2. Position-wise Feed-Forward Network
$$\text{FFN}(x) = \text{GeLU}(x W_1 + b_1) W_2 + b_2$$

### 3. Pre-LayerNorm Residual Connection
$$x^{(l)}_{attn} = x^{(l-1)} + \text{MHA}(\text{LayerNorm}(x^{(l-1)}))$$
$$x^{(l)} = x^{(l)}_{attn} + \text{FFN}(\text{LayerNorm}(x^{(l)}_{attn}))$$

---

## 🧪 Running Unit Tests

Verify the mathematical correctness and tensor shapes:
```bash
python3 test_llm.py
```
**Output:**
- `test_tokenizer_basic`: Verifies vocabulary mapping, `<bos>`, `<eos>`, and decoding.
- `test_transformer_forward_pass`: Verifies `[B, T, vocab_size]` output shape and cross-entropy loss.
- `test_causal_masking`: Verifies upper-triangular masking prevents future token leakage.
- `test_dataset_next_token_alignment`: Verifies shifted next-token alignment ($x_{1..t} \to y_{2..t+1}$).
- `test_engine_inference`: Verifies streaming generation on Apple Silicon GPU (`mps`).

---

## ⚡ Quick CLI Commands

```bash
# Print model summary
python3 run.py --summary

# Run automated clause risk and rewriting demo
python3 run.py --demo

# Generate continuation for a specific prompt
python3 run.py --prompt "Employee shall not engage in competing business"

# Retrain the LLM from scratch on local GPU
python3 trainer.py
```
