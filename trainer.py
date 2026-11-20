"""
trainer.py: Scaled Deep Training Pipeline for Legal Transformer LLM (~13M Parameters).

Trains a 6-layer Decoder-Only Transformer on:
1. IndicLegalQA Dataset (Indian Supreme Court QA records)
2. The Constitution of India (Articles 1-448)
3. Curated Indian Statutory Instruction Corpus:
   - Contract Risk Analysis (§27, §73, §28 ICA, §12(5) Arb Act, §74 ICA)
   - Role-Directed Clause Rewriting with Enforceable Caps
   - Citizen Legal Advisories:
     * Marriage of Choice & Adult Rights (Shafin Jahan / Lata Singh / Shakti Vahini)
     * Threat of Disowning & Property Rights (Hindu Succession Act, 1956)
     * WhatsApp Chat Privacy & Cyber Offenses (IT Act §72, §66E, Puttaswamy)
     * Non-Compete Bonds & Service Agreements (§27 ICA)
     * Landlord Security Deposit Disputes
     * Defamation & Online Harassment (IPC 499 / BNS 356)
     * Cheque Bounce (Section 138 NI Act)
     * E-Commerce Defective Product Refunds

Model Scale:
- Hidden Dimension (d_model): 384
- Multi-Head Attention: 6 Heads (d_k = 64)
- Transformer Blocks: 6 Layers
- FFN Dimension: 1536 (4 * d_model) with GeLU
- Max Context Length: 256 Tokens
- Vocabulary: 6,000+ Legal Subwords
- Total Parameters: ~13,000,000 (~13.05M Parameters)
- Compute Target: Apple Silicon Metal GPU (MPS) / Host CPU
"""

import json
import time
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LegalTransformerLM
from tokenizer import LegalTokenizer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VOCAB_PATH = DATA_DIR / "vocab.json"
WEIGHTS_PATH = DATA_DIR / "legal_transformer_weights.pt"
CURATED_CORPUS_PATH = DATA_DIR / "curated_legal_corpus.json"

# Datasets located in local data directory
RAW_DATA_DIR = DATA_DIR
CONSTITUTION_DIR = DATA_DIR
QA_FILE = RAW_DATA_DIR / "legal_qa_corpus.json"
CONST_FILE = CONSTITUTION_DIR / "constitution_corpus.json"


class LegalDataset(Dataset):
    """
    Causal Language Modeling Dataset with shifted next-token alignment:
      Input  x = [t_1, t_2, ..., t_{N-1}]
      Target y = [t_2, t_3, ..., t_N]
    """
    def __init__(self, tokenized_sequences: List[List[int]], max_seq_len: int = 256, pad_id: int = 0):
        self.samples = []
        for seq in tokenized_sequences:
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            else:
                seq = seq + [pad_id] * (max_seq_len - len(seq))
            self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        x = seq[:-1]
        y = seq[1:]  # Shifted right by 1 for next-token prediction
        return x, y


def load_training_corpora() -> List[str]:
    """Aggregates and balances Indian legal corpora for deep multi-task instruction learning."""
    sequences: List[str] = []

    # 1. Curated High-Value Legal Instruction Corpus (Primary Task Data)
    if CURATED_CORPUS_PATH.exists():
        with open(CURATED_CORPUS_PATH, "r", encoding="utf-8") as f:
            curated = json.load(f)

        # A. Contract Risk & Clause Rewriting
        for item in curated.get("contract_risk_and_rewrites", []):
            clause = item["clause"]
            risk = item["risk"]
            rewrite = item["rewrite"]

            seq_risk = f"<bos> <clause> {clause} <risk> {risk} <eos>"
            seq_rew = f"<bos> <clause> {clause} <rewrite> {rewrite} <eos>"
            seq_full = f"<bos> <clause> {clause} <risk> {risk} <rewrite> {rewrite} <eos>"
            seq_user_risk = f"<bos> <user> What is the legal risk in: {clause} <sep> <assistant> {risk} <eos>"
            seq_user_rew = f"<bos> <user> Rewrite this clause to protect my interests: {clause} <sep> <assistant> {rewrite} <eos>"

            # Oversample to give equal weight to transactional contract tasks
            for _ in range(10):
                sequences.extend([seq_risk, seq_rew, seq_full, seq_user_risk, seq_user_rew])

        # B. Citizen Legal Inquiries & Statutory Remedies
        for item in curated.get("citizen_legal_advisories", []):
            q = item["query"]
            ans = item["response"]
            seq_citizen = f"<bos> <user> {q} <sep> <assistant> {ans} <eos>"
            for _ in range(12):
                sequences.append(seq_citizen)

        # C. Key Constitutional Articles
        for item in curated.get("constitutional_articles", []):
            q = item["question"]
            ans = item["answer"]
            seq_const = f"<bos> <user> {q} <sep> <assistant> {ans} <eos>"
            for _ in range(10):
                sequences.append(seq_const)

    # 2. General Constitution Articles (Articles 1-448)
    if CONST_FILE.exists():
        with open(CONST_FILE, "r", encoding="utf-8") as f:
            const_data = json.load(f)
            articles = const_data if isinstance(const_data, list) else const_data.get("articles", [])
            for art in articles:
                num = art.get("article_number", "")
                title = art.get("title", "")
                summary = art.get("summary", "")
                text = f"<bos> <user> What is {num} of the Indian Constitution? <sep> <assistant> {title}: {summary} <eos>"
                sequences.append(text)

    # 3. IndicLegalQA Dataset (10,000 SC Case Records)
    if QA_FILE.exists():
        with open(QA_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
            count_qa = 0
            for d in qa_data:
                q = d.get("question", "").strip()
                a = d.get("answer", "").strip()
                case = d.get("case_name", "").strip()
                if 25 < len(a) < 220 and len(q) < 140:
                    sequences.append(f"<bos> <user> In {case}: {q} <sep> <assistant> {a} <eos>")
                    count_qa += 1
                    if count_qa >= 1500:
                        break

    return sequences


def train_legal_llm(
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 5e-4,
    d_model: int = 384,
    n_heads: int = 6,
    n_layers: int = 6,
    max_seq_len: int = 256
) -> Tuple[LegalTransformerLM, LegalTokenizer]:
    """
    Trains the scaled ~13M Parameter Legal Transformer LLM on full Indian legal corpora.
    """
    print("=" * 80)
    print("⚖️  STARTING SCALED LEGAL TRANSFORMER LLM TRAINING (~13M PARAMETERS)")
    print("=" * 80)

    # Hardware detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon Metal GPU (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "Host CPU"

    print(f"⚙️  Compute Device: {device} ({device_name})")

    # 1. Load balanced data from all corpora
    raw_texts = load_training_corpora()
    print(f"📚 Loaded {len(raw_texts)} formatted training sequences from IndicLegalQA, Constitution & Statutes.")

    # 2. Build domain vocabulary
    tokenizer = LegalTokenizer()
    tokenizer.build_vocab(raw_texts, max_vocab_size=8000, min_freq=1)
    tokenizer.save_vocab(VOCAB_PATH)
    print(f"📖 Optimized tokenizer vocabulary: {tokenizer.vocab_size} tokens.")

    # 3. Tokenize sequences
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in raw_texts]
    dataset = LegalDataset(tokenized, max_seq_len=max_seq_len, pad_id=tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Instantiate Scaled Transformer Model (13M Params)
    model = LegalTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        dropout=0.05
    ).to(device)

    total_params = model.get_num_params()
    print(f"📐 Transformer Architecture: {n_layers} Layers, {n_heads} Heads, d_model={d_model}, d_ff={d_model*4}")
    print(f"🔥 Total Scaled Parameters: {total_params:,} ({total_params / 1e6:.2f}M Parameters)")

    # 5. Optimizer & Cosine Annealing Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 6. Deep Training Loop
    model.train()
    print("-" * 80)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss = 0.0
        steps = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            _, loss = model(batch_x, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        elapsed = time.time() - start_time
        
        status_bar = "🟢" if avg_loss < 1.0 else "🟡" if avg_loss < 2.5 else "⚙️"
        print(f"{status_bar} Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s")

    # 7. Save Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_model * 4,
        "max_seq_len": max_seq_len,
        "total_params": total_params,
        "trained_epochs": epochs,
        "final_loss": avg_loss
    }
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, WEIGHTS_PATH)
    print("=" * 80)
    print(f"💾 Checkpoint successfully saved to: {WEIGHTS_PATH}")
    print(f"🎉 Final Loss: {avg_loss:.4f} (Model Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f})")
    print(f"🌟 Model successfully scaled to {total_params:,} parameters!")
    print("=" * 80)

    return model, tokenizer


if __name__ == "__main__":
    train_legal_llm(epochs=25, batch_size=16, lr=5e-4, d_model=384, n_heads=6, n_layers=6, max_seq_len=256)
