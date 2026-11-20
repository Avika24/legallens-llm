"""
test_llm.py: Standalone Unit Test Suite for the Legal Transformer LLM.

Run:
    python3 test_llm.py
or
    pytest test_llm.py
"""

import sys
from pathlib import Path
import torch

from model import LegalTransformerLM, MultiHeadSelfAttention
from tokenizer import LegalTokenizer
from engine import LegalLLMEngine
from trainer import LegalDataset


def test_tokenizer_basic():
    tok = LegalTokenizer()
    tok.build_vocab([
        "Section 27 of the Indian Contract Act 1872 declares restraint of trade void.",
        "Article 21 guarantees protection of life and personal liberty."
    ])
    assert tok.vocab_size > 15
    text = "Section 27 void"
    encoded = tok.encode(text, add_special_tokens=True)
    assert encoded[0] == tok.bos_id
    assert encoded[-1] == tok.eos_id
    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert "section" in decoded.lower()
    print("✅ test_tokenizer_basic passed")


def test_transformer_forward_pass():
    vocab_size = 500
    d_model = 64
    n_heads = 4
    n_layers = 2
    max_seq_len = 32

    model = LegalTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len
    )

    batch_size = 2
    seq_len = 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets=y)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0
    print("✅ test_transformer_forward_pass passed")


def test_causal_masking():
    d_model = 32
    n_heads = 2
    max_seq_len = 10
    attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)

    # Verify causal mask is lower-triangular
    mask = attn.causal_mask[0, 0]
    assert mask[0, 1].item() == 0.0  # Future token masked
    assert mask[1, 0].item() == 1.0  # Past token visible
    assert mask[1, 1].item() == 1.0  # Current token visible
    print("✅ test_causal_masking passed")


def test_dataset_next_token_alignment():
    sequences = [[10, 20, 30, 40], [5, 15, 25]]
    dataset = LegalDataset(sequences, max_seq_len=6, pad_id=0)

    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape[0] == 5
    assert y.shape[0] == 5
    # y must be shifted by 1 relative to x
    assert x[0].item() == 10
    assert y[0].item() == 20
    print("✅ test_dataset_next_token_alignment passed")


def test_engine_inference():
    engine = LegalLLMEngine.get_instance()
    tokens = list(engine.stream_generate("<bos> <clause> Employee shall not compete", max_tokens=10))
    assert len(tokens) > 0
    print(f"✅ test_engine_inference passed (generated {len(tokens)} tokens on {engine.device_name})")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RUNNING LEGAL TRANSFORMER UNIT TESTS")
    print("=" * 60)
    test_tokenizer_basic()
    test_transformer_forward_pass()
    test_causal_masking()
    test_dataset_next_token_alignment()
    test_engine_inference()
    print("=" * 60)
    print("🎉 ALL 5 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
