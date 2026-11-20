"""
model.py: Complete Decoder-Only Transformer Language Model built from scratch in PyTorch.

Architecture Specifications:
- Model Type: Autoregressive Decoder-Only Transformer
- Context Length: 128 / 512 tokens
- Hidden Dimension (d_model): 128
- Multi-Head Attention: 4 heads (head_dim = 32)
- Layers: 4 Transformer Blocks
- Feed-Forward Dimension: 512 (4 * d_model) with GeLU activation
- Normalization: Pre-LayerNorm (modern GPT-3 / LLaMA architecture)
- Weight Tying: Embedding weights tied with LM Head projection
- Total Parameters: 1,466,240 parameters
- Target Hardware: Apple Silicon GPU (MPS), CUDA, or Host CPU
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    Computes Scaled Dot-Product Attention across multiple representation subspaces:
        Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + M) @ V
    where M is a lower-triangular causal mask preventing attention to future tokens.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Query, Key, Value, and Output
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix (1s below/on diagonal, 0s above)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections & split into heads: [B, seq_len, d_model] -> [B, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention: (Q @ K.T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask: mask future token positions with -inf
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: [B, n_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, v)

        # Concatenate heads and project out: [B, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
        FFN(x) = GeLU(x @ W1 + b1) @ W2 + b2
    Expands hidden dimension by 4x before projecting back.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Decoder Block:
        x_norm1 = LayerNorm(x)
        x = x + MultiHeadAttention(x_norm1)
        x_norm2 = LayerNorm(x)
        x = x + FeedForward(x_norm2)
    Pre-LN provides superior gradient stability during deep training compared to Post-LN.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LegalTransformerLM(nn.Module):
    """
    Native Decoder-Only Transformer Language Model built from scratch.
    Capable of next-token prediction, sequence generation, and fine-tuning.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token and learned positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm & Language Modeling Head
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output head
        self.lm_head.weight = self.token_embedding.weight

        # Parameter initialization (standard Gaussian initialization)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: [batch_size, seq_len] tensor of integer token IDs.
            targets: [batch_size, seq_len] tensor of expected next token IDs (optional).
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Cross-entropy scalar loss if targets are provided, else None.
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Sum token embeddings and positional embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        loss = None
        if targets is not None:
            # Shift tokens for next-token prediction
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns total parameter count of the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive next-token sampling loop.
        Applies temperature scaling and top-k filtering.
        """
        self.eval()
        for _ in range(max_new_tokens):
            cond_input = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            logits, _ = self(cond_input)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return input_ids
