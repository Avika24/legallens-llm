"""
tokenizer.py: Custom Legal Tokenizer built from scratch for Indian Legal Texts.

Features:
- Handles Indian statutory terms, legal punctuation, and special tags:
  <pad>, <unk>, <bos>, <eos>, <sep>, <user>, <assistant>, <clause>, <risk>, <rewrite>, <law>
- Regex token segmentation preserving legal section markers (§, /, -, .)
- Encode text -> token IDs
- Decode token IDs -> clean human-readable text
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<user>",
    "<assistant>",
    "<clause>",
    "<risk>",
    "<rewrite>",
    "<law>"
]


class LegalTokenizer:
    """
    Subword & Word Tokenizer built specifically for legal agreements and Indian law.
    """
    def __init__(self, vocab_file: Optional[Path] = None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.sep_token = "<sep>"

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        # Pre-populate special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        self.pad_id = self.word2idx[self.pad_token]
        self.unk_id = self.word2idx[self.unk_token]
        self.bos_id = self.word2idx[self.bos_token]
        self.eos_id = self.word2idx[self.eos_token]
        self.sep_id = self.word2idx[self.sep_token]

        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)

    def _tokenize_text(self, text: str) -> List[str]:
        """Splits raw string into tokens while strictly preserving special XML tags."""
        pattern = r"(<[a-z]+>|[A-Za-z0-9_]+|[.,!?;:()\"'§/\-–—\[\]{}])"
        tokens = re.findall(pattern, text)
        return [t.strip() for t in tokens if t.strip()]

    def build_vocab(self, texts: List[str], max_vocab_size: int = 8000, min_freq: int = 1):
        """Constructs frequency dictionary and vocabulary index from raw text corpus."""
        freq: Dict[str, int] = {}
        for text in texts:
            for token in self._tokenize_text(text):
                token_lower = token if (token.startswith("<") and token.endswith(">")) else token.lower()
                freq[token_lower] = freq.get(token_lower, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        for token, count in sorted_tokens:
            if count < min_freq:
                continue
            if token not in self.word2idx:
                idx = len(self.word2idx)
                if idx >= max_vocab_size:
                    break
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string to list of integer token IDs."""
        tokens = self._tokenize_text(text)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for t in tokens:
            key = t if (t.startswith("<") and t.endswith(">")) else t.lower()
            ids.append(self.word2idx.get(key, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back into clean human-readable text."""
        words = []
        for tid in token_ids:
            if tid in self.idx2word:
                word = self.idx2word[tid]
                if skip_special_tokens and word in SPECIAL_TOKENS:
                    continue
                words.append(word)
            else:
                words.append(self.unk_token)

        # Normalize spacing around punctuation
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;:)\]'\"])", r"\1", text)
        text = re.sub(r"([(\[\"'])\s+", r"\1", text)
        return text

    def save_vocab(self, filepath: Path):
        """Serializes vocabulary dictionary to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "vocab_size": len(self.word2idx)
            }, f, indent=2, ensure_ascii=False)

    def load_vocab(self, filepath: Path):
        """Loads vocabulary dictionary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.word2idx = data.get("word2idx", {})
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
