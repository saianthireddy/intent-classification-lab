"""Word-level tokenizer fitted on the training split only.

Fitting the vocabulary on all splits is a quiet, common leak: test-set tokens
would get embeddings while genuinely unseen words at inference would not, so
the reported score reflects a model that saw more than it will in production.
Here ``fit`` takes training texts and nothing else, and everything unseen maps
to a single ``<unk>`` id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Tokenizer:
    max_length: int = 24
    min_freq: int = 1
    stoi: dict[str, int] = field(default_factory=dict)
    itos: list[str] = field(default_factory=list)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def fit(self, texts: list[str]) -> Tokenizer:
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        kept = sorted(tok for tok, n in counts.items() if n >= self.min_freq)
        self.itos = [PAD, UNK, *kept]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        return self

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)][: self.max_length]
        return ids + [self.pad_id] * (self.max_length - len(ids))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def unk_rate(self, texts: list[str]) -> float:
        """Share of tokens that fall outside the fitted vocabulary.

        Worth reporting next to accuracy: a low test score with a high unk rate
        is a vocabulary problem, not a modelling one.
        """
        total = unknown = 0
        for text in texts:
            for token in tokenize(text):
                total += 1
                unknown += token not in self.stoi
        return unknown / total if total else 0.0
