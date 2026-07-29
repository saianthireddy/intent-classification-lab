"""A small Transformer encoder written from scratch.

``nn.TransformerEncoder`` would be three lines and the right call in production.
It is the wrong call here, because the point of this module is to show the
mechanics: scaled dot-product attention, the mask, multi-head projection and
recombination, pre-norm residual blocks, and sinusoidal positions.

Two choices worth naming:

* **Pre-norm** (``x + attn(norm(x))``) rather than post-norm. Post-norm needs
  learning-rate warmup to train stably; pre-norm does not, which matters when
  the whole point is that this trains on a laptop CPU in seconds.
* **Masked mean pooling** rather than a ``[CLS]`` token. With ~200 training
  examples there is not enough signal to teach a dedicated token to summarise a
  sequence; averaging the real tokens is a stronger prior at this scale. Pad
  positions are excluded from both the attention softmax and the pool — leaving
  them in the pool is a silent bug that shrinks every embedding toward zero in
  proportion to how much padding it has.

**On exactness of the mask.** Padding contents provably cannot influence real
positions: at a fixed sequence length, replacing the pad rows with anything at
all leaves real-position outputs bit-identical, and comparing two different
padded lengths in float64 gives a difference of exactly zero. In float32 the
same comparison differs by ~1e-4, which is accumulation order in the matmul
kernel changing with the reduction length, not leakage. The tests assert the
first property exactly and the second within a tolerance, and say why.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Attention over the last two dims. Returns (output, weights).

    ``key_padding_mask`` is True where a position is padding. Masked scores go
    to -inf *before* the softmax so they contribute exactly zero, rather than
    being zeroed afterwards, which would leave the remaining weights unnormalised.
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if key_padding_mask is not None:
        # A row whose every key is masked softmaxes -inf/-inf -> NaN, and one NaN
        # poisons the rest of the forward pass. That happens for real inputs: any
        # text that tokenises to nothing (punctuation, an unknown script) encodes
        # to all-pad. Such a row carries no information, so zero is the honest
        # weight, and masked mean pooling drops the position anyway.
        weights = torch.nan_to_num(weights, nan=0.0)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        batch, seq, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_heads, self.d_head)
        # (batch, heads, seq, d_head)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attended, _ = scaled_dot_product_attention(q, k, v, key_padding_mask)
        merged = attended.transpose(1, 2).reshape(batch, seq, d_model)
        return self.dropout(self.out(merged))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        return x + self.dropout(self.ff(self.norm2(x)))


def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    position = torch.arange(max_len, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class IntentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_length: int = 24,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.register_buffer("positions", sinusoidal_positions(max_length, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def forward(self, ids: Tensor) -> Tensor:
        pad_mask = ids == self.pad_id
        x = self.dropout(self.embed(ids) + self.positions[: ids.size(1)])
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)

        keep = (~pad_mask).unsqueeze(-1).float()
        # clamp: a row that is entirely padding would divide by zero
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
