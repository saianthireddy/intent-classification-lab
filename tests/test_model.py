"""Attention mechanics: the mask is the part that is easy to get subtly wrong."""
from __future__ import annotations

import torch

from intent_lab.model import (
    IntentTransformer,
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
    sinusoidal_positions,
)


def test_attention_weights_are_zero_on_masked_keys_and_still_normalise():
    q = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[False, False, True, True]])
    _, weights = scaled_dot_product_attention(q, q, q, mask)
    assert torch.equal(weights[0, 0, :, 2:], torch.zeros(4, 2))
    assert torch.allclose(weights[0, 0].sum(dim=-1), torch.ones(4))


def test_attention_without_a_mask_attends_everywhere():
    q = torch.randn(1, 1, 3, 8)
    _, weights = scaled_dot_product_attention(q, q, q, None)
    assert (weights > 0).all()


def test_pad_contents_cannot_influence_real_positions():
    """Exact, not approximate: replacing pad rows with anything changes nothing.

    This is the property that actually matters — it proves no information leaks
    from padding into a real token's representation.
    """
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(8, 2, dropout=0.0).eval()
    real = torch.randn(1, 3, 8)
    mask = torch.tensor([[False] * 3 + [True] * 3])
    with torch.no_grad():
        quiet = attn(torch.cat([real, torch.zeros(1, 3, 8)], dim=1), mask)[0, :3]
        loud = attn(torch.cat([real, torch.randn(1, 3, 8) * 500], dim=1), mask)[0, :3]
    assert torch.equal(quiet, loud)


def test_sequence_length_is_invariant_up_to_float32_accumulation():
    """Same content, different amount of padding.

    Asserted with a tolerance rather than exactly, because the matmul reduction
    length changes and float32 addition is not associative. In float64 the
    difference is exactly zero — see the note in model.py — so this is rounding,
    not leakage.
    """
    torch.manual_seed(0)
    model = IntentTransformer(vocab_size=40, n_classes=6, max_length=32).eval()
    with torch.no_grad():
        short = model(torch.tensor([[5, 6, 7] + [0] * 3]))
        long = model(torch.tensor([[5, 6, 7] + [0] * 13]))
    assert torch.allclose(short, long, atol=1e-2)

    model64 = model.double()
    with torch.no_grad():
        s64 = model64(torch.tensor([[5, 6, 7] + [0] * 3]))
        l64 = model64(torch.tensor([[5, 6, 7] + [0] * 13]))
    assert torch.equal(s64, l64), "in float64 it must be bit-identical"


def test_head_count_must_divide_the_model_width():
    try:
        MultiHeadSelfAttention(d_model=10, n_heads=4)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_positional_encoding_is_bounded_and_position_dependent():
    pe = sinusoidal_positions(16, 8)
    assert pe.shape == (16, 8)
    assert pe.abs().max() <= 1.0
    assert not torch.allclose(pe[0], pe[1])


def test_forward_shape_and_pad_embedding_stays_zero():
    model = IntentTransformer(vocab_size=40, n_classes=6)
    out = model(torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]]))
    assert out.shape == (2, 6)
    assert torch.equal(model.embed.weight[model.pad_id], torch.zeros(model.embed.embedding_dim))


def test_an_all_padding_row_does_not_produce_nan():
    """Divide-by-zero in the masked mean is the failure this guards."""
    model = IntentTransformer(vocab_size=40, n_classes=6).eval()
    with torch.no_grad():
        out = model(torch.tensor([[0, 0, 0, 0]]))
    assert torch.isfinite(out).all()


def test_parameter_count_is_reported():
    assert IntentTransformer(vocab_size=86, n_classes=6).parameter_count() > 10_000
