"""ActiveQueryAttention: exact forward equivalence + correct sparse gradients."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.models import ToyMDM, ToyMDMConfig
from tabes.models.attention import ActiveQueryAttention


def make_inputs(B=3, L=12, D=32, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, L, D)
    active = torch.rand(B, L) < 0.3
    active[0, 0] = True  # ensure at least one active row
    return x, active


@pytest.mark.parametrize("mode", ["split", "mask"])
def test_forward_equivalence(mode):
    x, active = make_inputs()
    torch.manual_seed(1)
    attn = ActiveQueryAttention(32, 4, mode=mode)
    with torch.no_grad():
        ref = attn(x, active_mask=None)
    out = attn(x.requires_grad_(True), active_mask=active)
    assert torch.allclose(out, ref, atol=1e-5), "AQA must not change forward outputs"


def test_split_matches_mask_gradients():
    x, active = make_inputs()
    torch.manual_seed(1)
    a_split = ActiveQueryAttention(32, 4, mode="split")
    torch.manual_seed(1)
    a_mask = ActiveQueryAttention(32, 4, mode="mask")

    grads = []
    for attn in (a_split, a_mask):
        xi = x.clone().requires_grad_(True)
        attn(xi, active_mask=active).pow(2).sum().backward()
        grads.append(xi.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5), \
        "split implementation must match stop-grad reference gradients"


def test_inactive_query_rows_blocked():
    """With a single active row, gradients w.r.t. inputs must flow only through
    that query row's attention (plus key/value paths)."""
    B, L, D = 1, 8, 32
    torch.manual_seed(0)
    x = torch.randn(B, L, D, requires_grad=True)
    active = torch.zeros(B, L, dtype=torch.bool)
    active[0, 3] = True
    attn = ActiveQueryAttention(D, 4, mode="split")

    out = attn(x, active_mask=active)
    # loss touches only INACTIVE output rows -> no gradient should flow at all
    loss = out[0, [0, 1, 2, 4, 5, 6, 7]].sum()
    loss.backward()
    assert torch.all(x.grad.abs() < 1e-12), \
        "inactive query rows must be stop-gradded"


def test_model_threads_active_mask():
    cfg = ToyMDMConfig(vocab_size=14, mask_id=13, max_len=10, dim=32,
                       n_heads=4, n_layers=2)
    model = ToyMDM(cfg)
    tokens = torch.randint(0, 14, (2, 10))
    active = torch.rand(2, 10) < 0.5
    with torch.no_grad():
        ref = model(tokens)
    emb = model.embed(tokens).detach().requires_grad_(True)
    out = model.forward_from_embeddings(emb, active_mask=active)
    assert torch.allclose(out, ref, atol=1e-4)
    out.sum().backward()
    assert emb.grad is not None
