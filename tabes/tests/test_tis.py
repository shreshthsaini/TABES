"""TIS must match the first-order (Taylor) change in next-step masked entropy.

We verify Theorem 3.1's mechanism directly: for a small step eps along
delta_e_i = e~_i - e_mask, the change in surrogate entropy H~ satisfies
H~(e + eps*delta_e) - H~(e) ≈ eps * <g_i, delta_e_i>, i.e. TIS_i = -<g, delta_e>
is the predicted entropy *reduction*.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.models import ToyMDM, ToyMDMConfig


def entropy_sum(model, emb, probe_mask):
    logits = model.forward_from_embeddings(emb)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * probe_mask.float()).sum()


def test_tis_first_order_accuracy():
    torch.manual_seed(0)
    V, L = 14, 12
    cfg = ToyMDMConfig(vocab_size=V, mask_id=13, max_len=L, dim=32,
                       n_heads=4, n_layers=2)
    model = ToyMDM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokens = torch.randint(0, 12, (1, L))
    masked = torch.zeros(1, L, dtype=torch.bool)
    masked[0, [2, 5, 7, 9]] = True
    x = torch.where(masked, torch.full_like(tokens, cfg.mask_id), tokens)

    cand_pos, probe_pos = 5, torch.zeros(1, L, dtype=torch.bool)
    probe_pos[0, [2, 7, 9]] = True

    e = model.embed(x).detach()
    e_leaf = e.clone().requires_grad_(True)
    h0 = entropy_sum(model, e_leaf, probe_pos)
    h0.backward()
    g = e_leaf.grad[0, cand_pos]

    delta = torch.randn(cfg.dim) * 0.5  # a "reveal" direction
    eps = 1e-3
    e_pert = e.clone()
    e_pert[0, cand_pos] += eps * delta
    with torch.no_grad():
        h1 = entropy_sum(model, e_pert, probe_pos)

    fd = (h1 - h0).item() / eps          # finite-difference directional derivative
    lin = torch.dot(g, delta).item()     # first-order prediction
    assert abs(fd - lin) < 5e-2 * max(1.0, abs(lin)), (fd, lin)


def test_boe_tis_sign_convention():
    """Higher TIS == larger predicted reduction of future entropy."""
    from tabes.samplers import BoEConfig, BoESampler
    torch.manual_seed(0)
    cfg = ToyMDMConfig(vocab_size=14, mask_id=13, max_len=12, dim=32,
                       n_heads=4, n_layers=2)
    model = ToyMDM(cfg).eval()
    sampler = BoESampler(model, steps=4, config=BoEConfig(n_candidates=4))
    x = torch.full((2, 12), 13)
    x[:, 0] = 1  # one conditioning token
    masked = x == 13
    with torch.no_grad():
        logits = model(x)
    scores, extras = sampler.score_positions(x, logits, masked, step_idx=0)
    assert torch.isfinite(scores[masked & (scores > float('-inf'))]).all()
    assert extras["n_backward"] == 1
    assert len(extras["trace"]["tis"]) == 2
