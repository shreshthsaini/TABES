"""Sampler invariants: full unmasking, prompt preservation, determinism,
anti-collapse penalty math, and the training loss smoke test."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.data.synthetic import MASK_ID, VOCAB_SIZE, make_sudoku4, sudoku4_valid
from tabes.diffusion.loss import mdm_loss
from tabes.models import ToyMDM, ToyMDMConfig
from tabes.samplers import SAMPLERS, BoEConfig


def tiny_model(max_len=16):
    torch.manual_seed(0)
    cfg = ToyMDMConfig(vocab_size=VOCAB_SIZE, mask_id=MASK_ID, max_len=max_len,
                       dim=32, n_heads=4, n_layers=2)
    return ToyMDM(cfg).eval()


def prompts():
    p = torch.full((3, 16), MASK_ID)
    p[:, 0] = 1
    p[1, 5] = 2
    return p


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("steps", [1, 3, 8])
def test_full_unmask_and_prompt_preserved(name, steps):
    model = tiny_model()
    kw = {"config": BoEConfig(n_candidates=4)} if name == "boe" else {}
    sampler = SAMPLERS[name](model, steps=steps, seed=0, **kw)
    p = prompts()
    res = sampler.sample(p)
    assert not (res.tokens == MASK_ID).any()
    keep = p != MASK_ID
    assert torch.equal(res.tokens[keep], p[keep])
    assert res.n_forward >= 1


def test_determinism():
    model = tiny_model()
    outs = []
    for _ in range(2):
        s = SAMPLERS["boe"](model, steps=4, seed=7, config=BoEConfig(n_candidates=4))
        outs.append(s.sample(prompts()).tokens)
    assert torch.equal(outs[0], outs[1])


def test_anti_collapse_penalty():
    """Below-floor (overconfident) candidates are penalized; above floor not."""
    h_t = 0.8
    H = torch.tensor([0.2, 0.8, 1.5])
    pen = F.relu(h_t - H).pow(2)
    assert pen[0] > 0 and pen[1] == 0 and pen[2] == 0
    assert torch.isclose(pen[0], torch.tensor(0.36))


def test_anti_collapse_floor_decays():
    from tabes.samplers import BoESampler
    model = tiny_model()
    s = BoESampler(model, steps=4, config=BoEConfig(h_max=2.0, n_candidates=4))
    x = prompts()
    masked = x == MASK_ID
    with torch.no_grad():
        logits = model(x)
    h_ts = []
    for step_idx in range(4):
        _, extras = s.score_positions(x, logits, masked, step_idx)
        h_ts.append(extras["trace"]["h_t"])
    assert h_ts == sorted(h_ts, reverse=True) and h_ts[0] == 2.0


def test_mdm_loss_decreases():
    torch.manual_seed(0)
    data = make_sudoku4(n_eval=10).train[:128]
    model = ToyMDM(ToyMDMConfig(vocab_size=VOCAB_SIZE, mask_id=MASK_ID,
                                max_len=16, dim=32, n_heads=4, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for i in range(60):
        loss = mdm_loss(model, data, MASK_ID)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.8, (first, last)


def test_sudoku_checker():
    g = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
    assert sudoku4_valid(g)
    bad = list(g); bad[0] = 2
    assert not sudoku4_valid(bad)
