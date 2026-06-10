#!/usr/bin/env python3
"""ActiveQueryAttention microbenchmark: backward wall-time vs active fraction.

Validates the O(|A|*L*d) claim at realistic sequence lengths (the toy tasks'
L=16 is too small to show it): times forward+backward through a transformer
stack for full dense backward vs AQA at several rho, verifying forward outputs
stay exact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.models import ToyMDM, ToyMDMConfig
from tabes.utils.logging import get_logger, setup_logging

logger = get_logger("tabes.bench.aqa")


def time_backward(model, emb, active, iters=10):
    # warmup
    for _ in range(2):
        run_once(model, emb, active)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_once(model, emb, active)
    return (time.perf_counter() - t0) / iters


def run_once(model, emb, active):
    e = emb.clone().requires_grad_(True)
    logits = model.forward_from_embeddings(e, active_mask=active)
    logp = torch.log_softmax(logits, dim=-1)
    (-(logp.exp() * logp).sum()).backward()
    return e.grad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rhos", type=float, nargs="*",
                    default=[0.0625, 0.125, 0.25, 0.5])
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results/aqa_timing.json"))
    args = ap.parse_args()

    setup_logging("INFO")
    torch.manual_seed(0)
    cfg = ToyMDMConfig(vocab_size=64, mask_id=63, max_len=args.seq_len,
                       dim=args.dim, n_heads=8, n_layers=args.layers)
    model = ToyMDM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x = torch.randint(0, 62, (args.batch, args.seq_len))
    emb = model.embed(x).detach()

    rows = []
    t_full = time_backward(model, emb, None, args.iters)
    logger.info("dense backward (rho=1.00): %.1f ms/iter", t_full * 1e3)
    rows.append({"rho": 1.0, "mode": "dense", "ms": t_full * 1e3, "speedup": 1.0})

    # exactness reference
    with torch.no_grad():
        ref = model.forward_from_embeddings(emb)

    for rho in args.rhos:
        n_active = max(1, int(rho * args.seq_len))
        active = torch.zeros(args.batch, args.seq_len, dtype=torch.bool)
        for b in range(args.batch):
            active[b, torch.randperm(args.seq_len)[:n_active]] = True
        with torch.no_grad():
            pass
        out = model.forward_from_embeddings(emb.clone().requires_grad_(True),
                                            active_mask=active)
        assert torch.allclose(out, ref, atol=1e-4), "AQA changed forward outputs"
        t = time_backward(model, emb, active, args.iters)
        logger.info("AQA rho=%.4f (|A|=%d): %.1f ms/iter  (%.2fx vs dense)",
                    rho, n_active, t * 1e3, t_full / t)
        rows.append({"rho": rho, "mode": "aqa", "ms": t * 1e3,
                     "speedup": t_full / t})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"seq_len": args.seq_len, "dim": args.dim, "layers": args.layers,
         "batch": args.batch, "device": "cpu", "rows": rows}, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
