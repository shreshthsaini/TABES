#!/usr/bin/env python3
"""Demo: decode a few eval prompts with a chosen sampler and print the
step-by-step trajectory (what BoE reveals and why)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.data.synthetic import VOCAB, MASK_ID, make_task
from tabes.samplers import SAMPLERS, BoEConfig
from tabes.utils.logging import TraceWriter, setup_logging


def fmt(row: torch.Tensor) -> str:
    return "".join("□" if i == MASK_ID else VOCAB[i] for i in row.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/toy.yaml"))
    ap.add_argument("--runs", default=str(Path(__file__).parents[1] / "runs"))
    ap.add_argument("--task", default="sudoku4")
    ap.add_argument("--sampler", default="boe", choices=list(SAMPLERS))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--trace", default=None)
    args = ap.parse_args()

    setup_logging("INFO")
    cfg = yaml.safe_load(open(args.config))
    from scripts.run_benchmark import load_model

    task = make_task(args.task, **cfg["tasks"][args.task])
    model = load_model(Path(args.runs) / args.task)
    trace = TraceWriter(args.trace) if args.trace else None
    kw = {"config": BoEConfig(**cfg["boe"])} if args.sampler == "boe" else {}
    sampler = SAMPLERS[args.sampler](model, steps=args.steps, trace=trace, **kw)

    prompts = task.eval_prompts[: args.n]
    print(f"\n=== {args.sampler} on {args.task} ({args.steps} steps) ===")
    for i, p in enumerate(prompts):
        print(f"\nprompt {i}: {fmt(p)}")
    res = sampler.sample(prompts)
    for rec in res.step_records:
        revealed = [(int(b), p.tolist()) for b, p in enumerate(rec["revealed_pos"])]
        print(f"step {rec['step']}: mean masked entropy "
              f"{float(rec['mean_masked_entropy']):.3f} | revealed {revealed}")
    for i, row in enumerate(res.tokens):
        print(f"output {i}: {fmt(row)}")
    print(f"\nNFE: {res.n_forward} fwd / {res.n_backward} bwd | "
          f"{res.wall_time_s if hasattr(res, 'wall_time_s') else res.wall_time:.2f}s")


if __name__ == "__main__":
    main()
