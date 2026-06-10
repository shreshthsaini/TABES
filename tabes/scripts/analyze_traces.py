#!/usr/bin/env python3
"""Analyze decode traces: per-step residual-entropy trajectories per sampler.

Reads the JSONL traces produced by run_benchmark.py and writes a compact
markdown table showing how fast each sampler drives down future masked
entropy (the quantity BoE explicitly optimizes).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(Path(__file__).parents[1] / "results/traces"))
    ap.add_argument("--steps", type=int, default=8, help="step budget to analyze")
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results/entropy_trajectories.md"))
    args = ap.parse_args()

    # trajectories[task][sampler][step] -> list of mean masked entropies
    trajectories: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(Path(args.traces).glob(f"*_s{args.steps}.jsonl")):
        task, sampler = path.stem.replace(f"_s{args.steps}", "").split("_", 1)
        for line in path.open():
            rec = json.loads(line)
            trajectories[task][sampler][rec["step"]].append(rec["mean_masked_entropy"])

    lines = [f"# Mean residual masked entropy per denoising step (T={args.steps})", ""]
    for task, samplers in sorted(trajectories.items()):
        lines += [f"## {task}", "",
                  "| sampler | " + " | ".join(f"t={s}" for s in range(args.steps)) + " |",
                  "|---" * (args.steps + 1) + "|"]
        for name, steps in sorted(samplers.items()):
            row = [name]
            for s in range(args.steps):
                vals = steps.get(s, [])
                row.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
