"""Benchmarking: quality-compute Pareto sweeps and ablations.

Reproduces the paper's experimental axes at toy scale:
  * Pareto frontier: every sampler at several step budgets (accuracy vs NFE
    and wall-clock).
  * Ablations: gradient signal (use_tis), anti-collapse (lambda_ac=0),
    ActiveQueryAttention on/off, and the active-fraction rho sweep.
Outputs JSON records plus a markdown report.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tabes.eval.harness import evaluate
from tabes.samplers import SAMPLERS, BoEConfig, BoESampler
from tabes.utils.logging import TraceWriter, get_logger

logger = get_logger("tabes.bench")


def pareto_sweep(model, tasks, samplers: list[str], step_budgets: list[int],
                 boe_cfg: BoEConfig, out_dir: Path, seed: int = 0,
                 limit: int | None = None, trace_dir: Path | None = None) -> list[dict]:
    rows = []
    for task in tasks:
        for steps in step_budgets:
            for name in samplers:
                trace = None
                if trace_dir is not None:
                    trace = TraceWriter(trace_dir / f"{task.name}_{name}_s{steps}.jsonl")
                kw = {"config": boe_cfg} if name == "boe" else {}
                sampler = SAMPLERS[name](model, steps=steps, seed=seed, trace=trace, **kw)
                res = evaluate(sampler, task, limit=limit)
                rows.append({**asdict(res), "kind": "pareto"})
                if trace is not None:
                    trace.close()
    _dump(rows, out_dir / "pareto.json")
    return rows


def ablation_sweep(model, tasks, steps: int, base_cfg: BoEConfig, out_dir: Path,
                   rhos: list[float], seed: int = 0, limit: int | None = None) -> list[dict]:
    variants: list[tuple[str, BoEConfig]] = [
        ("boe (full)", base_cfg),
        ("boe -TIS", _replace(base_cfg, use_tis=False)),
        ("boe -anti-collapse", _replace(base_cfg, lambda_ac=0.0)),
        ("boe -AQA (full backward)", _replace(base_cfg, use_aqa=False)),
    ]
    for rho in rhos:
        variants.append((f"boe rho={rho}", _replace(base_cfg, rho=rho)))

    rows = []
    for task in tasks:
        for label, cfg in variants:
            sampler = BoESampler(model, steps=steps, seed=seed, config=cfg)
            res = evaluate(sampler, task, limit=limit)
            row = {**asdict(res), "kind": "ablation", "variant": label,
                   "config": asdict(cfg)}
            rows.append(row)
    _dump(rows, out_dir / "ablations.json")
    return rows


def write_report(pareto: list[dict], ablations: list[dict], out_path: Path,
                 header: str = "") -> None:
    lines = ["# TABES toy-scale benchmark report", ""]
    if header:
        lines += [header, ""]

    tasks = sorted({r["task"] for r in pareto})
    for task in tasks:
        lines += [f"## Pareto sweep — {task}", "",
                  "| sampler | steps | accuracy | fwd NFE | bwd NFE | wall (s) |",
                  "|---|---|---|---|---|---|"]
        rows = [r for r in pareto if r["task"] == task]
        for r in sorted(rows, key=lambda r: (r["steps"], r["sampler"])):
            lines.append(
                f"| {r['sampler']} | {r['steps']} | {r['accuracy']:.3f} "
                f"| {r['n_forward']} | {r['n_backward']} | {r['wall_time_s']:.2f} |")
        lines.append("")

    if ablations:
        for task in sorted({r["task"] for r in ablations}):
            lines += [f"## Ablations — {task}", "",
                      "| variant | accuracy | wall (s) |", "|---|---|---|"]
            for r in [r for r in ablations if r["task"] == task]:
                lines.append(f"| {r['variant']} | {r['accuracy']:.3f} "
                             f"| {r['wall_time_s']:.2f} |")
            lines.append("")

    out_path.write_text("\n".join(lines))
    logger.info("wrote report to %s", out_path)


def _replace(cfg: BoEConfig, **kw) -> BoEConfig:
    d = asdict(cfg)
    d.update(kw)
    return BoEConfig(**d)


def _dump(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    logger.info("wrote %d records to %s", len(rows), path)
