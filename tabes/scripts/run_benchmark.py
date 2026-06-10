#!/usr/bin/env python3
"""Run the full benchmark: Pareto sweep over samplers/steps + BoE ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.bench.benchmark import ablation_sweep, pareto_sweep, write_report
from tabes.data.synthetic import make_task
from tabes.models import ToyMDM
from tabes.samplers import BoEConfig
from tabes.utils.logging import get_logger, setup_logging
from tabes.utils.seed import set_seed

logger = get_logger("tabes.scripts.bench")


def load_model(run_dir: Path) -> ToyMDM:
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    model = ToyMDM(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/toy.yaml"))
    ap.add_argument("--runs", default=str(Path(__file__).parents[1] / "runs"))
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", out_dir / "benchmark.log")
    set_seed(cfg["seed"])

    bcfg = cfg["benchmark"]
    boe_cfg = BoEConfig(**cfg["boe"])
    limit = args.limit or bcfg.get("eval_limit")

    pareto_all, abl_all = [], []
    for name in cfg["tasks"]:
        task = make_task(name, **cfg["tasks"][name])
        model = load_model(Path(args.runs) / name)
        logger.info("=== task %s ===", name)
        pareto_all += pareto_sweep(
            model, [task], bcfg["samplers"], bcfg["step_budgets"], boe_cfg,
            out_dir, seed=cfg["seed"], limit=limit, trace_dir=out_dir / "traces")
        if not args.skip_ablations:
            abl_all += ablation_sweep(
                model, [task], bcfg["ablation_steps"], boe_cfg, out_dir,
                rhos=bcfg["rhos"], seed=cfg["seed"], limit=limit)

    from tabes.bench.benchmark import _dump
    _dump(pareto_all, out_dir / "pareto.json")
    if abl_all:
        _dump(abl_all, out_dir / "ablations.json")
    write_report(pareto_all, abl_all, out_dir / "report.md",
                 header=f"Toy MDM, CPU. Config: `{args.config}`.")


if __name__ == "__main__":
    main()
