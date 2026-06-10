#!/usr/bin/env python3
"""Train toy MDM denoisers for each synthetic task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.data.synthetic import MASK_ID, VOCAB_SIZE, make_task
from tabes.models import ToyMDM, ToyMDMConfig
from tabes.train.trainer import TrainConfig, train
from tabes.utils.logging import TraceWriter, get_logger, setup_logging
from tabes.utils.seed import set_seed

logger = get_logger("tabes.scripts.train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/toy.yaml"))
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "runs"))
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override train steps")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = Path(args.out)
    setup_logging("INFO", out_root / "train.log")
    set_seed(cfg["seed"])

    tasks = args.tasks or list(cfg["tasks"].keys())
    for name in tasks:
        data = make_task(name, **cfg["tasks"][name])
        mcfg = ToyMDMConfig(vocab_size=VOCAB_SIZE, mask_id=MASK_ID,
                            max_len=data.seq_len, **cfg["model"])
        model = ToyMDM(mcfg)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("task=%s | train sequences=%d | model params=%.2fM",
                    name, data.train.shape[0], n_params / 1e6)

        tcfg = TrainConfig(seed=cfg["seed"], device=cfg["device"], **cfg["train"])
        if args.steps is not None:
            tcfg.steps = args.steps
        with TraceWriter(out_root / name / "train_trace.jsonl") as trace:
            stats = train(model, data.train, tcfg, out_root / name, trace)
        logger.info("task=%s done: %s", name, stats)


if __name__ == "__main__":
    main()
