"""Training loop for the toy masked diffusion denoiser.

(BoE itself is training-free; this trains the base MDM the samplers steer.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tabes.diffusion.loss import mdm_loss
from tabes.utils.logging import TraceWriter, get_logger

logger = get_logger("tabes.train")


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 100
    log_every: int = 100
    seed: int = 0
    device: str = "cpu"


def train(model, data: torch.Tensor, cfg: TrainConfig,
          out_dir: str | Path, trace: TraceWriter | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    model.to(device).train()
    data = data.to(device)
    gen = torch.Generator().manual_seed(cfg.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup))

    t0 = time.perf_counter()
    losses = []
    for step in range(cfg.steps):
        idx = torch.randint(0, data.shape[0], (cfg.batch_size,), generator=gen)
        x0 = data[idx.to(data.device)]
        loss = mdm_loss(model, x0, model.mask_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if (step + 1) % cfg.log_every == 0 or step == 0:
            avg = sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:])
            logger.info("step %5d/%d | loss %.4f | %.1fs",
                        step + 1, cfg.steps, avg, time.perf_counter() - t0)
            if trace is not None:
                trace.write({"kind": "train", "step": step + 1, "loss": avg,
                             "lr": sched.get_last_lr()[0]})

    ckpt = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.cfg}, ckpt)
    logger.info("saved checkpoint to %s", ckpt)
    return {"final_loss": sum(losses[-50:]) / max(1, len(losses[-50:])),
            "wall_time_s": time.perf_counter() - t0, "ckpt": str(ckpt)}
