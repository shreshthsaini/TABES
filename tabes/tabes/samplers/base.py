"""Shared decoding loop for masked-diffusion samplers.

The loop runs ``steps`` denoising iterations t = T..1. At each step the model
is queried once; the sampler subclass scores every masked position (higher
score = reveal earlier); the loop reveals a linear-schedule budget of tokens
per sequence and writes a JSONL trace record (entropies, scores, selections,
timings, NFE counts).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tabes.utils.logging import TraceWriter, get_logger

logger = get_logger("tabes.sampler")

NEG_INF = float("-inf")


@dataclass
class SampleResult:
    tokens: torch.Tensor
    n_forward: int = 0
    n_backward: int = 0
    wall_time: float = 0.0
    step_records: list = field(default_factory=list)


class BaseSampler:
    """Subclasses implement ``score_positions``; everything else is shared."""

    name = "base"
    needs_grad = False

    def __init__(self, model, steps: int = 8, temperature: float = 0.0,
                 trace: TraceWriter | None = None, seed: int = 0):
        self.model = model
        self.steps = steps
        self.temperature = temperature
        self.trace = trace
        self.generator = torch.Generator().manual_seed(seed)
        for p in self.model.parameters():  # inference-time method: params frozen
            p.requires_grad_(False)
        self.model.eval()

    # ------------------------------------------------------------- interface
    def score_positions(
        self, x: torch.Tensor, logits: torch.Tensor, masked: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, dict]:
        """Return (scores [B, L] — NEG_INF at non-masked, extras dict).

        ``extras`` may include ``n_forward``/``n_backward`` for additional model
        queries and arbitrary trace payload under ``trace``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ loop
    def sample(self, prompts: torch.Tensor) -> SampleResult:
        x = prompts.clone()
        mask_id = self.model.mask_id
        res = SampleResult(tokens=x)
        t0 = time.perf_counter()

        for step_idx in range(self.steps):
            masked = x == mask_id
            n_masked = masked.sum(dim=1)
            if int(n_masked.sum()) == 0:
                break
            steps_left = self.steps - step_idx
            budget = torch.ceil(n_masked.float() / steps_left).long()  # [B]

            step_t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(x)
                logits[..., mask_id] = NEG_INF  # never commit the mask token
            res.n_forward += 1
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)

            scores, extras = self.score_positions(x, logits, masked, step_idx)
            res.n_forward += extras.get("n_forward", 0)
            res.n_backward += extras.get("n_backward", 0)

            commit = self._commit_tokens(logits)
            reveal = self._select(scores, budget, masked)
            x = torch.where(reveal, commit, x)

            record = {
                "step": step_idx,
                "steps_total": self.steps,
                "sampler": self.name,
                "n_masked": n_masked,
                "budget": budget,
                "revealed_pos": [r.nonzero(as_tuple=True)[0] for r in reveal],
                "revealed_tok": [commit[i][r] for i, r in enumerate(reveal)],
                "mean_masked_entropy": (entropy * masked).sum() / masked.sum().clamp_min(1),
                "step_wall_s": time.perf_counter() - step_t0,
                **extras.get("trace", {}),
            }
            res.step_records.append(record)
            if self.trace is not None:
                self.trace.write(record)

        res.tokens = x
        res.wall_time = time.perf_counter() - t0
        return res

    # --------------------------------------------------------------- helpers
    def _commit_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        gumbel = -torch.log(-torch.log(
            torch.rand(logits.shape, generator=self.generator).clamp_min(1e-12)
        ).clamp_min(1e-12))
        return (logits / self.temperature + gumbel.to(logits.device)).argmax(dim=-1)

    @staticmethod
    def _select(scores: torch.Tensor, budget: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """Per-row top-``budget`` positions by score, restricted to masked slots."""
        B, L = scores.shape
        # Masked positions must always outrank non-masked ones, even when a
        # sampler leaves some masked slots unscored (-inf).
        scores = torch.where(masked, scores.clamp_min(-1e30),
                             torch.full_like(scores, NEG_INF))
        order = scores.argsort(dim=1, descending=True)
        ranks = order.argsort(dim=1)
        take = torch.minimum(budget, masked.sum(dim=1))
        return (ranks < take[:, None]) & masked
