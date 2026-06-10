"""Evaluation harness: run a sampler over a task's eval prompts and score.

Metrics:
  * ``arithmetic`` — exact match of the generated answer digits.
  * ``sudoku4``    — fraction of completions satisfying all row/col/box
                     constraints (any valid completion counts; puzzles with few
                     clues admit multiple solutions).
Both also report mean residual-entropy trajectory and NFE/timing stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from tabes.data.synthetic import MASK_ID, STOI, VOCAB, TaskData, sudoku4_valid
from tabes.utils.logging import get_logger

logger = get_logger("tabes.eval")


@dataclass
class EvalResult:
    task: str
    sampler: str
    steps: int
    accuracy: float
    n_examples: int
    n_forward: int
    n_backward: int
    wall_time_s: float
    extra: dict = field(default_factory=dict)


_DIGIT_OF = {STOI[str(d)]: d for d in range(10)}


def _score_arithmetic(tokens: torch.Tensor, targets: list[str]) -> list[bool]:
    out = []
    for row, tgt in zip(tokens, targets):
        ans = "".join(VOCAB[i] if VOCAB[i].isdigit() else "?" for i in row[6:9].tolist())
        out.append(ans == tgt)
    return out


def _score_sudoku4(tokens: torch.Tensor, targets: list) -> list[bool]:
    out = []
    for row in tokens:
        cells = [_DIGIT_OF.get(int(i), -1) for i in row.tolist()]
        out.append(sudoku4_valid(cells))
    return out


SCORERS = {"arithmetic": _score_arithmetic, "sudoku4": _score_sudoku4}


def evaluate(sampler, task: TaskData, batch_size: int = 64,
             limit: int | None = None) -> EvalResult:
    prompts = task.eval_prompts
    targets = task.eval_targets
    if limit is not None:
        prompts, targets = prompts[:limit], targets[:limit]

    correct: list[bool] = []
    n_fwd = n_bwd = 0
    t0 = time.perf_counter()
    for i in range(0, prompts.shape[0], batch_size):
        batch = prompts[i : i + batch_size]
        res = sampler.sample(batch)
        assert not (res.tokens == MASK_ID).any(), "sampler left masks behind"
        correct += SCORERS[task.name](res.tokens, targets[i : i + batch_size])
        n_fwd += res.n_forward
        n_bwd += res.n_backward
    wall = time.perf_counter() - t0

    acc = sum(correct) / len(correct)
    result = EvalResult(
        task=task.name, sampler=sampler.name, steps=sampler.steps,
        accuracy=acc, n_examples=len(correct),
        n_forward=n_fwd, n_backward=n_bwd, wall_time_s=wall,
    )
    logger.info("eval %-10s | %-10s | steps=%2d | acc=%.3f | fwd=%d bwd=%d | %.1fs",
                task.name, sampler.name, sampler.steps, acc, n_fwd, n_bwd, wall)
    return result
