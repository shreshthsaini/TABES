"""Synthetic tasks for end-to-end validation of TABES at toy scale.

Two tasks where unmasking *order* matters, so trajectory lock-in is measurable:

* ``arithmetic``: sequences ``AA+BB=CCC`` (zero-padded). Eval masks the answer
  digits; metric is exact match. Carry chains make some digits load-bearing.
* ``sudoku4``: flattened 4x4 Sudoku solutions (digits 1-4, 2x2 boxes). Eval
  masks a random subset of cells; metric is whether the completion satisfies
  all row/column/box constraints. A wrong early commit provably locks the
  trajectory into constraint violations — the paper's motivating failure mode.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import torch

# Shared character-level vocabulary.
CHARS = list("0123456789+=")
PAD, MASK = "<pad>", "<mask>"
VOCAB = CHARS + [PAD, MASK]
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD_ID, MASK_ID = STOI[PAD], STOI[MASK]
VOCAB_SIZE = len(VOCAB)


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] if VOCAB[i] not in (PAD, MASK) else ("_" if VOCAB[i] == MASK else "") for i in ids)


@dataclass
class TaskData:
    name: str
    seq_len: int
    train: torch.Tensor                      # [N, L] token ids
    eval_prompts: torch.Tensor               # [M, L] with MASK_ID at generation slots
    eval_targets: list = field(default_factory=list)  # task-specific ground truth payloads


# ---------------------------------------------------------------- arithmetic

def make_arithmetic(n_train: int = 9000, n_eval: int = 300, seed: int = 0) -> TaskData:
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(100) for b in range(100)]
    rng.shuffle(pairs)
    train_pairs, eval_pairs = pairs[:n_train], pairs[n_train : n_train + n_eval]

    def seq(a: int, b: int) -> str:
        return f"{a:02d}+{b:02d}={a + b:03d}"  # length 9

    train = torch.tensor([encode(seq(a, b)) for a, b in train_pairs])
    prompts, targets = [], []
    for a, b in eval_pairs:
        ids = encode(seq(a, b))
        ids[6:9] = [MASK_ID] * 3  # mask the answer digits
        prompts.append(ids)
        targets.append(f"{a + b:03d}")
    return TaskData("arithmetic", 9, train, torch.tensor(prompts), targets)


# ------------------------------------------------------------------ sudoku4

def _all_sudoku4() -> list[tuple[int, ...]]:
    grids = []
    for perm in itertools.permutations(range(1, 5)):
        rows0 = [list(perm)]
        for r2 in itertools.permutations(range(1, 5)):
            if any(r2[c] == rows0[0][c] for c in range(4)):
                continue
            if {r2[0], r2[1]} & {rows0[0][0], rows0[0][1]} or {r2[2], r2[3]} & {rows0[0][2], rows0[0][3]}:
                continue
            for r3 in itertools.permutations(range(1, 5)):
                if any(r3[c] in (rows0[0][c], r2[c]) for c in range(4)):
                    continue
                r4 = [10 - rows0[0][c] - r2[c] - r3[c] for c in range(4)]
                if sorted(r4) != [1, 2, 3, 4]:
                    continue
                if {r4[0], r4[1]} & {r3[0], r3[1]} or {r4[2], r4[3]} & {r3[2], r3[3]}:
                    continue
                grids.append(tuple(rows0[0]) + tuple(r2) + tuple(r3) + tuple(r4))
    return sorted(set(grids))


def sudoku4_valid(cells: list[int]) -> bool:
    if any(c not in (1, 2, 3, 4) for c in cells):
        return False
    g = [cells[i * 4 : (i + 1) * 4] for i in range(4)]
    for i in range(4):
        if sorted(g[i]) != [1, 2, 3, 4] or sorted(r[i] for r in g) != [1, 2, 3, 4]:
            return False
    for br in (0, 2):
        for bc in (0, 2):
            box = [g[br][bc], g[br][bc + 1], g[br + 1][bc], g[br + 1][bc + 1]]
            if sorted(box) != [1, 2, 3, 4]:
                return False
    return True


def make_sudoku4(n_eval: int = 300, n_clues: int = 6, seed: int = 0,
                 holdout_frac: float = 0.3) -> TaskData:
    rng = random.Random(seed)
    grids = _all_sudoku4()
    assert all(sudoku4_valid(list(g)) for g in grids)
    rng.shuffle(grids)
    n_hold = int(len(grids) * holdout_frac)
    eval_grids, train_grids = grids[:n_hold], grids[n_hold:]

    train = torch.tensor([encode("".join(map(str, g))) for g in train_grids])
    prompts, targets = [], []
    for _ in range(n_eval):
        g = rng.choice(eval_grids)
        ids = encode("".join(map(str, g)))
        keep = set(rng.sample(range(16), n_clues))
        ids = [tok if i in keep else MASK_ID for i, tok in enumerate(ids)]
        prompts.append(ids)
        targets.append(list(g))
    return TaskData("sudoku4", 16, train, torch.tensor(prompts), targets)


def make_task(name: str, **kw) -> TaskData:
    if name == "arithmetic":
        return make_arithmetic(**kw)
    if name == "sudoku4":
        return make_sudoku4(**kw)
    raise ValueError(f"unknown task {name!r}")
