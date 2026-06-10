"""Local-heuristic unmasking baselines: random, confidence, margin, entropy."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tabes.samplers.base import NEG_INF, BaseSampler


class RandomSampler(BaseSampler):
    name = "random"

    def score_positions(self, x, logits, masked, step_idx):
        scores = torch.rand(masked.shape, generator=self.generator).to(logits.device)
        return torch.where(masked, scores, torch.full_like(scores, NEG_INF)), {}


class ConfidenceSampler(BaseSampler):
    name = "confidence"

    def score_positions(self, x, logits, masked, step_idx):
        conf = F.softmax(logits, dim=-1).max(dim=-1).values
        return torch.where(masked, conf, torch.full_like(conf, NEG_INF)), {}


class MarginSampler(BaseSampler):
    name = "margin"

    def score_positions(self, x, logits, masked, step_idx):
        top2 = F.softmax(logits, dim=-1).topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return torch.where(masked, margin, torch.full_like(margin, NEG_INF)), {}


class EntropySampler(BaseSampler):
    name = "entropy"

    def score_positions(self, x, logits, masked, step_idx):
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        return torch.where(masked, -ent, torch.full_like(ent, NEG_INF)), {}
