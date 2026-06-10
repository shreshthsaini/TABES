"""Masked-diffusion training objective (LLaDA-style).

Sample a masking level t ~ U(0, 1], mask each token independently with
probability t, and minimize the 1/t-weighted cross-entropy on masked positions
— a variational bound on the data log-likelihood for absorbing-state discrete
diffusion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_batch(
    x0: torch.Tensor, mask_id: int, generator: torch.Generator | None = None,
    prompt_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the forward absorbing process. Returns (x_t, masked, t).

    ``prompt_mask`` marks positions never masked (conditioning tokens).
    """
    B, L = x0.shape
    t = torch.rand(B, 1, device=x0.device, generator=generator).clamp_min(1e-3)
    masked = torch.rand(B, L, device=x0.device, generator=generator) < t
    if prompt_mask is not None:
        masked &= ~prompt_mask
    x_t = torch.where(masked, torch.full_like(x0, mask_id), x0)
    return x_t, masked, t.squeeze(1)


def mdm_loss(
    model, x0: torch.Tensor, mask_id: int, generator: torch.Generator | None = None,
    prompt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    x_t, masked, t = mask_batch(x0, mask_id, generator, prompt_mask)
    logits = model(x_t)
    ce = F.cross_entropy(
        logits.flatten(0, 1), x0.flatten(), reduction="none"
    ).view_as(x0)
    weights = masked.float() / t[:, None]
    denom = masked.float().sum().clamp_min(1.0)
    return (ce * weights).sum() / denom
