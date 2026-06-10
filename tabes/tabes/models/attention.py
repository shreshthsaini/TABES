"""ActiveQueryAttention: sparse-adjoint multi-head self-attention.

Forward outputs are *exactly* those of standard attention. The backward pass is
restricted to an "active" subset of query rows A_t: gradients only flow through
attention rows whose query position is active (stop-grad on inactive queries).
This reduces the attention backward cost from O(L^2 d) to O(|A| L d).

Two implementations with identical semantics:

* ``mode="split"`` (default): the full attention map is computed under
  ``no_grad`` (graph-free), and attention is recomputed *only for active query
  rows* with autograd enabled, then scattered back. This realizes the actual
  O(|A| L d) backward cost.
* ``mode="mask"``: a reference implementation using
  ``torch.where(active, q, q.detach())`` on the query tensor. Semantically
  identical gradients, but no wall-clock savings (used for testing).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_active_indices(active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return per-row active indices padded to the max active count.

    Padding uses *inactive* positions (stable argsort puts actives first), so
    indices within a row are unique and padded rows can be safely overwritten
    with detached values.
    """
    a_max = int(active_mask.sum(dim=-1).max().item())
    order = torch.argsort(active_mask.int(), dim=-1, descending=True, stable=True)
    idx = order[:, :a_max]                       # [B, A]
    valid = torch.gather(active_mask, 1, idx)    # [B, A]
    return idx, valid, a_max


class ActiveQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, mode: str = "split"):
        super().__init__()
        assert dim % n_heads == 0
        assert mode in ("split", "mask")
        self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
        self.mode = mode
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, D]; active_mask: optional bool [B, L] of active query rows."""
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,L,dh]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if active_mask is None or not torch.is_grad_enabled():
            out = F.scaled_dot_product_attention(q, k, v)
        elif self.mode == "mask":
            # Reference implementation: compute attention twice (with and
            # without graph) and select rows. Same gradients as "split", no
            # wall-clock savings.
            out_grad = F.scaled_dot_product_attention(q, k, v)
            with torch.no_grad():
                out_nograd = F.scaled_dot_product_attention(
                    q.detach(), k.detach(), v.detach())
            out = torch.where(active_mask[:, None, :, None], out_grad, out_nograd)
        else:
            out = self._split_forward(q, k, v, active_mask)

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

    def _split_forward(self, q, k, v, active_mask):
        B, h, L, dh = q.shape
        idx, valid, a_max = _pad_active_indices(active_mask)
        if a_max == 0:
            with torch.no_grad():
                return F.scaled_dot_product_attention(q, k, v).detach()
        if a_max >= L:  # everything active: no savings possible, plain attention
            return F.scaled_dot_product_attention(q, k, v)

        with torch.no_grad():  # graph-free full forward, O(L^2 d) but no adjoint
            out_base = F.scaled_dot_product_attention(q.detach(), k.detach(), v.detach())

        # Recompute only active rows with autograd: O(|A| L d) backward.
        idx_q = idx[:, None, :, None].expand(B, h, a_max, dh)
        q_a = torch.gather(q, 2, idx_q)                      # [B,h,A,dh]
        out_a = F.scaled_dot_product_attention(q_a, k, v)    # [B,h,A,dh]
        # Padded rows must contribute no gradient (values are identical anyway).
        out_a = torch.where(valid[:, None, :, None], out_a, out_a.detach())

        b_idx = torch.arange(B, device=q.device)[:, None].expand(B, a_max)
        base = out_base.permute(0, 2, 1, 3)                  # [B,L,h,dh]
        out = base.index_put((b_idx, idx), out_a.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3)
