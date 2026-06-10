"""Backward-on-Entropy (BoE) steering — the TABES sampler.

Per denoising step t (paper Sec. 3, pipeline figure):

1. Forward pass: denoiser predictions pi_i and entropies H_i for all masked i
   (done by the shared loop in :class:`BaseSampler`).
2. Prefilter: candidate set C_t = top-r masked positions by confidence.
3. Surrogate: build a relaxed next state by injecting soft token embeddings
   e~_i = sum_v pi_i(v)^{1/tau} e_v (normalized) at all candidate slots, run one
   forward at t-1, and take H~_{t-1} = total entropy over probe positions (the
   remaining masked positions, optionally restricted to the top-(rho*L) by
   entropy — the active fraction rho knob).
4. Backward: a single backward pass g_i = dH~_{t-1}/de~_i, sparsified with
   ActiveQueryAttention (queries outside A_t = C_t ∪ probes are stop-gradded).
5. TIS scoring: TIS_i = -<g_i, delta_e_i> with delta_e_i = e~_i - e_mask — the
   first-order predicted reduction in future masked entropy (Theorem 3.1).
6. Final score: confidence-gated TIS minus the anti-collapse penalty
   R_t(i) = [h_t - H_i]^2_+ with entropy floor h_t = h_max * (t / T).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tabes.samplers.base import NEG_INF, BaseSampler


@dataclass
class BoEConfig:
    n_candidates: int = 3       # r: prefilter size (keep tight: the gate IS the prefilter)
    tau: float = 0.5            # soft-embedding sharpening temperature
    gamma: float = 1.0          # confidence-gating exponent
    lambda_ac: float = 0.2      # anti-collapse penalty weight
    h_max: float = 0.3          # entropy floor at t = T (nats)
    rho: float = 1.0            # active fraction: probes capped at rho * L
    use_aqa: bool = True        # sparse adjoint (ablation: full backward)
    use_tis: bool = True        # set False to ablate gradient signal entirely


class BoESampler(BaseSampler):
    name = "boe"
    needs_grad = True

    def __init__(self, model, steps: int = 8, temperature: float = 0.0,
                 trace=None, seed: int = 0, config: BoEConfig | None = None):
        super().__init__(model, steps, temperature, trace, seed)
        self.cfg = config or BoEConfig()

    def score_positions(self, x, logits, masked, step_idx):
        cfg = self.cfg
        B, L = x.shape
        device = x.device
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values                                   # [B, L]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)    # [B, L]

        # ---- (2) prefilter: top-r masked positions by confidence -----------
        conf_masked = torch.where(masked, conf, torch.full_like(conf, NEG_INF))
        order = conf_masked.argsort(dim=1, descending=True)
        ranks = order.argsort(dim=1)
        r_per_row = torch.minimum(
            torch.full_like(masked.sum(1), cfg.n_candidates), masked.sum(1))
        cand = (ranks < r_per_row[:, None]) & masked                      # [B, L]

        # ---- anti-collapse penalty (entropy floor decays with t) -----------
        steps_left = self.steps - step_idx          # t = T..1
        h_t = cfg.h_max * steps_left / self.steps
        penalty = F.relu(h_t - entropy).pow(2)                            # [B, L]

        gated = conf.clamp_min(1e-12).pow(cfg.gamma)
        if not cfg.use_tis:  # ablation: confidence gate + anti-collapse only
            score = gated - cfg.lambda_ac * penalty
            fallback = torch.where(masked, conf - 1e6,
                                   torch.full_like(conf, NEG_INF))
            return torch.where(cand, score, fallback), {
                "trace": {"h_t": h_t, "tis": None}}

        # ---- (3) surrogate state with soft embeddings -----------------------
        E = self.model.token_embedding_matrix.detach()                    # [V, D]
        e_base = self.model.embed(x).detach()                             # [B, L, D]
        e_mask = E[self.model.mask_id]                                    # [D]

        cand_idx, cand_valid, R = _pad_indices(cand)                      # [B, R]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, R)

        p_cand = probs[b_idx, cand_idx]                                   # [B, R, V]
        if cfg.tau != 1.0:
            p_cand = p_cand.clamp_min(1e-12).pow(1.0 / cfg.tau)
            p_cand = p_cand / p_cand.sum(-1, keepdim=True)
        soft_emb = p_cand @ E                                             # [B, R, D]
        base_at = e_base[b_idx, cand_idx]
        inject = torch.where(cand_valid[..., None], soft_emb, base_at)
        e_leaf = inject.detach().requires_grad_(True)
        e_sur = e_base.index_put((b_idx, cand_idx), e_leaf)               # [B, L, D]

        # ---- probe set: remaining masked uncertainty carriers ---------------
        probe_pool = masked & ~cand
        if cfg.rho < 1.0:
            cap = max(1, int(math.ceil(cfg.rho * L)))
            ent_pool = torch.where(probe_pool, entropy,
                                   torch.full_like(entropy, NEG_INF))
            p_ranks = ent_pool.argsort(dim=1, descending=True).argsort(dim=1)
            probes = (p_ranks < cap) & probe_pool
        else:
            probes = probe_pool
        active = cand | probes  # A_t: stop_grad on all other query rows

        # ---- (4) single surrogate forward + backward ------------------------
        with torch.enable_grad():
            logits_sur = self.model.forward_from_embeddings(
                e_sur, active_mask=active if cfg.use_aqa else None)
            logp = F.log_softmax(logits_sur, dim=-1)
            ent_sur = -(logp.exp() * logp).sum(-1)                        # [B, L]
            h_next = (ent_sur * probes.float()).sum()
            h_next.backward()
        g = e_leaf.grad                                                   # [B, R, D]

        # ---- (5) TIS + (6) gating and anti-collapse --------------------------
        delta_e = inject.detach() - e_mask                                # [B, R, D]
        tis = -(g * delta_e).sum(-1)                                      # [B, R]
        tis = torch.where(cand_valid, tis, torch.zeros_like(tis))
        tis_n = _minmax_norm(tis, cand_valid)

        # Rows with no probes left (endgame) carry no TIS signal; fall back to
        # pure confidence there.
        has_probe = probes.any(dim=1, keepdim=True)
        tis_n = torch.where(has_probe.expand_as(tis_n), tis_n, torch.ones_like(tis_n))

        s_cand = gated[b_idx, cand_idx] * tis_n \
            - cfg.lambda_ac * penalty[b_idx, cand_idx]
        # Non-candidate masked positions fall back to confidence ordering at a
        # large negative offset (only reached when budget > |C_t|).
        scores = torch.where(masked, conf - 1e6,
                             torch.full((B, L), NEG_INF, device=device))
        scores[b_idx[cand_valid], cand_idx[cand_valid]] = s_cand[cand_valid]

        extras = {
            "n_forward": 1,
            "n_backward": 1,
            "trace": {
                "h_t": h_t,
                "n_candidates": cand.sum(dim=1),
                "n_probes": probes.sum(dim=1),
                "cand_pos": [c.nonzero(as_tuple=True)[0] for c in cand],
                "tis": [t[v] for t, v in zip(tis, cand_valid)],
                "cand_conf": [conf[i][c] for i, c in enumerate(cand)],
                "cand_entropy": [entropy[i][c] for i, c in enumerate(cand)],
                "h_next_total": h_next.detach(),
            },
        }
        return scores, extras


def _pad_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-row indices of True entries, padded (with unique False positions)."""
    r_max = max(1, int(mask.sum(dim=1).max().item()))
    order = torch.argsort(mask.int(), dim=-1, descending=True, stable=True)
    idx = order[:, :r_max]
    valid = torch.gather(mask, 1, idx)
    return idx, valid, r_max


def _minmax_norm(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per row over valid entries -> [0, 1]."""
    big = torch.finfo(x.dtype).max
    lo = torch.where(valid, x, torch.full_like(x, big)).min(dim=1, keepdim=True).values
    hi = torch.where(valid, x, torch.full_like(x, -big)).max(dim=1, keepdim=True).values
    span = (hi - lo).clamp_min(1e-12)
    return torch.where(valid, (x - lo) / span, torch.zeros_like(x))
