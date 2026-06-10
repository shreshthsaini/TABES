"""A small bidirectional transformer masked diffusion model (LLaDA-style denoiser).

The model predicts token distributions at every position given a partially
masked sequence. It exposes the interface required by the samplers:

* ``forward(tokens, active_mask=None) -> logits [B, L, V]``
* ``forward_from_embeddings(emb, active_mask=None) -> logits``
* ``embed(tokens) -> [B, L, D]`` and ``token_embedding_matrix -> [V, D]``
* ``mask_id`` attribute

``active_mask`` is threaded to every ActiveQueryAttention layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from tabes.models.attention import ActiveQueryAttention


@dataclass
class ToyMDMConfig:
    vocab_size: int
    mask_id: int
    max_len: int = 32
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    attn_mode: str = "split"  # ActiveQueryAttention implementation


class Block(nn.Module):
    def __init__(self, cfg: ToyMDMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = ActiveQueryAttention(cfg.dim, cfg.n_heads, mode=cfg.attn_mode)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.mlp_ratio * cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), active_mask)
        h = h + self.mlp(self.ln2(h))
        return h


class ToyMDM(nn.Module):
    def __init__(self, cfg: ToyMDMConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.mask_id
        self.vocab_size = cfg.vocab_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.dim))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        nn.init.normal_(self.pos_emb, std=0.02)

    @property
    def token_embedding_matrix(self) -> torch.Tensor:
        return self.tok_emb.weight

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(tokens)

    def forward_from_embeddings(
        self, emb: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        L = emb.shape[1]
        h = emb + self.pos_emb[:, :L]
        for blk in self.blocks:
            h = blk(h, active_mask)
        return self.head(self.ln_f(h))

    def forward(
        self, tokens: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_from_embeddings(self.embed(tokens), active_mask)
