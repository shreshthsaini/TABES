"""LLaDA adapter: run TABES samplers on LLaDA-8B / LLaDA-1.5 (GPU required).

Wraps the HuggingFace checkpoints (e.g. ``GSAI-ML/LLaDA-8B-Instruct``) behind
the same interface the samplers expect. ActiveQueryAttention requires patching
the model's attention modules, which is checkpoint-specific; this adapter
reports ``supports_aqa = False`` so BoE falls back to a full (dense) backward —
functionally identical, just without the sparse-adjoint speedup. Untested in
this CPU-only environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LLADA_MASK_ID = 126336  # [MASK] token id used by LLaDA checkpoints


class LLaDAAdapter(nn.Module):
    supports_aqa = False

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # lazy import

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.mask_id = LLADA_MASK_ID
        self.vocab_size = self.model.config.vocab_size
        self.device = device

    @property
    def token_embedding_matrix(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(tokens)

    def forward_from_embeddings(self, emb: torch.Tensor,
                                active_mask: torch.Tensor | None = None) -> torch.Tensor:
        # active_mask ignored (supports_aqa = False): dense backward fallback.
        return self.model(inputs_embeds=emb).logits

    def forward(self, tokens: torch.Tensor,
                active_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=tokens).logits
