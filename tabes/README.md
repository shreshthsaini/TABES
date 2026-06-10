# TABES — Reference Implementation

Implementation of **TABES: Trajectory-Aware Backward-on-Entropy Steering for
Masked Diffusion Models** (arXiv:2602.00250) with training, structured trace
logging, evaluation, and benchmarking.

BoE steering is a **training-free, model-agnostic inference framework** for
masked diffusion language models. Instead of greedily unmasking the most
confident tokens, it spends one extra forward + one **backward** pass per
denoising step to pick the tokens that **most reduce future masked entropy**.

## What's implemented

| Paper component | Where |
|---|---|
| BoE steering pipeline (prefilter → surrogate → backward → TIS → unmask) | `tabes/samplers/boe.py` |
| Token Importance Score `TIS_i = −⟨g_i, Δe_i⟩` (first-order expansion) | `tabes/samplers/boe.py` |
| ActiveQueryAttention (sparse adjoint, exact forward, `O(\|A\|·L·d)` backward) | `tabes/models/attention.py` |
| Anti-collapse regularizer `R_t = Σ [h_t − H_i]²₊`, floor `h_t = h_max·t/T` | `tabes/samplers/boe.py` |
| Active-fraction knob ρ (probe set capped at ρ·L) | `tabes/samplers/boe.py` |
| Baselines: random / confidence / margin / entropy unmasking | `tabes/samplers/heuristics.py` |
| MDM training (LLaDA-style 1/t-weighted masked CE) | `tabes/diffusion/loss.py`, `tabes/train/trainer.py` |
| Toy bidirectional MDM denoiser (runs the full pipeline on CPU) | `tabes/models/toy_mdm.py` |
| LLaDA-8B / LLaDA-1.5 adapter + GSM8K/MATH500/MBPP/HumanEval harness (GPU) | `tabes/models/llada.py`, `scripts/run_llada.py` |
| Step-level JSONL trace logging (entropies, candidates, TIS, selections, timings) | `tabes/utils/logging.py` |
| Eval harness + metrics | `tabes/eval/harness.py` |
| Benchmarks: Pareto sweep, ablations (−TIS, −anti-collapse, −AQA, ρ sweep) | `tabes/bench/benchmark.py` |

## Quickstart (CPU, ~15 min end-to-end)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 27 tests: TIS vs finite differences,
                                           # AQA forward-exactness & grad sparsity, ...
python scripts/train_toy.py                # train toy MDMs (arithmetic + sudoku4)
python scripts/run_benchmark.py            # Pareto sweep + ablations -> results/report.md
python scripts/generate.py --task sudoku4 --sampler boe   # watch a trajectory
```

## Why these toy tasks

There is no GPU in the dev environment, so the pipeline is validated end-to-end
on tasks where **unmasking order is causally load-bearing** — the failure mode
(trajectory lock-in) TABES targets:

* **`sudoku4`** — complete a 4×4 Sudoku from a few clues. A single wrong early
  commit provably forces constraint violations later: lock-in in its purest
  form. Metric: fraction of completions satisfying all constraints.
* **`arithmetic`** — `AA+BB=CCC` with masked answers; carry chains make digit
  order matter. Metric: exact match.

The same sampler code runs unchanged on LLaDA checkpoints via
`scripts/run_llada.py` (GPU; the adapter falls back to a dense backward since
AQA requires patching the checkpoint's attention modules).

## BoE step (as implemented)

For each denoising step `t = T..1`:

1. **Forward**: logits → `π_i`, entropy `H_i`, confidence `c_i` for masked `i`
   (the mask-token logit is suppressed).
2. **Prefilter**: `C_t` = top-`r` masked positions by confidence.
3. **Surrogate**: inject soft embeddings `ẽ_i = Σ_v π_i(v)^{1/τ} e_v` (normalized)
   at all candidate slots at once; probe set = remaining masked positions
   (top-`ρ·L` by entropy if `ρ < 1`); one forward at `t−1` gives
   `H̃_{t−1} = Σ_{j∈probes} H(π̃_j)`.
4. **Backward**: single backward, `g_i = ∇_{ẽ_i} H̃_{t−1}`, with
   ActiveQueryAttention stop-gradding all query rows outside `A_t = C_t ∪ probes`.
5. **TIS**: `TIS_i = −⟨g_i, ẽ_i − e_mask⟩`, min-max normalized over `C_t`.
6. **Score & unmask**: `s_i = c_i^γ · TIS̃_i − λ_ac·[h_t − H_i]²₊`; reveal the
   per-row budget `⌈masked/steps_left⌉` of top-scoring positions (greedy commit,
   or Gumbel sampling with `--temperature`).

Cost per step: 2 forwards + 1 (sparse) backward, vs 1 forward for heuristics —
matching the paper's compute model.

### Implementation notes (deviations / interpretations)

* The paper PDF was unreachable from this environment (network policy); the
  implementation follows the method spec in this repo's project page and
  figures. Two interpretation points to verify against the paper:
  * **Active set**: taken as `A_t = C_t ∪ probes` — gradients must reach
    candidate embeddings *from* probe entropies, so probe query rows stay
    grad-active. With `A_t = C_t` strictly, `∂H̃/∂ẽ_i` would be zero in a
    stop-grad-everywhere reading.
  * **ρ** caps the probe set by top entropy (the dominant uncertainty
    carriers), which is what makes `|A_t| ≪ L`.
* `mask` mode of AQA is a slow reference implementation used to test that the
  `split` mode (which realizes the actual sparse-backward savings) computes
  identical gradients.

## Layout

```
tabes/
├── tabes/                  # package
│   ├── models/             # toy MDM, ActiveQueryAttention, LLaDA adapter
│   ├── samplers/           # base loop + heuristics + BoE
│   ├── diffusion/          # forward masking + training loss
│   ├── data/               # synthetic tasks (arithmetic, sudoku4)
│   ├── train/ eval/ bench/ # trainer, eval harness, benchmark sweeps
│   └── utils/              # logging + JSONL tracing, seeding
├── scripts/                # train_toy, run_benchmark, generate, run_llada
├── configs/toy.yaml        # one config drives the whole toy experiment
├── tests/                  # 27 unit tests
├── runs/                   # checkpoints + train traces (gitignored)
└── results/                # benchmark JSON + report.md + decode traces
```

## Trace format

Every decode step appends one JSON line:
`{step, sampler, n_masked, budget, revealed_pos, revealed_tok,
mean_masked_entropy, step_wall_s, n_candidates, n_probes, cand_pos, tis,
cand_conf, cand_entropy, h_next_total, h_t}` — enough to replay and audit any
trajectory offline (e.g., verify BoE picks pivots over easy function tokens).
