#!/usr/bin/env python3
"""Run TABES samplers on LLaDA-8B / LLaDA-1.5 (requires GPU + transformers).

Reproduces the paper's large-scale setting: GSM8K / MBPP / HumanEval / MATH500
prompts decoded with confidence vs BoE under compute-matched budgets.

Example:
    python scripts/run_llada.py --model GSAI-ML/LLaDA-8B-Instruct \
        --task gsm8k --sampler boe --gen-len 256 --steps 128 --limit 100

Note: written for the paper's setting but untested in this CPU-only
environment — expect to adjust prompt formatting / answer extraction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabes.models.llada import LLaDAAdapter
from tabes.samplers import SAMPLERS, BoEConfig
from tabes.utils.logging import TraceWriter, get_logger, setup_logging

logger = get_logger("tabes.scripts.llada")


def load_task(name: str, limit: int):
    from datasets import load_dataset  # optional dep

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"prompt": ex["question"],
                  "answer": ex["answer"].split("####")[-1].strip()} for ex in ds]
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"prompt": ex["problem"], "answer": ex["answer"]} for ex in ds]
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = [{"prompt": ex["prompt"], "tests": ex["test_list"]} for ex in ds]
    elif name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        items = [{"prompt": ex["prompt"], "tests": ex["test"],
                  "entry_point": ex["entry_point"]} for ex in ds]
    else:
        raise ValueError(name)
    return items[:limit]


def extract_numeric(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--task", default="gsm8k",
                    choices=["gsm8k", "math500", "mbpp", "humaneval"])
    ap.add_argument("--sampler", default="boe", choices=list(SAMPLERS))
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results/llada"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", out_dir / f"{args.task}_{args.sampler}.log")

    model = LLaDAAdapter(args.model)
    trace = TraceWriter(out_dir / f"{args.task}_{args.sampler}_trace.jsonl")
    kw = {}
    if args.sampler == "boe":
        kw["config"] = BoEConfig(n_candidates=args.n_candidates, rho=args.rho,
                                 use_aqa=False)  # adapter: dense backward
    sampler = SAMPLERS[args.sampler](model, steps=args.steps, trace=trace, **kw)

    items = load_task(args.task, args.limit)
    records = []
    for i, item in enumerate(items):
        msgs = [{"role": "user", "content": item["prompt"]}]
        ids = model.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")[0]
        prompt = torch.cat([
            ids, torch.full((args.gen_len,), model.mask_id, dtype=torch.long)
        ]).unsqueeze(0).to(model.device)

        res = sampler.sample(prompt)
        text = model.tokenizer.decode(res.tokens[0, ids.shape[0]:],
                                      skip_special_tokens=True)
        rec = {"idx": i, "completion": text, **{k: item.get(k) for k in
               ("answer", "tests", "entry_point") if k in item}}
        if "answer" in item:
            rec["correct"] = extract_numeric(text) == extract_numeric(item["answer"])
        records.append(rec)
        logger.info("[%d/%d] correct=%s", i + 1, len(items), rec.get("correct"))

    out = out_dir / f"{args.task}_{args.sampler}_results.json"
    out.write_text(json.dumps(records, indent=2))
    scored = [r for r in records if "correct" in r]
    if scored:
        logger.info("accuracy: %.4f (%d examples)",
                    sum(r["correct"] for r in scored) / len(scored), len(scored))
    logger.info("wrote %s (code tasks: run completions in a sandbox to score)", out)


if __name__ == "__main__":
    main()
