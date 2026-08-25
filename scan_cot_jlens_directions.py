#!/usr/bin/env python3
"""
scan_cot_jlens_directions.py

Scan GPT-OSS-20B's generated chain of thought for activation of J-Lens
right-singular directions. This is a generation driver for the statistics and
ranking machinery in scan_sparse_jlens_directions.py.

Why generation and activation capture are separate passes
---------------------------------------------------------
During generation, retaining every selected layer's activation at every step
is needlessly expensive. Instead, this script:

  1. generates one complete Harmony response with the KV cache;
  2. locates only the generated ``analysis``-channel content tokens;
  3. releases the generation cache;
  4. teacher-forces the exact prompt + response once and records those tokens.

Because the transformer is causal and in eval mode, the hidden state at each
replayed token position is the same state obtained when that token was
processed autoregressively (up to ordinary kernel-level numerical variation).
The resulting projections therefore reflect the model's actual reasoning
context, rather than a separately tokenized plain-text version of its CoT.

This script deliberately reuses the supplied sparse scanner's tested code for
SVD construction, streaming moments, robust tail/selectivity metrics, extreme
context heaps, unembedding geometry, and CSV/JSONL output. Keep the two scripts
in the same directory, or pass --scanner-base PATH.

Expected prompt JSONL (either form may be mixed):

    {"id":"logic_01", "category":"logic", "prompt":"..."}
    {"id":"chat_01", "messages":[{"role":"user","content":"..."}]}

If --prompts-jsonl is omitted, a small built-in set of moderately difficult,
diverse reasoning tasks is used for a smoke test / first run.

L4-friendly starting runs
-------------------------
Low effort (recommended first):

    python scan_cot_jlens_directions.py \
      --scanner-base scan_sparse_jlens_directions.py \
      --prompts-jsonl reasoning_prompts.jsonl \
      --reasoning-effort low \
      --max-new-tokens 512 \
      --k 64 \
      --out cot_unrealized_low

Medium effort after the low run is stable:

    python scan_cot_jlens_directions.py \
      --scanner-base scan_sparse_jlens_directions.py \
      --prompts-jsonl reasoning_prompts.jsonl \
      --reasoning-effort medium \
      --max-new-tokens 768 \
      --max-replay-tokens 1536 \
      --out cot_unrealized_medium

Outputs
-------
    OUT/sv_rankings.csv
    OUT/selectivity_rankings.csv
    OUT/top_contexts.jsonl
    OUT/rollouts.jsonl
    OUT/unembedding_neighbors.jsonl      (unless --skip-unembedding)
    OUT/metadata.json
    OUT/checkpoint.pkl
    OUT/directions/LXX.npz

Layer numbers are 0-based. SV labels/ranks are 1-based, matching the supplied
scanners (for example L09_SV10).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"


# These are intentionally moderate rather than impossible. They elicit real
# multi-step reasoning without encouraging very long failed searches.
BUILTIN_PROMPTS: list[dict[str, str]] = [
    {
        "id": "logic_schedule",
        "category": "logic",
        "prompt": (
            "Ava, Ben, Cara, and Dev each present once from Monday through Thursday. "
            "Ava presents before Cara. Ben is not on Monday. Dev presents immediately "
            "after Ben. Determine the schedule and explain your reasoning."
        ),
    },
    {
        "id": "probability_cards",
        "category": "probability",
        "prompt": (
            "A box contains 4 red, 3 blue, and 2 green cards. Two cards are drawn "
            "without replacement. Given that at least one is red, what is the "
            "probability both are red? Work it out carefully."
        ),
    },
    {
        "id": "causal_machine",
        "category": "causal_reasoning",
        "prompt": (
            "A machine overheats only on humid afternoons. Replacing its fan changes "
            "nothing, cleaning a clogged intake eliminates the problem, and opening a "
            "nearby window brings it back. What is the most likely causal explanation, "
            "and what observation would best distinguish it from the main alternative?"
        ),
    },
    {
        "id": "code_trace",
        "category": "programming",
        "prompt": (
            "Trace this Python mentally and explain the surprising part: "
            "x=[[0]*2]*3; x[1][0]=7; y=[row[:] for row in x]; y[2][1]=9. "
            "What are x and y at the end?"
        ),
    },
    {
        "id": "spatial_cube",
        "category": "spatial",
        "prompt": (
            "A cube has opposite face pairs red/blue, green/yellow, and black/white. "
            "It rests with red on top and green facing you. The cube rolls away from "
            "you, then to its right, then toward you. Which color is on top? Explain."
        ),
    },
    {
        "id": "linguistic_scope",
        "category": "language",
        "prompt": (
            "The sentence 'Every student read a book' has two classic scope readings. "
            "State both readings, give a situation where one is true and the other "
            "false, and explain the difference without using formal notation."
        ),
    },
    {
        "id": "planning_errands",
        "category": "planning",
        "prompt": (
            "You must visit a pharmacy before it closes at 5, collect a package that "
            "will not be ready until 4:20, and attend a 30-minute appointment at 3:30. "
            "Travel takes 15 minutes between any two places. Starting at 3:00, make a "
            "feasible plan and identify any assumption you need."
        ),
    },
    {
        "id": "bayes_testing",
        "category": "probability",
        "prompt": (
            "A condition affects 1% of people. A test has 95% sensitivity and a 5% "
            "false-positive rate. Someone tests positive twice using independent test "
            "errors. Estimate the probability they have the condition and show the steps."
        ),
    },
    {
        "id": "debug_experiment",
        "category": "scientific_reasoning",
        "prompt": (
            "An intervention appears to improve accuracy by 8 points, but only when "
            "examples are evaluated in the same order used during tuning. List the most "
            "likely failure modes, rank them, and propose the fastest diagnostic sequence."
        ),
    },
    {
        "id": "counterfactual_policy",
        "category": "counterfactual",
        "prompt": (
            "A city introduces congestion pricing and traffic falls 15%, while a nearby "
            "similar city sees a 5% fall during the same month. What can and cannot be "
            "inferred about the policy's effect? Reason through the counterfactual."
        ),
    },
    {
        "id": "number_pattern",
        "category": "induction",
        "prompt": (
            "A sequence begins 2, 6, 12, 20, 30. Give the simplest continuation, then "
            "construct a different defensible rule that agrees so far but predicts a "
            "different next term. What does this demonstrate?"
        ),
    },
    {
        "id": "mechanism_analogy",
        "category": "analogy",
        "prompt": (
            "Compare a thermostat, a biological homeostatic loop, and gradient descent. "
            "Identify the shared abstract structure, then explain exactly where the "
            "analogy breaks for each pair."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan J-Lens right-SV directions over GPT-OSS analysis-channel tokens.",
    )

    # Reuse / model / lens.
    p.add_argument(
        "--scanner-base",
        default=None,
        help=(
            "Path to scan_sparse_jlens_directions.py. If omitted, common filenames "
            "are searched beside this script and in the current directory."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    p.add_argument("--layers", default="all")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--directions-dir", default=None)
    p.add_argument("--exact-svd", action="store_true")
    p.add_argument("--svd-oversample", type=int, default=16)
    p.add_argument("--svd-niter", type=int, default=4)
    p.add_argument("--recompute-directions", action="store_true")

    # Prompts / generation.
    p.add_argument(
        "--prompts-jsonl",
        default=None,
        help="JSONL with prompt or messages fields. Omit to use built-in diagnostic tasks.",
    )
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=768,
        help="Skip (do not truncate) formatted prompts longer than this many tokens.",
    )
    p.add_argument(
        "--max-replay-tokens",
        type=int,
        default=1536,
        help="Skip activation replay if prompt + generation exceeds this L4 safety cap.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding. Sampling is the default so repeated samples can differ.",
    )

    # Direction statistics.
    p.add_argument("--top-r", type=int, default=5)
    p.add_argument("--top-contexts", type=int, default=64)
    p.add_argument("--local-context-candidates", type=int, default=3)
    p.add_argument("--context-radius", type=int, default=24)
    p.add_argument("--token-reservoir-size", type=int, default=32768)
    p.add_argument("--min-tail-docs", type=int, default=5)

    # Unembedding geometry.
    p.add_argument("--skip-unembedding", action="store_true")
    p.add_argument("--unembedding-neighbors", type=int, default=32)
    p.add_argument("--unembedding-chunk-size", type=int, default=4096)
    p.add_argument("--include-special-unembedding-tokens", action="store_true")

    # Run / checkpoint.
    p.add_argument("--out", default="cot_unrealized_words_scan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=1)
    args = p.parse_args()

    for name in (
        "k",
        "samples_per_prompt",
        "max_new_tokens",
        "max_input_tokens",
        "max_replay_tokens",
        "top_r",
        "top_contexts",
        "local_context_candidates",
        "context_radius",
        "token_reservoir_size",
        "min_tail_docs",
        "unembedding_neighbors",
        "unembedding_chunk_size",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_prompts is not None and args.max_prompts <= 0:
        p.error("--max-prompts must be positive")
    if not args.greedy and args.temperature <= 0:
        p.error("--temperature must be positive unless --greedy is used")
    if not (0 < args.top_p <= 1):
        p.error("--top-p must be in (0, 1]")
    return args


def locate_scanner_base(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--scanner-base does not exist: {path}")
        return path

    roots = [Path(__file__).resolve().parent, Path.cwd()]
    names = [
        "scan_sparse_jlens_directions.py",
        "scan_sparse_jlens_directions(1).py",
        "scan_sparse_jlens_directions (1).py",
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.resolve() != Path(__file__).resolve():
                return candidate.resolve()
        upload = root / "upload"
        for name in names:
            candidate = upload / name
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not find scan_sparse_jlens_directions.py. Put it beside this script "
        "or pass --scanner-base /path/to/scan_sparse_jlens_directions.py"
    )


def load_scanner_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sparse_jlens_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import scanner base from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "parse_layers",
        "prepare_direction_bank",
        "init_stats",
        "update_stats",
        "collect_context_candidates",
        "analyze_unembedding_geometry",
        "build_ranking_rows",
        "add_context_diversity_columns",
        "add_unembedding_columns",
        "write_csv",
        "write_contexts",
        "write_unembedding_neighbors",
        "gpu_status",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Scanner base {path} is missing required helpers: {', '.join(missing)}"
        )
    return module


def load_prompts(path: str | None, max_prompts: int | None) -> list[dict[str, Any]]:
    if path is None:
        records: list[dict[str, Any]] = [dict(x) for x in BUILTIN_PROMPTS]
    else:
        prompt_path = Path(path)
        records = []
        with prompt_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{prompt_path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{prompt_path}:{line_no}: each line must be an object")
                records.append(item)

    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(records):
        prompt_id = str(item.get("id", f"prompt_{i:04d}"))
        if "messages" in item:
            messages = item["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{prompt_id}: messages must be a nonempty list")
            clean_messages = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{prompt_id}: every message must be an object")
                role = message.get("role")
                content = message.get("content")
                if role not in ("system", "developer", "user", "assistant"):
                    raise ValueError(f"{prompt_id}: unsupported role {role!r}")
                if not isinstance(content, str):
                    raise ValueError(f"{prompt_id}: message content must be a string")
                clean_messages.append({"role": role, "content": content})
            messages = clean_messages
        else:
            prompt = item.get("prompt", item.get("question", item.get("text")))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{prompt_id}: expected a nonempty prompt, question, text, or messages field"
                )
            messages = [{"role": "user", "content": prompt}]

        source = {
            key: value
            for key, value in item.items()
            if key not in ("messages", "prompt", "question", "text")
            and isinstance(value, (str, int, float, bool))
        }
        source["id"] = prompt_id
        normalized.append({"id": prompt_id, "messages": messages, "source": source})

    if max_prompts is not None:
        normalized = normalized[:max_prompts]
    if not normalized:
        raise ValueError("No prompts were loaded")
    return normalized


def make_jobs(prompts: list[dict[str, Any]], samples_per_prompt: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(samples_per_prompt):
            jobs.append(
                {
                    **prompt,
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "job_index": len(jobs),
                }
            )
    return jobs


def marker_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer produced no ids for Harmony marker {text!r}")
    return [int(x) for x in ids]


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return start
    limit = len(sequence) - len(pattern) + 1
    for i in range(start, max(start, limit)):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def harmony_channel_positions(
    tokenizer: Any,
    generated_ids: list[int],
    channel: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return content-token positions local to generated_ids and segment bounds."""
    start_pattern = marker_ids(tokenizer, f"<|channel|>{channel}<|message|>")
    end_patterns = [
        marker_ids(tokenizer, "<|end|>"),
        marker_ids(tokenizer, "<|return|>"),
    ]
    special_ids = set(int(x) for x in tokenizer.all_special_ids)

    positions: list[int] = []
    segments: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(generated_ids):
        marker_at = find_subsequence(generated_ids, start_pattern, cursor)
        if marker_at < 0:
            break
        content_start = marker_at + len(start_pattern)
        ends = [
            pos
            for pattern in end_patterns
            if (pos := find_subsequence(generated_ids, pattern, content_start)) >= 0
        ]
        content_end = min(ends) if ends else len(generated_ids)
        segments.append((content_start, content_end))
        positions.extend(
            i
            for i in range(content_start, content_end)
            if generated_ids[i] not in special_ids
        )
        cursor = max(content_end + 1, content_start + 1)
    return positions, segments


def decode_positions(tokenizer: Any, ids: list[int], positions: list[int]) -> str:
    if not positions:
        return ""
    pieces: list[str] = []
    last = -2
    for pos in positions:
        if pos != last + 1 and pieces:
            pieces.append("\n\n[...next analysis segment...]\n\n")
        pieces.append(
            tokenizer.decode(
                [ids[pos]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        last = pos
    return "".join(pieces)


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def atomic_json_dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_signature(args: argparse.Namespace, jobs: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    prompt_payload = json.dumps(
        [{"messages": j["messages"], "sample_index": j["sample_index"]} for j in jobs],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": layers,
        "k": args.k,
        "directions_dir": args.directions_dir,
        "reasoning_effort": args.reasoning_effort,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_replay_tokens": args.max_replay_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "top_r": args.top_r,
        "top_contexts": args.top_contexts,
        "local_context_candidates": args.local_context_candidates,
        "context_radius": args.context_radius,
        "token_reservoir_size": args.token_reservoir_size,
        "min_tail_docs": args.min_tail_docs,
        "seed": args.seed,
        "jobs_sha256": hashlib.sha256(prompt_payload).hexdigest(),
        "n_jobs": len(jobs),
    }


def save_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    next_job_index: int,
    analysis_docs: int,
    total_analysis_tokens: int,
    stats: dict[int, dict[str, Any]],
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
    heap_counter: int,
    rollout_records: list[dict[str, Any]],
) -> None:
    atomic_pickle_dump(
        {
            "signature": signature,
            "next_job_index": next_job_index,
            "analysis_docs": analysis_docs,
            "total_analysis_tokens": total_analysis_tokens,
            "stats": stats,
            "heaps": heaps,
            "heap_counter": heap_counter,
            "rollout_records": rollout_records,
        },
        path,
    )


def load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("signature") != signature:
        raise ValueError(
            "Checkpoint configuration does not match this run. Use a new --out "
            "directory or restore the original arguments."
        )
    return state


def safe_generation(
    *,
    hf_model: Any,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    job_seed: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort=args.reasoning_effort,
    )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if prompt_tokens > args.max_input_tokens:
        return None, {
            "status": "skipped_prompt_too_long",
            "prompt_tokens": prompt_tokens,
        }

    inputs = {key: value.to(model.input_device) for key, value in inputs.items()}
    torch.manual_seed(job_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job_seed)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
        "output_hidden_states": False,
    }
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if not args.greedy:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    with torch.inference_mode():
        sequences = hf_model.generate(**inputs, **generation_kwargs)
    sequence = sequences[0].detach().cpu()
    generated_tokens = int(sequence.numel() - prompt_tokens)
    del sequences, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sequence, {
        "status": "generated",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pkl"
    rollouts_path = out_dir / "rollouts.jsonl"

    scanner_path = locate_scanner_base(args.scanner_base)
    base = load_scanner_module(scanner_path)
    prompts = load_prompts(args.prompts_jsonl, args.max_prompts)
    jobs = make_jobs(prompts, args.samples_per_prompt)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[base] {scanner_path}")
    print(f"[jobs] prompts={len(prompts)} samples={args.samples_per_prompt} total={len(jobs)}")
    print("[1/4] Loading J-Lens and preparing right-SV bank...")
    import jlens

    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    layers = base.parse_layers(args.layers, lens.source_layers)
    if args.k > lens.d_model:
        raise ValueError(f"k={args.k} exceeds d_model={lens.d_model}")
    V_bank, S_bank, direction_source = base.prepare_direction_bank(
        args, lens, layers, out_dir
    )
    d_model = int(lens.d_model)
    del lens
    gc.collect()

    print("[2/4] Loading GPT-OSS model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    model = jlens.from_hf(hf_model, tokenizer)
    if int(model.d_model) != d_model:
        raise ValueError(
            f"Model d_model={model.d_model} does not match J-Lens d_model={d_model}"
        )
    print(base.gpu_status())

    signature = build_signature(args, jobs, layers)
    estimated_max_analysis_tokens = max(1, len(jobs) * args.max_new_tokens)
    token_sample_stride = max(
        1, estimated_max_analysis_tokens // args.token_reservoir_size
    )
    token_sample_capacity = (
        int(math.ceil(estimated_max_analysis_tokens / token_sample_stride)) + 2
    )
    sample_offset_rng = random.Random(args.seed + 917_531)
    token_sample_offset = sample_offset_rng.randrange(token_sample_stride)

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but no {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, signature)
        next_job_index = int(state["next_job_index"])
        analysis_docs = int(state["analysis_docs"])
        total_analysis_tokens = int(state["total_analysis_tokens"])
        stats = state["stats"]
        heaps = state["heaps"]
        heap_counter = int(state["heap_counter"])
        rollout_records = state["rollout_records"]
        print(
            f"[resume] next_job={next_job_index}/{len(jobs)} "
            f"analysis_rollouts={analysis_docs} tokens={total_analysis_tokens:,}"
        )
    else:
        next_job_index = 0
        analysis_docs = 0
        total_analysis_tokens = 0
        stats = base.init_stats(
            layers, args.k, len(jobs), token_sample_capacity
        )
        heaps: dict[
            tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]
        ] = {}
        heap_counter = 0
        rollout_records: list[dict[str, Any]] = []

    V_device_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    print("[3/4] Generating and replaying analysis-channel tokens...")
    for job_index in range(next_job_index, len(jobs)):
        job = jobs[job_index]
        replay_started = False
        source = dict(job["source"])
        source.update(
            {
                "sample_index": job["sample_index"],
                "reasoning_effort": args.reasoning_effort,
            }
        )
        record: dict[str, Any] = {
            "job_index": job_index,
            "prompt_index": job["prompt_index"],
            "sample_index": job["sample_index"],
            "prompt_id": job["id"],
            "source": source,
            "messages": job["messages"],
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed + job_index,
        }

        try:
            sequence, generation_meta = safe_generation(
                hf_model=hf_model,
                model=model,
                tokenizer=tokenizer,
                messages=job["messages"],
                args=args,
                job_seed=args.seed + job_index,
            )
            record.update(generation_meta)
            if sequence is None:
                rollout_records.append(record)
                next_job_index = job_index + 1
                continue

            full_ids_cpu = [int(x) for x in sequence.tolist()]
            prompt_len = int(record["prompt_tokens"])
            generated_ids = full_ids_cpu[prompt_len:]
            analysis_local, analysis_segments = harmony_channel_positions(
                tokenizer, generated_ids, "analysis"
            )
            final_local, final_segments = harmony_channel_positions(
                tokenizer, generated_ids, "final"
            )
            analysis_positions = [prompt_len + x for x in analysis_local]

            record.update(
                {
                    "raw_response": tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "reasoning": decode_positions(tokenizer, generated_ids, analysis_local),
                    "final": decode_positions(tokenizer, generated_ids, final_local),
                    "analysis_tokens": len(analysis_positions),
                    "final_tokens": len(final_local),
                    "analysis_segments": len(analysis_segments),
                    "final_segments": len(final_segments),
                    "sequence_tokens": len(full_ids_cpu),
                    "hit_max_new_tokens": len(generated_ids) >= args.max_new_tokens,
                }
            )

            if not analysis_positions:
                record["status"] = "no_analysis_channel_found"
                rollout_records.append(record)
                next_job_index = job_index + 1
                continue
            if len(full_ids_cpu) > args.max_replay_tokens:
                record["status"] = "skipped_replay_too_long"
                rollout_records.append(record)
                next_job_index = job_index + 1
                continue

            input_ids = sequence.unsqueeze(0).to(model.input_device)
            # From here onward an exception could leave only a prefix of layers
            # updated. Fail fast rather than checkpointing a cross-layer-
            # inconsistent state. --resume will restart from the last clean
            # checkpoint.
            replay_started = True
            with torch.inference_mode(), jlens.ActivationRecorder(
                model.layers, at=layers
            ) as recorder:
                model.forward(input_ids)

                for layer in layers:
                    hidden_full = recorder.activations[layer][0].detach()
                    valid_positions = torch.tensor(
                        analysis_positions,
                        dtype=torch.long,
                        device=hidden_full.device,
                    )
                    hidden = hidden_full.index_select(0, valid_positions)
                    cache_key = (layer, str(hidden.device), hidden.dtype)
                    V = V_device_cache.get(cache_key)
                    if V is None:
                        V = V_bank[layer].to(hidden.device, dtype=hidden.dtype)
                        V_device_cache[cache_key] = V

                    proj = (hidden @ V).float()
                    residual_norm = hidden.float().norm(dim=-1)
                    cos = proj / residual_norm.clamp_min(1e-12).unsqueeze(1)

                    first_sample_row = (
                        token_sample_offset - total_analysis_tokens
                    ) % token_sample_stride
                    token_sample_rows = torch.arange(
                        first_sample_row,
                        proj.shape[0],
                        token_sample_stride,
                        dtype=torch.long,
                        device=proj.device,
                    )
                    base.update_stats(
                        stats[layer],
                        proj,
                        residual_norm,
                        args.top_r,
                        analysis_docs,
                        token_sample_rows,
                    )
                    heap_counter = base.collect_context_candidates(
                        layer=layer,
                        proj=proj,
                        cos=cos,
                        valid_positions=valid_positions,
                        input_ids_cpu=full_ids_cpu,
                        source=source,
                        doc_index=analysis_docs,
                        window_meta={
                            "prompt_tokens": prompt_len,
                            "generated_tokens": len(generated_ids),
                            "analysis_tokens": len(analysis_positions),
                            "analysis_segments": len(analysis_segments),
                        },
                        heaps=heaps,
                        counter=heap_counter,
                        top_contexts=args.top_contexts,
                        local_candidates=args.local_context_candidates,
                        context_radius=args.context_radius,
                    )
                    del proj, cos, residual_norm, hidden, valid_positions, token_sample_rows

            del input_ids, sequence
            total_analysis_tokens += len(analysis_positions)
            analysis_docs += 1
            record["status"] = "scanned"
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM. The current in-memory job was not checkpointed, so the "
                "last saved checkpoint remains clean. Lower --max-new-tokens and "
                "--max-replay-tokens, keep --reasoning-effort low, or scan fewer "
                "--layers; then use a new --out because those settings change the "
                "run signature."
            ) from exc
        except Exception as exc:
            if replay_started:
                raise RuntimeError(
                    "Activation replay failed after statistics may have begun updating. "
                    "Stopping preserves the last clean checkpoint for --resume."
                ) from exc
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")

        rollout_records.append(record)
        next_job_index = job_index + 1

        if args.progress_every > 0 and next_job_index % args.progress_every == 0:
            print(
                f"[scan] jobs={next_job_index}/{len(jobs)} "
                f"analysis_rollouts={analysis_docs} tokens={total_analysis_tokens:,} "
                f"last={record['status']}" + base.gpu_status(),
                flush=True,
            )

        if args.checkpoint_every > 0 and next_job_index % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                signature=signature,
                next_job_index=next_job_index,
                analysis_docs=analysis_docs,
                total_analysis_tokens=total_analysis_tokens,
                stats=stats,
                heaps=heaps,
                heap_counter=heap_counter,
                rollout_records=rollout_records,
            )
            write_jsonl(rollout_records, rollouts_path)
            print(f"[checkpoint] {checkpoint_path}", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_checkpoint(
        checkpoint_path,
        signature=signature,
        next_job_index=next_job_index,
        analysis_docs=analysis_docs,
        total_analysis_tokens=total_analysis_tokens,
        stats=stats,
        heaps=heaps,
        heap_counter=heap_counter,
        rollout_records=rollout_records,
    )
    write_jsonl(rollout_records, rollouts_path)

    if analysis_docs == 0:
        raise RuntimeError(
            "No rollouts with an identifiable analysis channel were scanned. "
            f"Inspect {rollouts_path} for raw responses and status/error fields."
        )

    print("[4/4] Building rankings and optional unembedding geometry...")
    rows = base.build_ranking_rows(
        stats, S_bank, layers, args.k, args.top_r, args.min_tail_docs
    )
    base.add_context_diversity_columns(rows, heaps)

    unembedding_geometry: dict[tuple[int, int], dict[str, Any]] = {}
    if not args.skip_unembedding:
        unembedding_geometry = base.analyze_unembedding_geometry(
            hf_model=hf_model,
            tokenizer=tokenizer,
            V_bank=V_bank,
            layers=layers,
            k=args.k,
            n_neighbors=args.unembedding_neighbors,
            chunk_size=args.unembedding_chunk_size,
            include_special_tokens=args.include_special_unembedding_tokens,
        )
        base.add_unembedding_columns(rows, unembedding_geometry)

    rankings_path = out_dir / "sv_rankings.csv"
    selectivity_path = out_dir / "selectivity_rankings.csv"
    contexts_path = out_dir / "top_contexts.jsonl"
    unembedding_path = out_dir / "unembedding_neighbors.jsonl"
    metadata_path = out_dir / "metadata.json"

    base.write_csv(rows, rankings_path)
    base.write_csv(
        sorted(rows, key=lambda row: row["tail_selectivity_score"], reverse=True),
        selectivity_path,
    )
    base.write_contexts(contexts_path, tokenizer, heaps, S_bank, layers, args.k)
    if unembedding_geometry:
        base.write_unembedding_neighbors(
            unembedding_path, unembedding_geometry, S_bank, layers, args.k
        )

    status_counts: dict[str, int] = {}
    for record in rollout_records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    metadata = {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": layers,
        "d_model": d_model,
        "k": args.k,
        "direction_source": direction_source,
        "scanner_base_filename": scanner_path.name,
        "prompts_source": args.prompts_jsonl or "built_in_moderate_reasoning_set",
        "prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "jobs": len(jobs),
        "status_counts": status_counts,
        "reasoning_effort": args.reasoning_effort,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_replay_tokens": args.max_replay_tokens,
        "sampling": not args.greedy,
        "temperature": None if args.greedy else args.temperature,
        "top_p": None if args.greedy else args.top_p,
        "analysis_rollouts_scanned": analysis_docs,
        "analysis_tokens_scanned": total_analysis_tokens,
        "token_sample_target_per_layer": args.token_reservoir_size,
        "token_sample_stride": token_sample_stride,
        "token_sample_offset": token_sample_offset,
        "token_sample_actual_per_layer": {
            str(layer): int(stats[layer]["token_sample_n"]) for layer in layers
        },
        "channel_filter": "Harmony analysis content only; prompt, control, and final tokens excluded",
        "activation_alignment": "hidden state at the current generated reasoning token",
        "capture_method": "generate with KV cache, then exact causal teacher-forced replay",
        "projection_definition": "a = h_layer_reasoning_token @ V[:, sv]",
        "candidate_naming": "LXX_SVYY: layer 0-based, SV rank 1-based",
        "primary_csv_sort": "rank_global_mean_abs_cosine",
        "selectivity_csv_sort": "rank_global_tail_selectivity",
        "unembedding_analysis": not args.skip_unembedding,
        "seed": args.seed,
        "versions": {"python": sys.version, "torch": torch.__version__},
    }
    try:
        import transformers

        metadata["versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    atomic_json_dump(metadata, metadata_path)

    print("\nDone.")
    print(f"  rankings: {rankings_path}")
    print(f"  selectivity rankings: {selectivity_path}")
    print(f"  contexts: {contexts_path}")
    print(f"  rollouts: {rollouts_path}")
    if unembedding_geometry:
        print(f"  unembedding: {unembedding_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
