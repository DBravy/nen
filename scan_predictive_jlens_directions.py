#!/usr/bin/env python3
"""
scan_predictive_jlens_directions.py

Build a database of *predictively* activated J-Lens right-singular directions
("unrealized word" candidates) in GPT-OSS-20B, in one self-contained script.

Predictive alignment
--------------------
For every fitted J-Lens source layer l:
    J_l = U_l diag(S_l) V_l^T

we take the top-k columns V_l[:, i]. Unlike the earlier scanners, which
projected the residual state captured during the forward pass of the target
token itself, this script projects the residual state at the position
*immediately before* each target token:

    a_l,i,t = <h_l,t-1, V_l[:, i]>        (attributed to the target token t)

In a causal model, h_l,t-1 is the state from which token t is predicted, so a
direction that is strongly active there is predictive of the target token
rather than merely coincident with it. Target tokens are non-special tokens at
positions t >= 1; the state that is read may sit on any token, including
special/marker tokens (that state is still the genuine prediction context).

All statistics, rankings, context records, and file formats match the earlier
scanners so runs remain directly comparable; only the state/token pairing
differs. Context records additionally carry "read_position" and "prev_token"
fields identifying the state that was actually projected. The context span and
the ⟦...⟧ mark still center on the target token.

Two data sources, one script
----------------------------
This file merges the three earlier scripts:

  * --mode corpus   (default) streams a text corpus and scans random windows,
                    exactly like scan_sparse_jlens_directions.py (which itself
                    superseded scan_unrealized_words.py).
  * --mode cot      generates Harmony responses with GPT-OSS, then
                    teacher-forces the exact sequence once and scans only the
                    generated ``analysis``-channel content tokens, exactly like
                    scan_cot_jlens_directions.py. Because the transformer is
                    causal and in eval mode, the replayed hidden state at each
                    position equals the autoregressive one (up to ordinary
                    kernel-level numerical variation).

Both modes feed the identical statistics/ranking machinery:
  * raw activation moments
  * residual-normalized activation moments (cosine-like usage)
  * top-1 / top-r recruitment frequencies within each layer's top-k bank
  * singular-value-weighted usage
  * strongest positive and negative token contexts for each direction
  * nearest and most anti-aligned vocabulary tokens in unembedding space
  * stable online variance/skew/kurtosis moments
  * a bounded systematic token sample for median/MAD, quantiles, and tail rates
  * every document's positive/negative peak for cross-document selectivity
  * positive/negative top-tail energy concentration and effective support

Outputs
-------
  OUT/directions/LXX.npz
  OUT/sv_rankings.csv                 (broad-usage ordering)
  OUT/selectivity_rankings.csv        (rare-but-sharp ordering)
  OUT/top_contexts.jsonl
  OUT/unembedding_neighbors.jsonl     (unless --skip-unembedding)
  OUT/rollouts.jsonl                  (cot mode only)
  OUT/metadata.json
  OUT/checkpoint.pkl

Examples
--------
Corpus scan (default fineweb sample-10BT, streamed):

    python scan_predictive_jlens_directions.py \
        --out predictive_words_fineweb \
        --k 64 \
        --n-docs 2000 \
        --max-seq-len 256

Chain-of-thought scan, L4-friendly low-effort first run:

    python scan_predictive_jlens_directions.py \
        --mode cot \
        --prompts-jsonl reasoning_prompts.jsonl \
        --reasoning-effort low \
        --max-new-tokens 512 \
        --k 64 \
        --out predictive_cot_low

Reuse a direction bank you already generated:

    python scan_predictive_jlens_directions.py \
        --directions-dir task_gaming_jlens/directions \
        --out predictive_words_fineweb \
        --k 64 \
        --n-docs 2000

Resume an interrupted scan (either mode):

    python scan_predictive_jlens_directions.py \
        --out predictive_words_fineweb \
        --k 64 \
        --n-docs 2000 \
        --resume

Layer numbers are 0-based. SV labels/ranks are 1-based, matching the earlier
scanners (for example L09_SV10).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_CONFIG = "sample-10BT"


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


# ----------------------------- CLI -----------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Scan J-Lens right-SV directions with predictive alignment: the state "
            "at position t-1 is projected and attributed to the target token at t."
        ),
    )

    p.add_argument(
        "--mode",
        choices=("corpus", "cot"),
        default="corpus",
        help=(
            "corpus: stream a text corpus and scan random windows. "
            "cot: generate Harmony responses and scan analysis-channel tokens."
        ),
    )

    # Model / lens
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    p.add_argument(
        "--layers",
        default="all",
        help="Comma-separated fitted source layers, e.g. 3,5,6,9,10,11,12; or all.",
    )
    p.add_argument("--k", type=int, default=64, help="Top singular directions per layer.")

    # Direction bank / SVD
    p.add_argument(
        "--directions-dir",
        type=str,
        default=None,
        help=(
            "Optional existing LXX.npz direction bank. Expected arrays S and V. "
            "If omitted, directions are computed from the J-Lens and saved under OUT/directions."
        ),
    )
    p.add_argument(
        "--exact-svd",
        action="store_true",
        help="Use exact torch.linalg.svd instead of randomized torch.svd_lowrank.",
    )
    p.add_argument("--svd-oversample", type=int, default=16)
    p.add_argument("--svd-niter", type=int, default=4)
    p.add_argument(
        "--recompute-directions",
        action="store_true",
        help="Ignore cached OUT/directions/LXX.npz files and recompute.",
    )

    # Corpus (corpus mode)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Optional local JSONL corpus; overrides --dataset/--dataset-config.",
    )
    p.add_argument("--n-docs", type=int, default=2000, help="Number of accepted windows to scan.")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument(
        "--sample-char-cap",
        type=int,
        default=30000,
        help="For very long docs, first choose a random character crop of this size before tokenizing.",
    )
    p.add_argument("--shuffle-buffer", type=int, default=10000)

    # Prompts / generation (cot mode)
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

    # Usage / context statistics
    p.add_argument(
        "--top-r",
        type=int,
        default=5,
        help="Count how often each SV is among the r largest |projections| at its layer.",
    )
    p.add_argument(
        "--top-contexts",
        type=int,
        default=64,
        help="Global positive and negative contexts retained per direction.",
    )
    p.add_argument(
        "--local-context-candidates",
        type=int,
        default=3,
        help="Per document/direction/polarity candidates offered to the global heap.",
    )
    p.add_argument(
        "--context-radius",
        type=int,
        default=20,
        help="Tokens on each side of the target token in stored contexts.",
    )
    p.add_argument(
        "--token-reservoir-size",
        type=int,
        default=32768,
        help=(
            "Approximate number of uniformly spaced target tokens retained per layer "
            "for robust quantiles and tail-rate estimates. This stores projections only, "
            "not hidden states. Use 16384 to reduce checkpoint size."
        ),
    )
    p.add_argument(
        "--min-tail-docs",
        type=int,
        default=5,
        help=(
            "Distinct z>5 documents needed for full support weight in the composite "
            "selectivity score; candidates below this are retained but downweighted."
        ),
    )

    # Unembedding geometry
    p.add_argument(
        "--unembedding-neighbors",
        type=int,
        default=32,
        help="Nearest and most anti-aligned vocabulary tokens retained per SV.",
    )
    p.add_argument(
        "--unembedding-chunk-size",
        type=int,
        default=4096,
        help="Vocabulary rows processed at once for SV-vs-unembedding cosine search.",
    )
    p.add_argument(
        "--skip-unembedding",
        action="store_true",
        help="Skip SV-to-unembedding cosine analysis.",
    )
    p.add_argument(
        "--unembedding-only",
        action="store_true",
        help=(
            "Compute unembedding neighbors from the saved/computed direction bank and exit "
            "without scanning. If OUT/sv_rankings.csv exists, augment it in place."
        ),
    )
    p.add_argument(
        "--include-special-unembedding-tokens",
        action="store_true",
        help="Allow tokenizer special tokens to appear among unembedding neighbors.",
    )

    # Run / checkpointing
    p.add_argument("--out", type=str, default="predictive_words_scan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Windows/jobs between checkpoints. Default: 100 in corpus mode, 5 in cot mode.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Windows/jobs between progress lines. Default: 10 in corpus mode, 1 in cot mode.",
    )

    args = p.parse_args()

    if args.checkpoint_every is None:
        args.checkpoint_every = 100 if args.mode == "corpus" else 5
    if args.progress_every is None:
        args.progress_every = 10 if args.mode == "corpus" else 1

    if args.k <= 0:
        p.error("--k must be positive")
    if args.top_r <= 0:
        p.error("--top-r must be positive")
    if args.top_contexts <= 0:
        p.error("--top-contexts must be positive")
    if args.local_context_candidates <= 0:
        p.error("--local-context-candidates must be positive")
    if args.context_radius <= 0:
        p.error("--context-radius must be positive")
    if args.token_reservoir_size <= 0:
        p.error("--token-reservoir-size must be positive")
    if args.min_tail_docs <= 0:
        p.error("--min-tail-docs must be positive")
    if args.unembedding_neighbors <= 0:
        p.error("--unembedding-neighbors must be positive")
    if args.unembedding_chunk_size <= 0:
        p.error("--unembedding-chunk-size must be positive")
    if args.unembedding_only and args.skip_unembedding:
        p.error("--unembedding-only and --skip-unembedding cannot be used together")

    if args.mode == "corpus":
        if args.n_docs <= 0:
            p.error("--n-docs must be positive")
        if args.max_seq_len < 8:
            p.error("--max-seq-len is too small")
        if args.min_tokens <= 0:
            p.error("--min-tokens must be positive")
    else:
        for name in (
            "samples_per_prompt",
            "max_new_tokens",
            "max_input_tokens",
            "max_replay_tokens",
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


# ----------------------------- Helpers -------------------------------------


def parse_layers(spec: str, available: list[int]) -> list[int]:
    if spec.strip().lower() == "all":
        return list(available)
    wanted = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(
            f"Requested layers {missing} are not fitted J-Lens source layers. "
            f"Available: {available}"
        )
    return wanted


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


def jsonable_source(item: dict[str, Any]) -> dict[str, Any]:
    """Keep a few useful source fields without copying whole dataset records."""
    out: dict[str, Any] = {}
    for key in ("id", "url", "dump", "date", "file_path"):
        val = item.get(key)
        if val is not None and isinstance(val, (str, int, float, bool)):
            out[key] = val
    return out


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return f" | cuda alloc={allocated:.1f}G reserved={reserved:.1f}G peak={peak:.1f}G"


# ------------------------- J-Lens directions -------------------------------


def load_npz_direction_file(path: Path, d_model: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if "V" not in z or "S" not in z:
        raise ValueError(f"{path} must contain arrays named V and S")

    V = np.asarray(z["V"])
    S = np.asarray(z["S"]).reshape(-1)

    # Existing scripts use V[:, i], but accept Vh-style storage too.
    if V.ndim != 2:
        raise ValueError(f"{path}: V must be 2D, got {V.shape}")
    if V.shape[0] == d_model:
        pass
    elif V.shape[1] == d_model:
        V = V.T
    else:
        raise ValueError(
            f"{path}: neither axis of V matches d_model={d_model}; shape={V.shape}"
        )

    if V.shape[1] < k or S.shape[0] < k:
        raise ValueError(
            f"{path}: requested k={k}, but V has {V.shape[1]} directions and S has {S.shape[0]} values"
        )

    return V[:, :k].astype(np.float32, copy=False), S[:k].astype(np.float32, copy=False)


def canonicalize_svd_signs(U: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the arbitrary SVD sign so the largest-|coordinate| of each v is positive."""
    U = U.clone()
    V = V.clone()
    for i in range(V.shape[1]):
        anchor = int(torch.argmax(torch.abs(V[:, i])).item())
        if V[anchor, i] < 0:
            V[:, i].mul_(-1)
            U[:, i].mul_(-1)
    return U, V


def compute_top_svd(
    J: torch.Tensor,
    k: int,
    exact: bool,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    J = J.float().cpu()
    d = min(J.shape)
    if k > d:
        raise ValueError(f"k={k} exceeds matrix dimension {d}")

    if exact:
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        V = Vh[:k, :].T.contiguous()
    else:
        q = min(d, k + max(0, oversample))
        U, S, V = torch.svd_lowrank(J, q=q, niter=niter)
        order = torch.argsort(S, descending=True)
        U = U[:, order][:, :k]
        S = S[order][:k]
        V = V[:, order][:, :k]

    U, V = canonicalize_svd_signs(U, V)
    return U.contiguous(), S.contiguous(), V.contiguous()


def prepare_direction_bank(
    args: argparse.Namespace,
    lens: Any,
    layers: list[int],
    out_dir: Path,
) -> tuple[dict[int, torch.Tensor], dict[int, np.ndarray], str]:
    """Return CPU V[layer] = [d_model, k], S[layer] = [k]."""
    out_direction_dir = out_dir / "directions"
    out_direction_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.directions_dir) if args.directions_dir else out_direction_dir
    V_bank: dict[int, torch.Tensor] = {}
    S_bank: dict[int, np.ndarray] = {}

    if args.directions_dir:
        direction_source = str(source_dir)
    else:
        direction_source = "computed_from_jlens"

    for layer in layers:
        source_path = source_dir / f"L{layer:02d}.npz"
        cached_out = out_direction_dir / f"L{layer:02d}.npz"

        # Explicit external direction bank always wins.
        if args.directions_dir:
            if not source_path.exists():
                raise FileNotFoundError(f"Missing direction file: {source_path}")
            V_np, S_np = load_npz_direction_file(source_path, lens.d_model, args.k)
            V_bank[layer] = torch.from_numpy(np.array(V_np, copy=True))
            S_bank[layer] = np.array(S_np, copy=True)
            # Save a compact copy into this scan for provenance/reproducibility.
            np.savez_compressed(cached_out, V=V_np, S=S_np)
            print(f"[directions] L{layer:02d}: loaded {source_path}")
            continue

        # Resume/re-run can reuse directions already computed in OUT.
        if cached_out.exists() and not args.recompute_directions:
            try:
                V_np, S_np = load_npz_direction_file(cached_out, lens.d_model, args.k)
            except ValueError as exc:
                print(f"[directions] L{layer:02d}: cached bank unusable ({exc}); recomputing")
            else:
                V_bank[layer] = torch.from_numpy(np.array(V_np, copy=True))
                S_bank[layer] = np.array(S_np, copy=True)
                print(f"[directions] L{layer:02d}: reused {cached_out}")
                continue

        print(
            f"[directions] L{layer:02d}: computing top-{args.k} "
            f"{'exact' if args.exact_svd else 'randomized'} SVD"
        )
        J = lens.jacobians[layer]
        U, S, V = compute_top_svd(
            J,
            k=args.k,
            exact=args.exact_svd,
            oversample=args.svd_oversample,
            niter=args.svd_niter,
        )
        U_np = U.numpy().astype(np.float32, copy=False)
        S_np = S.numpy().astype(np.float32, copy=False)
        V_np = V.numpy().astype(np.float32, copy=False)
        np.savez_compressed(cached_out, U=U_np, S=S_np, V=V_np)
        V_bank[layer] = V.cpu()
        S_bank[layer] = S_np

    return V_bank, S_bank, direction_source


# --------------------------- Corpus (corpus mode) ---------------------------


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                yield {"text": obj}
            elif isinstance(obj, dict):
                yield obj
            else:
                raise ValueError(f"Expected JSON object/string, got {type(obj).__name__}")


def build_corpus(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_jsonl:
        return iter_jsonl(Path(args.input_jsonl))

    from datasets import load_dataset

    print(
        f"[corpus] streaming {args.dataset} / {args.dataset_config} / {args.split}"
    )
    ds = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return ds


def sample_token_window(
    tokenizer: Any,
    text: str,
    rng: random.Random,
    max_seq_len: int,
    min_tokens: int,
    char_cap: int,
) -> tuple[torch.Tensor, dict[str, int]] | None:
    """
    Choose one random window per document so we do not systematically scan only
    document beginnings. For very long documents, choose a random character crop
    before tokenization to keep tokenizer work bounded.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    char_start = 0
    cropped = text
    if char_cap > 0 and len(cropped) > char_cap:
        char_start = rng.randint(0, len(cropped) - char_cap)
        cropped = cropped[char_start : char_start + char_cap]

    ids = tokenizer.encode(cropped, add_special_tokens=False)
    if len(ids) < min_tokens:
        return None

    bos_id = tokenizer.bos_token_id
    payload = max_seq_len - (1 if bos_id is not None else 0)
    payload = max(1, payload)

    token_start = 0
    if len(ids) > payload:
        token_start = rng.randint(0, len(ids) - payload)
        ids = ids[token_start : token_start + payload]

    if bos_id is not None:
        ids = [int(bos_id)] + [int(x) for x in ids]
    else:
        ids = [int(x) for x in ids]

    return torch.tensor([ids], dtype=torch.long), {
        "char_crop_start": char_start,
        "token_window_start_in_crop": token_start,
    }


# ------------------- Prompts and generation (cot mode) ----------------------


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


# --------------------------- Statistics ------------------------------------


def init_stats(
    layers: list[int],
    k: int,
    n_docs: int,
    token_sample_capacity: int,
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for layer in layers:
        stats[layer] = {
            "n": 0,
            "n_docs": 0,
            "sum": np.zeros(k, dtype=np.float64),
            "abs_sum": np.zeros(k, dtype=np.float64),
            "sq_sum": np.zeros(k, dtype=np.float64),
            # Numerically stable mergeable central moments. Mj is the sum of
            # (x - running_mean)^j, not the normalized moment.
            "moment_mean": np.zeros(k, dtype=np.float64),
            "M2": np.zeros(k, dtype=np.float64),
            "M3": np.zeros(k, dtype=np.float64),
            "M4": np.zeros(k, dtype=np.float64),
            "cos_abs_sum": np.zeros(k, dtype=np.float64),
            "cos_sq_sum": np.zeros(k, dtype=np.float64),
            "positive_count": np.zeros(k, dtype=np.int64),
            "top1_count": np.zeros(k, dtype=np.int64),
            "topr_count": np.zeros(k, dtype=np.int64),
            "doc_topr_count": np.zeros(k, dtype=np.int64),
            "doc_peak_abs_sum": np.zeros(k, dtype=np.float64),
            "max": np.full(k, -np.inf, dtype=np.float64),
            "min": np.full(k, np.inf, dtype=np.float64),
            "residual_norm_sum": 0.0,
            # Exact per-document extrema. Rows beyond n_docs processed remain
            # unused and are ignored when outputs are built.
            "doc_max": np.full((n_docs, k), np.nan, dtype=np.float32),
            "doc_min": np.full((n_docs, k), np.nan, dtype=np.float32),
            # Bounded cross-corpus token sample used for robust statistics.
            "token_sample": np.empty(
                (token_sample_capacity, k), dtype=np.float32
            ),
            "token_sample_n": 0,
        }
    return stats


def merge_batch_central_moments(st: dict[str, Any], proj: torch.Tensor) -> None:
    """Merge one [T,k] batch into stable running central moments (Pébay)."""
    n1 = int(st["n"])
    n2 = int(proj.shape[0])
    if n2 == 0:
        return

    # Center before powers to avoid the cancellation of raw third/fourth sums.
    batch_mean_t = proj.mean(dim=0)
    centered = proj - batch_mean_t
    centered2 = centered.square()
    mean2 = batch_mean_t.cpu().double().numpy()
    bM2 = centered2.sum(0).cpu().double().numpy()
    bM3 = (centered2 * centered).sum(0).cpu().double().numpy()
    bM4 = centered2.square().sum(0).cpu().double().numpy()

    if n1 == 0:
        st["moment_mean"] = mean2
        st["M2"] = bM2
        st["M3"] = bM3
        st["M4"] = bM4
        return

    mean1 = st["moment_mean"]
    M2_1 = st["M2"]
    M3_1 = st["M3"]
    M4_1 = st["M4"]
    n = n1 + n2
    delta = mean2 - mean1
    delta2 = delta * delta
    delta3 = delta2 * delta
    delta4 = delta2 * delta2

    st["moment_mean"] = mean1 + delta * (n2 / n)
    st["M2"] = M2_1 + bM2 + delta2 * n1 * n2 / n
    st["M3"] = (
        M3_1
        + bM3
        + delta3 * n1 * n2 * (n1 - n2) / (n * n)
        + 3.0 * delta * (n1 * bM2 - n2 * M2_1) / n
    )
    st["M4"] = (
        M4_1
        + bM4
        + delta4
        * n1
        * n2
        * (n1 * n1 - n1 * n2 + n2 * n2)
        / (n * n * n)
        + 6.0 * delta2 * (n1 * n1 * bM2 + n2 * n2 * M2_1) / (n * n)
        + 4.0 * delta * (n1 * bM3 - n2 * M3_1) / n
    )


def update_stats(
    st: dict[str, Any],
    proj: torch.Tensor,
    residual_norm: torch.Tensor,
    top_r: int,
    doc_index: int,
    token_sample_rows: torch.Tensor,
) -> None:
    """proj [T,k] and residual_norm [T], both fp32 on the layer device.

    Rows are aligned to target tokens; the projected state is the residual
    stream at each target's immediately preceding position.
    """
    T, k = proj.shape
    if T == 0:
        return

    abs_proj = proj.abs()
    cos = proj / residual_norm.clamp_min(1e-12).unsqueeze(1)

    merge_batch_central_moments(st, proj)

    st["doc_max"][doc_index] = proj.max(0).values.cpu().float().numpy()
    st["doc_min"][doc_index] = proj.min(0).values.cpu().float().numpy()
    if token_sample_rows.numel() > 0:
        sampled = proj.index_select(0, token_sample_rows).cpu().float().numpy()
        start = int(st["token_sample_n"])
        stop = start + sampled.shape[0]
        if stop > st["token_sample"].shape[0]:
            raise RuntimeError(
                "Internal token-sample capacity exceeded; this indicates a sampling bug"
            )
        st["token_sample"][start:stop] = sampled
        st["token_sample_n"] = stop

    st["n"] += int(T)
    st["n_docs"] += 1
    st["sum"] += proj.sum(0).cpu().double().numpy()
    st["abs_sum"] += abs_proj.sum(0).cpu().double().numpy()
    st["sq_sum"] += proj.square().sum(0).cpu().double().numpy()
    st["cos_abs_sum"] += cos.abs().sum(0).cpu().double().numpy()
    st["cos_sq_sum"] += cos.square().sum(0).cpu().double().numpy()
    st["positive_count"] += (proj > 0).sum(0).cpu().numpy().astype(np.int64)
    st["max"] = np.maximum(st["max"], proj.max(0).values.cpu().double().numpy())
    st["min"] = np.minimum(st["min"], proj.min(0).values.cpu().double().numpy())
    st["doc_peak_abs_sum"] += abs_proj.max(0).values.cpu().double().numpy()
    st["residual_norm_sum"] += float(residual_norm.sum().item())

    top1 = torch.argmax(abs_proj, dim=1)
    st["top1_count"] += torch.bincount(top1, minlength=k).cpu().numpy().astype(np.int64)

    r = min(max(1, top_r), k)
    top_idx = torch.topk(abs_proj, k=r, dim=1, largest=True, sorted=False).indices
    st["topr_count"] += (
        torch.bincount(top_idx.reshape(-1), minlength=k).cpu().numpy().astype(np.int64)
    )

    present = torch.zeros(k, dtype=torch.bool, device=proj.device)
    present[top_idx.reshape(-1).unique()] = True
    st["doc_topr_count"] += present.cpu().numpy().astype(np.int64)


# Heap entries are (strength, unique_counter, record).
# strength is positive for both positive and negative polarity heaps; for the
# negative side it is -raw_activation, so larger always means more extreme.


def heap_push(
    heap: list[tuple[float, int, dict[str, Any]]],
    strength: float,
    counter: int,
    record: dict[str, Any],
    keep: int,
) -> None:
    entry = (float(strength), int(counter), record)
    if len(heap) < keep:
        heapq.heappush(heap, entry)
    elif strength > heap[0][0]:
        heapq.heapreplace(heap, entry)


def collect_context_candidates(
    *,
    layer: int,
    proj: torch.Tensor,
    cos: torch.Tensor,
    target_positions: torch.Tensor,
    input_ids_cpu: list[int],
    source: dict[str, Any],
    doc_index: int,
    window_meta: dict[str, int],
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
    counter: int,
    top_contexts: int,
    local_candidates: int,
    context_radius: int,
) -> int:
    """Rows of proj/cos correspond one-to-one with target_positions.

    The stored context span and center mark the *target* token; the recorded
    activation was read from the residual state one position earlier, which is
    identified by read_position / prev_token_id.
    """
    T, k = proj.shape
    if T == 0:
        return counter

    local_n = min(local_candidates, T)
    pos_vals, pos_rows = torch.topk(proj, k=local_n, dim=0, largest=True, sorted=False)
    neg_strength, neg_rows = torch.topk(-proj, k=local_n, dim=0, largest=True, sorted=False)

    pos_vals = pos_vals.cpu()
    pos_rows = pos_rows.cpu()
    neg_strength = neg_strength.cpu()
    neg_rows = neg_rows.cpu()
    cos_cpu = cos.cpu()
    target_positions_cpu = target_positions.cpu()

    def make_record(row: int, sv0: int, activation: float) -> dict[str, Any]:
        absolute_pos = int(target_positions_cpu[row].item())
        read_pos = absolute_pos - 1
        left = max(0, absolute_pos - context_radius)
        right = min(len(input_ids_cpu), absolute_pos + context_radius + 1)
        span = tuple(int(x) for x in input_ids_cpu[left:right])
        return {
            "layer": int(layer),
            "sv_index_0": int(sv0),
            "sv_rank_1based": int(sv0 + 1),
            "activation": float(activation),
            "cosine_activation": float(cos_cpu[row, sv0].item()),
            "document_index": int(doc_index),
            "token_position": int(absolute_pos),
            "read_position": int(read_pos),
            "prev_token_id": int(input_ids_cpu[read_pos]),
            "span_token_ids": span,
            "center_offset": int(absolute_pos - left),
            "source": source,
            "window_meta": window_meta,
        }

    for sv0 in range(k):
        pos_heap = heaps.setdefault((layer, sv0, "positive"), [])
        neg_heap = heaps.setdefault((layer, sv0, "negative"), [])

        for q in range(local_n):
            row = int(pos_rows[q, sv0].item())
            activation = float(pos_vals[q, sv0].item())
            rec = make_record(row, sv0, activation)
            heap_push(pos_heap, activation, counter, rec, top_contexts)
            counter += 1

            row = int(neg_rows[q, sv0].item())
            strength = float(neg_strength[q, sv0].item())
            activation = -strength
            rec = make_record(row, sv0, activation)
            heap_push(neg_heap, strength, counter, rec, top_contexts)
            counter += 1

    return counter


# ---------------------- Shared per-window predictive scan -------------------


def predictive_target_positions(
    input_ids_cpu: list[int],
    special_ids: set[int],
) -> list[int]:
    """Non-special target tokens at t >= 1; position t-1 supplies the state."""
    return [
        t
        for t, token_id in enumerate(input_ids_cpu)
        if t >= 1 and token_id not in special_ids
    ]


def scan_targets_over_layers(
    *,
    recorder: Any,
    layers: list[int],
    target_positions_list: list[int],
    input_ids_cpu: list[int],
    V_bank: dict[int, torch.Tensor],
    V_device_cache: dict[tuple[int, str, torch.dtype], torch.Tensor],
    stats: dict[int, dict[str, Any]],
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
    heap_counter: int,
    args: argparse.Namespace,
    doc_index: int,
    tokens_before: int,
    token_sample_stride: int,
    token_sample_offset: int,
    source: dict[str, Any],
    window_meta: dict[str, int],
) -> int:
    """Update statistics and context heaps for one recorded window.

    For each target token at absolute position t, the projected state is the
    layer residual stream at position t-1 (the state from which t is
    predicted). Rows of every derived tensor stay aligned to targets, so all
    downstream statistics attribute predictive activation to the target token.
    """
    for layer in layers:
        hidden_full = recorder.activations[layer][0].detach()  # [seq, d_model]
        target_positions = torch.tensor(
            target_positions_list,
            dtype=torch.long,
            device=hidden_full.device,
        )
        read_positions = target_positions - 1
        hidden = hidden_full.index_select(0, read_positions)

        cache_key = (layer, str(hidden.device), hidden.dtype)
        V = V_device_cache.get(cache_key)
        if V is None:
            V = V_bank[layer].to(device=hidden.device, dtype=hidden.dtype)
            V_device_cache[cache_key] = V

        # h_{t-1} @ V gives predictive projections onto the orthonormal
        # right-SV directions, one row per target token t.
        proj = (hidden @ V).float()  # [T, k]
        residual_norm = hidden.float().norm(dim=-1)
        cos = proj / residual_norm.clamp_min(1e-12).unsqueeze(1)

        first_sample_row = (
            token_sample_offset - tokens_before
        ) % token_sample_stride
        token_sample_rows = torch.arange(
            first_sample_row,
            proj.shape[0],
            token_sample_stride,
            dtype=torch.long,
            device=proj.device,
        )

        update_stats(
            stats[layer],
            proj,
            residual_norm,
            args.top_r,
            doc_index,
            token_sample_rows,
        )
        heap_counter = collect_context_candidates(
            layer=layer,
            proj=proj,
            cos=cos,
            target_positions=target_positions,
            input_ids_cpu=input_ids_cpu,
            source=source,
            doc_index=doc_index,
            window_meta=window_meta,
            heaps=heaps,
            counter=heap_counter,
            top_contexts=args.top_contexts,
            local_candidates=args.local_context_candidates,
            context_radius=args.context_radius,
        )

        del proj, cos, residual_norm, hidden, target_positions, read_positions
        del token_sample_rows

    return heap_counter


# ------------------------- Unembedding geometry ----------------------------


def token_record(tokenizer: Any, token_id: int, cosine: float) -> dict[str, Any]:
    """Human-readable representations for a single vocabulary row."""
    try:
        token_string = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token_string = None
    try:
        decoded = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded = None
    return {
        "token_id": int(token_id),
        "token": token_string,
        "decoded": decoded,
        "cosine": float(cosine),
        "is_special": int(token_id) in set(int(x) for x in tokenizer.all_special_ids),
    }


def analyze_unembedding_geometry(
    *,
    hf_model: Any,
    tokenizer: Any,
    V_bank: dict[int, torch.Tensor],
    layers: list[int],
    k: int,
    n_neighbors: int,
    chunk_size: int,
    include_special_tokens: bool,
) -> dict[tuple[int, int], dict[str, Any]]:
    """
    Compare every unit right-SV direction directly with every row of the model's
    output embedding / lm_head matrix using cosine similarity.

    This is deliberately the raw unembedding geometry: if W is [vocab,d_model],
        cosine(token, v) = <W[token], v> / ||W[token]||
    because v already has unit norm.

    The scan is chunked over vocabulary rows, so it does not create a normalized
    copy of the full unembedding matrix.
    """
    output_embeddings = hf_model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose get_output_embeddings().weight")
    W = output_embeddings.weight
    if W.ndim != 2:
        raise ValueError(f"Expected 2D unembedding weight, got shape={tuple(W.shape)}")

    vocab_size, d_model = int(W.shape[0]), int(W.shape[1])
    expected_d = int(next(iter(V_bank.values())).shape[0])
    if d_model != expected_d:
        raise ValueError(
            f"Unembedding d_model={d_model} does not match direction d_model={expected_d}"
        )

    # Concatenate all candidate directions once: [d_model, C].
    candidate_keys: list[tuple[int, int]] = []
    candidate_cols: list[torch.Tensor] = []
    for layer in layers:
        V = V_bank[layer][:, :k].float()
        V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
        for sv0 in range(k):
            candidate_keys.append((layer, sv0))
        candidate_cols.append(V)
    V_all_cpu = torch.cat(candidate_cols, dim=1).contiguous()
    C = int(V_all_cpu.shape[1])

    keep = min(int(n_neighbors), vocab_size)
    top_values = torch.full((keep, C), -float("inf"), dtype=torch.float32)
    top_ids = torch.full((keep, C), -1, dtype=torch.long)
    bottom_values = torch.full((keep, C), float("inf"), dtype=torch.float32)
    bottom_ids = torch.full((keep, C), -1, dtype=torch.long)
    cos_sum = torch.zeros(C, dtype=torch.float64)
    cos_sq_sum = torch.zeros(C, dtype=torch.float64)
    valid_count = 0

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    device = W.device
    V_all = V_all_cpu.to(device=device, dtype=torch.float32)

    print(
        f"[unembedding] vocab={vocab_size:,} d_model={d_model} candidates={C:,} "
        f"device={device} neighbors={keep}"
    )

    with torch.inference_mode():
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            rows = W[start:end].detach().to(device=device, dtype=torch.float32)
            row_norm = rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
            sims = (rows @ V_all) / row_norm  # [chunk, C]

            if not include_special_tokens and special_ids:
                local_special = [
                    tok_id - start
                    for tok_id in special_ids
                    if start <= tok_id < end
                ]
                if local_special:
                    idx = torch.tensor(local_special, device=device, dtype=torch.long)
                    sims.index_fill_(0, idx, float("nan"))

            finite = torch.isfinite(sims)
            safe = torch.where(finite, sims, torch.zeros_like(sims))
            cos_sum += safe.sum(dim=0).cpu().double()
            cos_sq_sum += safe.square().sum(dim=0).cpu().double()
            if C > 0:
                valid_count += int(finite[:, 0].sum().item())

            # Merge this chunk's extrema with the running top/bottom K.
            top_input = torch.where(finite, sims, torch.full_like(sims, -float("inf")))
            bottom_input = torch.where(finite, sims, torch.full_like(sims, float("inf")))
            local_k = min(keep, end - start)

            chunk_top_v, chunk_top_i = torch.topk(
                top_input, k=local_k, dim=0, largest=True, sorted=False
            )
            chunk_bottom_v, chunk_bottom_i = torch.topk(
                bottom_input, k=local_k, dim=0, largest=False, sorted=False
            )
            chunk_top_i = chunk_top_i + start
            chunk_bottom_i = chunk_bottom_i + start

            merged_v = torch.cat([top_values.to(device), chunk_top_v], dim=0)
            merged_i = torch.cat([top_ids.to(device), chunk_top_i], dim=0)
            sel_v, sel_pos = torch.topk(
                merged_v, k=keep, dim=0, largest=True, sorted=False
            )
            sel_i = torch.gather(merged_i, 0, sel_pos)
            top_values, top_ids = sel_v.cpu(), sel_i.cpu()

            merged_v = torch.cat([bottom_values.to(device), chunk_bottom_v], dim=0)
            merged_i = torch.cat([bottom_ids.to(device), chunk_bottom_i], dim=0)
            sel_v, sel_pos = torch.topk(
                merged_v, k=keep, dim=0, largest=False, sorted=False
            )
            sel_i = torch.gather(merged_i, 0, sel_pos)
            bottom_values, bottom_ids = sel_v.cpu(), sel_i.cpu()

            del rows, row_norm, sims, finite, safe, top_input, bottom_input

    # Sort retained extrema exactly for readable output.
    top_order = torch.argsort(top_values, dim=0, descending=True)
    bottom_order = torch.argsort(bottom_values, dim=0, descending=False)
    top_values = torch.gather(top_values, 0, top_order)
    top_ids = torch.gather(top_ids, 0, top_order)
    bottom_values = torch.gather(bottom_values, 0, bottom_order)
    bottom_ids = torch.gather(bottom_ids, 0, bottom_order)

    denom = max(1, valid_count)
    mean = cos_sum.numpy() / denom
    var = np.maximum(cos_sq_sum.numpy() / denom - mean**2, 0.0)
    std = np.sqrt(var)

    results: dict[tuple[int, int], dict[str, Any]] = {}
    for col, key in enumerate(candidate_keys):
        nearest = [
            token_record(tokenizer, int(top_ids[r, col]), float(top_values[r, col]))
            for r in range(keep)
            if int(top_ids[r, col]) >= 0 and math.isfinite(float(top_values[r, col]))
        ]
        farthest = [
            token_record(tokenizer, int(bottom_ids[r, col]), float(bottom_values[r, col]))
            for r in range(keep)
            if int(bottom_ids[r, col]) >= 0 and math.isfinite(float(bottom_values[r, col]))
        ]
        top1 = nearest[0] if nearest else None
        bottom1 = farthest[0] if farthest else None
        top2_cos = nearest[1]["cosine"] if len(nearest) > 1 else float("nan")
        bottom2_cos = farthest[1]["cosine"] if len(farthest) > 1 else float("nan")

        results[key] = {
            "nearest_tokens": nearest,
            "farthest_tokens": farthest,
            "unembedding_vocab_rows_considered": int(valid_count),
            "unembedding_cosine_mean": float(mean[col]),
            "unembedding_cosine_std": float(std[col]),
            "nearest_token": top1,
            "farthest_token": bottom1,
            "nearest_cosine_margin": (
                float(top1["cosine"] - top2_cos)
                if top1 is not None and math.isfinite(top2_cos)
                else float("nan")
            ),
            "farthest_cosine_margin": (
                float(bottom2_cos - bottom1["cosine"])
                if bottom1 is not None and math.isfinite(bottom2_cos)
                else float("nan")
            ),
            "max_abs_token_cosine": float(
                max(
                    abs(top1["cosine"]) if top1 is not None else 0.0,
                    abs(bottom1["cosine"]) if bottom1 is not None else 0.0,
                )
            ),
            "nearest_token_z": float(
                (top1["cosine"] - mean[col]) / max(std[col], 1e-12)
            ) if top1 is not None else float("nan"),
            "farthest_token_z": float(
                (bottom1["cosine"] - mean[col]) / max(std[col], 1e-12)
            ) if bottom1 is not None else float("nan"),
        }

    del V_all
    return results


def write_unembedding_neighbors(
    path: Path,
    geometry: dict[tuple[int, int], dict[str, Any]],
    S_bank: dict[int, np.ndarray],
    layers: list[int],
    k: int,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for layer in layers:
            for sv0 in range(k):
                g = geometry[(layer, sv0)]
                rec = {
                    "candidate": f"L{layer:02d}_SV{sv0 + 1:02d}",
                    "layer": int(layer),
                    "sv_index_0": int(sv0),
                    "sv_rank_1based": int(sv0 + 1),
                    "singular_value": float(S_bank[layer][sv0]),
                    "space": "raw_unembedding_row_cosine",
                    **g,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def add_unembedding_columns(
    rows: list[dict[str, Any]],
    geometry: dict[tuple[int, int], dict[str, Any]],
) -> None:
    """Add compact geometry columns to the main rankings CSV in place."""
    for row in rows:
        key = (int(row["layer"]), int(row["sv_index_0"]))
        g = geometry.get(key)
        if g is None:
            continue
        near = g["nearest_token"]
        far = g["farthest_token"]
        row.update(
            {
                "nearest_unembed_token_id": None if near is None else near["token_id"],
                "nearest_unembed_token": None if near is None else near["token"],
                "nearest_unembed_decoded": None if near is None else near["decoded"],
                "nearest_unembed_cosine": None if near is None else near["cosine"],
                "nearest_unembed_margin": g["nearest_cosine_margin"],
                "farthest_unembed_token_id": None if far is None else far["token_id"],
                "farthest_unembed_token": None if far is None else far["token"],
                "farthest_unembed_decoded": None if far is None else far["decoded"],
                "farthest_unembed_cosine": None if far is None else far["cosine"],
                "farthest_unembed_margin": g["farthest_cosine_margin"],
                "max_abs_unembed_token_cosine": g["max_abs_token_cosine"],
                "nearest_unembed_token_z": g["nearest_token_z"],
                "farthest_unembed_token_z": g["farthest_token_z"],
                "unembedding_cosine_mean": g["unembedding_cosine_mean"],
                "unembedding_cosine_std": g["unembedding_cosine_std"],
            }
        )

    # Rank lexical anchoring after every row has been augmented.
    augmented = [
        i for i, row in enumerate(rows) if "max_abs_unembed_token_cosine" in row
    ]
    augmented.sort(
        key=lambda i: rows[i]["max_abs_unembed_token_cosine"], reverse=True
    )
    for rank, i in enumerate(augmented, 1):
        rows[i]["rank_global_max_abs_unembed_token_cosine"] = rank


# ------------------------- Checkpoint / signature ---------------------------


def run_signature(
    args: argparse.Namespace,
    layers: list[int],
    jobs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "script": "scan_predictive_jlens_directions",
        "alignment": "predictive_prev_token",
        "mode": args.mode,
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": list(layers),
        "k": int(args.k),
        "directions_dir": args.directions_dir,
        "top_r": int(args.top_r),
        "top_contexts": int(args.top_contexts),
        "local_context_candidates": int(args.local_context_candidates),
        "context_radius": int(args.context_radius),
        "token_reservoir_size": int(args.token_reservoir_size),
        "min_tail_docs": int(args.min_tail_docs),
        "seed": int(args.seed),
    }
    if args.mode == "corpus":
        signature.update(
            {
                "n_docs": int(args.n_docs),
                "dataset": None if args.input_jsonl else args.dataset,
                "dataset_config": None if args.input_jsonl else args.dataset_config,
                "split": None if args.input_jsonl else args.split,
                "input_jsonl": args.input_jsonl,
                "text_field": args.text_field,
                "max_seq_len": int(args.max_seq_len),
                "min_tokens": int(args.min_tokens),
                "sample_char_cap": int(args.sample_char_cap),
                "shuffle_buffer": int(args.shuffle_buffer),
            }
        )
    else:
        prompt_payload = json.dumps(
            [
                {"messages": j["messages"], "sample_index": j["sample_index"]}
                for j in (jobs or [])
            ],
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        signature.update(
            {
                "reasoning_effort": args.reasoning_effort,
                "max_new_tokens": int(args.max_new_tokens),
                "max_input_tokens": int(args.max_input_tokens),
                "max_replay_tokens": int(args.max_replay_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "greedy": bool(args.greedy),
                "jobs_sha256": hashlib.sha256(prompt_payload).hexdigest(),
                "n_jobs": len(jobs or []),
            }
        )
    return signature


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    atomic_pickle_dump(payload, path)


def load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("signature") != signature:
        raise ValueError(
            "Checkpoint configuration does not match this run.\n"
            f"checkpoint: {state.get('signature')}\n"
            f"current:    {signature}"
        )
    return state


# ------------------------- Tail statistics / rankings -----------------------


def column_quantile(x: np.ndarray, q: float) -> np.ndarray:
    if x.shape[0] == 0:
        return np.full(x.shape[1], np.nan, dtype=np.float64)
    return np.quantile(x, q, axis=0)


def top_energy_share(magnitude: np.ndarray, fraction: float) -> np.ndarray:
    """Share of per-column squared energy held by the largest fraction of rows."""
    n, k = magnitude.shape
    if n == 0:
        return np.full(k, np.nan, dtype=np.float64)
    keep = min(n, max(1, int(math.ceil(fraction * n))))
    energy = np.square(magnitude, dtype=np.float64)
    cutoff = n - keep
    top = np.partition(energy, cutoff, axis=0)[cutoff:].sum(axis=0)
    total = energy.sum(axis=0)
    return np.divide(top, total, out=np.zeros_like(top), where=total > 0)


def robust_tail_statistics(
    st: dict[str, Any],
    min_tail_docs: int,
) -> dict[str, np.ndarray]:
    """Derive scale-free token and document tail statistics for one layer."""
    sample_n = int(st["token_sample_n"])
    sample = st["token_sample"][:sample_n].astype(np.float64, copy=False)
    k = st["token_sample"].shape[1]
    nd = int(st["n_docs"])
    n = max(1, int(st["n"]))

    if sample_n == 0:
        # Keep the output schema valid for tiny diagnostic runs whose random
        # systematic offset happens to fall beyond all processed tokens.
        sample = st["moment_mean"].reshape(1, k).astype(np.float64, copy=False)

    median = np.median(sample, axis=0)
    mad = np.median(np.abs(sample - median), axis=0)
    robust_scale = 1.4826 * mad
    ordinary_std = np.sqrt(np.maximum(st["M2"] / n, 0.0))
    robust_scale = np.where(
        robust_scale > 1e-12,
        robust_scale,
        np.where(ordinary_std > 1e-12, ordinary_std, 1.0),
    )

    z = (sample - median) / robust_scale
    positive = np.maximum(z, 0.0)
    negative = np.maximum(-z, 0.0)
    abs_z = np.abs(z)

    doc_max = st["doc_max"][:nd].astype(np.float64, copy=False)
    doc_min = st["doc_min"][:nd].astype(np.float64, copy=False)
    doc_pos_z = (doc_max - median) / robust_scale
    doc_neg_z = (median - doc_min) / robust_scale

    M2 = np.maximum(st["M2"], 0.0)
    M3 = st["M3"]
    M4 = np.maximum(st["M4"], 0.0)
    variance = M2 / n
    skewness = np.divide(
        math.sqrt(n) * M3,
        np.power(M2, 1.5),
        out=np.zeros_like(M3),
        where=M2 > 1e-24,
    )
    kurtosis = np.divide(
        n * M4,
        np.square(M2),
        out=np.zeros_like(M4),
        where=M2 > 1e-24,
    )
    excess_kurtosis = kurtosis - 3.0
    effective_support_fraction = np.divide(
        np.square(M2),
        n * M4,
        out=np.ones_like(M4),
        where=M4 > 1e-24,
    )

    out: dict[str, np.ndarray] = {
        "sample_n": np.full(k, sample_n, dtype=np.float64),
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "stable_variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": excess_kurtosis,
        "effective_support_fraction": effective_support_fraction,
        "positive_max_z": (st["max"] - median) / robust_scale,
        "negative_max_z": (median - st["min"]) / robust_scale,
        "positive_q99_z": column_quantile(z, 0.99),
        "positive_q999_z": column_quantile(z, 0.999),
        "positive_q9999_z": column_quantile(z, 0.9999),
        "negative_q99_z": column_quantile(-z, 0.99),
        "negative_q999_z": column_quantile(-z, 0.999),
        "negative_q9999_z": column_quantile(-z, 0.9999),
        "abs_q99_z": column_quantile(abs_z, 0.99),
        "abs_q999_z": column_quantile(abs_z, 0.999),
        "abs_q9999_z": column_quantile(abs_z, 0.9999),
        "positive_top1pct_energy_share": top_energy_share(positive, 0.01),
        "positive_top0_1pct_energy_share": top_energy_share(positive, 0.001),
        "negative_top1pct_energy_share": top_energy_share(negative, 0.01),
        "negative_top0_1pct_energy_share": top_energy_share(negative, 0.001),
        "abs_top1pct_energy_share": top_energy_share(abs_z, 0.01),
        "abs_top0_1pct_energy_share": top_energy_share(abs_z, 0.001),
        "doc_positive_peak_q95_z": column_quantile(doc_pos_z, 0.95),
        "doc_positive_peak_q99_z": column_quantile(doc_pos_z, 0.99),
        "doc_negative_peak_q95_z": column_quantile(doc_neg_z, 0.95),
        "doc_negative_peak_q99_z": column_quantile(doc_neg_z, 0.99),
    }

    for threshold in (3, 5, 8):
        out[f"positive_token_rate_z{threshold}"] = (z > threshold).mean(axis=0)
        out[f"negative_token_rate_z{threshold}"] = (z < -threshold).mean(axis=0)
        pos_count = (doc_pos_z > threshold).sum(axis=0).astype(np.float64)
        neg_count = (doc_neg_z > threshold).sum(axis=0).astype(np.float64)
        out[f"positive_doc_count_z{threshold}"] = pos_count
        out[f"negative_doc_count_z{threshold}"] = neg_count
        denom = max(1, nd)
        out[f"positive_doc_rate_z{threshold}"] = pos_count / denom
        out[f"negative_doc_rate_z{threshold}"] = neg_count / denom

    # One pseudodocument prevents an otherwise interesting tail shape from
    # collapsing to an exact zero while still strongly penalizing no support.
    pos_support = np.minimum(
        1.0, (out["positive_doc_count_z5"] + 1.0) / (min_tail_docs + 1.0)
    )
    neg_support = np.minimum(
        1.0, (out["negative_doc_count_z5"] + 1.0) / (min_tail_docs + 1.0)
    )
    pos_shape = (
        np.maximum(out["positive_q999_z"], 0.0)
        * np.sqrt(out["positive_top0_1pct_energy_share"])
    )
    neg_shape = (
        np.maximum(out["negative_q999_z"], 0.0)
        * np.sqrt(out["negative_top0_1pct_energy_share"])
    )
    pos_score = pos_shape * pos_support
    neg_score = neg_shape * neg_support
    out["positive_tail_shape_score"] = pos_shape
    out["negative_tail_shape_score"] = neg_shape
    out["positive_tail_support_weight"] = pos_support
    out["negative_tail_support_weight"] = neg_support
    out["positive_tail_selectivity_score"] = pos_score
    out["negative_tail_selectivity_score"] = neg_score
    out["tail_selectivity_score"] = np.maximum(pos_score, neg_score)
    out["tail_polarity_is_positive"] = (pos_score >= neg_score).astype(np.float64)
    return out


def build_ranking_rows(
    stats: dict[int, dict[str, Any]],
    S_bank: dict[int, np.ndarray],
    layers: list[int],
    k: int,
    top_r: int,
    min_tail_docs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for layer in layers:
        st = stats[layer]
        tail = robust_tail_statistics(st, min_tail_docs)
        n = max(1, int(st["n"]))
        nd = max(1, int(st["n_docs"]))
        mean = st["sum"] / n
        mean_abs = st["abs_sum"] / n
        rms = np.sqrt(st["sq_sum"] / n)
        var = np.maximum(st["sq_sum"] / n - mean**2, 0.0)
        std = np.sqrt(var)
        mean_abs_cos = st["cos_abs_sum"] / n
        rms_cos = np.sqrt(st["cos_sq_sum"] / n)
        positive_rate = st["positive_count"] / n
        top1_rate = st["top1_count"] / n
        # A token contributes r memberships, so this is the literal probability
        # that the direction is among the token's top-r bank members.
        topr_rate = st["topr_count"] / n
        doc_topr_rate = st["doc_topr_count"] / nd
        mean_doc_peak_abs = st["doc_peak_abs_sum"] / nd
        mean_residual_norm = st["residual_norm_sum"] / n

        S = np.asarray(S_bank[layer])[:k]
        sigma0 = max(float(S[0]), 1e-12)

        for sv0 in range(k):
            rows.append(
                {
                    "candidate": f"L{layer:02d}_SV{sv0 + 1:02d}",
                    "layer": layer,
                    "sv_index_0": sv0,
                    "sv_rank_1based": sv0 + 1,
                    "singular_value": float(S[sv0]),
                    "singular_value_over_sv1": float(S[sv0] / sigma0),
                    "n_tokens": int(st["n"]),
                    "n_documents": int(st["n_docs"]),
                    "mean_activation": float(mean[sv0]),
                    "mean_abs_activation": float(mean_abs[sv0]),
                    "rms_activation": float(rms[sv0]),
                    "std_activation": float(std[sv0]),
                    "positive_rate": float(positive_rate[sv0]),
                    "max_activation": float(st["max"][sv0]),
                    "min_activation": float(st["min"][sv0]),
                    "mean_abs_cosine": float(mean_abs_cos[sv0]),
                    "rms_cosine": float(rms_cos[sv0]),
                    "top1_abs_rate": float(top1_rate[sv0]),
                    f"top{min(top_r, k)}_abs_rate": float(topr_rate[sv0]),
                    f"doc_top{min(top_r, k)}_presence_rate": float(doc_topr_rate[sv0]),
                    "mean_document_peak_abs": float(mean_doc_peak_abs[sv0]),
                    "mean_layer_residual_norm": float(mean_residual_norm),
                    "sigma_weighted_mean_abs": float(S[sv0] * mean_abs[sv0]),
                    "sigma_weighted_std": float(S[sv0] * std[sv0]),
                    "dynamicity_std_over_abs_mean": float(
                        std[sv0] / (abs(mean[sv0]) + 1e-12)
                    ),
                    "token_sample_n": int(tail["sample_n"][sv0]),
                    "median_activation": float(tail["median"][sv0]),
                    "mad_activation": float(tail["mad"][sv0]),
                    "robust_activation_scale": float(tail["robust_scale"][sv0]),
                    "stable_skewness": float(tail["skewness"][sv0]),
                    "stable_kurtosis": float(tail["kurtosis"][sv0]),
                    "stable_excess_kurtosis": float(tail["excess_kurtosis"][sv0]),
                    "effective_support_fraction": float(
                        tail["effective_support_fraction"][sv0]
                    ),
                    "positive_max_robust_z": float(tail["positive_max_z"][sv0]),
                    "negative_max_robust_z": float(tail["negative_max_z"][sv0]),
                    "positive_q99_robust_z": float(tail["positive_q99_z"][sv0]),
                    "positive_q999_robust_z": float(tail["positive_q999_z"][sv0]),
                    "positive_q9999_robust_z": float(tail["positive_q9999_z"][sv0]),
                    "negative_q99_robust_z": float(tail["negative_q99_z"][sv0]),
                    "negative_q999_robust_z": float(tail["negative_q999_z"][sv0]),
                    "negative_q9999_robust_z": float(tail["negative_q9999_z"][sv0]),
                    "abs_q99_robust_z": float(tail["abs_q99_z"][sv0]),
                    "abs_q999_robust_z": float(tail["abs_q999_z"][sv0]),
                    "abs_q9999_robust_z": float(tail["abs_q9999_z"][sv0]),
                    "positive_top1pct_energy_share": float(
                        tail["positive_top1pct_energy_share"][sv0]
                    ),
                    "positive_top0_1pct_energy_share": float(
                        tail["positive_top0_1pct_energy_share"][sv0]
                    ),
                    "negative_top1pct_energy_share": float(
                        tail["negative_top1pct_energy_share"][sv0]
                    ),
                    "negative_top0_1pct_energy_share": float(
                        tail["negative_top0_1pct_energy_share"][sv0]
                    ),
                    "abs_top1pct_energy_share": float(
                        tail["abs_top1pct_energy_share"][sv0]
                    ),
                    "abs_top0_1pct_energy_share": float(
                        tail["abs_top0_1pct_energy_share"][sv0]
                    ),
                    "positive_token_rate_z3": float(tail["positive_token_rate_z3"][sv0]),
                    "positive_token_rate_z5": float(tail["positive_token_rate_z5"][sv0]),
                    "positive_token_rate_z8": float(tail["positive_token_rate_z8"][sv0]),
                    "negative_token_rate_z3": float(tail["negative_token_rate_z3"][sv0]),
                    "negative_token_rate_z5": float(tail["negative_token_rate_z5"][sv0]),
                    "negative_token_rate_z8": float(tail["negative_token_rate_z8"][sv0]),
                    "positive_doc_count_z3": int(tail["positive_doc_count_z3"][sv0]),
                    "positive_doc_count_z5": int(tail["positive_doc_count_z5"][sv0]),
                    "positive_doc_count_z8": int(tail["positive_doc_count_z8"][sv0]),
                    "negative_doc_count_z3": int(tail["negative_doc_count_z3"][sv0]),
                    "negative_doc_count_z5": int(tail["negative_doc_count_z5"][sv0]),
                    "negative_doc_count_z8": int(tail["negative_doc_count_z8"][sv0]),
                    "positive_doc_rate_z3": float(tail["positive_doc_rate_z3"][sv0]),
                    "positive_doc_rate_z5": float(tail["positive_doc_rate_z5"][sv0]),
                    "positive_doc_rate_z8": float(tail["positive_doc_rate_z8"][sv0]),
                    "negative_doc_rate_z3": float(tail["negative_doc_rate_z3"][sv0]),
                    "negative_doc_rate_z5": float(tail["negative_doc_rate_z5"][sv0]),
                    "negative_doc_rate_z8": float(tail["negative_doc_rate_z8"][sv0]),
                    "doc_positive_peak_q95_z": float(tail["doc_positive_peak_q95_z"][sv0]),
                    "doc_positive_peak_q99_z": float(tail["doc_positive_peak_q99_z"][sv0]),
                    "doc_negative_peak_q95_z": float(tail["doc_negative_peak_q95_z"][sv0]),
                    "doc_negative_peak_q99_z": float(tail["doc_negative_peak_q99_z"][sv0]),
                    "positive_tail_selectivity_score": float(
                        tail["positive_tail_selectivity_score"][sv0]
                    ),
                    "negative_tail_selectivity_score": float(
                        tail["negative_tail_selectivity_score"][sv0]
                    ),
                    "positive_tail_shape_score": float(
                        tail["positive_tail_shape_score"][sv0]
                    ),
                    "negative_tail_shape_score": float(
                        tail["negative_tail_shape_score"][sv0]
                    ),
                    "positive_tail_support_weight": float(
                        tail["positive_tail_support_weight"][sv0]
                    ),
                    "negative_tail_support_weight": float(
                        tail["negative_tail_support_weight"][sv0]
                    ),
                    "tail_selectivity_score": float(tail["tail_selectivity_score"][sv0]),
                    "selected_tail_polarity": (
                        "positive" if tail["tail_polarity_is_positive"][sv0] else "negative"
                    ),
                }
            )

    def assign_global_rank(metric: str, col: str) -> None:
        ordered = sorted(range(len(rows)), key=lambda i: rows[i][metric], reverse=True)
        for rank, i in enumerate(ordered, 1):
            rows[i][col] = rank

    def assign_layer_rank(metric: str, col: str) -> None:
        for layer in layers:
            idxs = [i for i, row in enumerate(rows) if row["layer"] == layer]
            idxs.sort(key=lambda i: rows[i][metric], reverse=True)
            for rank, i in enumerate(idxs, 1):
                rows[i][col] = rank

    top_r_col = f"top{min(top_r, k)}_abs_rate"
    assign_global_rank("mean_abs_cosine", "rank_global_mean_abs_cosine")
    assign_global_rank("top1_abs_rate", "rank_global_top1_rate")
    assign_global_rank(top_r_col, f"rank_global_{top_r_col}")
    assign_global_rank("sigma_weighted_mean_abs", "rank_global_sigma_weighted_mean_abs")
    assign_global_rank("std_activation", "rank_global_std_activation")
    assign_global_rank("stable_excess_kurtosis", "rank_global_excess_kurtosis")
    assign_global_rank("tail_selectivity_score", "rank_global_tail_selectivity")

    assign_layer_rank("mean_abs_activation", "rank_layer_mean_abs_activation")
    assign_layer_rank("mean_abs_cosine", "rank_layer_mean_abs_cosine")
    assign_layer_rank("top1_abs_rate", "rank_layer_top1_rate")
    assign_layer_rank(top_r_col, f"rank_layer_{top_r_col}")
    assign_layer_rank("std_activation", "rank_layer_std_activation")
    assign_layer_rank("stable_excess_kurtosis", "rank_layer_excess_kurtosis")
    assign_layer_rank("tail_selectivity_score", "rank_layer_tail_selectivity")

    # Primary ordering for the CSV is mean |cosine| because it is the cleanest
    # scale-normalized quantity for comparing unit directions across layers.
    rows.sort(key=lambda row: row["rank_global_mean_abs_cosine"])
    return rows


# ------------------------------ Outputs ------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


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


def decode_context(tokenizer: Any, rec: dict[str, Any]) -> dict[str, Any]:
    ids = list(rec.pop("span_token_ids"))
    center = int(rec.pop("center_offset"))
    left_ids = ids[:center]
    center_ids = ids[center : center + 1]
    right_ids = ids[center + 1 :]

    left = tokenizer.decode(
        left_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    token = tokenizer.decode(
        center_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    right = tokenizer.decode(
        right_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    rec["token"] = token
    rec["context"] = left + token + right
    rec["context_marked"] = left + "\u27e6" + token + "\u27e7" + right
    # The state that was actually projected sits one position earlier; expose
    # that token in decoded form so contexts read unambiguously.
    prev_token_id = rec.get("prev_token_id")
    if prev_token_id is not None:
        rec["prev_token"] = tokenizer.decode(
            [int(prev_token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return rec


def add_context_diversity_columns(
    rows: list[dict[str, Any]],
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
) -> None:
    """Summarize lexical/document diversity among each retained extreme tail."""
    for row in rows:
        layer = int(row["layer"])
        sv0 = int(row["sv_index_0"])
        summaries: dict[str, dict[str, float | int]] = {}
        for polarity in ("positive", "negative"):
            heap = heaps.get((layer, sv0, polarity), [])
            ordered = sorted(heap, key=lambda x: x[0], reverse=True)
            # Do not allow an accidentally duplicated event to inflate lexicality.
            unique_records: list[dict[str, Any]] = []
            seen_events: set[tuple[int, int]] = set()
            for _, _, rec in ordered:
                event = (int(rec["document_index"]), int(rec["token_position"]))
                if event in seen_events:
                    continue
                seen_events.add(event)
                unique_records.append(rec)

            docs = {int(rec["document_index"]) for rec in unique_records}
            token_counts: dict[int, int] = {}
            for rec in unique_records:
                ids = rec["span_token_ids"]
                center = int(rec["center_offset"])
                token_id = int(ids[center])
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

            total = len(unique_records)
            if total:
                probabilities = np.asarray(
                    [count / total for count in token_counts.values()], dtype=np.float64
                )
                entropy = float(-(probabilities * np.log(probabilities)).sum())
                max_share = float(max(token_counts.values()) / total)
                effective_tokens = float(math.exp(entropy))
            else:
                entropy = 0.0
                max_share = 0.0
                effective_tokens = 0.0

            summary: dict[str, float | int] = {
                "retained_events": total,
                "unique_documents": len(docs),
                "unique_center_tokens": len(token_counts),
                "center_token_entropy": entropy,
                "effective_center_tokens": effective_tokens,
                "largest_center_token_share": max_share,
            }
            summaries[polarity] = summary
            for key, value in summary.items():
                row[f"{polarity}_top_context_{key}"] = value

        selected = str(row["selected_tail_polarity"])
        for key, value in summaries[selected].items():
            row[f"selected_tail_top_context_{key}"] = value


def write_contexts(
    path: Path,
    tokenizer: Any,
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
    S_bank: dict[int, np.ndarray],
    layers: list[int],
    k: int,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for layer in layers:
            for sv0 in range(k):
                candidate = f"L{layer:02d}_SV{sv0 + 1:02d}"
                for polarity in ("positive", "negative"):
                    heap = heaps.get((layer, sv0, polarity), [])
                    ordered = sorted(heap, key=lambda x: x[0], reverse=True)
                    for rank, (_, _, stored) in enumerate(ordered, 1):
                        rec = dict(stored)
                        rec = decode_context(tokenizer, rec)
                        rec = {
                            "candidate": candidate,
                            "layer": layer,
                            "sv_index_0": sv0,
                            "sv_rank_1based": sv0 + 1,
                            "singular_value": float(S_bank[layer][sv0]),
                            "polarity": polarity,
                            "rank_within_polarity": rank,
                            **rec,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------- Mode drivers ----------------------------------


def token_sampling_plan(args: argparse.Namespace, n_units: int, unit_tokens: int) -> tuple[int, int, int]:
    """Return (stride, capacity, offset) for the systematic token sample.

    A random-offset systematic sample is uniform over the concatenated target
    token stream and has a hard memory bound. All layers sample the same
    targets.
    """
    estimated_max_tokens = max(1, n_units * unit_tokens)
    stride = max(1, estimated_max_tokens // args.token_reservoir_size)
    capacity = int(math.ceil(estimated_max_tokens / stride)) + 2
    offset_rng = random.Random(args.seed + 917_531)
    offset = offset_rng.randrange(stride)
    return stride, capacity, offset


def run_corpus_mode(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    tokenizer: Any,
    model: Any,
    layers: list[int],
    V_bank: dict[int, torch.Tensor],
    signature: dict[str, Any],
) -> dict[str, Any]:
    import jlens

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    rng = random.Random(args.seed)

    token_sample_stride, token_sample_capacity, token_sample_offset = token_sampling_plan(
        args, args.n_docs, args.max_seq_len
    )

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but no {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, signature)
        docs_processed = int(state["docs_scanned"])
        stream_items_seen = int(state["stream_items_seen"])
        total_tokens = int(state["target_tokens_total"])
        stats = state["stats"]
        heaps = state["heaps"]
        heap_counter = int(state["heap_counter"])
        rng.setstate(state["rng_state"])
        print(
            f"[resume] docs={docs_processed}, stream_items_seen={stream_items_seen}, "
            f"tokens={total_tokens}"
        )
    else:
        docs_processed = 0
        stream_items_seen = 0
        total_tokens = 0
        stats = init_stats(layers, args.k, args.n_docs, token_sample_capacity)
        heaps: dict[
            tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]
        ] = {}
        heap_counter = 0

    corpus = build_corpus(args)

    # Keep one device/dtype copy of each V bank per actual layer placement.
    # With device_map="auto", layers can in principle live on different devices.
    V_device_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def checkpoint_now() -> None:
        save_checkpoint(
            checkpoint_path,
            {
                "signature": signature,
                "docs_scanned": docs_processed,
                "stream_items_seen": stream_items_seen,
                "target_tokens_total": total_tokens,
                "stats": stats,
                "heaps": heaps,
                "heap_counter": heap_counter,
                "rng_state": rng.getstate(),
            },
        )

    print("[3/4] Scanning corpus with predictive alignment...")
    with torch.inference_mode():
        for stream_index, item in enumerate(corpus):
            if stream_index < stream_items_seen:
                continue
            stream_items_seen = stream_index + 1

            if docs_processed >= args.n_docs:
                break
            if not isinstance(item, dict):
                continue
            text = item.get(args.text_field)
            if not isinstance(text, str):
                continue

            sampled = sample_token_window(
                tokenizer,
                text,
                rng,
                max_seq_len=args.max_seq_len,
                min_tokens=args.min_tokens,
                char_cap=args.sample_char_cap,
            )
            if sampled is None:
                continue
            input_ids_cpu_tensor, window_meta = sampled
            input_ids_cpu = input_ids_cpu_tensor[0].tolist()

            # Target tokens exclude special tokens and position 0 (which has no
            # predecessor state to read). Special tokens remain in model context
            # and may serve as read positions.
            target_positions_list = predictive_target_positions(
                input_ids_cpu, special_ids
            )
            if len(target_positions_list) < args.min_tokens:
                continue

            input_ids = input_ids_cpu_tensor.to(model.input_device)
            source = jsonable_source(item)

            with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids)
                heap_counter = scan_targets_over_layers(
                    recorder=recorder,
                    layers=layers,
                    target_positions_list=target_positions_list,
                    input_ids_cpu=input_ids_cpu,
                    V_bank=V_bank,
                    V_device_cache=V_device_cache,
                    stats=stats,
                    heaps=heaps,
                    heap_counter=heap_counter,
                    args=args,
                    doc_index=docs_processed,
                    tokens_before=total_tokens,
                    token_sample_stride=token_sample_stride,
                    token_sample_offset=token_sample_offset,
                    source=source,
                    window_meta=window_meta,
                )

            total_tokens += len(target_positions_list)
            docs_processed += 1

            if args.progress_every > 0 and docs_processed % args.progress_every == 0:
                print(
                    f"[scan] docs={docs_processed}/{args.n_docs} "
                    f"tokens={total_tokens:,} stream_items={stream_items_seen:,}"
                    + gpu_status(),
                    flush=True,
                )

            if args.checkpoint_every > 0 and docs_processed % args.checkpoint_every == 0:
                checkpoint_now()
                print(f"[checkpoint] {checkpoint_path}", flush=True)

    # Always save final state before producing human-readable outputs.
    checkpoint_now()

    return {
        "stats": stats,
        "heaps": heaps,
        "heap_counter": heap_counter,
        "docs_scanned": docs_processed,
        "target_tokens_total": total_tokens,
        "token_sample_stride": token_sample_stride,
        "token_sample_offset": token_sample_offset,
        "token_sample_capacity": token_sample_capacity,
        "metadata_extras": {
            "svd_method": (
                "external_direction_bank"
                if args.directions_dir
                else ("exact" if args.exact_svd else "randomized_lowrank")
            ),
            "dataset": args.input_jsonl or args.dataset,
            "dataset_config": None if args.input_jsonl else args.dataset_config,
            "split": None if args.input_jsonl else args.split,
            "text_field": args.text_field,
            "documents_processed": docs_processed,
            "target_tokens_processed": total_tokens,
            "max_seq_len": args.max_seq_len,
            "min_tokens": args.min_tokens,
            "sample_char_cap": args.sample_char_cap,
            "shuffle_buffer": args.shuffle_buffer,
        },
    }


def run_cot_mode(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    rollouts_path: Path,
    tokenizer: Any,
    hf_model: Any,
    model: Any,
    layers: list[int],
    V_bank: dict[int, torch.Tensor],
    jobs: list[dict[str, Any]],
    signature: dict[str, Any],
) -> dict[str, Any]:
    import jlens

    token_sample_stride, token_sample_capacity, token_sample_offset = token_sampling_plan(
        args, len(jobs), args.max_new_tokens
    )

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but no {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, signature)
        next_job_index = int(state["next_job_index"])
        analysis_docs = int(state["docs_scanned"])
        total_target_tokens = int(state["target_tokens_total"])
        stats = state["stats"]
        heaps = state["heaps"]
        heap_counter = int(state["heap_counter"])
        rollout_records = state["rollout_records"]
        print(
            f"[resume] next_job={next_job_index}/{len(jobs)} "
            f"analysis_rollouts={analysis_docs} tokens={total_target_tokens:,}"
        )
    else:
        next_job_index = 0
        analysis_docs = 0
        total_target_tokens = 0
        stats = init_stats(layers, args.k, len(jobs), token_sample_capacity)
        heaps: dict[
            tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]
        ] = {}
        heap_counter = 0
        rollout_records: list[dict[str, Any]] = []

    V_device_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def checkpoint_now() -> None:
        save_checkpoint(
            checkpoint_path,
            {
                "signature": signature,
                "next_job_index": next_job_index,
                "docs_scanned": analysis_docs,
                "target_tokens_total": total_target_tokens,
                "stats": stats,
                "heaps": heaps,
                "heap_counter": heap_counter,
                "rollout_records": rollout_records,
            },
        )

    print("[3/4] Generating and replaying analysis-channel tokens (predictive)...")
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
            # Every generated analysis token sits at t >= prompt_len >= 1, so
            # each has a predecessor state; the guard is defensive only.
            target_positions_list = [t for t in analysis_positions if t >= 1]

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
                    "target_tokens": len(target_positions_list),
                    "final_tokens": len(final_local),
                    "analysis_segments": len(analysis_segments),
                    "final_segments": len(final_segments),
                    "sequence_tokens": len(full_ids_cpu),
                    "hit_max_new_tokens": len(generated_ids) >= args.max_new_tokens,
                }
            )

            if not target_positions_list:
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
                heap_counter = scan_targets_over_layers(
                    recorder=recorder,
                    layers=layers,
                    target_positions_list=target_positions_list,
                    input_ids_cpu=full_ids_cpu,
                    V_bank=V_bank,
                    V_device_cache=V_device_cache,
                    stats=stats,
                    heaps=heaps,
                    heap_counter=heap_counter,
                    args=args,
                    doc_index=analysis_docs,
                    tokens_before=total_target_tokens,
                    token_sample_stride=token_sample_stride,
                    token_sample_offset=token_sample_offset,
                    source=source,
                    window_meta={
                        "prompt_tokens": prompt_len,
                        "generated_tokens": len(generated_ids),
                        "analysis_tokens": len(analysis_positions),
                        "analysis_segments": len(analysis_segments),
                    },
                )

            del input_ids, sequence
            total_target_tokens += len(target_positions_list)
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
                f"analysis_rollouts={analysis_docs} tokens={total_target_tokens:,} "
                f"last={record['status']}" + gpu_status(),
                flush=True,
            )

        if args.checkpoint_every > 0 and next_job_index % args.checkpoint_every == 0:
            checkpoint_now()
            write_jsonl(rollout_records, rollouts_path)
            print(f"[checkpoint] {checkpoint_path}", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint_now()
    write_jsonl(rollout_records, rollouts_path)

    if analysis_docs == 0:
        raise RuntimeError(
            "No rollouts with an identifiable analysis channel were scanned. "
            f"Inspect {rollouts_path} for raw responses and status/error fields."
        )

    status_counts: dict[str, int] = {}
    for record in rollout_records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "stats": stats,
        "heaps": heaps,
        "heap_counter": heap_counter,
        "docs_scanned": analysis_docs,
        "target_tokens_total": total_target_tokens,
        "token_sample_stride": token_sample_stride,
        "token_sample_offset": token_sample_offset,
        "token_sample_capacity": token_sample_capacity,
        "metadata_extras": {
            "prompts_source": args.prompts_jsonl or "built_in_moderate_reasoning_set",
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
            "analysis_target_tokens_scanned": total_target_tokens,
            "channel_filter": (
                "Harmony analysis content only; prompt, control, and final tokens excluded"
            ),
            "capture_method": "generate with KV cache, then exact causal teacher-forced replay",
        },
    }


# ----------------------------- Finalize / main ------------------------------


def finalize_outputs(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    tokenizer: Any,
    hf_model: Any,
    V_bank: dict[int, torch.Tensor],
    S_bank: dict[int, np.ndarray],
    layers: list[int],
    d_model: int,
    direction_source: str,
    scan_state: dict[str, Any],
    prompts_count: int | None,
) -> None:
    print("[4/4] Building rankings and optional unembedding geometry...")
    stats = scan_state["stats"]
    heaps = scan_state["heaps"]

    rows = build_ranking_rows(
        stats, S_bank, layers, args.k, args.top_r, args.min_tail_docs
    )
    add_context_diversity_columns(rows, heaps)

    unembedding_geometry: dict[tuple[int, int], dict[str, Any]] = {}
    if not args.skip_unembedding:
        unembedding_geometry = analyze_unembedding_geometry(
            hf_model=hf_model,
            tokenizer=tokenizer,
            V_bank=V_bank,
            layers=layers,
            k=args.k,
            n_neighbors=args.unembedding_neighbors,
            chunk_size=args.unembedding_chunk_size,
            include_special_tokens=args.include_special_unembedding_tokens,
        )
        add_unembedding_columns(rows, unembedding_geometry)

    rankings_path = out_dir / "sv_rankings.csv"
    selectivity_path = out_dir / "selectivity_rankings.csv"
    contexts_path = out_dir / "top_contexts.jsonl"
    unembedding_path = out_dir / "unembedding_neighbors.jsonl"
    metadata_path = out_dir / "metadata.json"
    checkpoint_path = out_dir / "checkpoint.pkl"
    rollouts_path = out_dir / "rollouts.jsonl"

    write_csv(rows, rankings_path)
    write_csv(
        sorted(rows, key=lambda row: row["tail_selectivity_score"], reverse=True),
        selectivity_path,
    )
    write_contexts(contexts_path, tokenizer, heaps, S_bank, layers, args.k)
    if unembedding_geometry:
        write_unembedding_neighbors(
            unembedding_path, unembedding_geometry, S_bank, layers, args.k
        )

    metadata: dict[str, Any] = {
        "script": "scan_predictive_jlens_directions",
        "mode": args.mode,
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": layers,
        "d_model": d_model,
        "k": args.k,
        "direction_source": direction_source,
        "activation_alignment": (
            "predictive: residual state at position t-1, attributed to the "
            "target token at position t"
        ),
        "projection_definition": (
            "a = h_layer_{t-1} @ V[:, sv], attributed to the target token at t"
        ),
        "target_token_rule": (
            "non-special tokens at positions t >= 1; the read position t-1 may "
            "be any token, including special/marker tokens"
        ),
        "top_r": min(args.top_r, args.k),
        "top_contexts_per_polarity": args.top_contexts,
        "context_radius_tokens": args.context_radius,
        "token_sample_target_per_layer": args.token_reservoir_size,
        "token_sample_stride": scan_state["token_sample_stride"],
        "token_sample_offset": scan_state["token_sample_offset"],
        "token_sample_capacity_per_layer": scan_state["token_sample_capacity"],
        "token_sample_actual_per_layer": {
            str(layer): int(stats[layer]["token_sample_n"]) for layer in layers
        },
        "min_tail_docs_for_full_score_weight": args.min_tail_docs,
        "unembedding_analysis": not args.skip_unembedding,
        "unembedding_neighbors_per_polarity": (
            None if args.skip_unembedding else args.unembedding_neighbors
        ),
        "unembedding_space": (
            None
            if args.skip_unembedding
            else "raw cosine between V[:,sv] and lm_head.weight[token]"
        ),
        "unembedding_special_tokens_included": args.include_special_unembedding_tokens,
        "seed": args.seed,
        "primary_csv_sort": "rank_global_mean_abs_cosine",
        "selectivity_csv_sort": "rank_global_tail_selectivity",
        "candidate_naming": "LXX_SVYY where layer is 0-based and SV rank is 1-based",
        "notes": {
            "predictive_alignment": (
                "every statistic is computed on projections of h_{t-1}; a "
                "direction that scores highly is active in the state that "
                "predicts the target token, not necessarily at the token itself"
            ),
            "context_records": (
                "token/context/context_marked center on the target token; "
                "read_position and prev_token identify the state that was "
                "actually projected"
            ),
            "mean_abs_cosine": "mean(|a| / ||h_{t-1}||); useful for cross-layer scale normalization",
            "top1_abs_rate": "fraction of target tokens where this SV has the largest |a| among the top-k bank",
            f"top{min(args.top_r, args.k)}_abs_rate": "fraction of target tokens where this SV is among the r largest |a| in the top-k bank",
            "sigma_weighted_mean_abs": "singular_value * mean_abs_activation",
            "std_activation": "variation across target-token activations; useful for spotting nearly constant/baseline directions",
            "robust_activation_scale": "1.4826 * MAD from the bounded cross-corpus token sample; falls back to ordinary std if MAD is zero",
            "stable_excess_kurtosis": "exact streaming fourth-central-moment tail weight; high values may indicate selectivity or artifacts",
            "effective_support_fraction": "M2^2 / (N*M4); smaller means activation energy is supported by fewer tokens",
            "positive/negative_q999_robust_z": "one-sided 99.9th percentile after median/MAD normalization",
            "positive/negative_top0_1pct_energy_share": "share of one-sided squared robust-z energy in the strongest 0.1% of sampled tokens",
            "positive/negative_doc_count_z5": "number of documents whose exact polarity-specific peak exceeds robust z=5",
            "tail_selectivity_score": "max over polarities of q999_z * sqrt(top0.1pct_energy_share) * min(1, (doc_z5_count+1)/(min_tail_docs+1)); the pseudocount retains unsupported shapes but downweights them",
            "selected_tail_top_context_largest_center_token_share": "lexical concentration among retained extreme events; high values flag token-specific candidates",
            "nearest_unembed_cosine": "largest cosine between the unit SV direction and any unembedding row",
            "farthest_unembed_cosine": "most negative cosine between the unit SV direction and any unembedding row",
            "max_abs_unembed_token_cosine": "max absolute token cosine; a rough token-likeness / lexical anchoring score",
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    if prompts_count is not None:
        metadata["prompts"] = prompts_count
    metadata.update(scan_state["metadata_extras"])
    try:
        import transformers

        metadata["versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets

        metadata["versions"]["datasets"] = datasets.__version__
    except Exception:
        pass
    atomic_json_dump(metadata, metadata_path)

    print("\nDone.")
    print(f"  rankings: {rankings_path}")
    print(f"  selectivity rankings: {selectivity_path}")
    print(f"  contexts: {contexts_path}")
    if args.mode == "cot":
        print(f"  rollouts: {rollouts_path}")
    if unembedding_geometry:
        print(f"  unembedding: {unembedding_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  directions: {out_dir / 'directions'}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pkl"
    rollouts_path = out_dir / "rollouts.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[mode] {args.mode} (predictive alignment: h_{{t-1}} attributed to target t)")
    print("[1/4] Loading J-Lens and preparing right-SV bank...")
    import jlens

    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    layers = parse_layers(args.layers, lens.source_layers)
    if args.k > lens.d_model:
        raise ValueError(f"k={args.k} exceeds d_model={lens.d_model}")
    print(
        f"[lens] d_model={lens.d_model}, fitted_layers={lens.source_layers}, "
        f"scanning_layers={layers}"
    )
    V_bank, S_bank, direction_source = prepare_direction_bank(
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
    print(gpu_status())

    if args.unembedding_only:
        unembedding_geometry = analyze_unembedding_geometry(
            hf_model=hf_model,
            tokenizer=tokenizer,
            V_bank=V_bank,
            layers=layers,
            k=args.k,
            n_neighbors=args.unembedding_neighbors,
            chunk_size=args.unembedding_chunk_size,
            include_special_tokens=args.include_special_unembedding_tokens,
        )
        print(f"[unembedding] analyzed {len(unembedding_geometry):,} SV directions")

        unembedding_path = out_dir / "unembedding_neighbors.jsonl"
        write_unembedding_neighbors(
            unembedding_path, unembedding_geometry, S_bank, layers, args.k
        )

        rankings_path = out_dir / "sv_rankings.csv"
        if rankings_path.exists():
            rows = read_csv_rows(rankings_path)
            add_unembedding_columns(rows, unembedding_geometry)
            write_csv(rows, rankings_path)
            print(f"[unembedding-only] augmented {rankings_path}")
        else:
            rows = []
            for layer in layers:
                for sv0 in range(args.k):
                    rows.append({
                        "candidate": f"L{layer:02d}_SV{sv0 + 1:02d}",
                        "layer": layer,
                        "sv_index_0": sv0,
                        "sv_rank_1based": sv0 + 1,
                        "singular_value": float(S_bank[layer][sv0]),
                    })
            add_unembedding_columns(rows, unembedding_geometry)
            summary_path = out_dir / "unembedding_summary.csv"
            write_csv(rows, summary_path)
            print(f"[unembedding-only] wrote {summary_path}")

        print("\nDone (unembedding-only).")
        print(f"  unembedding: {unembedding_path}")
        print(f"  directions: {out_dir / 'directions'}")
        return

    prompts_count: int | None = None
    if args.mode == "cot":
        prompts = load_prompts(args.prompts_jsonl, args.max_prompts)
        jobs = make_jobs(prompts, args.samples_per_prompt)
        prompts_count = len(prompts)
        print(
            f"[jobs] prompts={len(prompts)} samples={args.samples_per_prompt} "
            f"total={len(jobs)}"
        )
        signature = run_signature(args, layers, jobs)
        scan_state = run_cot_mode(
            args=args,
            checkpoint_path=checkpoint_path,
            rollouts_path=rollouts_path,
            tokenizer=tokenizer,
            hf_model=hf_model,
            model=model,
            layers=layers,
            V_bank=V_bank,
            jobs=jobs,
            signature=signature,
        )
    else:
        signature = run_signature(args, layers, None)
        scan_state = run_corpus_mode(
            args=args,
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            model=model,
            layers=layers,
            V_bank=V_bank,
            signature=signature,
        )

    finalize_outputs(
        args=args,
        out_dir=out_dir,
        tokenizer=tokenizer,
        hf_model=hf_model,
        V_bank=V_bank,
        S_bank=S_bank,
        layers=layers,
        d_model=d_model,
        direction_source=direction_source,
        scan_state=scan_state,
        prompts_count=prompts_count,
    )


if __name__ == "__main__":
    main()
