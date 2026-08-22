#!/usr/bin/env python3
"""
scan_unrealized_words.py

Build a first-pass database of heavily used J-Lens right-singular directions
("unrealized word" candidates) in GPT-OSS-20B.

For every fitted J-Lens source layer l:
    J_l = U_l diag(S_l) V_l^T

we take the top-k columns V_l[:, i] and scan residual-stream activations h_l,t:
    a_l,i,t = <h_l,t, V_l[:, i]>

The scanner streams a text corpus, never writes full hidden states, and keeps:
  * raw activation moments
  * residual-normalized activation moments (cosine-like usage)
  * top-1 / top-r recruitment frequencies within each layer's top-k bank
  * singular-value-weighted usage
  * strongest positive and negative token contexts for each direction
  * nearest and most anti-aligned vocabulary tokens in unembedding space

Outputs:
  OUT/directions/LXX.npz
  OUT/sv_rankings.csv
  OUT/top_contexts.jsonl
  OUT/unembedding_neighbors.jsonl
  OUT/metadata.json
  OUT/checkpoint.pkl

Default corpus: HuggingFaceFW/fineweb, config sample-10BT, streamed.

Example:
    python scan_unrealized_words.py \
        --out unrealized_words_fineweb \
        --k 64 \
        --n-docs 2000 \
        --max-seq-len 256

Reuse a direction bank you already generated:
    python scan_unrealized_words.py \
        --directions-dir task_gaming_jlens/directions \
        --out unrealized_words_fineweb \
        --k 64 \
        --n-docs 2000

Resume an interrupted scan:
    python scan_unrealized_words.py \
        --out unrealized_words_fineweb \
        --k 64 \
        --n-docs 2000 \
        --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
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


# ----------------------------- CLI -----------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan J-Lens right-singular directions over diverse text.",
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

    # Corpus
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
        default=24,
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
        help="Tokens on each side of the peak token in stored contexts.",
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
            "without scanning the text corpus. If OUT/sv_rankings.csv exists, augment it in place."
        ),
    )
    p.add_argument(
        "--include-special-unembedding-tokens",
        action="store_true",
        help="Allow tokenizer special tokens to appear among unembedding neighbors.",
    )

    # Run / checkpointing
    p.add_argument("--out", type=str, default="unrealized_words_scan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=10)

    args = p.parse_args()

    if args.k <= 0:
        p.error("--k must be positive")
    if args.n_docs <= 0:
        p.error("--n-docs must be positive")
    if args.max_seq_len < 8:
        p.error("--max-seq-len is too small")
    if args.min_tokens <= 0:
        p.error("--min-tokens must be positive")
    if args.top_contexts <= 0:
        p.error("--top-contexts must be positive")
    if args.local_context_candidates <= 0:
        p.error("--local-context-candidates must be positive")
    if args.unembedding_neighbors <= 0:
        p.error("--unembedding-neighbors must be positive")
    if args.unembedding_chunk_size <= 0:
        p.error("--unembedding-chunk-size must be positive")
    if args.unembedding_only and args.skip_unembedding:
        p.error("--unembedding-only and --skip-unembedding cannot be used together")
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


# --------------------------- Corpus ----------------------------------------


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


# --------------------------- Statistics ------------------------------------


def init_stats(layers: list[int], k: int) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for layer in layers:
        stats[layer] = {
            "n": 0,
            "n_docs": 0,
            "sum": np.zeros(k, dtype=np.float64),
            "abs_sum": np.zeros(k, dtype=np.float64),
            "sq_sum": np.zeros(k, dtype=np.float64),
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
        }
    return stats


def update_stats(
    st: dict[str, Any],
    proj: torch.Tensor,
    residual_norm: torch.Tensor,
    top_r: int,
) -> None:
    """proj [T,k] and residual_norm [T], both fp32 on the layer device."""
    T, k = proj.shape
    if T == 0:
        return

    abs_proj = proj.abs()
    cos = proj / residual_norm.clamp_min(1e-12).unsqueeze(1)

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
    valid_positions: torch.Tensor,
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
    valid_positions_cpu = valid_positions.cpu()

    def make_record(row: int, sv0: int, activation: float) -> dict[str, Any]:
        absolute_pos = int(valid_positions_cpu[row].item())
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


# ------------------------- Checkpoint / output -----------------------------


def run_signature(args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    return {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": list(layers),
        "k": int(args.k),
        "dataset": None if args.input_jsonl else args.dataset,
        "dataset_config": None if args.input_jsonl else args.dataset_config,
        "split": None if args.input_jsonl else args.split,
        "input_jsonl": args.input_jsonl,
        "text_field": args.text_field,
        "max_seq_len": int(args.max_seq_len),
        "min_tokens": int(args.min_tokens),
        "sample_char_cap": int(args.sample_char_cap),
        "shuffle_buffer": int(args.shuffle_buffer),
        "top_r": int(args.top_r),
        "top_contexts": int(args.top_contexts),
        "local_context_candidates": int(args.local_context_candidates),
        "context_radius": int(args.context_radius),
        "directions_dir": args.directions_dir,
        "seed": int(args.seed),
    }


def save_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    docs_processed: int,
    stream_items_seen: int,
    total_tokens: int,
    stats: dict[int, dict[str, Any]],
    heaps: dict[tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]],
    heap_counter: int,
    rng: random.Random,
) -> None:
    atomic_pickle_dump(
        {
            "signature": signature,
            "docs_processed": docs_processed,
            "stream_items_seen": stream_items_seen,
            "total_tokens": total_tokens,
            "stats": stats,
            "heaps": heaps,
            "heap_counter": heap_counter,
            "rng_state": rng.getstate(),
        },
        path,
    )


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


def build_ranking_rows(
    stats: dict[int, dict[str, Any]],
    S_bank: dict[int, np.ndarray],
    layers: list[int],
    k: int,
    top_r: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for layer in layers:
        st = stats[layer]
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

    assign_layer_rank("mean_abs_activation", "rank_layer_mean_abs_activation")
    assign_layer_rank("mean_abs_cosine", "rank_layer_mean_abs_cosine")
    assign_layer_rank("top1_abs_rate", "rank_layer_top1_rate")
    assign_layer_rank(top_r_col, f"rank_layer_{top_r_col}")
    assign_layer_rank("std_activation", "rank_layer_std_activation")

    # Primary ordering for the CSV is mean |cosine| because it is the cleanest
    # scale-normalized quantity for comparing unit directions across layers.
    rows.sort(key=lambda row: row["rank_global_mean_abs_cosine"])
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    rec["context_marked"] = left + "⟦" + token + "⟧" + right
    return rec


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


# ----------------------------- Main ----------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pkl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/4] Loading J-Lens...")
    import jlens

    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_file,
    )
    layers = parse_layers(args.layers, lens.source_layers)
    if args.k > lens.d_model:
        raise ValueError(f"k={args.k} exceeds d_model={lens.d_model}")
    print(
        f"[lens] d_model={lens.d_model}, fitted_layers={lens.source_layers}, "
        f"scanning_layers={layers}"
    )

    print("[2/4] Preparing right-singular direction bank...")
    V_bank, S_bank, direction_source = prepare_direction_bank(
        args, lens, layers, out_dir
    )
    d_model = int(lens.d_model)
    del lens
    gc.collect()

    print("[3/4] Loading GPT-OSS model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = jlens.from_hf(hf_model, tokenizer)
    if model.d_model != d_model:
        raise ValueError(
            f"Model d_model={model.d_model} does not match J-Lens d_model={d_model}"
        )
    print(model)
    print(gpu_status())

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
        print(f"[unembedding] analyzed {len(unembedding_geometry):,} SV directions")

    if args.unembedding_only:
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

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    signature = run_signature(args, layers)
    rng = random.Random(args.seed)

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but no {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, signature)
        docs_processed = int(state["docs_processed"])
        stream_items_seen = int(state["stream_items_seen"])
        total_tokens = int(state["total_tokens"])
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
        stats = init_stats(layers, args.k)
        heaps: dict[
            tuple[int, int, str], list[tuple[float, int, dict[str, Any]]]
        ] = {}
        heap_counter = 0

    corpus = build_corpus(args)

    # Keep one device/dtype copy of each V bank per actual layer placement.
    # With device_map="auto", layers can in principle live on different devices.
    V_device_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    print("[4/4] Scanning corpus...")
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

            # Exclude special tokens from statistics but leave them in model context.
            valid_positions_list = [
                i for i, token_id in enumerate(input_ids_cpu) if token_id not in special_ids
            ]
            if len(valid_positions_list) < args.min_tokens:
                continue

            input_ids = input_ids_cpu_tensor.to(model.input_device)
            source = jsonable_source(item)

            with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids)

                for layer in layers:
                    hidden_full = recorder.activations[layer][0].detach()  # [seq, d_model]
                    valid_positions = torch.tensor(
                        valid_positions_list,
                        dtype=torch.long,
                        device=hidden_full.device,
                    )
                    hidden = hidden_full.index_select(0, valid_positions)

                    cache_key = (layer, str(hidden.device), hidden.dtype)
                    V = V_device_cache.get(cache_key)
                    if V is None:
                        V = V_bank[layer].to(device=hidden.device, dtype=hidden.dtype)
                        V_device_cache[cache_key] = V

                    # h @ V gives projections onto orthonormal right-SV directions.
                    proj = (hidden @ V).float()  # [T, k]
                    residual_norm = hidden.float().norm(dim=-1)
                    cos = proj / residual_norm.clamp_min(1e-12).unsqueeze(1)

                    update_stats(stats[layer], proj, residual_norm, args.top_r)
                    heap_counter = collect_context_candidates(
                        layer=layer,
                        proj=proj,
                        cos=cos,
                        valid_positions=valid_positions,
                        input_ids_cpu=input_ids_cpu,
                        source=source,
                        doc_index=docs_processed,
                        window_meta=window_meta,
                        heaps=heaps,
                        counter=heap_counter,
                        top_contexts=args.top_contexts,
                        local_candidates=args.local_context_candidates,
                        context_radius=args.context_radius,
                    )

                    del proj, cos, residual_norm, hidden

            n_content_tokens = len(valid_positions_list)
            total_tokens += n_content_tokens
            docs_processed += 1

            if args.progress_every > 0 and docs_processed % args.progress_every == 0:
                print(
                    f"[scan] docs={docs_processed}/{args.n_docs} "
                    f"tokens={total_tokens:,} stream_items={stream_items_seen:,}"
                    + gpu_status(),
                    flush=True,
                )

            if args.checkpoint_every > 0 and docs_processed % args.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path,
                    signature=signature,
                    docs_processed=docs_processed,
                    stream_items_seen=stream_items_seen,
                    total_tokens=total_tokens,
                    stats=stats,
                    heaps=heaps,
                    heap_counter=heap_counter,
                    rng=rng,
                )
                print(f"[checkpoint] {checkpoint_path}", flush=True)

    # Always save final state before producing human-readable outputs.
    save_checkpoint(
        checkpoint_path,
        signature=signature,
        docs_processed=docs_processed,
        stream_items_seen=stream_items_seen,
        total_tokens=total_tokens,
        stats=stats,
        heaps=heaps,
        heap_counter=heap_counter,
        rng=rng,
    )

    rows = build_ranking_rows(stats, S_bank, layers, args.k, args.top_r)
    rankings_path = out_dir / "sv_rankings.csv"
    contexts_path = out_dir / "top_contexts.jsonl"
    unembedding_path = out_dir / "unembedding_neighbors.jsonl"
    metadata_path = out_dir / "metadata.json"

    if unembedding_geometry:
        add_unembedding_columns(rows, unembedding_geometry)
    write_csv(rows, rankings_path)
    write_contexts(contexts_path, tokenizer, heaps, S_bank, layers, args.k)
    if unembedding_geometry:
        write_unembedding_neighbors(
            unembedding_path, unembedding_geometry, S_bank, layers, args.k
        )

    metadata = {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": layers,
        "d_model": d_model,
        "k": args.k,
        "direction_source": direction_source,
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
        "content_tokens_processed": total_tokens,
        "max_seq_len": args.max_seq_len,
        "top_r": min(args.top_r, args.k),
        "top_contexts_per_polarity": args.top_contexts,
        "context_radius_tokens": args.context_radius,
        "unembedding_analysis": not args.skip_unembedding,
        "unembedding_neighbors_per_polarity": (
            None if args.skip_unembedding else args.unembedding_neighbors
        ),
        "unembedding_space": (
            None if args.skip_unembedding else "raw cosine between V[:,sv] and lm_head.weight[token]"
        ),
        "unembedding_special_tokens_included": args.include_special_unembedding_tokens,
        "seed": args.seed,
        "primary_csv_sort": "rank_global_mean_abs_cosine",
        "candidate_naming": "LXX_SVYY where layer is 0-based and SV rank is 1-based",
        "projection_definition": "a = h_layer_token @ V[:, sv]",
        "notes": {
            "mean_abs_cosine": "mean(|a| / ||h||); useful for cross-layer scale normalization",
            "top1_abs_rate": "fraction of tokens where this SV has the largest |a| among the top-k bank",
            f"top{min(args.top_r, args.k)}_abs_rate": "fraction of tokens where this SV is among the r largest |a| in the top-k bank",
            "sigma_weighted_mean_abs": "singular_value * mean_abs_activation",
            "std_activation": "variation across token activations; useful for spotting nearly constant/baseline directions",
            "nearest_unembed_cosine": "largest cosine between the unit SV direction and any unembedding row",
            "farthest_unembed_cosine": "most negative cosine between the unit SV direction and any unembedding row",
            "max_abs_unembed_token_cosine": "max absolute token cosine; a rough token-likeness / lexical anchoring score",
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
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

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"  rankings: {rankings_path}")
    print(f"  contexts: {contexts_path}")
    if unembedding_geometry:
        print(f"  unembedding: {unembedding_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  directions: {out_dir / 'directions'}")


if __name__ == "__main__":
    main()
