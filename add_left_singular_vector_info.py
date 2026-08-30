#!/usr/bin/env python3
"""Add transported (left-SV) geometry to a J-Lens scan directory.

This is a post-processor for artifacts produced by:

* scan_unrealized_words.py
* scan_sparse_jlens_directions.py
* scan_cot_jlens_directions.py
* scan_predictive_jlens_directions.py

Those scanners project residual states onto right singular vectors ``v`` of a
J-Lens matrix ``J``.  This script reconstructs the direction on the output side
of the lens without scanning text or running the language model:

    transported = J @ v
    gain        = ||transported||
    u_exact     = transported / gain

For an exact SVD, ``gain == sigma`` and ``u_exact == u``.  For a randomized
low-rank SVD, ``J @ v`` is the most faithful output direction for the *saved*
right vector, while the saved U/S pair can differ slightly.  Both the actual
transport gain and the agreement with a stored U/S pair are reported.

The optional token analysis compares every reconstructed unit ``u_exact`` with
rows of the model's output embedding (lm_head) in chunks.  It reports cosine
neighbors (token-direction geometry) and raw dot-product neighbors (linear
lm-head score geometry).  Loading the model weights is necessary to obtain the
lm_head matrix, but ``forward``/``generate`` are never called.

Outputs (under --out-dir, which defaults to --scan-dir):

    left_singular_vectors.jsonl
    left_singular_vectors_metadata.json
    left_directions/LXX.npz

Typical dashboard run:

    python add_left_singular_vector_info.py \
      --scan-dir predictive_words_scan \
      --compare-scan-dir predictive_words_fineweb \
      --compare-scan-dir unrealized_words_fineweb \
      --compare-scan-dir predictive_cot_low \
      --compare-scan-dir predictive_cot_medium

Fast geometry-only run (no model weights/token neighbors):

    python add_left_singular_vector_info.py \
      --scan-dir cot_unrealized_low \
      --skip-token-geometry

If J-Lens is unavailable but the direction files contain U, ``--use-stored-u``
can build a provisional artifact.  It cannot validate ``J @ v = sigma * u``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"
DEFAULT_SCAN_DIR = "predictive_words_scan"


@dataclass
class LayerDirections:
    layer: int
    path: Path
    V: np.ndarray
    S: np.ndarray
    stored_U: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(DEFAULT_SCAN_DIR),
        help="Scanner output directory containing directions/LXX.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --scan-dir.",
    )
    parser.add_argument(
        "--compare-scan-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional scanner directory whose V/S basis is audited against "
            "--scan-dir. Repeat for multiple datasets."
        ),
    )
    parser.add_argument(
        "--require-shared-basis",
        action="store_true",
        help="Fail if any --compare-scan-dir does not have the exact same V/S bank.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated source layers or 'all'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Directions per layer; by default use every saved V/S column.",
    )

    parser.add_argument(
        "--lens-repo",
        default=None,
        help=f"J-Lens repository; defaults to scan metadata, then {DEFAULT_LENS_REPO}.",
    )
    parser.add_argument(
        "--lens-file",
        default=None,
        help=f"J-Lens filename; defaults to scan metadata, then {DEFAULT_LENS_FILE}.",
    )
    parser.add_argument(
        "--use-stored-u",
        action="store_true",
        help=(
            "Use U already stored in each NPZ instead of loading J-Lens and "
            "reconstructing J@V. Faster, but transport identity cannot be validated."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=f"Model/tokenizer; defaults to scan metadata, then {DEFAULT_MODEL}.",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map used only to load the model's output embedding.",
    )
    parser.add_argument(
        "--skip-token-geometry",
        action="store_true",
        help="Do not load tokenizer/model weights or calculate token neighbors.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=32,
        help="Cosine and dot-product token neighbors retained per side/direction.",
    )
    parser.add_argument(
        "--unembedding-chunk-size",
        type=int,
        default=4096,
        help="Vocabulary rows processed at once.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Allow tokenizer special tokens in token-neighbor results.",
    )
    parser.add_argument(
        "--right-neighbors",
        type=Path,
        default=None,
        help=(
            "Right-SV unembedding_neighbors.jsonl used for input/output token-overlap "
            "diagnostics. Defaults to SCAN_DIR/unembedding_neighbors.jsonl when present."
        ),
    )
    parser.add_argument(
        "--overlap-k",
        type=int,
        default=8,
        help="Top token count per side used for V-vs-U overlap diagnostics.",
    )
    args = parser.parse_args()
    if args.k is not None and args.k <= 0:
        parser.error("--k must be positive")
    if args.neighbors <= 0:
        parser.error("--neighbors must be positive")
    if args.unembedding_chunk_size <= 0:
        parser.error("--unembedding-chunk-size must be positive")
    if args.overlap_k <= 0:
        parser.error("--overlap-k must be positive")
    return args


def finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def parse_layer_number(path: Path) -> int:
    stem = path.stem
    if len(stem) < 2 or stem[0] != "L" or not stem[1:].isdigit():
        raise ValueError(f"Not an LXX direction file: {path}")
    return int(stem[1:])


def selected_layers(spec: str, available: Iterable[int]) -> list[int]:
    available_list = sorted(set(int(layer) for layer in available))
    if spec.strip().lower() == "all":
        return available_list
    wanted = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    missing = sorted(set(wanted) - set(available_list))
    if missing:
        raise ValueError(f"Requested layers {missing}; available layers are {available_list}")
    return wanted


def orient_columns(array: np.ndarray, d_model: int | None, name: str, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{path}: {name} must be 2D, got {array.shape}")
    if d_model is None:
        # Scanner files conventionally store [d_model, k], and d_model is much
        # larger than k. This also handles full square matrices without change.
        return array if array.shape[0] >= array.shape[1] else array.T
    if array.shape[0] == d_model:
        return array
    if array.shape[1] == d_model:
        return array.T
    raise ValueError(f"{path}: neither {name} axis matches d_model={d_model}: {array.shape}")


def load_direction_bank(scan_dir: Path, layers_spec: str, k: int | None) -> dict[int, LayerDirections]:
    direction_dir = scan_dir / "directions"
    paths = sorted(direction_dir.glob("L*.npz"))
    if not paths:
        raise FileNotFoundError(f"No LXX.npz files found under {direction_dir}")
    path_by_layer = {parse_layer_number(path): path for path in paths}
    layers = selected_layers(layers_spec, path_by_layer)
    result: dict[int, LayerDirections] = {}
    expected_d: int | None = None
    for layer in layers:
        path = path_by_layer[layer]
        with np.load(path) as saved:
            if "V" not in saved or "S" not in saved:
                raise ValueError(f"{path} must contain V and S")
            V = orient_columns(np.asarray(saved["V"]), expected_d, "V", path)
            expected_d = int(V.shape[0])
            S = np.asarray(saved["S"]).reshape(-1)
            U = (
                orient_columns(np.asarray(saved["U"]), expected_d, "U", path)
                if "U" in saved
                else None
            )
        use_k = min(V.shape[1], S.shape[0]) if k is None else k
        if V.shape[1] < use_k or S.shape[0] < use_k:
            raise ValueError(
                f"{path}: requested k={use_k}, V has {V.shape[1]} columns and S has {S.shape[0]}"
            )
        if U is not None and U.shape[1] < use_k:
            raise ValueError(f"{path}: stored U has only {U.shape[1]} columns; requested {use_k}")
        result[layer] = LayerDirections(
            layer=layer,
            path=path,
            V=np.ascontiguousarray(V[:, :use_k], dtype=np.float32),
            S=np.ascontiguousarray(S[:use_k], dtype=np.float32),
            stored_U=(
                None if U is None else np.ascontiguousarray(U[:, :use_k], dtype=np.float32)
            ),
        )
    dims = {item.V.shape[0] for item in result.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent d_model values across layers: {sorted(dims)}")
    return result


def update_array_digest(digest: Any, key: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(key.encode("ascii") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())


def basis_fingerprint(bank: dict[int, LayerDirections]) -> str:
    digest = hashlib.sha256()
    digest.update(b"jlens-vs-bank-v1\0")
    for layer, item in sorted(bank.items()):
        digest.update(f"L{layer:02d}\0".encode("ascii"))
        update_array_digest(digest, "V", item.V)
        update_array_digest(digest, "S", item.S)
    return digest.hexdigest()


def audit_other_bases(
    canonical: dict[int, LayerDirections],
    canonical_fingerprint: str,
    scan_dirs: list[Path],
) -> list[dict[str, Any]]:
    layers_spec = ",".join(str(layer) for layer in sorted(canonical))
    canonical_k = {item.V.shape[1] for item in canonical.values()}
    if len(canonical_k) != 1:
        raise ValueError("The canonical bank has varying k across layers")
    k = next(iter(canonical_k))
    audits: list[dict[str, Any]] = []
    for directory in scan_dirs:
        resolved = directory.expanduser().resolve()
        try:
            other = load_direction_bank(resolved, layers_spec, k)
            fingerprint = basis_fingerprint(other)
            exact = fingerprint == canonical_fingerprint
            error = None
        except Exception as exc:
            fingerprint = None
            exact = False
            error = f"{type(exc).__name__}: {exc}"
        audits.append(
            {
                "scan_dir": str(resolved),
                "basis_fingerprint_sha256": fingerprint,
                "exact_v_s_match": exact,
                "error": error,
            }
        )
    return audits


def unit_columns(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array.astype(np.float64), axis=0).astype(np.float32)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms <= 1e-12).tolist()
        raise ValueError(f"Zero-norm direction columns: {bad[:10]}")
    return np.ascontiguousarray(array / norms[None, :]), norms


def torch_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def reconstruct_left_bank(
    bank: dict[int, LayerDirections],
    lens_repo: str,
    lens_file: str,
    use_stored_u: bool,
) -> tuple[dict[int, np.ndarray], dict[int, list[dict[str, Any]]], str]:
    left: dict[int, np.ndarray] = {}
    diagnostics: dict[int, list[dict[str, Any]]] = {}

    lens = None
    if not use_stored_u:
        try:
            import jlens
        except ImportError as exc:
            raise SystemExit(
                "J-Lens is required to reconstruct J@V. Install the scan environment "
                "or rerun with --use-stored-u when every NPZ contains U."
            ) from exc
        print(f"[lens] loading {lens_repo} / {lens_file}")
        lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
        lens_layers = set(int(value) for value in lens.source_layers)
    else:
        lens_layers = set()

    for layer, item in sorted(bank.items()):
        V_unit, v_norms = unit_columns(item.V)
        if use_stored_u:
            if item.stored_U is None:
                raise ValueError(
                    f"{item.path} has no U. Use J-Lens reconstruction (omit --use-stored-u)."
                )
            U_exact, stored_u_norms = unit_columns(item.stored_U)
            transported_gain = item.S.astype(np.float32, copy=True)
            Jv = U_exact * transported_gain[None, :]
            reconstruction_method = "stored_svd_u_unvalidated"
        else:
            if layer not in lens_layers:
                raise ValueError(f"J-Lens has no source layer {layer}")
            J = torch_to_numpy(lens.jacobians[layer])
            if J.ndim != 2 or J.shape[1] != V_unit.shape[0]:
                raise ValueError(
                    f"L{layer:02d}: J shape {J.shape} cannot multiply V shape {V_unit.shape}"
                )
            Jv = np.asarray(J @ V_unit, dtype=np.float32)
            U_exact, transported_gain = unit_columns(Jv)
            stored_u_norms = (
                np.linalg.norm(item.stored_U.astype(np.float64), axis=0).astype(np.float32)
                if item.stored_U is not None
                else None
            )
            reconstruction_method = "normalized_jlens_times_saved_v"

        left[layer] = U_exact
        layer_rows: list[dict[str, Any]] = []
        for sv0 in range(V_unit.shape[1]):
            sigma = float(item.S[sv0])
            gain = float(transported_gain[sv0])
            row: dict[str, Any] = {
                "source_v_norm_before_normalization": float(v_norms[sv0]),
                "stored_singular_value": sigma,
                "actual_transport_gain": gain,
                "gain_over_stored_singular_value": finite_float(gain / sigma) if sigma else None,
                "left_right_coordinate_cosine": finite_float(
                    np.dot(U_exact[:, sv0].astype(np.float64), V_unit[:, sv0].astype(np.float64))
                ),
                "reconstruction_method": reconstruction_method,
                "stored_u_present": item.stored_U is not None,
            }
            if item.stored_U is not None:
                stored_u_unit, _ = unit_columns(item.stored_U[:, sv0 : sv0 + 1])
                target = sigma * stored_u_unit[:, 0]
                transported = Jv[:, sv0]
                denom = max(float(np.linalg.norm(target.astype(np.float64))), 1e-12)
                row.update(
                    {
                        "stored_u_norm_before_normalization": float(stored_u_norms[sv0]),
                        "transport_vs_stored_u_cosine": finite_float(
                            np.dot(
                                U_exact[:, sv0].astype(np.float64),
                                stored_u_unit[:, 0].astype(np.float64),
                            )
                        ),
                        "transport_vs_sigma_stored_u_relative_error": (
                            None
                            if use_stored_u
                            else finite_float(
                                np.linalg.norm((transported - target).astype(np.float64)) / denom
                            )
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "stored_u_norm_before_normalization": None,
                        "transport_vs_stored_u_cosine": None,
                        "transport_vs_sigma_stored_u_relative_error": None,
                    }
                )
            layer_rows.append(row)
        diagnostics[layer] = layer_rows
        print(
            f"[transport] L{layer:02d}: d={V_unit.shape[0]} k={V_unit.shape[1]} "
            f"gain=[{transported_gain.min():.4g}, {transported_gain.max():.4g}]"
        )

    del lens
    return left, diagnostics, reconstruction_method


def add_within_layer_diagnostics(
    left: dict[int, np.ndarray], diagnostics: dict[int, list[dict[str, Any]]]
) -> None:
    for layer, U in sorted(left.items()):
        gram = np.asarray(U.T @ U, dtype=np.float32)
        np.fill_diagonal(gram, 0.0)
        for sv0, row in enumerate(diagnostics[layer]):
            abs_col = np.abs(gram[:, sv0])
            match = int(np.argmax(abs_col)) if len(abs_col) else -1
            row["largest_abs_left_cosine_with_other_sv_same_layer"] = (
                finite_float(abs_col[match]) if match >= 0 else None
            )
            row["most_overlapping_left_sv_index_0_same_layer"] = match if match >= 0 else None
            row["most_overlapping_left_sv_cosine_same_layer"] = (
                finite_float(gram[match, sv0]) if match >= 0 else None
            )


def cross_layer_match(source: np.ndarray, target: np.ndarray, sv0: int) -> dict[str, Any]:
    sims = np.asarray(target.T @ source[:, sv0], dtype=np.float32)
    match = int(np.argmax(np.abs(sims)))
    return {
        "sv_index_0": match,
        "cosine": finite_float(sims[match]),
        "abs_cosine": finite_float(abs(sims[match])),
        "same_index_cosine": (
            finite_float(sims[sv0]) if sv0 < sims.shape[0] else None
        ),
    }


def add_adjacent_layer_diagnostics(
    left: dict[int, np.ndarray], diagnostics: dict[int, list[dict[str, Any]]]
) -> None:
    layers = sorted(left)
    for position, layer in enumerate(layers):
        for sv0, row in enumerate(diagnostics[layer]):
            row["previous_layer_left_match"] = None
            row["next_layer_left_match"] = None
            if position > 0:
                other = layers[position - 1]
                row["previous_layer_left_match"] = {
                    "layer": other,
                    **cross_layer_match(left[layer], left[other], sv0),
                }
            if position + 1 < len(layers):
                other = layers[position + 1]
                row["next_layer_left_match"] = {
                    "layer": other,
                    **cross_layer_match(left[layer], left[other], sv0),
                }


def load_right_neighbors(path: Path | None) -> dict[str, dict[str, list[int]]]:
    if path is None or not path.is_file():
        return {}
    output: dict[str, dict[str, list[int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            candidate = str(raw.get("candidate") or "")
            if not candidate:
                raise ValueError(f"{path}:{line_number}: missing candidate")
            output[candidate] = {
                "positive": [int(item["token_id"]) for item in raw.get("nearest_tokens", [])],
                "negative": [int(item["token_id"]) for item in raw.get("farthest_tokens", [])],
            }
    return output


def token_record(tokenizer: Any, token_id: int, cosine: float, dot: float) -> dict[str, Any]:
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token = None
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
        "token": token,
        "decoded": decoded,
        "cosine": finite_float(cosine),
        "dot_product": finite_float(dot),
        "is_special": int(token_id) in set(int(x) for x in tokenizer.all_special_ids),
    }


def merge_extrema(
    running_values: Any,
    running_ids: Any,
    chunk_values: Any,
    start: int,
    keep: int,
    largest: bool,
) -> tuple[Any, Any]:
    import torch

    local_k = min(keep, int(chunk_values.shape[0]))
    values, local_ids = torch.topk(
        chunk_values, k=local_k, dim=0, largest=largest, sorted=False
    )
    local_ids = local_ids + start
    merged_values = torch.cat([running_values.to(values.device), values], dim=0)
    merged_ids = torch.cat([running_ids.to(values.device), local_ids], dim=0)
    selected_values, selected_positions = torch.topk(
        merged_values, k=keep, dim=0, largest=largest, sorted=False
    )
    selected_ids = torch.gather(merged_ids, 0, selected_positions)
    return selected_values.cpu(), selected_ids.cpu()


def analyze_token_geometry(
    *,
    left: dict[int, np.ndarray],
    model_name: str,
    model_revision: str | None,
    device_map: str,
    n_neighbors: int,
    chunk_size: int,
    include_special_tokens: bool,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Token geometry requires torch and transformers. Use the same environment as "
            "the scanners, or pass --skip-token-geometry."
        ) from exc

    print(f"[tokenizer] loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    print(f"[model] loading output-embedding owner {model_name}; no forward pass will run")
    load_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if device_map.lower() not in {"none", "null", ""}:
        load_kwargs["device_map"] = device_map
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError as exc:
        # Older Transformers releases named this kwarg torch_dtype. Keep the
        # post-processor usable in the environments that produced older scans.
        if "dtype" not in str(exc):
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    hf_model.eval()
    output_embeddings = hf_model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose get_output_embeddings().weight")
    W = output_embeddings.weight
    if W.ndim != 2:
        raise ValueError(f"Expected a 2D output embedding, got {tuple(W.shape)}")

    keys: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    for layer, U in sorted(left.items()):
        columns.append(U)
        keys.extend((layer, sv0) for sv0 in range(U.shape[1]))
    all_u_np = np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float32)
    if W.shape[1] != all_u_np.shape[0]:
        raise ValueError(
            f"Output embedding d_model={W.shape[1]} but left vectors have d={all_u_np.shape[0]}"
        )

    vocab_size, candidate_count = int(W.shape[0]), int(all_u_np.shape[1])
    keep = min(n_neighbors, vocab_size)
    device = W.device
    all_u = torch.from_numpy(all_u_np).to(device=device, dtype=torch.float32)
    neg_inf = -float("inf")
    pos_inf = float("inf")
    cos_top_values = torch.full((keep, candidate_count), neg_inf, dtype=torch.float32)
    cos_top_ids = torch.full((keep, candidate_count), -1, dtype=torch.long)
    cos_bottom_values = torch.full((keep, candidate_count), pos_inf, dtype=torch.float32)
    cos_bottom_ids = torch.full((keep, candidate_count), -1, dtype=torch.long)
    dot_top_values = torch.full((keep, candidate_count), neg_inf, dtype=torch.float32)
    dot_top_ids = torch.full((keep, candidate_count), -1, dtype=torch.long)
    dot_bottom_values = torch.full((keep, candidate_count), pos_inf, dtype=torch.float32)
    dot_bottom_ids = torch.full((keep, candidate_count), -1, dtype=torch.long)
    cos_sum = torch.zeros(candidate_count, dtype=torch.float64)
    cos_sq_sum = torch.zeros(candidate_count, dtype=torch.float64)
    dot_sum = torch.zeros(candidate_count, dtype=torch.float64)
    dot_sq_sum = torch.zeros(candidate_count, dtype=torch.float64)
    valid_count = 0
    special_ids = set(int(value) for value in tokenizer.all_special_ids)

    print(
        f"[tokens] vocab={vocab_size:,} d_model={W.shape[1]} "
        f"directions={candidate_count:,} device={device}"
    )
    with torch.inference_mode():
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            rows = W[start:end].detach().to(device=device, dtype=torch.float32)
            dots = rows @ all_u
            cosines = dots / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if not include_special_tokens and special_ids:
                local_special = [token_id - start for token_id in special_ids if start <= token_id < end]
                if local_special:
                    indices = torch.tensor(local_special, device=device, dtype=torch.long)
                    cosines.index_fill_(0, indices, float("nan"))
                    dots.index_fill_(0, indices, float("nan"))
            finite = torch.isfinite(cosines) & torch.isfinite(dots)
            safe_cos = torch.where(finite, cosines, torch.zeros_like(cosines))
            safe_dot = torch.where(finite, dots, torch.zeros_like(dots))
            cos_sum += safe_cos.sum(dim=0).cpu().double()
            cos_sq_sum += safe_cos.square().sum(dim=0).cpu().double()
            dot_sum += safe_dot.sum(dim=0).cpu().double()
            dot_sq_sum += safe_dot.square().sum(dim=0).cpu().double()
            if candidate_count:
                valid_count += int(finite[:, 0].sum().item())

            cos_top_values, cos_top_ids = merge_extrema(
                cos_top_values,
                cos_top_ids,
                torch.where(finite, cosines, torch.full_like(cosines, neg_inf)),
                start,
                keep,
                True,
            )
            cos_bottom_values, cos_bottom_ids = merge_extrema(
                cos_bottom_values,
                cos_bottom_ids,
                torch.where(finite, cosines, torch.full_like(cosines, pos_inf)),
                start,
                keep,
                False,
            )
            dot_top_values, dot_top_ids = merge_extrema(
                dot_top_values,
                dot_top_ids,
                torch.where(finite, dots, torch.full_like(dots, neg_inf)),
                start,
                keep,
                True,
            )
            dot_bottom_values, dot_bottom_ids = merge_extrema(
                dot_bottom_values,
                dot_bottom_ids,
                torch.where(finite, dots, torch.full_like(dots, pos_inf)),
                start,
                keep,
                False,
            )
            del rows, dots, cosines, finite, safe_cos, safe_dot
            if start == 0 or end == vocab_size or (start // chunk_size) % 10 == 0:
                print(f"[tokens] {end:,}/{vocab_size:,}", flush=True)

    def sort_extrema(values: Any, ids: Any, descending: bool) -> tuple[Any, Any]:
        order = torch.argsort(values, dim=0, descending=descending)
        return torch.gather(values, 0, order), torch.gather(ids, 0, order)

    cos_top_values, cos_top_ids = sort_extrema(cos_top_values, cos_top_ids, True)
    cos_bottom_values, cos_bottom_ids = sort_extrema(cos_bottom_values, cos_bottom_ids, False)
    dot_top_values, dot_top_ids = sort_extrema(dot_top_values, dot_top_ids, True)
    dot_bottom_values, dot_bottom_ids = sort_extrema(dot_bottom_values, dot_bottom_ids, False)

    denom = max(valid_count, 1)
    cos_mean = cos_sum.numpy() / denom
    cos_std = np.sqrt(np.maximum(cos_sq_sum.numpy() / denom - cos_mean**2, 0.0))
    dot_mean = dot_sum.numpy() / denom
    dot_std = np.sqrt(np.maximum(dot_sq_sum.numpy() / denom - dot_mean**2, 0.0))

    # Token records include both values irrespective of which metric selected
    # them. Gather the companion metric directly from W for the retained ids.
    def companion_values(ids: Any) -> tuple[Any, Any]:
        # Chunk across directions. Materializing K*C full d_model rows can be
        # hundreds of MB even though only K scalar companions are needed.
        rows_per_direction, n_columns = int(ids.shape[0]), int(ids.shape[1])
        all_cos = torch.empty((rows_per_direction, n_columns), dtype=torch.float32)
        all_dot = torch.empty((rows_per_direction, n_columns), dtype=torch.float32)
        column_batch = 32
        for column_start in range(0, n_columns, column_batch):
            column_end = min(n_columns, column_start + column_batch)
            ids_part = ids[:, column_start:column_end]
            flat = ids_part.T.reshape(-1).to(device)
            rows = W.index_select(0, flat).detach().to(device=device, dtype=torch.float32)
            repeated_u = all_u[:, column_start:column_end].T.repeat_interleave(
                rows_per_direction, dim=0
            )
            dots = (rows * repeated_u).sum(dim=1)
            cos = dots / rows.norm(dim=1).clamp_min(1e-12)
            shape = (column_end - column_start, rows_per_direction)
            all_cos[:, column_start:column_end] = cos.reshape(shape).T.cpu()
            all_dot[:, column_start:column_end] = dots.reshape(shape).T.cpu()
            del flat, rows, repeated_u, dots, cos
        return all_cos, all_dot

    cos_top_cos, cos_top_dot = companion_values(cos_top_ids)
    cos_bottom_cos, cos_bottom_dot = companion_values(cos_bottom_ids)
    dot_top_cos, dot_top_dot = companion_values(dot_top_ids)
    dot_bottom_cos, dot_bottom_dot = companion_values(dot_bottom_ids)

    results: dict[tuple[int, int], dict[str, Any]] = {}
    for column, key in enumerate(keys):
        def records(ids: Any, cos: Any, dot: Any) -> list[dict[str, Any]]:
            return [
                token_record(
                    tokenizer,
                    int(ids[row, column]),
                    float(cos[row, column]),
                    float(dot[row, column]),
                )
                for row in range(ids.shape[0])
                if int(ids[row, column]) >= 0
            ]

        nearest = records(cos_top_ids, cos_top_cos, cos_top_dot)
        farthest = records(cos_bottom_ids, cos_bottom_cos, cos_bottom_dot)
        highest_dot = records(dot_top_ids, dot_top_cos, dot_top_dot)
        lowest_dot = records(dot_bottom_ids, dot_bottom_cos, dot_bottom_dot)
        max_abs = max(
            abs(float(nearest[0]["cosine"])) if nearest else 0.0,
            abs(float(farthest[0]["cosine"])) if farthest else 0.0,
        )
        results[key] = {
            "nearest_tokens": nearest,
            "farthest_tokens": farthest,
            "highest_dot_product_tokens": highest_dot,
            "lowest_dot_product_tokens": lowest_dot,
            "vocab_rows_considered": valid_count,
            "cosine_mean": finite_float(cos_mean[column]),
            "cosine_std": finite_float(cos_std[column]),
            "dot_product_mean": finite_float(dot_mean[column]),
            "dot_product_std": finite_float(dot_std[column]),
            "max_abs_token_cosine": finite_float(max_abs),
            "nearest_token_z": (
                finite_float((float(nearest[0]["cosine"]) - cos_mean[column]) / max(cos_std[column], 1e-12))
                if nearest
                else None
            ),
            "farthest_token_z": (
                finite_float((float(farthest[0]["cosine"]) - cos_mean[column]) / max(cos_std[column], 1e-12))
                if farthest
                else None
            ),
        }

    meta = {
        "available": True,
        "model": model_name,
        "model_revision": model_revision,
        "space": "raw_lm_head_row_vs_normalized_J_times_V",
        "vocab_rows_considered": valid_count,
        "neighbors_per_side": keep,
        "special_tokens_included": include_special_tokens,
        "forward_passes": 0,
    }
    del all_u, hf_model
    return results, meta


def overlap_summary(
    right: dict[str, list[int]] | None,
    left_geometry: dict[str, Any] | None,
    overlap_k: int,
) -> dict[str, Any] | None:
    if not right or not left_geometry:
        return None
    left = {
        "positive": [int(item["token_id"]) for item in left_geometry["nearest_tokens"]],
        "negative": [int(item["token_id"]) for item in left_geometry["farthest_tokens"]],
    }

    def comparison(a: list[int], b: list[int]) -> dict[str, Any]:
        aset, bset = set(a[:overlap_k]), set(b[:overlap_k])
        union = aset | bset
        return {
            "count": len(aset & bset),
            "jaccard": finite_float(len(aset & bset) / len(union)) if union else None,
            "token_ids": sorted(aset & bset),
        }

    return {
        "k": overlap_k,
        "positive_to_positive": comparison(right["positive"], left["positive"]),
        "negative_to_negative": comparison(right["negative"], left["negative"]),
        "positive_to_negative": comparison(right["positive"], left["negative"]),
        "negative_to_positive": comparison(right["negative"], left["positive"]),
    }


def write_outputs(
    *,
    out_dir: Path,
    scan_dir: Path,
    bank: dict[int, LayerDirections],
    left: dict[int, np.ndarray],
    diagnostics: dict[int, list[dict[str, Any]]],
    token_geometry: dict[tuple[int, int], dict[str, Any]],
    token_meta: dict[str, Any],
    right_neighbors: dict[str, dict[str, list[int]]],
    overlap_k: int,
    fingerprint: str,
    basis_audits: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    left_dir = out_dir / "left_directions"
    left_dir.mkdir(parents=True, exist_ok=True)

    for layer, U in sorted(left.items()):
        gains = np.asarray(
            [row["actual_transport_gain"] for row in diagnostics[layer]], dtype=np.float32
        )
        output_path = left_dir / f"L{layer:02d}.npz"
        temp_path = output_path.with_suffix(".npz.tmp")
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                U=U.astype(np.float32, copy=False),
                transport_gain=gains,
                stored_S=bank[layer].S,
            )
        os.replace(temp_path, output_path)

    jsonl_path = out_dir / "left_singular_vectors.jsonl"
    jsonl_temp = jsonl_path.with_suffix(".jsonl.tmp")
    with jsonl_temp.open("w", encoding="utf-8") as handle:
        for layer, item in sorted(bank.items()):
            for sv0 in range(item.V.shape[1]):
                candidate = f"L{layer:02d}_SV{sv0 + 1:02d}"
                geometry = token_geometry.get((layer, sv0))
                record = {
                    "candidate": candidate,
                    "layer": layer,
                    "sv_index_0": sv0,
                    "sv_rank_1based": sv0 + 1,
                    "basis_fingerprint_sha256": fingerprint,
                    **diagnostics[layer][sv0],
                    "left_token_geometry": geometry,
                    "right_left_token_overlap": overlap_summary(
                        right_neighbors.get(candidate), geometry, overlap_k
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(jsonl_temp, jsonl_path)

    metadata_path = out_dir / "left_singular_vectors_metadata.json"
    source_metadata: dict[str, Any] = {}
    scan_metadata_path = scan_dir / "metadata.json"
    if scan_metadata_path.is_file():
        source_metadata = json.loads(scan_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "source_scan_dir": str(scan_dir),
        "source_scanner_family": (
            "scan_cot_jlens_directions / scan_predictive_jlens_directions / "
            "scan_sparse_jlens_directions / scan_unrealized_words"
        ),
        "model": source_metadata.get("model") or args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "layers": sorted(bank),
        "directions_per_layer": {str(layer): bank[layer].V.shape[1] for layer in sorted(bank)},
        "d_model": next(iter(bank.values())).V.shape[0],
        "candidate_count": sum(item.V.shape[1] for item in bank.values()),
        "basis_fingerprint_algorithm": "sha256(jlens-vs-bank-v1, layer, raw float32 V/S)",
        "basis_fingerprint_sha256": fingerprint,
        "basis_audits": basis_audits,
        "reconstruction": {
            "definition": "u_exact = (J @ v) / ||J @ v||",
            "gain_definition": "actual_transport_gain = ||J @ v||",
            "used_stored_u_without_jlens": bool(args.use_stored_u),
            "forward_passes": 0,
            "stored_u_note": (
                "Stored U is compared with normalized J@V when present. Randomized SVD "
                "can make stored U/S approximate for an individual saved V."
            ),
        },
        "token_geometry": token_meta,
        "right_neighbor_artifact": (
            str(args.right_neighbors) if args.right_neighbors is not None else None
        ),
        "outputs": {
            "records": str(jsonl_path),
            "left_arrays": str(left_dir),
        },
    }
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temp, metadata_path)
    return jsonl_path, metadata_path, left_dir


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir.expanduser().resolve()
    out_dir = (args.out_dir or args.scan_dir).expanduser().resolve()
    scan_metadata_path = scan_dir / "metadata.json"
    scan_metadata = (
        json.loads(scan_metadata_path.read_text(encoding="utf-8"))
        if scan_metadata_path.is_file()
        else {}
    )
    args.model = args.model or scan_metadata.get("model") or DEFAULT_MODEL
    args.lens_repo = args.lens_repo or scan_metadata.get("lens_repo") or DEFAULT_LENS_REPO
    args.lens_file = args.lens_file or scan_metadata.get("lens_file") or DEFAULT_LENS_FILE
    args.compare_scan_dir = [path.expanduser().resolve() for path in args.compare_scan_dir]
    if args.right_neighbors is None:
        possible = scan_dir / "unembedding_neighbors.jsonl"
        args.right_neighbors = possible if possible.is_file() else None
    elif args.right_neighbors is not None:
        args.right_neighbors = args.right_neighbors.expanduser().resolve()

    print(f"[scan] loading direction bank from {scan_dir}")
    bank = load_direction_bank(scan_dir, args.layers, args.k)
    fingerprint = basis_fingerprint(bank)
    print(
        f"[basis] layers={len(bank)} candidates={sum(item.V.shape[1] for item in bank.values()):,} "
        f"fingerprint={fingerprint}"
    )
    audits = audit_other_bases(bank, fingerprint, args.compare_scan_dir)
    for audit in audits:
        status = "EXACT" if audit["exact_v_s_match"] else "DIFFERENT"
        suffix = f" ({audit['error']})" if audit["error"] else ""
        print(f"[basis] {status}: {audit['scan_dir']}{suffix}")
    if args.require_shared_basis and any(not audit["exact_v_s_match"] for audit in audits):
        raise SystemExit("At least one compared scan directory does not share the exact V/S bank")

    left, diagnostics, _ = reconstruct_left_bank(
        bank,
        lens_repo=args.lens_repo,
        lens_file=args.lens_file,
        use_stored_u=args.use_stored_u,
    )
    add_within_layer_diagnostics(left, diagnostics)
    add_adjacent_layer_diagnostics(left, diagnostics)

    if args.skip_token_geometry:
        token_geometry: dict[tuple[int, int], dict[str, Any]] = {}
        token_meta = {
            "available": False,
            "reason": "--skip-token-geometry",
            "forward_passes": 0,
        }
    else:
        token_geometry, token_meta = analyze_token_geometry(
            left=left,
            model_name=args.model,
            model_revision=args.model_revision,
            device_map=args.device_map,
            n_neighbors=args.neighbors,
            chunk_size=args.unembedding_chunk_size,
            include_special_tokens=args.include_special_tokens,
        )

    right_neighbors = load_right_neighbors(args.right_neighbors)
    jsonl_path, metadata_path, left_dir = write_outputs(
        out_dir=out_dir,
        scan_dir=scan_dir,
        bank=bank,
        left=left,
        diagnostics=diagnostics,
        token_geometry=token_geometry,
        token_meta=token_meta,
        right_neighbors=right_neighbors,
        overlap_k=args.overlap_k,
        fingerprint=fingerprint,
        basis_audits=audits,
        args=args,
    )
    print("\nDone. No corpus scan, model forward, or generation was run.")
    print(f"  records: {jsonl_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  arrays: {left_dir}")


if __name__ == "__main__":
    main()
