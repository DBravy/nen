#!/usr/bin/env python3
"""
analyze_two_sv_trajectories.py

Focused trajectory analysis for the two zero-indexed J-Lens right singular
vectors the user identified:

    SV_02 : default window L2..L11
    SV_09 : default window L9..L15

The point is to distinguish two geometries that can look identical if we only
ask whether adjacent layers preserve the same SV rank:

  1) a nearly fixed/persistent axis (high adjacent AND high endpoint retention)
  2) a continuously rotating axis (high adjacent but low long-range retention)

IMPORTANT INDEXING CONVENTION
-----------------------------
This script is ZERO-INDEXED throughout.

    SV_02 == column 2 of V
    SV_09 == column 9 of V

The earlier scanner, scan_unrealized_words_with_unembedding.py, saved explicit
fields `sv_index_0` and `sv_rank_1based`, but its human-readable `candidate`
string was ONE-BASED:

    zero-index SV_02 -> scanner candidate Lxx_SV03
    zero-index SV_09 -> scanner candidate Lxx_SV10

When this script reads unembedding_neighbors.jsonl it uses `sv_index_0` when
available and otherwise converts the scanner's one-based candidate/rank. It
never interprets the candidate suffix as zero-indexed unless explicitly told
via --lexical-labeling zero.

EXPECTED DIRECTION INPUT
------------------------
A directory containing the direction banks produced by the scanner:

    directions/
      L00.npz
      L01.npz
      ...

Each file must contain:
    V : right singular vectors, [d_model, n_sv] or [n_sv, d_model]
    S : singular values, [n_sv]

OPTIONAL LEXICAL INPUT
----------------------
Pass the scanner's unembedding_neighbors.jsonl with --unembedding-neighbors.
The script then quantifies how much the positive/negative token neighborhoods
are retained after canonicalizing the arbitrary SVD sign along each chain.
Because that JSONL stores only the retained top/bottom token lists rather than
the full vocabulary cosine profile, the lexical statistic is a top-N overlap
(Jaccard), not a full-vocabulary correlation.

OUTPUTS
-------
OUT/
  trajectory_steps.csv
      adjacent-layer geometry: raw sign, |cos|, angle, reciprocal best match,
      rank drift, runner-up margin, singular values, spectral gaps

  trajectory_layers.csv
      per-layer state relative to the trajectory anchor: anchor retention,
      cumulative path rotation, path-vs-displacement ratio, canonical sign,
      singular value and spectral gap

  pairwise_retention.csv
      every pair within each trajectory window; this is the key file for
      comparing adjacent continuity with long-range endpoint retention

  retention_by_distance.csv
      mean/median/min/max |cos| as a function of layer separation

  greedy_identity_trace.csv
      a best-absolute-cosine identity trace through the requested window,
      allowing the SV rank to move; useful for detecting swaps

  lexical_pairwise_retention.csv   (if lexical JSONL supplied)
      canonicalized positive/negative top-token Jaccard overlaps for all pairs

  lexical_layers.csv               (if lexical JSONL supplied)
      readable canonicalized token neighborhoods for each layer

  summary.md
      concise comparison of SV_02 and SV_09

  plots/
      pairwise cosine heatmaps, anchor-retention curves, step-angle curves,
      cumulative path-vs-displacement curves, singular-value curves, and
      lexical-retention curves when available

EXAMPLE
-------
Geometry only:

    python analyze_two_sv_trajectories.py \
        --directions-dir unrealized_words_fineweb/directions \
        --out two_sv_trajectories

Geometry + lexical-neighborhood retention:

    python analyze_two_sv_trajectories.py \
        --directions-dir unrealized_words_fineweb/directions \
        --unembedding-neighbors unrealized_words_fineweb/unembedding_neighbors.jsonl \
        --out two_sv_trajectories

Extend SV_09 far enough to inspect its later rank slip:

    python analyze_two_sv_trajectories.py \
        --directions-dir unrealized_words_fineweb/directions \
        --sv09-layers 9-18 \
        --out two_sv_trajectories_extended
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Focused zero-indexed trajectory analysis for J-Lens SV_02 and SV_09."
        )
    )
    p.add_argument(
        "--directions-dir",
        type=Path,
        required=True,
        help="Directory containing LXX.npz files with V and S.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("two_sv_trajectories"),
        help="Output directory.",
    )
    p.add_argument(
        "--sv02-layers",
        default="2-11",
        help="Inclusive layer window for zero-index SV_02. Default: 2-11.",
    )
    p.add_argument(
        "--sv09-layers",
        default="9-15",
        help="Inclusive layer window for zero-index SV_09. Default: 9-15.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=64,
        help="Number of SVs used when testing best/reciprocal matches. Default: 64.",
    )
    p.add_argument(
        "--unembedding-neighbors",
        type=Path,
        default=None,
        help="Optional scanner unembedding_neighbors.jsonl for lexical-overlap analysis.",
    )
    p.add_argument(
        "--lexical-labeling",
        choices=("auto", "scanner-one", "zero"),
        default="auto",
        help=(
            "How to interpret lexical JSONL if explicit sv_index_0 is absent. "
            "'scanner-one' means Lxx_SV03 is zero-index SV_02. Default: auto."
        ),
    )
    p.add_argument(
        "--lexical-top-n",
        type=int,
        default=32,
        help="Use at most this many nearest/farthest token IDs per direction. Default: 32.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


LAYER_RE = re.compile(r"^L(\d+)\.npz$")
CANDIDATE_RE = re.compile(r"^L(\d+)_SV(\d+)$")


@dataclass
class LayerBank:
    layer: int
    V: np.ndarray  # [d_model, n_sv], unit columns
    S: np.ndarray  # [n_sv]


@dataclass(frozen=True)
class TrajectorySpec:
    sv0: int
    layers: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"SV_{self.sv0:02d}"



def parse_layer_window(spec: str) -> tuple[int, ...]:
    """Parse '2-11', '2:11', or comma-separated layer integers."""
    s = spec.strip()
    for sep in ("-", ":"):
        if sep in s and "," not in s:
            a, b = s.split(sep, 1)
            a, b = int(a.strip()), int(b.strip())
            if b < a:
                raise ValueError(f"Invalid decreasing layer range: {spec}")
            return tuple(range(a, b + 1))
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(vals) < 2:
        raise ValueError(f"Need >=2 layers, got: {spec}")
    return vals



def discover_files(directory: Path) -> dict[int, Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    found: dict[int, Path] = {}
    for path in directory.glob("L*.npz"):
        m = LAYER_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path
    if not found:
        raise FileNotFoundError(f"No LXX.npz files found in {directory}")
    return dict(sorted(found.items()))



def infer_v_orientation(V: np.ndarray, S: np.ndarray, path: Path) -> np.ndarray:
    if V.ndim != 2:
        raise ValueError(f"{path}: V must be 2D, got {V.shape}")
    n_s = int(S.size)
    if V.shape[1] == n_s:
        return V
    if V.shape[0] == n_s:
        return V.T
    if V.shape[1] < V.shape[0] and V.shape[1] >= n_s:
        return V
    if V.shape[0] < V.shape[1] and V.shape[0] >= n_s:
        return V.T
    raise ValueError(
        f"{path}: cannot infer V orientation from V.shape={V.shape}, len(S)={n_s}"
    )



def load_banks(directory: Path, needed_layers: Iterable[int]) -> dict[int, LayerBank]:
    files = discover_files(directory)
    needed = sorted(set(int(x) for x in needed_layers))
    missing = [x for x in needed if x not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing direction-bank layers {missing}. Available={sorted(files)}"
        )

    banks: dict[int, LayerBank] = {}
    d_model = None
    for layer in needed:
        path = files[layer]
        with np.load(path) as z:
            if "V" not in z or "S" not in z:
                raise KeyError(f"{path}: expected arrays V and S")
            V = np.asarray(z["V"], dtype=np.float64)
            S = np.asarray(z["S"], dtype=np.float64).reshape(-1)
        V = infer_v_orientation(V, S, path)
        n = np.linalg.norm(V, axis=0, keepdims=True)
        if np.any(n <= 1e-12):
            raise ValueError(f"{path}: found zero-length singular vector")
        V = V / n
        if d_model is None:
            d_model = V.shape[0]
        elif V.shape[0] != d_model:
            raise ValueError(
                f"d_model mismatch: layer {layer} has {V.shape[0]}, expected {d_model}"
            )
        banks[layer] = LayerBank(layer=layer, V=V, S=S)
    return banks


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------



def clamp_cos(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))



def axis_angle_deg(abs_cos: float) -> float:
    """Angle between unoriented axes, in [0, 90] degrees."""
    return math.degrees(math.acos(float(np.clip(abs_cos, 0.0, 1.0))))



def vector_angle_deg(cos: float) -> float:
    return math.degrees(math.acos(clamp_cos(cos)))



def local_gap(S: np.ndarray, i: int) -> tuple[float, float, float, float]:
    """
    Return (gap_up, gap_down, min_abs_gap, min_relative_gap).

    SVD convention has S descending. For boundaries, missing side is inf so the
    existing neighbor controls the minimum.
    """
    s = float(S[i])
    up = float(S[i - 1] - S[i]) if i > 0 else float("inf")
    down = float(S[i] - S[i + 1]) if i + 1 < len(S) else float("inf")
    g = min(up, down)
    rel = g / max(abs(s), 1e-12)
    return up, down, g, rel



def match_vector_to_layer(
    v: np.ndarray, target: LayerBank, k: int
) -> dict[str, Any]:
    kk = min(k, target.V.shape[1])
    signed = v @ target.V[:, :kk]
    absolute = np.abs(signed)
    order = np.argsort(-absolute)
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return {
        "best_rank_0": best,
        "best_signed_cosine": float(signed[best]),
        "best_abs_cosine": float(absolute[best]),
        "second_rank_0": second,
        "second_abs_cosine": float(absolute[second]),
        "best_margin": float(absolute[best] - absolute[second]) if len(order) > 1 else float("nan"),
    }



def same_rank_reciprocal_match(
    a: LayerBank, b: LayerBank, sv0: int, k: int
) -> dict[str, Any]:
    va = a.V[:, sv0]
    vb = b.V[:, sv0]
    raw = float(va @ vb)
    ab = match_vector_to_layer(va, b, k)
    ba = match_vector_to_layer(vb, a, k)
    return {
        "raw_same_rank_cosine": raw,
        "same_rank_abs_cosine": abs(raw),
        "same_rank_axis_angle_deg": axis_angle_deg(abs(raw)),
        "raw_sign_flip": bool(raw < 0),
        "forward_best_rank_0": ab["best_rank_0"],
        "forward_best_abs_cosine": ab["best_abs_cosine"],
        "forward_second_rank_0": ab["second_rank_0"],
        "forward_second_abs_cosine": ab["second_abs_cosine"],
        "forward_margin": ab["best_margin"],
        "backward_best_rank_0": ba["best_rank_0"],
        "backward_best_abs_cosine": ba["best_abs_cosine"],
        "backward_second_rank_0": ba["second_rank_0"],
        "backward_second_abs_cosine": ba["second_abs_cosine"],
        "backward_margin": ba["best_margin"],
        "same_rank_is_forward_best": bool(ab["best_rank_0"] == sv0),
        "same_rank_is_backward_best": bool(ba["best_rank_0"] == sv0),
        "same_rank_is_reciprocal_best": bool(
            ab["best_rank_0"] == sv0 and ba["best_rank_0"] == sv0
        ),
        "same_rank_fraction_of_forward_best": float(
            abs(raw) / max(ab["best_abs_cosine"], 1e-12)
        ),
    }



def canonical_signs(banks: dict[int, LayerBank], spec: TrajectorySpec) -> dict[int, int]:
    """
    Parallel-transport a sign convention along the FIXED-RANK chain.

    signs[layer] multiplies the raw SVD vector. The first layer is +1; each
    subsequent sign is chosen so its canonical vector has positive dot product
    with the previous canonical vector.
    """
    signs: dict[int, int] = {spec.layers[0]: 1}
    prev_layer = spec.layers[0]
    prev_v = banks[prev_layer].V[:, spec.sv0].copy()
    for layer in spec.layers[1:]:
        raw_v = banks[layer].V[:, spec.sv0]
        s = 1 if float(prev_v @ raw_v) >= 0 else -1
        signs[layer] = s
        prev_v = s * raw_v
        prev_layer = layer
    return signs



def analyze_trajectory_geometry(
    banks: dict[int, LayerBank], spec: TrajectorySpec, k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (step_rows, layer_rows, pair_rows, greedy_rows)."""
    sv0 = spec.sv0
    layers = list(spec.layers)

    for l in layers:
        if sv0 >= banks[l].V.shape[1] or sv0 >= len(banks[l].S):
            raise ValueError(
                f"{spec.label}: layer {l} does not contain zero-index column {sv0}"
            )

    signs = canonical_signs(banks, spec)
    anchor_layer = layers[0]
    anchor_raw = banks[anchor_layer].V[:, sv0]
    anchor_can = signs[anchor_layer] * anchor_raw

    # Adjacent steps.
    step_rows: list[dict[str, Any]] = []
    cumulative_path = 0.0
    cumulative_by_layer: dict[int, float] = {anchor_layer: 0.0}
    for la, lb in zip(layers[:-1], layers[1:]):
        a, b = banks[la], banks[lb]
        m = same_rank_reciprocal_match(a, b, sv0, k)
        cumulative_path += m["same_rank_axis_angle_deg"]
        cumulative_by_layer[lb] = cumulative_path
        _, _, ga, gra = local_gap(a.S, sv0)
        _, _, gb, grb = local_gap(b.S, sv0)
        can_cos = float((signs[la] * a.V[:, sv0]) @ (signs[lb] * b.V[:, sv0]))
        step_rows.append(
            {
                "trajectory": spec.label,
                "sv_index_0": sv0,
                "layer_from": la,
                "layer_to": lb,
                "layer_delta": lb - la,
                **m,
                "canonical_same_rank_cosine": can_cos,
                "canonical_angle_deg": vector_angle_deg(can_cos),
                "sigma_from": float(a.S[sv0]),
                "sigma_to": float(b.S[sv0]),
                "sigma_change": float(b.S[sv0] - a.S[sv0]),
                "rel_local_gap_from": gra,
                "rel_local_gap_to": grb,
                "min_rel_local_gap_pair": min(gra, grb),
                "cumulative_path_angle_deg_at_to": cumulative_path,
            }
        )

    # Per-layer anchor retention / path-vs-displacement.
    layer_rows: list[dict[str, Any]] = []
    for l in layers:
        bank = banks[l]
        raw_v = bank.V[:, sv0]
        can_v = signs[l] * raw_v
        raw_anchor_cos = float(anchor_raw @ raw_v)
        can_anchor_cos = float(anchor_can @ can_v)
        abs_anchor_cos = abs(raw_anchor_cos)
        displacement = axis_angle_deg(abs_anchor_cos)
        path = cumulative_by_layer[l]
        up, down, gap, rel = local_gap(bank.S, sv0)
        layer_rows.append(
            {
                "trajectory": spec.label,
                "sv_index_0": sv0,
                "layer": l,
                "canonical_sign_multiplier": signs[l],
                "sigma": float(bank.S[sv0]),
                "gap_to_prev_sigma": up,
                "gap_to_next_sigma": down,
                "min_local_gap": gap,
                "relative_local_gap": rel,
                "raw_anchor_cosine": raw_anchor_cos,
                "anchor_abs_cosine": abs_anchor_cos,
                "canonical_anchor_cosine": can_anchor_cos,
                "anchor_axis_displacement_deg": displacement,
                "cumulative_path_angle_deg": path,
                "path_over_displacement": (
                    path / displacement if displacement > 1e-9 else (1.0 if path < 1e-9 else float("inf"))
                ),
            }
        )

    # Every within-window pair.
    pair_rows: list[dict[str, Any]] = []
    for ia, la in enumerate(layers):
        for lb in layers[ia + 1 :]:
            a, b = banks[la], banks[lb]
            va, vb = a.V[:, sv0], b.V[:, sv0]
            raw = float(va @ vb)
            can = float((signs[la] * va) @ (signs[lb] * vb))
            ab = match_vector_to_layer(va, b, k)
            ba = match_vector_to_layer(vb, a, k)
            pair_rows.append(
                {
                    "trajectory": spec.label,
                    "sv_index_0": sv0,
                    "layer_a": la,
                    "layer_b": lb,
                    "layer_distance": lb - la,
                    "raw_cosine": raw,
                    "abs_cosine": abs(raw),
                    "canonical_cosine": can,
                    "axis_angle_deg": axis_angle_deg(abs(raw)),
                    "raw_sign_flip": bool(raw < 0),
                    "a_to_b_best_rank_0": ab["best_rank_0"],
                    "a_to_b_best_abs_cosine": ab["best_abs_cosine"],
                    "a_to_b_second_rank_0": ab["second_rank_0"],
                    "a_to_b_second_abs_cosine": ab["second_abs_cosine"],
                    "a_to_b_margin": ab["best_margin"],
                    "b_to_a_best_rank_0": ba["best_rank_0"],
                    "b_to_a_best_abs_cosine": ba["best_abs_cosine"],
                    "b_to_a_second_rank_0": ba["second_rank_0"],
                    "b_to_a_second_abs_cosine": ba["second_abs_cosine"],
                    "b_to_a_margin": ba["best_margin"],
                    "same_rank_is_reciprocal_best": bool(
                        ab["best_rank_0"] == sv0 and ba["best_rank_0"] == sv0
                    ),
                    "sigma_a": float(a.S[sv0]),
                    "sigma_b": float(b.S[sv0]),
                    "relative_gap_a": local_gap(a.S, sv0)[3],
                    "relative_gap_b": local_gap(b.S, sv0)[3],
                }
            )

    # Greedy identity trace: start at fixed-rank SV in first layer, then choose
    # the most aligned top-k vector at each next layer. This is intentionally
    # separate from the fixed-rank analysis so rank swaps are visible.
    greedy_rows: list[dict[str, Any]] = []
    current_rank = sv0
    current_layer = layers[0]
    current_v = banks[current_layer].V[:, current_rank]
    current_sign = 1
    greedy_rows.append(
        {
            "trajectory": spec.label,
            "start_sv_index_0": sv0,
            "layer": current_layer,
            "matched_rank_0": current_rank,
            "rank_offset_from_start": current_rank - sv0,
            "step_raw_cosine": 1.0,
            "step_abs_cosine": 1.0,
            "step_axis_angle_deg": 0.0,
            "canonical_sign_multiplier": current_sign,
            "reciprocal_to_previous": True,
            "best_margin": float("nan"),
            "sigma": float(banks[current_layer].S[current_rank]),
        }
    )
    for next_layer in layers[1:]:
        target = banks[next_layer]
        match = match_vector_to_layer(current_v, target, k)
        next_rank = int(match["best_rank_0"])
        next_raw = target.V[:, next_rank]
        raw_cos = float(current_v @ next_raw)
        next_sign = current_sign * (1 if raw_cos >= 0 else -1)

        # Reciprocal check uses the newly selected raw vector back to previous layer.
        back = match_vector_to_layer(next_raw, banks[current_layer], k)
        reciprocal = bool(back["best_rank_0"] == current_rank)

        greedy_rows.append(
            {
                "trajectory": spec.label,
                "start_sv_index_0": sv0,
                "layer": next_layer,
                "matched_rank_0": next_rank,
                "rank_offset_from_start": next_rank - sv0,
                "step_raw_cosine": raw_cos,
                "step_abs_cosine": abs(raw_cos),
                "step_axis_angle_deg": axis_angle_deg(abs(raw_cos)),
                "canonical_sign_multiplier": next_sign,
                "reciprocal_to_previous": reciprocal,
                "best_margin": match["best_margin"],
                "sigma": float(target.S[next_rank]),
            }
        )
        current_layer = next_layer
        current_rank = next_rank
        current_v = next_raw
        current_sign = next_sign

    return step_rows, layer_rows, pair_rows, greedy_rows


# -----------------------------------------------------------------------------
# Lexical-neighborhood analysis
# -----------------------------------------------------------------------------



def infer_lexical_key(rec: dict[str, Any], labeling: str) -> tuple[int, int]:
    """Return (layer, zero-index SV)."""
    if "layer" in rec and "sv_index_0" in rec:
        return int(rec["layer"]), int(rec["sv_index_0"])
    if "layer" in rec and "sv_rank_1based" in rec:
        return int(rec["layer"]), int(rec["sv_rank_1based"]) - 1

    cand = str(rec.get("candidate", ""))
    m = CANDIDATE_RE.match(cand)
    if not m:
        raise ValueError(
            "Lexical record lacks sv_index_0/sv_rank_1based and candidate is not Lxx_SVyy"
        )
    layer, suffix = int(m.group(1)), int(m.group(2))

    if labeling == "zero":
        return layer, suffix
    # Auto defaults to the scanner convention because that is the file this
    # script was designed to consume. The explicit fields above always win.
    return layer, suffix - 1



def load_lexical_neighbors(
    path: Path, labeling: str, top_n: int
) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                key = infer_lexical_key(rec, labeling)
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e

            near = list(rec.get("nearest_tokens") or [])[:top_n]
            far = list(rec.get("farthest_tokens") or [])[:top_n]
            out[key] = {
                "nearest_tokens": near,
                "farthest_tokens": far,
            }
    return out



def token_id_set(rows: list[dict[str, Any]]) -> set[int]:
    return {int(x["token_id"]) for x in rows if "token_id" in x}



def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return float("nan")
    u = a | b
    return len(a & b) / len(u) if u else float("nan")



def pretty_tokens(rows: list[dict[str, Any]], limit: int = 12) -> str:
    vals = []
    for r in rows[:limit]:
        txt = r.get("decoded", r.get("token", str(r.get("token_id", "?"))))
        txt = str(txt).replace("\n", "\\n").replace("\r", "\\r")
        vals.append(txt)
    return " | ".join(vals)



def canonicalized_lexical_lists(
    lex: dict[tuple[int, int], dict[str, Any]],
    banks: dict[int, LayerBank],
    spec: TrajectorySpec,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    signs = canonical_signs(banks, spec)
    out: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for l in spec.layers:
        rec = lex.get((l, spec.sv0))
        if rec is None:
            continue
        # If the raw SVD vector is multiplied by -1, positive and negative
        # unembedding neighborhoods swap semantic sides.
        if signs[l] > 0:
            pos = rec["nearest_tokens"]
            neg = rec["farthest_tokens"]
        else:
            pos = rec["farthest_tokens"]
            neg = rec["nearest_tokens"]
        out[l] = {"positive": pos, "negative": neg}
    return out



def analyze_lexical(
    lex: dict[tuple[int, int], dict[str, Any]],
    banks: dict[int, LayerBank],
    spec: TrajectorySpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    can = canonicalized_lexical_lists(lex, banks, spec)
    layer_rows: list[dict[str, Any]] = []
    for l in spec.layers:
        if l not in can:
            continue
        layer_rows.append(
            {
                "trajectory": spec.label,
                "sv_index_0": spec.sv0,
                "layer": l,
                "canonical_positive_tokens": pretty_tokens(can[l]["positive"]),
                "canonical_negative_tokens": pretty_tokens(can[l]["negative"]),
                "n_positive_tokens": len(can[l]["positive"]),
                "n_negative_tokens": len(can[l]["negative"]),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    layers = [l for l in spec.layers if l in can]
    for ia, la in enumerate(layers):
        pa = token_id_set(can[la]["positive"])
        na = token_id_set(can[la]["negative"])
        for lb in layers[ia + 1 :]:
            pb = token_id_set(can[lb]["positive"])
            nb = token_id_set(can[lb]["negative"])
            pos_j = jaccard(pa, pb)
            neg_j = jaccard(na, nb)
            vals = [x for x in (pos_j, neg_j) if math.isfinite(x)]
            pair_rows.append(
                {
                    "trajectory": spec.label,
                    "sv_index_0": spec.sv0,
                    "layer_a": la,
                    "layer_b": lb,
                    "layer_distance": lb - la,
                    "positive_top_token_jaccard": pos_j,
                    "negative_top_token_jaccard": neg_j,
                    "mean_signed_side_jaccard": float(np.mean(vals)) if vals else float("nan"),
                }
            )
    return layer_rows, pair_rows


# -----------------------------------------------------------------------------
# Aggregation / writing
# -----------------------------------------------------------------------------



def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)



def retention_by_distance(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for r in pair_rows:
        grouped.setdefault((r["trajectory"], int(r["layer_distance"])), []).append(
            float(r["abs_cosine"])
        )
    rows = []
    for (traj, d), vals in sorted(grouped.items()):
        a = np.asarray(vals, dtype=float)
        rows.append(
            {
                "trajectory": traj,
                "layer_distance": d,
                "n_pairs": len(a),
                "mean_abs_cosine": float(np.mean(a)),
                "median_abs_cosine": float(np.median(a)),
                "min_abs_cosine": float(np.min(a)),
                "max_abs_cosine": float(np.max(a)),
                "mean_axis_angle_deg": float(np.mean([axis_angle_deg(x) for x in a])),
            }
        )
    return rows



def trajectory_stats(
    spec: TrajectorySpec,
    steps: list[dict[str, Any]],
    layers: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    greedy: list[dict[str, Any]],
) -> dict[str, Any]:
    srows = [r for r in steps if r["trajectory"] == spec.label]
    lrows = [r for r in layers if r["trajectory"] == spec.label]
    prows = [r for r in pairs if r["trajectory"] == spec.label]
    grows = [r for r in greedy if r["trajectory"] == spec.label]
    endpoint = max(lrows, key=lambda r: r["layer"])
    adjacent = np.asarray([float(r["same_rank_abs_cosine"]) for r in srows])
    reciprocal = np.asarray([bool(r["same_rank_is_reciprocal_best"]) for r in srows])
    signs = sum(bool(r["raw_sign_flip"]) for r in srows)
    drift_steps = sum(int(r["matched_rank_0"]) != spec.sv0 for r in grows)
    return {
        "trajectory": spec.label,
        "sv_index_0": spec.sv0,
        "layer_start": spec.layers[0],
        "layer_end": spec.layers[-1],
        "n_steps": len(srows),
        "mean_adjacent_abs_cosine": float(np.mean(adjacent)),
        "min_adjacent_abs_cosine": float(np.min(adjacent)),
        "max_adjacent_abs_cosine": float(np.max(adjacent)),
        "adjacent_reciprocal_same_rank_rate": float(np.mean(reciprocal)),
        "raw_adjacent_sign_flips": signs,
        "endpoint_abs_cosine": float(endpoint["anchor_abs_cosine"]),
        "endpoint_axis_displacement_deg": float(endpoint["anchor_axis_displacement_deg"]),
        "cumulative_path_angle_deg": float(endpoint["cumulative_path_angle_deg"]),
        "path_over_displacement": float(endpoint["path_over_displacement"]),
        "sigma_start": float(lrows[0]["sigma"]),
        "sigma_end": float(endpoint["sigma"]),
        "min_relative_local_gap": float(min(r["relative_local_gap"] for r in lrows)),
        "max_relative_local_gap": float(max(r["relative_local_gap"] for r in lrows)),
        "greedy_trace_rank_drift_layers": drift_steps,
        "all_pair_reciprocal_same_rank_rate": float(
            np.mean([bool(r["same_rank_is_reciprocal_best"]) for r in prows])
        ),
    }



def make_summary(
    path: Path,
    specs: list[TrajectorySpec],
    stats: list[dict[str, Any]],
    lexical_pairs: list[dict[str, Any]],
) -> None:
    lines = [
        "# Focused SV trajectory analysis",
        "",
        "**Indexing:** zero-based throughout this report. `SV_02` is column 2; `SV_09` is column 9.",
        "",
        "The earlier scanner's human-readable candidate labels were one-based, so its `SV03` maps to this report's `SV_02`, and its `SV10` maps to this report's `SV_09`.",
        "",
        "## Headline geometry",
        "",
        "| trajectory | layers | mean adjacent |cos| | min adjacent |cos| | endpoint |cos| | path angle | endpoint angle | reciprocal adjacent | sign flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s['trajectory']} | L{s['layer_start']}–L{s['layer_end']} | "
            f"{s['mean_adjacent_abs_cosine']:.3f} | {s['min_adjacent_abs_cosine']:.3f} | "
            f"{s['endpoint_abs_cosine']:.3f} | {s['cumulative_path_angle_deg']:.1f}° | "
            f"{s['endpoint_axis_displacement_deg']:.1f}° | "
            f"{s['adjacent_reciprocal_same_rank_rate']:.3f} | {s['raw_adjacent_sign_flips']} |"
        )

    lines += [
        "",
        "## Interpretation aids",
        "",
        "- **High adjacent + high endpoint retention** indicates a nearly fixed persistent axis.",
        "- **High adjacent + low endpoint retention** indicates a continuous rotating trajectory: local identity survives even though the endpoint eventually becomes geometrically different.",
        "- `cumulative_path_angle_deg` sums the unoriented adjacent angles; `endpoint_axis_displacement_deg` measures only start-to-current displacement. Their ratio helps distinguish a short/direct path from a long curved one.",
        "- Reciprocal best-match checks are performed among the top-K SVs; they are stricter than same-rank cosine alone.",
        "- Raw SVD sign flips are reported but are not treated as identity changes. Canonical signs are parallel-transported along each fixed-rank chain.",
        "",
        "## Spectral context",
        "",
    ]
    for s in stats:
        lines.append(
            f"- **{s['trajectory']}**: sigma {s['sigma_start']:.3f} → {s['sigma_end']:.3f}; "
            f"relative local gap range {s['min_relative_local_gap']:.4f}–{s['max_relative_local_gap']:.4f}; "
            f"greedy identity trace leaves the starting rank on {s['greedy_trace_rank_drift_layers']} layer(s) in the requested window."
        )

    if lexical_pairs:
        lines += ["", "## Lexical-neighborhood retention", ""]
        for spec in specs:
            rows = [r for r in lexical_pairs if r["trajectory"] == spec.label]
            if not rows:
                continue
            adj = [r["mean_signed_side_jaccard"] for r in rows if r["layer_distance"] == 1]
            endpoint = max(rows, key=lambda r: r["layer_distance"])
            lines.append(
                f"- **{spec.label}**: mean adjacent canonical top-token Jaccard "
                f"{np.nanmean(adj):.3f}; longest-span Jaccard {endpoint['mean_signed_side_jaccard']:.3f}."
            )
        lines += [
            "",
            "Lexical overlap uses only the retained nearest/farthest token lists from the scanner, not the full-vocabulary cosine profile.",
        ]

    lines += [
        "",
        "## Files to inspect",
        "",
        "- `trajectory_layers.csv`: anchor retention and cumulative rotation at each layer.",
        "- `trajectory_steps.csv`: adjacent continuity, reciprocal matching, margins, sign flips, and gaps.",
        "- `pairwise_retention.csv`: all within-window pairwise comparisons.",
        "- `greedy_identity_trace.csv`: whether the direction changes singular-value rank when followed by identity.",
    ]
    if lexical_pairs:
        lines += [
            "- `lexical_pairwise_retention.csv`: sign-canonicalized positive/negative token-neighborhood overlap.",
            "- `lexical_layers.csv`: readable canonical token neighborhoods by layer.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------



def make_plots(
    out_dir: Path,
    banks: dict[int, LayerBank],
    specs: list[TrajectorySpec],
    step_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    lexical_pair_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plots")
        return

    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        layers = list(spec.layers)
        # Fixed-rank pairwise absolute-cosine matrix.
        M = np.empty((len(layers), len(layers)), dtype=float)
        for i, la in enumerate(layers):
            va = banks[la].V[:, spec.sv0]
            for j, lb in enumerate(layers):
                M[i, j] = abs(float(va @ banks[lb].V[:, spec.sv0]))
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        im = ax.imshow(M, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(layers)), [f"L{x}" for x in layers], rotation=45, ha="right")
        ax.set_yticks(range(len(layers)), [f"L{x}" for x in layers])
        ax.set_title(f"{spec.label}: fixed-rank |cosine| retention")
        fig.colorbar(im, ax=ax, label="|cosine|")
        fig.tight_layout()
        fig.savefig(pdir / f"{spec.label}_pairwise_retention.png", dpi=180)
        plt.close(fig)

        lr = [r for r in layer_rows if r["trajectory"] == spec.label]
        xs = [r["layer"] for r in lr]

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(xs, [r["anchor_abs_cosine"] for r in lr], marker="o")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Layer")
        ax.set_ylabel("|cos(anchor, current)|")
        ax.set_title(f"{spec.label}: retention to starting direction")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(pdir / f"{spec.label}_anchor_retention.png", dpi=180)
        plt.close(fig)

        sr = [r for r in step_rows if r["trajectory"] == spec.label]
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot([r["layer_to"] for r in sr], [r["same_rank_axis_angle_deg"] for r in sr], marker="o")
        ax.set_xlabel("Destination layer")
        ax.set_ylabel("Adjacent axis rotation (degrees)")
        ax.set_title(f"{spec.label}: local rotation per layer step")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(pdir / f"{spec.label}_step_angles.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(xs, [r["cumulative_path_angle_deg"] for r in lr], marker="o", label="Cumulative path")
        ax.plot(xs, [r["anchor_axis_displacement_deg"] for r in lr], marker="o", label="Endpoint displacement")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Degrees")
        ax.set_title(f"{spec.label}: path length vs endpoint displacement")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(pdir / f"{spec.label}_path_vs_displacement.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(xs, [r["sigma"] for r in lr], marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Singular value")
        ax.set_title(f"{spec.label}: singular value along fixed-rank chain")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(pdir / f"{spec.label}_singular_values.png", dpi=180)
        plt.close(fig)

        lex = [r for r in lexical_pair_rows if r["trajectory"] == spec.label]
        if lex:
            grouped: dict[int, list[float]] = {}
            for r in lex:
                grouped.setdefault(int(r["layer_distance"]), []).append(float(r["mean_signed_side_jaccard"]))
            ds = sorted(grouped)
            ys = [float(np.nanmean(grouped[d])) for d in ds]
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.plot(ds, ys, marker="o")
            ax.set_xlabel("Layer distance")
            ax.set_ylabel("Mean canonical top-token Jaccard")
            ax.set_title(f"{spec.label}: lexical-neighborhood retention")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(pdir / f"{spec.label}_lexical_retention.png", dpi=180)
            plt.close(fig)

    # Direct comparison plot: anchor retention.
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for spec in specs:
        lr = [r for r in layer_rows if r["trajectory"] == spec.label]
        rel_x = [r["layer"] - spec.layers[0] for r in lr]
        ax.plot(rel_x, [r["anchor_abs_cosine"] for r in lr], marker="o", label=spec.label)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Layers from trajectory start")
    ax.set_ylabel("|cos(start, current)|")
    ax.set_title("Fixed-axis retention: SV_02 vs SV_09")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdir / "SV_02_vs_SV_09_anchor_retention.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def main() -> None:
    args = parse_args()
    if args.k < 2:
        raise ValueError("--k must be >= 2")
    if args.lexical_top_n < 1:
        raise ValueError("--lexical-top-n must be >= 1")

    spec02 = TrajectorySpec(sv0=2, layers=parse_layer_window(args.sv02_layers))
    spec09 = TrajectorySpec(sv0=9, layers=parse_layer_window(args.sv09_layers))
    specs = [spec02, spec09]

    needed_layers = sorted(set(spec02.layers) | set(spec09.layers))
    banks = load_banks(args.directions_dir, needed_layers)

    # Confirm top-k match search is feasible everywhere.
    min_cols = min(banks[l].V.shape[1] for l in needed_layers)
    k = min(args.k, min_cols)
    if k <= 9:
        raise ValueError(
            f"Need at least 10 saved SVs to analyze zero-index SV_09; common count={min_cols}"
        )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps: list[dict[str, Any]] = []
    all_layers: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    all_greedy: list[dict[str, Any]] = []

    for spec in specs:
        print(f"[geometry] {spec.label}: L{spec.layers[0]}..L{spec.layers[-1]} (zero-index column {spec.sv0})")
        steps, layers, pairs, greedy = analyze_trajectory_geometry(banks, spec, k)
        all_steps.extend(steps)
        all_layers.extend(layers)
        all_pairs.extend(pairs)
        all_greedy.extend(greedy)

    write_csv(out_dir / "trajectory_steps.csv", all_steps)
    write_csv(out_dir / "trajectory_layers.csv", all_layers)
    write_csv(out_dir / "pairwise_retention.csv", all_pairs)
    write_csv(out_dir / "retention_by_distance.csv", retention_by_distance(all_pairs))
    write_csv(out_dir / "greedy_identity_trace.csv", all_greedy)

    lexical_layers: list[dict[str, Any]] = []
    lexical_pairs: list[dict[str, Any]] = []
    if args.unembedding_neighbors is not None:
        print(f"[lexical] reading {args.unembedding_neighbors}")
        lex = load_lexical_neighbors(
            args.unembedding_neighbors,
            args.lexical_labeling,
            args.lexical_top_n,
        )
        for spec in specs:
            missing = [l for l in spec.layers if (l, spec.sv0) not in lex]
            if missing:
                print(
                    f"[warn] lexical data missing {spec.label} at layers {missing}; "
                    "geometry is unaffected"
                )
            ll, lp = analyze_lexical(lex, banks, spec)
            lexical_layers.extend(ll)
            lexical_pairs.extend(lp)
        write_csv(out_dir / "lexical_layers.csv", lexical_layers)
        write_csv(out_dir / "lexical_pairwise_retention.csv", lexical_pairs)

    stats = [trajectory_stats(s, all_steps, all_layers, all_pairs, all_greedy) for s in specs]
    write_csv(out_dir / "trajectory_summary.csv", stats)
    make_summary(out_dir / "summary.md", specs, stats, lexical_pairs)

    metadata = {
        "sv_numbering": "zero-based",
        "targets": {
            "SV_02": {"sv_index_0": 2, "layers": list(spec02.layers)},
            "SV_09": {"sv_index_0": 9, "layers": list(spec09.layers)},
        },
        "match_search_top_k": k,
        "scanner_candidate_label_warning": (
            "scan_unrealized_words_with_unembedding.py used one-based candidate strings: "
            "zero SV_02 == scanner SV03; zero SV_09 == scanner SV10"
        ),
        "unembedding_neighbors": None if args.unembedding_neighbors is None else str(args.unembedding_neighbors),
        "lexical_labeling": args.lexical_labeling,
        "lexical_top_n": args.lexical_top_n,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not args.no_plots:
        make_plots(out_dir, banks, specs, all_steps, all_layers, lexical_pairs)

    print("\nDone.")
    for s in stats:
        print(
            f"  {s['trajectory']} L{s['layer_start']}..L{s['layer_end']}: "
            f"adjacent mean |cos|={s['mean_adjacent_abs_cosine']:.3f}, "
            f"endpoint |cos|={s['endpoint_abs_cosine']:.3f}, "
            f"path={s['cumulative_path_angle_deg']:.1f}°, "
            f"endpoint displacement={s['endpoint_axis_displacement_deg']:.1f}°"
        )
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
