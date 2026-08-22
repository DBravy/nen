#!/usr/bin/env python3
"""
analyze_cross_layer_sv_alignment.py

Analyze whether J-Lens right singular vectors preserve their identities and
singular-value ranks across layers.

Expected input
--------------
A directory containing direction banks produced by scan_unrealized_words.py:

    directions/
      L00.npz
      L01.npz
      ...

Each .npz must contain:
    V : right singular vectors, either [d_model, n_sv] or [n_sv, d_model]
    S : singular values, [n_sv]

The script asks questions such as:

  * Is SV10 at one layer most aligned with SV10 at another layer?
  * How often is the best cross-layer match the exact same singular-value rank?
  * Is there a diagonal band in |V_l^T V_m|?
  * How large is same-rank alignment relative to the best available match?
  * Are sign flips common for otherwise aligned same-rank directions?
  * Does same-rank persistence correlate with the local singular-value gap?
  * Is the diagonal stronger than expected under random rank relabeling?

Outputs
-------
OUT/
  pair_summary.csv
      one row per layer pair

  sv_pair_matches.csv
      one row per (layer pair, source SV rank)

  sv_rank_summary.csv
      aggregates persistence statistics by SV rank across layers

  singular_value_gaps.csv
      singular value and local spectral-gap statistics for every (layer, SV)

  gap_alignment_summary.csv
      correlations between spectral isolation and same-rank persistence

  alignment_matrices.npz
      signed and absolute cosine matrices for every analyzed layer pair

  heatmaps/
      alignment heatmaps (adjacent pairs by default)

  summary.md
      compact human-readable report

Important sign convention
-------------------------
SVD vectors are defined only up to sign. Therefore the main identity metric is
ABSOLUTE cosine similarity. Signed cosine is saved separately so you can see
where the raw SVD orientation flips.

Examples
--------
Basic analysis of all saved layers, top 64 SVs:

    python analyze_cross_layer_sv_alignment.py \
        --directions-dir unrealized_words_fineweb/directions \
        --out sv_cross_layer_analysis \
        --k 64

Only adjacent layers:

    python analyze_cross_layer_sv_alignment.py \
        --directions-dir unrealized_words_fineweb/directions \
        --out sv_cross_layer_analysis \
        --k 64 \
        --pairs adjacent

Compare all layer pairs but only plot adjacent heatmaps:

    python analyze_cross_layer_sv_alignment.py \
        --directions-dir unrealized_words_fineweb/directions \
        --out sv_cross_layer_analysis \
        --k 64 \
        --pairs all \
        --heatmaps adjacent

If you specifically care about a handful of ranks:

    python analyze_cross_layer_sv_alignment.py \
        --directions-dir unrealized_words_fineweb/directions \
        --out sv_cross_layer_analysis \
        --k 64 \
        --focus-svs 3,6,8,10

Notes on numbering
------------------
By default SV ranks in output are ZERO-BASED to match filenames/names such as
SV03 meaning column index 3. If your convention is one-based, pass:

    --sv-numbering one

This changes labels only; it does not change which columns are analyzed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze cross-layer alignment and rank persistence of J-Lens right singular vectors."
    )
    p.add_argument("--directions-dir", type=Path, required=True,
                   help="Directory containing LXX.npz files with arrays V and S.")
    p.add_argument("--out", type=Path, default=Path("sv_cross_layer_analysis"),
                   help="Output directory.")
    p.add_argument("--k", type=int, default=None,
                   help="Analyze top K SVs. Default: maximum K shared by every selected layer.")
    p.add_argument("--layers", type=str, default="all",
                   help="Comma-separated layer numbers, or 'all'.")
    p.add_argument("--pairs", choices=("adjacent", "all"), default="all",
                   help="Which layer pairs to analyze numerically. Default: all.")
    p.add_argument("--heatmaps", choices=("none", "adjacent", "all"), default="adjacent",
                   help="Which pair matrices to render as PNG heatmaps.")
    p.add_argument("--focus-svs", type=str, default="",
                   help="Optional comma-separated SV labels/ranks to emphasize in summary.md, e.g. 3,6,8,10.")
    p.add_argument("--sv-numbering", choices=("zero", "one"), default="zero",
                   help="How SV labels are displayed. Default: zero (SV00 is first vector).")
    p.add_argument("--permutations", type=int, default=2000,
                   help="Random rank relabelings per layer pair for diagonal null tests. Default: 2000.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for permutation null tests.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip all matplotlib plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------


LAYER_RE = re.compile(r"^L(\d+)\.npz$")


@dataclass
class LayerBank:
    layer: int
    V: np.ndarray   # [d_model, k], unit-normalized columns
    S: np.ndarray   # [k]


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


def parse_layers(spec: str, available: Iterable[int]) -> list[int]:
    available = sorted(available)
    if spec.strip().lower() == "all":
        return available
    wanted = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"Requested layers not found: {missing}; available={available}")
    if len(wanted) < 2:
        raise ValueError("Need at least two layers for cross-layer analysis.")
    return wanted


def infer_v_orientation(V: np.ndarray, S: np.ndarray, path: Path) -> np.ndarray:
    """Return V as [d_model, n_sv]."""
    if V.ndim != 2:
        raise ValueError(f"{path}: V must be 2D, got {V.shape}")
    n_s = int(S.size)

    # The SV axis is usually the one equal to len(S). If both axes match
    # (full square SVD), the conventional saved format in this project is
    # [d_model, n_sv], so leave it unchanged.
    if V.shape[1] == n_s:
        return V
    if V.shape[0] == n_s:
        return V.T

    # A truncated S array can accompany a larger V. Prefer the smaller axis as
    # the SV axis when the shape is clearly rectangular.
    if V.shape[0] < V.shape[1] and V.shape[0] >= n_s:
        return V.T
    if V.shape[1] < V.shape[0] and V.shape[1] >= n_s:
        return V

    raise ValueError(
        f"{path}: cannot infer V orientation from V.shape={V.shape}, len(S)={n_s}"
    )


def inspect_capacity(path: Path) -> tuple[int, int]:
    z = np.load(path, mmap_mode="r")
    if "V" not in z or "S" not in z:
        raise ValueError(f"{path} must contain arrays V and S")
    S = np.asarray(z["S"]).reshape(-1)
    V = infer_v_orientation(np.asarray(z["V"]), S, path)
    return V.shape[1], V.shape[0]


def load_bank(path: Path, layer: int, k: int) -> LayerBank:
    z = np.load(path)
    if "V" not in z or "S" not in z:
        raise ValueError(f"{path} must contain arrays V and S")
    S = np.asarray(z["S"], dtype=np.float64).reshape(-1)
    V = infer_v_orientation(np.asarray(z["V"], dtype=np.float64), S, path)
    if V.shape[1] < k or S.size < k:
        raise ValueError(f"{path}: requested k={k}, available V={V.shape[1]}, S={S.size}")

    V = np.array(V[:, :k], dtype=np.float64, copy=True)
    S = np.array(S[:k], dtype=np.float64, copy=True)

    norms = np.linalg.norm(V, axis=0)
    if np.any(norms < 1e-12):
        bad = np.flatnonzero(norms < 1e-12).tolist()
        raise ValueError(f"{path}: zero-norm directions at columns {bad[:10]}")
    V /= norms[None, :]
    return LayerBank(layer=layer, V=V, S=S)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def label_sv(index: int, numbering: str) -> str:
    display = index if numbering == "zero" else index + 1
    return f"SV{display:02d}"


def display_rank(index: int, numbering: str) -> int:
    return index if numbering == "zero" else index + 1


def layer_pairs(layers: list[int], mode: str) -> list[tuple[int, int]]:
    if mode == "adjacent":
        return list(zip(layers[:-1], layers[1:]))
    return [(layers[i], layers[j]) for i in range(len(layers)) for j in range(i + 1, len(layers))]


def spectral_gap_rows(bank: LayerBank, numbering: str) -> list[dict]:
    """
    Local gap is the smaller separation from neighboring singular values.
    Edge ranks have only one neighbor.
    Relative gap divides by the focal singular value.
    """
    S = bank.S
    rows = []
    k = len(S)
    for i, s in enumerate(S):
        upper = np.nan if i == 0 else S[i - 1] - s
        lower = np.nan if i == k - 1 else s - S[i + 1]
        finite = [x for x in (upper, lower) if np.isfinite(x)]
        local = min(finite) if finite else np.nan
        rel = local / max(abs(s), 1e-30) if np.isfinite(local) else np.nan
        rows.append({
            "layer": bank.layer,
            "sv_index_zero_based": i,
            "sv_rank": display_rank(i, numbering),
            "sv_label": label_sv(i, numbering),
            "singular_value": float(s),
            "gap_to_higher_rank": float(upper) if np.isfinite(upper) else np.nan,
            "gap_to_lower_rank": float(lower) if np.isfinite(lower) else np.nan,
            "local_gap": float(local) if np.isfinite(local) else np.nan,
            "relative_local_gap": float(rel) if np.isfinite(rel) else np.nan,
        })
    return rows


def top2_indices(row: np.ndarray) -> tuple[int, int]:
    if row.size == 1:
        return 0, 0
    inds = np.argpartition(row, -2)[-2:]
    inds = inds[np.argsort(row[inds])[::-1]]
    return int(inds[0]), int(inds[1])


def permutation_null(
    absC: np.ndarray,
    best_j: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Randomly relabel target singular-value ranks.

    Under a permutation p, source rank i's nominal same-rank target is the
    original target column p[i]. This preserves the actual alignment matrix
    while destroying the meaning of the target rank labels.
    """
    k = absC.shape[0]
    diag_idx = np.arange(k)
    obs_diag_mean = float(np.mean(np.diag(absC)))
    obs_exact = float(np.mean(best_j == diag_idx))

    if n_perm <= 0:
        return {
            "null_diag_mean": np.nan,
            "null_diag_std": np.nan,
            "diag_perm_p": np.nan,
            "null_exact_rate_mean": np.nan,
            "null_exact_rate_std": np.nan,
            "exact_perm_p": np.nan,
        }

    null_diag = np.empty(n_perm, dtype=np.float64)
    null_exact = np.empty(n_perm, dtype=np.float64)
    for r in range(n_perm):
        p = rng.permutation(k)
        null_diag[r] = np.mean(absC[diag_idx, p])
        null_exact[r] = np.mean(best_j == p)

    return {
        "null_diag_mean": float(null_diag.mean()),
        "null_diag_std": float(null_diag.std(ddof=1)) if n_perm > 1 else 0.0,
        "diag_perm_p": float((1 + np.sum(null_diag >= obs_diag_mean)) / (n_perm + 1)),
        "null_exact_rate_mean": float(null_exact.mean()),
        "null_exact_rate_std": float(null_exact.std(ddof=1)) if n_perm > 1 else 0.0,
        "exact_perm_p": float((1 + np.sum(null_exact >= obs_exact)) / (n_perm + 1)),
    }


def analyze_pair(
    a: LayerBank,
    b: LayerBank,
    numbering: str,
    gap_lookup: dict[tuple[int, int], dict],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict], dict]:
    C = a.V.T @ b.V
    C = np.clip(C, -1.0, 1.0)
    absC = np.abs(C)
    k = absC.shape[0]
    diag_idx = np.arange(k)
    best_j = np.argmax(absC, axis=1)
    best_i_for_target = np.argmax(absC, axis=0)

    rows = []
    for i in range(k):
        j1, j2 = top2_indices(absC[i])
        same_signed = float(C[i, i])
        same_abs = float(absC[i, i])
        best_abs = float(absC[i, j1])
        second_abs = float(absC[i, j2]) if k > 1 else np.nan
        exact = (j1 == i)
        reciprocal = bool(best_i_for_target[j1] == i)

        ga = gap_lookup[(a.layer, i)]
        gb_same = gap_lookup[(b.layer, i)]
        gb_best = gap_lookup[(b.layer, j1)]

        rows.append({
            "source_layer": a.layer,
            "target_layer": b.layer,
            "layer_distance": b.layer - a.layer,
            "source_sv_index_zero_based": i,
            "source_sv_rank": display_rank(i, numbering),
            "source_sv_label": label_sv(i, numbering),
            "target_same_rank_sv_label": label_sv(i, numbering),
            "same_rank_signed_cosine": same_signed,
            "same_rank_abs_cosine": same_abs,
            "same_rank_sign_flipped": same_signed < 0,
            "best_target_sv_index_zero_based": j1,
            "best_target_sv_rank": display_rank(j1, numbering),
            "best_target_sv_label": label_sv(j1, numbering),
            "best_signed_cosine": float(C[i, j1]),
            "best_abs_cosine": best_abs,
            "second_best_abs_cosine": second_abs,
            "best_match_margin": best_abs - second_abs if np.isfinite(second_abs) else np.nan,
            "same_rank_is_best": exact,
            "best_match_is_reciprocal": reciprocal,
            "rank_offset_zero_based": j1 - i,
            "abs_rank_offset": abs(j1 - i),
            "same_rank_fraction_of_best": same_abs / max(best_abs, 1e-30),
            "same_rank_deficit_from_best": best_abs - same_abs,
            "source_singular_value": float(a.S[i]),
            "same_rank_target_singular_value": float(b.S[i]),
            "best_target_singular_value": float(b.S[j1]),
            "source_relative_local_gap": ga["relative_local_gap"],
            "same_rank_target_relative_local_gap": gb_same["relative_local_gap"],
            "best_target_relative_local_gap": gb_best["relative_local_gap"],
            "pair_min_same_rank_relative_gap": float(np.nanmin([
                ga["relative_local_gap"], gb_same["relative_local_gap"]
            ])),
            "pair_mean_same_rank_relative_gap": float(np.nanmean([
                ga["relative_local_gap"], gb_same["relative_local_gap"]
            ])),
        })

    diag_abs = np.diag(absC)
    best_abs_rows = np.max(absC, axis=1)
    exact_rate = float(np.mean(best_j == diag_idx))
    reciprocal_exact_rate = float(np.mean((best_j == diag_idx) & (best_i_for_target == diag_idx)))
    perm = permutation_null(absC, best_j, n_perm, rng)

    offdiag = absC.copy()
    np.fill_diagonal(offdiag, np.nan)

    summary = {
        "source_layer": a.layer,
        "target_layer": b.layer,
        "layer_distance": b.layer - a.layer,
        "k": k,
        "exact_rank_best_match_rate": exact_rate,
        "chance_exact_rate_1_over_k": 1.0 / k,
        "exact_rate_over_chance": exact_rate / (1.0 / k),
        "reciprocal_exact_rank_rate": reciprocal_exact_rate,
        "mean_same_rank_abs_cosine": float(np.mean(diag_abs)),
        "median_same_rank_abs_cosine": float(np.median(diag_abs)),
        "min_same_rank_abs_cosine": float(np.min(diag_abs)),
        "max_same_rank_abs_cosine": float(np.max(diag_abs)),
        "mean_best_abs_cosine": float(np.mean(best_abs_rows)),
        "mean_same_rank_fraction_of_best": float(np.mean(diag_abs / np.maximum(best_abs_rows, 1e-30))),
        "mean_same_rank_deficit_from_best": float(np.mean(best_abs_rows - diag_abs)),
        "mean_offdiag_abs_cosine": float(np.nanmean(offdiag)) if k > 1 else np.nan,
        "diagonal_to_offdiag_ratio": float(np.mean(diag_abs) / max(np.nanmean(offdiag), 1e-30)) if k > 1 else np.nan,
        "same_rank_raw_sign_flip_rate": float(np.mean(np.diag(C) < 0)),
        "mean_abs_rank_offset_of_best": float(np.mean(np.abs(best_j - diag_idx))),
        "median_abs_rank_offset_of_best": float(np.median(np.abs(best_j - diag_idx))),
        **perm,
    }
    return C, rows, summary


# ---------------------------------------------------------------------------
# Correlations / aggregation
# ---------------------------------------------------------------------------


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Small scipy-free rankdata implementation (average ranks for ties)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    return pearson(rankdata_average_ties(x), rankdata_average_ties(y))


def aggregate_rank_summary(rows: list[dict], k: int, numbering: str) -> list[dict]:
    out = []
    for i in range(k):
        rr = [r for r in rows if r["source_sv_index_zero_based"] == i]
        if not rr:
            continue
        same = np.array([r["same_rank_abs_cosine"] for r in rr], dtype=float)
        best = np.array([r["best_abs_cosine"] for r in rr], dtype=float)
        exact = np.array([float(r["same_rank_is_best"]) for r in rr], dtype=float)
        flips = np.array([float(r["same_rank_sign_flipped"]) for r in rr], dtype=float)
        offsets = np.array([r["abs_rank_offset"] for r in rr], dtype=float)
        frac = np.array([r["same_rank_fraction_of_best"] for r in rr], dtype=float)

        adj = [r for r in rr if r["layer_distance"] == 1]
        adj_same = np.array([r["same_rank_abs_cosine"] for r in adj], dtype=float) if adj else np.array([])
        adj_exact = np.array([float(r["same_rank_is_best"]) for r in adj], dtype=float) if adj else np.array([])
        adj_flips = np.array([float(r["same_rank_sign_flipped"]) for r in adj], dtype=float) if adj else np.array([])

        out.append({
            "sv_index_zero_based": i,
            "sv_rank": display_rank(i, numbering),
            "sv_label": label_sv(i, numbering),
            "n_layer_pairs": len(rr),
            "mean_same_rank_abs_cosine": float(np.mean(same)),
            "median_same_rank_abs_cosine": float(np.median(same)),
            "min_same_rank_abs_cosine": float(np.min(same)),
            "max_same_rank_abs_cosine": float(np.max(same)),
            "exact_rank_best_match_rate": float(np.mean(exact)),
            "mean_same_rank_fraction_of_best": float(np.mean(frac)),
            "mean_best_abs_cosine": float(np.mean(best)),
            "raw_sign_flip_rate": float(np.mean(flips)),
            "mean_abs_rank_offset": float(np.mean(offsets)),
            "n_adjacent_pairs": len(adj),
            "adjacent_mean_same_rank_abs_cosine": float(np.mean(adj_same)) if len(adj_same) else np.nan,
            "adjacent_exact_rank_best_match_rate": float(np.mean(adj_exact)) if len(adj_exact) else np.nan,
            "adjacent_raw_sign_flip_rate": float(np.mean(adj_flips)) if len(adj_flips) else np.nan,
        })
    return out


def gap_correlation_summary(rows: list[dict]) -> list[dict]:
    subsets = {
        "all_pairs": rows,
        "adjacent_pairs": [r for r in rows if r["layer_distance"] == 1],
    }
    results = []
    for name, rr in subsets.items():
        if not rr:
            continue
        gap = np.array([r["pair_min_same_rank_relative_gap"] for r in rr], dtype=float)
        same = np.array([r["same_rank_abs_cosine"] for r in rr], dtype=float)
        frac = np.array([r["same_rank_fraction_of_best"] for r in rr], dtype=float)
        exact = np.array([float(r["same_rank_is_best"]) for r in rr], dtype=float)
        deficit = np.array([r["same_rank_deficit_from_best"] for r in rr], dtype=float)
        for metric_name, metric in [
            ("same_rank_abs_cosine", same),
            ("same_rank_fraction_of_best", frac),
            ("same_rank_is_best", exact),
            ("same_rank_deficit_from_best", deficit),
        ]:
            results.append({
                "subset": name,
                "n": int(np.sum(np.isfinite(gap) & np.isfinite(metric))),
                "gap_metric": "min_relative_local_gap_across_pair",
                "alignment_metric": metric_name,
                "pearson_r": pearson(gap, metric),
                "spearman_rho": spearman(gap, metric),
            })
    return results


# ---------------------------------------------------------------------------
# I/O and plots
# ---------------------------------------------------------------------------


def clean_value(v):
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return ""
        return float(v)
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: clean_value(row.get(k)) for k in fields})


def should_plot_pair(a: int, b: int, heatmaps: str) -> bool:
    if heatmaps == "none":
        return False
    if heatmaps == "all":
        return True
    return b == a + 1


def plot_heatmap(C: np.ndarray, a: int, b: int, out_path: Path, numbering: str) -> None:
    import matplotlib.pyplot as plt

    absC = np.abs(C)
    k = absC.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(absC, origin="upper", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="|cosine similarity|")
    ax.set_title(f"Right-SV alignment: L{a:02d} → L{b:02d}")
    ax.set_xlabel(f"L{b:02d} singular-vector rank")
    ax.set_ylabel(f"L{a:02d} singular-vector rank")

    # Keep labels readable for large K.
    step = 1 if k <= 20 else 4 if k <= 64 else 8
    ticks = np.arange(0, k, step)
    labels = [str(display_rank(int(i), numbering)) for i in ticks]
    ax.set_xticks(ticks, labels, rotation=90)
    ax.set_yticks(ticks, labels)

    # Diagonal guide.
    ax.plot(np.arange(k), np.arange(k), linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pair_summary(pair_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Exact-rank preservation vs layer distance.
    distances = sorted({int(r["layer_distance"]) for r in pair_rows})
    xs, ys, ydiag = [], [], []
    for d in distances:
        rr = [r for r in pair_rows if int(r["layer_distance"]) == d]
        xs.append(d)
        ys.append(float(np.mean([r["exact_rank_best_match_rate"] for r in rr])))
        ydiag.append(float(np.mean([r["mean_same_rank_abs_cosine"] for r in rr])))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Layer distance")
    ax.set_ylabel("P(best match has same SV rank)")
    ax.set_ylim(0, 1)
    ax.set_title("Exact singular-rank preservation vs layer distance")
    fig.tight_layout()
    fig.savefig(out_dir / "exact_rank_rate_vs_layer_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xs, ydiag, marker="o")
    ax.set_xlabel("Layer distance")
    ax.set_ylabel("Mean same-rank |cosine|")
    ax.set_ylim(0, 1)
    ax.set_title("Same-rank alignment vs layer distance")
    fig.tight_layout()
    fig.savefig(out_dir / "same_rank_cosine_vs_layer_distance.png", dpi=180)
    plt.close(fig)


def plot_rank_summary(rank_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    x = [r["sv_rank"] for r in rank_rows]
    y = [r["exact_rank_best_match_rate"] for r in rank_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y, marker="o", markersize=3)
    ax.set_xlabel("Singular-vector rank")
    ax.set_ylabel("Exact-rank best-match rate")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-layer rank persistence by singular-vector rank")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_persistence_by_sv.png", dpi=180)
    plt.close(fig)

    y2 = [r["mean_same_rank_abs_cosine"] for r in rank_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y2, marker="o", markersize=3)
    ax.set_xlabel("Singular-vector rank")
    ax.set_ylabel("Mean same-rank |cosine|")
    ax.set_ylim(0, 1)
    ax.set_title("Same-rank cosine by singular-vector rank")
    fig.tight_layout()
    fig.savefig(out_dir / "same_rank_cosine_by_sv.png", dpi=180)
    plt.close(fig)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def make_summary(
    layers: list[int],
    k: int,
    pair_summaries: list[dict],
    rank_summaries: list[dict],
    gap_corr: list[dict],
    focus_indices: list[int],
    numbering: str,
    n_perm: int,
) -> str:
    adj = [r for r in pair_summaries if r["layer_distance"] == 1]
    all_exact = np.mean([r["exact_rank_best_match_rate"] for r in pair_summaries])
    all_diag = np.mean([r["mean_same_rank_abs_cosine"] for r in pair_summaries])
    adj_exact = np.mean([r["exact_rank_best_match_rate"] for r in adj]) if adj else np.nan
    adj_diag = np.mean([r["mean_same_rank_abs_cosine"] for r in adj]) if adj else np.nan
    adj_flip = np.mean([r["same_rank_raw_sign_flip_rate"] for r in adj]) if adj else np.nan

    lines = [
        "# Cross-layer singular-vector alignment",
        "",
        f"Layers: {layers}",
        f"Top K directions: {k}",
        f"Layer pairs analyzed: {len(pair_summaries)}",
        f"Random rank-label permutations per pair: {n_perm}",
        "",
        "## Headline statistics",
        "",
        f"- Chance exact-rank rate for K={k}: **{1/k:.4f}** ({100/k:.2f}%).",
        f"- Mean exact-rank best-match rate across analyzed pairs: **{all_exact:.3f}**.",
        f"- Mean same-rank |cosine| across analyzed pairs: **{all_diag:.3f}**.",
        f"- Adjacent-layer exact-rank best-match rate: **{fmt(adj_exact)}**.",
        f"- Adjacent-layer same-rank |cosine|: **{fmt(adj_diag)}**.",
        f"- Adjacent-layer raw same-rank sign-flip rate: **{fmt(adj_flip)}**.",
        "",
        "The primary identity metric is absolute cosine because SVD sign is arbitrary. "
        "The raw sign-flip statistic is reported only to characterize orientation changes.",
        "",
        "## Most rank-persistent SVs",
        "",
        "| SV | exact-rank rate | mean same-rank |cos| | adjacent exact rate | adjacent |cos| |",
        "|---:|---:|---:|---:|---:|",
    ]

    top = sorted(rank_summaries, key=lambda r: (
        r["exact_rank_best_match_rate"], r["mean_same_rank_abs_cosine"]
    ), reverse=True)[:15]
    for r in top:
        lines.append(
            f"| {r['sv_label']} | {r['exact_rank_best_match_rate']:.3f} | "
            f"{r['mean_same_rank_abs_cosine']:.3f} | "
            f"{fmt(r['adjacent_exact_rank_best_match_rate'])} | "
            f"{fmt(r['adjacent_mean_same_rank_abs_cosine'])} |"
        )

    if focus_indices:
        by_i = {r["sv_index_zero_based"]: r for r in rank_summaries}
        lines += ["", "## Focus SVs", ""]
        for i in focus_indices:
            r = by_i.get(i)
            if r is None:
                continue
            lines.append(
                f"- **{r['sv_label']}**: exact-rank rate={r['exact_rank_best_match_rate']:.3f}, "
                f"mean same-rank |cos|={r['mean_same_rank_abs_cosine']:.3f}, "
                f"adjacent exact-rank rate={fmt(r['adjacent_exact_rank_best_match_rate'])}, "
                f"raw sign-flip rate={r['raw_sign_flip_rate']:.3f}."
            )

    lines += ["", "## Spectral-gap relationship", ""]
    if gap_corr:
        lines += [
            "The table below correlates the minimum relative local singular-value gap across "
            "a same-rank layer pair with alignment persistence.",
            "",
            "| subset | alignment metric | Pearson r | Spearman rho | n |",
            "|---|---|---:|---:|---:|",
        ]
        for r in gap_corr:
            lines.append(
                f"| {r['subset']} | {r['alignment_metric']} | {fmt(r['pearson_r'])} | "
                f"{fmt(r['spearman_rho'])} | {r['n']} |"
            )

    sig_diag = [r for r in pair_summaries if np.isfinite(r["diag_perm_p"]) and r["diag_perm_p"] <= 0.05]
    sig_exact = [r for r in pair_summaries if np.isfinite(r["exact_perm_p"]) and r["exact_perm_p"] <= 0.05]
    lines += [
        "",
        "## Rank-label permutation tests",
        "",
        f"- Pairs with mean diagonal |cosine| above the random-rank null at p<=0.05: "
        f"**{len(sig_diag)}/{len(pair_summaries)}**.",
        f"- Pairs with exact-rank best-match rate above the random-rank null at p<=0.05: "
        f"**{len(sig_exact)}/{len(pair_summaries)}**.",
        "",
        "See `pair_summary.csv` for pair-level permutation p-values and `sv_pair_matches.csv` "
        "for the full source-SV -> target-SV matching table.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    files = discover_files(args.directions_dir)
    layers = parse_layers(args.layers, files.keys())

    capacities = {l: inspect_capacity(files[l]) for l in layers}
    d_models = {d for _, d in capacities.values()}
    if len(d_models) != 1:
        raise ValueError(f"Direction files disagree on d_model: {capacities}")

    shared_k = min(n for n, _ in capacities.values())
    k = shared_k if args.k is None else args.k
    if k <= 0 or k > shared_k:
        raise ValueError(f"k={k} invalid; maximum shared K across selected layers is {shared_k}")

    args.out.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out / "heatmaps"

    print(f"Loading {len(layers)} layers, d_model={next(iter(d_models))}, k={k}")
    banks = {l: load_bank(files[l], l, k) for l in layers}

    # Spectral gaps.
    gap_rows: list[dict] = []
    for l in layers:
        gap_rows.extend(spectral_gap_rows(banks[l], args.sv_numbering))
    gap_lookup = {(r["layer"], r["sv_index_zero_based"]): r for r in gap_rows}
    write_csv(args.out / "singular_value_gaps.csv", gap_rows)

    pairs = layer_pairs(layers, args.pairs)
    print(f"Analyzing {len(pairs)} layer pairs ({args.pairs})")

    all_match_rows: list[dict] = []
    pair_summaries: list[dict] = []
    matrix_payload: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)

    for n, (la, lb) in enumerate(pairs, start=1):
        C, match_rows, pair_summary = analyze_pair(
            banks[la], banks[lb], args.sv_numbering,
            gap_lookup, args.permutations, rng,
        )
        all_match_rows.extend(match_rows)
        pair_summaries.append(pair_summary)
        matrix_payload[f"L{la:02d}_L{lb:02d}_signed"] = C.astype(np.float32)
        matrix_payload[f"L{la:02d}_L{lb:02d}_abs"] = np.abs(C).astype(np.float32)

        if not args.no_plots and should_plot_pair(la, lb, args.heatmaps):
            plot_heatmap(
                C, la, lb,
                heatmap_dir / f"L{la:02d}_L{lb:02d}_abs_cosine.png",
                args.sv_numbering,
            )

        if n == 1 or n % 20 == 0 or n == len(pairs):
            print(
                f"[{n:>3}/{len(pairs)}] L{la:02d}->L{lb:02d}: "
                f"exact-rank={pair_summary['exact_rank_best_match_rate']:.3f}, "
                f"diag|cos|={pair_summary['mean_same_rank_abs_cosine']:.3f}, "
                f"p_diag={pair_summary['diag_perm_p']:.4g}"
            )

    write_csv(args.out / "sv_pair_matches.csv", all_match_rows)
    write_csv(args.out / "pair_summary.csv", pair_summaries)
    np.savez_compressed(args.out / "alignment_matrices.npz", **matrix_payload)

    rank_summaries = aggregate_rank_summary(all_match_rows, k, args.sv_numbering)
    write_csv(args.out / "sv_rank_summary.csv", rank_summaries)

    gap_corr = gap_correlation_summary(all_match_rows)
    write_csv(args.out / "gap_alignment_summary.csv", gap_corr)

    if not args.no_plots:
        plot_pair_summary(pair_summaries, args.out)
        plot_rank_summary(rank_summaries, args.out)

    # Focus SV parsing uses the user's DISPLAY numbering.
    focus_indices: list[int] = []
    if args.focus_svs.strip():
        vals = [int(x.strip()) for x in args.focus_svs.split(",") if x.strip()]
        focus_indices = [v if args.sv_numbering == "zero" else v - 1 for v in vals]
        focus_indices = [i for i in focus_indices if 0 <= i < k]

    summary = make_summary(
        layers, k, pair_summaries, rank_summaries, gap_corr,
        focus_indices, args.sv_numbering, args.permutations,
    )
    (args.out / "summary.md").write_text(summary, encoding="utf-8")

    metadata = {
        "directions_dir": str(args.directions_dir),
        "layers": layers,
        "d_model": next(iter(d_models)),
        "k": k,
        "pairs": args.pairs,
        "heatmaps": args.heatmaps,
        "sv_numbering": args.sv_numbering,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  {args.out / 'pair_summary.csv'}")
    print(f"  {args.out / 'sv_pair_matches.csv'}")
    print(f"  {args.out / 'sv_rank_summary.csv'}")
    print(f"  {args.out / 'singular_value_gaps.csv'}")
    print(f"  {args.out / 'gap_alignment_summary.csv'}")
    print(f"  {args.out / 'alignment_matrices.npz'}")
    print(f"  {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
