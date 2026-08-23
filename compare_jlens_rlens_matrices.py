#!/usr/bin/env python3
"""
compare_jlens_rlens_matrices.py

Matrix analysis of matched J-lens / R-lens pairs for:
  - Qwen/Qwen3.6-27B
  - google/gemma-3-27b-it

Source repo:
  camilablank/workspace-lenses

Writes ONE JSON file containing:
  * per-layer matrix diagnostics for J-lens and R-lens
  * within-lens adjacent-layer geometry
  * direct same-layer J-vs-R comparisons
  * full singular/eigen spectra (unless --skip-eigen)
  * rank-band summaries over zero-based SV ranges

Expected checkpoint format (per repo README):
  dict with keys J, source_layers, d_model, n_prompts, provenance

SVD indices are ZERO-BASED everywhere.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO_ID = "camilablank/workspace-lenses"
MODEL_SPECS = {
    "qwen3.6-27b": {
        "display_name": "Qwen/Qwen3.6-27B",
        "j_path": "qwen3.6-27b/j-lens/lens.pt",
        "r_path": "qwen3.6-27b/r-lens/lens.pt",
    },
    "gemma-3-27b-it": {
        "display_name": "google/gemma-3-27b-it",
        "j_path": "gemma-3-27b-it/j-lens/lens.pt",
        "r_path": "gemma-3-27b-it/r-lens/lens.pt",
    },
}
RANK_BANDS = [(0, 3), (4, 7), (8, 15), (16, 31), (32, 63)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="jlens_rlens_matrix_analysis.json")
    p.add_argument("--models", default="qwen3.6-27b,gemma-3-27b-it")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--layers", default=None,
                   help="Optional subset, e.g. 0-12,20,30-40")
    p.add_argument("--complex-rel-tol", type=float, default=1e-6)
    p.add_argument("--zero-rel-tol", type=float, default=1e-7)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-eigen", action="store_true",
                   help="Skip eigvals for a faster run; SVD-based analysis still runs.")
    return p.parse_args()


def device_from_arg(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def dtype_from_arg(s):
    return torch.float64 if s == "float64" else torch.float32


def parse_layers(spec: Optional[str], available: List[int]) -> List[int]:
    if not spec:
        return list(available)
    wanted = set()
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        if "-" in x:
            a, b = map(int, x.split("-", 1))
            wanted.update(range(min(a, b), max(a, b) + 1))
        else:
            wanted.add(int(x))
    missing = sorted(wanted - set(available))
    if missing:
        raise ValueError(f"Unavailable requested source layers: {missing}")
    return [x for x in available if x in wanted]


def ff(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().item()
    if isinstance(x, np.generic):
        x = x.item()
    return float(x)


def finite_or_none(x):
    x = ff(x)
    return x if math.isfinite(x) else None


def safe_acos(x):
    return torch.acos(torch.clamp(x, -1.0, 1.0))


def cosine_flat(a, b, eps=1e-30):
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return ff(num / den.clamp_min(eps))


def quantiles(x, qs=(0.25, 0.5, 0.75, 0.9, 0.99)):
    q = torch.tensor(qs, device=x.device, dtype=x.dtype)
    vals = torch.quantile(x, q)
    return {f"q{int(100*p):02d}": ff(v) for p, v in zip(qs, vals)}


def entropy_effective_rank(s):
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return ff(torch.exp(-(p * torch.log(p)).sum()))


def stable_rank(s):
    return ff(torch.sum(s * s) / (s[0] * s[0])) if s[0] > 0 else 0.0


def participation_ratio(s):
    den = torch.sum(s * s)
    return ff((s.sum() ** 2) / den) if den > 0 else 0.0


def local_relative_gaps(s, k):
    a = s.detach().cpu().double().numpy()
    n = min(k, len(a))
    out = []
    for i in range(n):
        gs = []
        if i > 0:
            gs.append(abs(a[i-1] - a[i]))
        if i + 1 < len(a):
            gs.append(abs(a[i] - a[i+1]))
        out.append(None if not gs or a[i] == 0 else float(min(gs) / abs(a[i])))
    return out


def consecutive_relative_gaps(s, k):
    n = min(k, s.numel() - 1)
    out = []
    for i in range(n):
        out.append(None if s[i] == 0 else ff((s[i] - s[i+1]) / s[i]))
    return out


def valid_bands(k):
    out = []
    for a, b in RANK_BANDS:
        if a < k:
            out.append((a, min(b, k-1)))
    return out


def download(filename, args):
    print(f"[download] {REPO_ID}/{filename}")
    return hf_hub_download(
        REPO_ID,
        filename=filename,
        repo_type="model",
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(ckpt)}")
    for key in ("J", "source_layers", "d_model"):
        if key not in ckpt:
            raise KeyError(f"Missing checkpoint key {key!r}")
    J = ckpt["J"] if torch.is_tensor(ckpt["J"]) else torch.as_tensor(ckpt["J"])
    layers = [int(x) for x in ckpt["source_layers"]]
    if J.ndim != 3 or J.shape[0] != len(layers) or J.shape[1] != J.shape[2]:
        raise ValueError(f"Unexpected J stack shape {tuple(J.shape)} for {len(layers)} source layers")
    return ckpt, layers, J


@torch.no_grad()
def characterize_matrix(layer, M_cpu, device, dtype, args):
    t0 = time.time()
    M = M_cpu.detach().to(device=device, dtype=dtype)
    d = M.shape[0]
    k = min(args.top_k, d)
    eps = torch.finfo(dtype).eps

    fro = torch.linalg.vector_norm(M)
    fro2 = fro * fro
    tr = torch.trace(M)

    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    smax, smin = s[0], s[-1]
    cond = smax / smin if smin > 0 else torch.tensor(float("inf"), device=device)

    # Identity diagnostics without allocating I.
    ident_dist = torch.sqrt(torch.clamp(fro2 + d - 2.0 * tr, min=0.0))
    ident_cos = tr / (fro * math.sqrt(d)).clamp_min(eps)

    # Polar J = QH from SVD.
    trace_q = torch.sum(U * Vh.T)
    q_dist = torch.sqrt(torch.clamp(2.0*d - 2.0*trace_q, min=0.0))
    h_dist = torch.linalg.vector_norm(s - 1.0)

    # Non-normality using the SVD.
    V = Vh.T
    C = U.T @ V
    s2 = s * s
    gram_norm_sq = torch.sum(s2 * s2)
    cross = torch.sum((s2[:, None] * s2[None, :]) * (C * C))
    comm_norm = torch.sqrt(torch.clamp(2*gram_norm_sq - 2*cross, min=0.0))
    nonnorm = comm_norm / fro2.clamp_min(eps)

    row = {
        "layer": int(layer),
        "dimension": int(d),
        "frobenius_norm": ff(fro),
        "trace": ff(tr),
        "trace_over_dim": ff(tr/d),
        "top_singular_value": ff(smax),
        "smallest_singular_value": ff(smin),
        "condition_number": finite_or_none(cond),
        "log10_condition_number": finite_or_none(torch.log10(cond)),
        "stable_rank": stable_rank(s),
        "entropy_effective_rank": entropy_effective_rank(s),
        "participation_ratio_rank": participation_ratio(s),
        "identity": {
            "frobenius_distance": ff(ident_dist),
            "distance_over_sqrt_dim": ff(ident_dist/math.sqrt(d)),
            "frobenius_cosine_with_identity": ff(ident_cos),
        },
        "polar": {
            "orthogonal_factor_trace_over_dim": ff(trace_q/d),
            "orthogonal_factor_distance_to_identity_over_sqrt_dim": ff(q_dist/math.sqrt(d)),
            "stretch_factor_distance_to_identity_over_sqrt_dim": ff(h_dist/math.sqrt(d)),
            "stretch_factor_mean_eigenvalue": ff(s.mean()),
            "stretch_factor_top_eigenvalue": ff(smax),
            "stretch_factor_min_eigenvalue": ff(smin),
        },
        "non_normality": {
            "normalized_commutator": ff(nonnorm),
            "commutator_frobenius_norm": ff(comm_norm),
            "definition": "||M^T M - M M^T||_F / ||M||_F^2",
        },
        "singular_values": s.detach().cpu().float().tolist(),
        "top_k_relative_local_singular_gaps": local_relative_gaps(s, k),
        "top_k_consecutive_relative_singular_gaps": consecutive_relative_gaps(s, k),
    }

    if not args.skip_eigen:
        eig = torch.linalg.eigvals(M)
        ea, er, ei = torch.abs(eig), eig.real, eig.imag
        phase = torch.angle(eig)
        ai = torch.abs(ei)
        complex_thresh = args.complex_rel_tol * torch.maximum(torch.ones_like(ea), ea)
        cmask = ai > complex_thresh
        real_scale = torch.maximum(torch.ones_like(ea), ea)
        zmask = torch.abs(er) <= args.zero_rel_tol * real_scale

        row["eigen"] = {
            "spectral_radius": ff(ea.max()),
            "spectral_radius_over_top_singular_value": ff(ea.max()/smax.clamp_min(eps)),
            "mean_abs_eigenvalue": ff(ea.mean()),
            "median_abs_eigenvalue": ff(ea.median()),
            "abs_eigenvalue_quantiles": quantiles(ea),
            "complex_fraction": ff(cmask.float().mean()),
            "complex_count": int(cmask.sum().item()),
            "real_only_count": int((~cmask).sum().item()),
            "positive_real_fraction": ff((er > args.zero_rel_tol*real_scale).float().mean()),
            "negative_real_fraction": ff((er < -args.zero_rel_tol*real_scale).float().mean()),
            "near_zero_real_fraction": ff(zmask.float().mean()),
            "mean_abs_imaginary_part": ff(ai.mean()),
            "max_abs_imaginary_part": ff(ai.max()),
            "mean_abs_phase_deg_all": ff(torch.rad2deg(torch.abs(phase)).mean()),
            "mean_abs_phase_deg_complex_only": ff(torch.rad2deg(torch.abs(phase[cmask])).mean()) if cmask.any() else 0.0,
        }
        row["eigenvalues_real"] = er.detach().cpu().float().tolist()
        row["eigenvalues_imag"] = ei.detach().cpu().float().tolist()
        del eig, ea, er, ei, phase, ai

    row["seconds_for_layer"] = time.time() - t0

    carry = {
        "M": M,
        "s": s,
        "Vh_top": Vh[:k].detach(),
    }

    del U, V, Vh, C
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, carry


@torch.no_grad()
def compare_carry(layer_a, A, layer_b, B, top_k):
    Ma, Mb = A["M"], B["M"]
    sa, sb = A["s"], B["s"]
    Va, Vb = A["Vh_top"], B["Vh_top"]
    k = min(top_k, Va.shape[0], Vb.shape[0])

    diff = torch.linalg.vector_norm(Mb - Ma)
    na, nb = torch.linalg.vector_norm(Ma), torch.linalg.vector_norm(Mb)

    signed = torch.sum(Va[:k] * Vb[:k], dim=1)
    ab = torch.abs(signed).clamp(0, 1)
    ang = torch.rad2deg(safe_acos(ab))

    cross = Va[:k] @ Vb[:k].T
    pc = torch.linalg.svdvals(cross).clamp(0, 1)
    pa = torch.rad2deg(safe_acos(pc))

    band_alignment = []
    for a, b in valid_bands(k):
        band_alignment.append({
            "rank_start": a,
            "rank_end": b,
            "label": f"SV{a:02d}-SV{b:02d}",
            "mean_abs_cosine": ff(ab[a:b+1].mean()),
            "mean_projective_angle_deg": ff(ang[a:b+1].mean()),
        })

    ga = consecutive_relative_gaps(sa, k)
    gb = consecutive_relative_gaps(sb, k)
    gap_bands = []
    max_gap_rank = min(len(ga), len(gb))
    for a, b in valid_bands(max_gap_rank):
        xa = [x for x in ga[a:b+1] if x is not None]
        xb = [x for x in gb[a:b+1] if x is not None]
        gap_bands.append({
            "rank_start": a,
            "rank_end": b,
            "label": f"SV{a:02d}-SV{b:02d}",
            "mean_consecutive_relative_gap_a": float(np.mean(xa)) if xa else None,
            "mean_consecutive_relative_gap_b": float(np.mean(xb)) if xb else None,
        })

    return {
        "layer_a": int(layer_a),
        "layer_b": int(layer_b),
        "matrix": {
            "frobenius_cosine": cosine_flat(Ma, Mb),
            "frobenius_difference": ff(diff),
            "relative_difference_vs_a": ff(diff/na.clamp_min(1e-30)),
            "symmetric_relative_difference": ff(2*diff/(na+nb).clamp_min(1e-30)),
        },
        "singular_spectrum": {
            "cosine": cosine_flat(sa, sb),
            "log_spectrum_cosine": cosine_flat(
                torch.log(sa.clamp_min(torch.finfo(sa.dtype).tiny)),
                torch.log(sb.clamp_min(torch.finfo(sb.dtype).tiny)),
            ),
            "top_singular_value_ratio_b_over_a": ff(sb[0]/sa[0].clamp_min(1e-30)),
        },
        "right_svd_basis": {
            "top_k": int(k),
            "mean_same_rank_abs_cosine": ff(ab.mean()),
            "median_same_rank_abs_cosine": ff(ab.median()),
            "mean_projective_angle_deg": ff(ang.mean()),
            "median_projective_angle_deg": ff(ang.median()),
            "raw_sign_flip_fraction": ff((signed < 0).float().mean()),
            "same_rank_signed_cosines": signed.detach().cpu().float().tolist(),
            "same_rank_abs_cosines": ab.detach().cpu().float().tolist(),
            "projective_angles_deg": ang.detach().cpu().float().tolist(),
            "rank_band_alignment": band_alignment,
        },
        "top_k_subspace": {
            "principal_cosines": pc.detach().cpu().float().tolist(),
            "principal_angles_deg": pa.detach().cpu().float().tolist(),
            "mean_principal_cosine": ff(pc.mean()),
            "min_principal_cosine": ff(pc.min()),
            "mean_principal_angle_deg": ff(pa.mean()),
            "max_principal_angle_deg": ff(pa.max()),
        },
        "rank_band_spectral_gaps": gap_bands,
    }


def release(c):
    if c is None:
        return
    c.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_lens(path, lens_kind, model_key, args, device, dtype):
    print(f"\n[load] {model_key} / {lens_kind}")
    ckpt, source_layers, stack = load_ckpt(path)
    selected = parse_layers(args.layers, source_layers)
    idx = {l: i for i, l in enumerate(source_layers)}

    provenance = ckpt.get("provenance", {})
    if isinstance(provenance, str):
        provenance = {"raw": provenance}

    result = {
        "lens_kind": lens_kind,
        "source_layers": selected,
        "d_model": int(ckpt["d_model"]),
        "n_prompts": int(ckpt.get("n_prompts", -1)),
        "provenance": provenance,
        "layers": [],
        "adjacent_pairs": [],
    }

    prev_layer, prev = None, None
    for n, layer in enumerate(selected, 1):
        print(f"[{model_key} {lens_kind}] {n}/{len(selected)} layer {layer}")
        row, carry = characterize_matrix(layer, stack[idx[layer]], device, dtype, args)
        result["layers"].append(row)

        if prev is not None and prev_layer is not None and layer == prev_layer + 1:
            cmp = compare_carry(prev_layer, prev, layer, carry, args.top_k)
            cmp["comparison_type"] = "adjacent_within_lens"
            result["adjacent_pairs"].append(cmp)
            print(
                f"  matrix={cmp['matrix']['frobenius_cosine']:.4f} "
                f"basis={cmp['right_svd_basis']['mean_same_rank_abs_cosine']:.4f} "
                f"vel={cmp['right_svd_basis']['mean_projective_angle_deg']:.1f}° "
                f"subspace={cmp['top_k_subspace']['mean_principal_angle_deg']:.1f}°"
            )
        release(prev)
        prev_layer, prev = layer, carry

    release(prev)
    del stack, ckpt
    gc.collect()
    return result


def analyze_j_vs_r(j_path, r_path, model_key, args, device, dtype):
    print(f"\n[cross] matched J-vs-R: {model_key}")
    jck, jl, js = load_ckpt(j_path)
    rck, rl, rs = load_ckpt(r_path)
    common = sorted(set(jl) & set(rl))
    common = parse_layers(args.layers, common)
    ji, ri = {l:i for i,l in enumerate(jl)}, {l:i for i,l in enumerate(rl)}

    out = []
    for n, layer in enumerate(common, 1):
        print(f"[{model_key} J-vs-R] {n}/{len(common)} layer {layer}")
        # Reuse full characterizer but omit eigvals for this direct comparison pass.
        old = args.skip_eigen
        args.skip_eigen = True
        _, jc = characterize_matrix(layer, js[ji[layer]], device, dtype, args)
        _, rc = characterize_matrix(layer, rs[ri[layer]], device, dtype, args)
        args.skip_eigen = old

        cmp = compare_carry(layer, jc, layer, rc, args.top_k)
        cmp["comparison_type"] = "matched_j_vs_r_same_layer"
        cmp["a"] = "j-lens"
        cmp["b"] = "r-lens"
        out.append(cmp)
        print(
            f"  J/R matrix={cmp['matrix']['frobenius_cosine']:.4f} "
            f"basis={cmp['right_svd_basis']['mean_same_rank_abs_cosine']:.4f} "
            f"subspace={cmp['top_k_subspace']['mean_principal_angle_deg']:.1f}°"
        )
        release(jc); release(rc)

    del js, rs, jck, rck
    gc.collect()
    return out


def mean_path(rows, path):
    vals = []
    for row in rows:
        x = row
        ok = True
        for k in path:
            if not isinstance(x, dict) or k not in x:
                ok = False; break
            x = x[k]
        if ok and isinstance(x, (int, float)) and math.isfinite(float(x)):
            vals.append(float(x))
    return float(np.mean(vals)) if vals else None


def summarize_lens(res):
    L, A = res["layers"], res["adjacent_pairs"]
    return {
        "num_layers": len(L),
        "layer_range": [L[0]["layer"], L[-1]["layer"]] if L else None,
        "means": {
            "non_normality": mean_path(L, ("non_normality", "normalized_commutator")),
            "identity_cosine": mean_path(L, ("identity", "frobenius_cosine_with_identity")),
            "polar_q_distance_over_sqrt_dim": mean_path(L, ("polar", "orthogonal_factor_distance_to_identity_over_sqrt_dim")),
            "entropy_effective_rank": mean_path(L, ("entropy_effective_rank",)),
            "participation_ratio_rank": mean_path(L, ("participation_ratio_rank",)),
            "eigen_mean_abs_phase_deg": mean_path(L, ("eigen", "mean_abs_phase_deg_all")),
            "eigen_complex_fraction": mean_path(L, ("eigen", "complex_fraction")),
            "adjacent_matrix_cosine": mean_path(A, ("matrix", "frobenius_cosine")),
            "adjacent_topk_basis_abs_cosine": mean_path(A, ("right_svd_basis", "mean_same_rank_abs_cosine")),
            "adjacent_topk_basis_velocity_deg": mean_path(A, ("right_svd_basis", "mean_projective_angle_deg")),
            "adjacent_topk_subspace_angle_deg": mean_path(A, ("top_k_subspace", "mean_principal_angle_deg")),
        },
        "adjacent_timeline": [
            {
                "layer_a": x["layer_a"],
                "layer_b": x["layer_b"],
                "matrix_cosine": x["matrix"]["frobenius_cosine"],
                "basis_abs_cosine": x["right_svd_basis"]["mean_same_rank_abs_cosine"],
                "basis_velocity_deg": x["right_svd_basis"]["mean_projective_angle_deg"],
                "subspace_angle_deg": x["top_k_subspace"]["mean_principal_angle_deg"],
            } for x in A
        ],
    }


def summarize_jr(rows):
    return {
        "num_common_layers": len(rows),
        "means": {
            "matrix_cosine": mean_path(rows, ("matrix", "frobenius_cosine")),
            "spectrum_cosine": mean_path(rows, ("singular_spectrum", "cosine")),
            "basis_abs_cosine": mean_path(rows, ("right_svd_basis", "mean_same_rank_abs_cosine")),
            "basis_angle_deg": mean_path(rows, ("right_svd_basis", "mean_projective_angle_deg")),
            "subspace_angle_deg": mean_path(rows, ("top_k_subspace", "mean_principal_angle_deg")),
        },
        "timeline": [
            {
                "layer": x["layer_a"],
                "matrix_cosine": x["matrix"]["frobenius_cosine"],
                "spectrum_cosine": x["singular_spectrum"]["cosine"],
                "basis_abs_cosine": x["right_svd_basis"]["mean_same_rank_abs_cosine"],
                "basis_angle_deg": x["right_svd_basis"]["mean_projective_angle_deg"],
                "subspace_angle_deg": x["top_k_subspace"]["mean_principal_angle_deg"],
            } for x in rows
        ],
    }


def sanitize(x):
    if isinstance(x, dict): return {str(k): sanitize(v) for k,v in x.items()}
    if isinstance(x, list): return [sanitize(v) for v in x]
    if isinstance(x, tuple): return [sanitize(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (np.floating, float)):
        y = float(x)
        return y if math.isfinite(y) else None
    if isinstance(x, (int, str, bool)) or x is None: return x
    return x


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    dtype = dtype_from_arg(args.dtype)
    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    bad = [x for x in requested if x not in MODEL_SPECS]
    if bad:
        raise SystemExit(f"Unknown model keys: {bad}; valid={list(MODEL_SPECS)}")

    t0 = time.time()
    out = {
        "format_version": 1,
        "description": "Matched J-lens/R-lens matrix analysis; SVD ranks are zero-based.",
        "metadata": {
            "repo_id": REPO_ID,
            "models": requested,
            "device": str(device),
            "dtype": args.dtype,
            "top_k": args.top_k,
            "rank_bands_zero_based": [list(x) for x in valid_bands(args.top_k)],
            "skip_eigen": args.skip_eigen,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "interpretation_notes": {
            "spectral_cosine": "Cosine of the ordered singular-value vectors; spectrum shape up to scale, not singular-vector geometry.",
            "svd_basis_velocity": "arccos(|v_l,i dot v_l+1,i|); sign invariant.",
            "complex_eigenvalues": "Complex eigenvalues describe rotational structure of the linear map; not nonlinearity by themselves.",
            "subspace_warning": "When singular values are near-degenerate, individual SVs may rotate while their subspace remains stable; inspect principal angles.",
        },
        "models": {},
    }

    print(f"[info] repo={REPO_ID}")
    print(f"[info] device={device} dtype={dtype} top_k={args.top_k}")

    for key in requested:
        spec = MODEL_SPECS[key]
        print("\n" + "="*80)
        print(f"[model] {key} ({spec['display_name']})")
        print("="*80)

        jpath = download(spec["j_path"], args)
        rpath = download(spec["r_path"], args)

        jres = analyze_lens(jpath, "j-lens", key, args, device, dtype)
        rres = analyze_lens(rpath, "r-lens", key, args, device, dtype)
        jr = analyze_j_vs_r(jpath, rpath, key, args, device, dtype)

        out["models"][key] = {
            "display_name": spec["display_name"],
            "files": {"j_lens": spec["j_path"], "r_lens": spec["r_path"]},
            "j_lens": jres,
            "r_lens": rres,
            "j_vs_r_same_layer": jr,
            "summary": {
                "j_lens": summarize_lens(jres),
                "r_lens": summarize_lens(rres),
                "j_vs_r": summarize_jr(jr),
            },
        }

        # Checkpoint after each model.
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")
        print(f"[checkpoint] {p}")

        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    out["metadata"]["total_seconds"] = time.time() - t0
    p = Path(args.out)
    p.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")
    print(f"\n[done] wrote {p}")
    print(f"[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
