#!/usr/bin/env python3
"""
characterize_jlens_matrices.py

Characterize every square Jacobian J_l in a trained J-Lens and write ONE JSON
file containing both full spectra and compact depthwise diagnostics.

Per layer:
  - full singular-value spectrum
  - effective rank, stable rank, participation ratio, condition number
  - full eigenvalue spectrum (real/imaginary parts)
  - complex-eigenvalue statistics and spectral radius
  - normalized non-normality
  - polar-decomposition diagnostics J = QH
  - identity proximity
  - matrix norms / trace

Adjacent layer pairs:
  - Frobenius matrix similarity and relative change
  - singular-spectrum similarity
  - same-rank right-SV alignment
  - SVD-basis velocity (projective angle; sign-invariant)
  - top-k subspace principal-angle diagnostics

Default lens:
  jlens.JacobianLens.from_pretrained(
      "solarkyle/jspace-lenses",
      filename="gpt-oss-20b/lens.pt",
  )

Example:
  python characterize_jlens_matrices.py \
      --out jlens_matrix_characterization.json
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
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Characterize every Jacobian in a trained J-Lens.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--lens-path", type=str, default=None,
                     help="Optional local lens checkpoint/path.")
    src.add_argument("--hf-repo", type=str, default="solarkyle/jspace-lenses",
                     help="Hugging Face repo containing the lens.")
    p.add_argument("--hf-filename", type=str, default="gpt-oss-20b/lens.pt")
    p.add_argument("--out", type=str, default="jlens_matrix_characterization.json")
    p.add_argument("--device", type=str, default="auto",
                   help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--top-k", type=int, default=64,
                   help="Top right-SVs used for basis/subspace diagnostics.")
    p.add_argument("--complex-rel-tol", type=float, default=1e-6,
                   help="Complex if |Im(lambda)| > tol * max(1, |lambda|).")
    p.add_argument("--zero-rel-tol", type=float, default=1e-7,
                   help="Threshold for calling eigenvalue real parts near zero.")
    p.add_argument("--layers", type=str, default=None,
                   help="Optional subset, e.g. 0-5,9,10,15-22. Default: all.")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def parse_layer_spec(spec: str | None, available: List[int]) -> List[int]:
    if not spec:
        return available
    wanted = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            wanted.update(range(min(a, b), max(a, b) + 1))
        else:
            wanted.add(int(part))
    missing = sorted(wanted.difference(available))
    if missing:
        raise ValueError(f"Requested unavailable layers: {missing}. Available: {available}")
    return [x for x in available if x in wanted]


def as_int_layer_key(k: Any) -> int:
    if isinstance(k, (int, np.integer)):
        return int(k)
    s = str(k)
    digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    if digits and digits != "-":
        return int(digits)
    raise ValueError(f"Could not interpret Jacobian layer key {k!r}")


def get_jacobian_dict(lens: Any) -> Dict[int, torch.Tensor]:
    if not hasattr(lens, "jacobians"):
        raise AttributeError("Expected a trained jlens.JacobianLens with `.jacobians`.")
    raw = lens.jacobians
    if isinstance(raw, dict):
        out = {as_int_layer_key(k): torch.as_tensor(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        out = {i: torch.as_tensor(v) for i, v in enumerate(raw)}
    elif torch.is_tensor(raw) and raw.ndim == 3:
        out = {i: raw[i] for i in range(raw.shape[0])}
    else:
        raise TypeError(f"Unsupported lens.jacobians type: {type(raw)}")

    for layer, J in out.items():
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"Layer {layer} Jacobian must be square; got {tuple(J.shape)}")
    return dict(sorted(out.items()))


def load_lens(args: argparse.Namespace) -> Any:
    try:
        import jlens
    except ImportError as e:
        raise SystemExit("Could not import `jlens`; use the environment where your J-Lens code runs.") from e

    JL = jlens.JacobianLens
    if args.lens_path:
        path = args.lens_path
        if hasattr(JL, "from_pretrained"):
            try:
                return JL.from_pretrained(path)
            except Exception:
                pass
        if hasattr(JL, "load"):
            try:
                return JL.load(path)
            except Exception:
                pass
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(obj, "jacobians"):
            return obj
        raise RuntimeError(f"Could not load local lens from {path!r}.")

    return JL.from_pretrained(args.hf_repo, filename=args.hf_filename)


def f64(x: Any) -> float:
    if torch.is_tensor(x):
        x = x.detach().cpu().item()
    if isinstance(x, np.generic):
        x = x.item()
    return float(x)


def finite_or_none(x: Any) -> float | None:
    x = f64(x)
    return x if math.isfinite(x) else None


def quantile_tensor(x: torch.Tensor, qs: Iterable[float]) -> Dict[str, float]:
    qlist = list(qs)
    xf = x.detach().float()
    qv = torch.quantile(xf, torch.tensor(qlist, device=xf.device, dtype=xf.dtype))
    return {f"q{int(round(q*100)):02d}": f64(v) for q, v in zip(qlist, qv)}


def cosine_1d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> float:
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return f64(num / den.clamp_min(eps))


def safe_acos(x: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(x, -1.0, 1.0))


def entropy_effective_rank(s: torch.Tensor) -> float:
    s = s.clamp_min(0)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return f64(torch.exp(-(p * torch.log(p)).sum()))


def participation_ratio(s: torch.Tensor) -> float:
    den = torch.sum(s * s)
    if den <= 0:
        return 0.0
    return f64((s.sum() ** 2) / den)


def stable_rank(s: torch.Tensor) -> float:
    if s.numel() == 0 or s[0] <= 0:
        return 0.0
    return f64(torch.sum(s * s) / (s[0] * s[0]))


def relative_local_gaps(s: torch.Tensor, k: int) -> List[float | None]:
    vals = s.detach().cpu().double().numpy()
    n = min(k, len(vals))
    out: List[float | None] = []
    for i in range(n):
        gaps = []
        if i > 0:
            gaps.append(abs(vals[i - 1] - vals[i]))
        if i + 1 < len(vals):
            gaps.append(abs(vals[i] - vals[i + 1]))
        out.append(None if not gaps or vals[i] == 0 else float(min(gaps) / abs(vals[i])))
    return out


@torch.no_grad()
def characterize_layer(
    layer: int,
    J_in: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    top_k: int,
    complex_rel_tol: float,
    zero_rel_tol: float,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    t0 = time.time()
    J = J_in.detach().to(device=device, dtype=dtype)
    d = J.shape[0]
    k = min(top_k, d)
    eps = torch.finfo(dtype).eps

    fro = torch.linalg.vector_norm(J)
    fro_sq = fro * fro
    tr = torch.trace(J)
    sqrt_d = math.sqrt(d)

    # J = U diag(s) Vh. Row i of Vh is right singular vector v_i^T.
    U, s, Vh = torch.linalg.svd(J, full_matrices=False)
    smax, smin = s[0], s[-1]
    if smin > 0:
        cond = smax / smin
        log10_cond = torch.log10(cond)
    else:
        cond = torch.tensor(float("inf"), device=device)
        log10_cond = torch.tensor(float("inf"), device=device)

    # Identity proximity without allocating I.
    ji_sq = torch.clamp(fro_sq + d - 2.0 * tr, min=0.0)
    identity_dist = torch.sqrt(ji_sq)
    identity_cos = tr / (fro * sqrt_d) if fro > 0 else torch.tensor(0.0, device=device)

    # Polar decomposition J = QH.
    # Q = U Vh, H = V diag(s) V^T.
    # ||Q-I||_F^2 = 2d - 2tr(Q); ||H-I||_F^2 = sum_i (s_i-1)^2.
    trace_Q = torch.sum(U * Vh.T)
    q_i = torch.sqrt(torch.clamp(2.0 * d - 2.0 * trace_Q, min=0.0))
    h_i = torch.linalg.vector_norm(s - 1.0)

    # Normality: ||J^T J - J J^T||_F / ||J||_F^2.
    # Use the SVD identity instead of forming both Gram matrices.
    V = Vh.T
    Cuv = U.T @ V
    s2 = s * s
    gram_norm_sq = torch.sum(s2 * s2)
    cross = torch.sum((s2[:, None] * s2[None, :]) * (Cuv * Cuv))
    comm_sq = torch.clamp(2.0 * gram_norm_sq - 2.0 * cross, min=0.0)
    comm_norm = torch.sqrt(comm_sq)
    nonnormality = comm_norm / fro_sq.clamp_min(eps)

    # Full eigenspectrum. We do not store the d x d complex eigenvector matrix;
    # the current requested characterization only needs eigenvalues.
    eig = torch.linalg.eigvals(J)
    eig_abs = torch.abs(eig)
    eig_real = eig.real
    eig_imag = eig.imag
    eig_phase = torch.angle(eig)
    abs_imag = torch.abs(eig_imag)

    complex_threshold = complex_rel_tol * torch.maximum(torch.ones_like(eig_abs), eig_abs)
    complex_mask = abs_imag > complex_threshold
    real_scale = torch.maximum(torch.ones_like(eig_abs), eig_abs)
    zero_real_mask = torch.abs(eig_real) <= zero_rel_tol * real_scale

    spectral_radius = torch.max(eig_abs)
    rho_over_smax = spectral_radius / smax.clamp_min(eps)
    complex_abs_imag = abs_imag[complex_mask]
    complex_phase_abs_deg = torch.rad2deg(torch.abs(eig_phase[complex_mask]))

    eigen_stats: Dict[str, Any] = {
        "spectral_radius": f64(spectral_radius),
        "spectral_radius_over_top_singular_value": f64(rho_over_smax),
        "mean_abs_eigenvalue": f64(eig_abs.mean()),
        "median_abs_eigenvalue": f64(eig_abs.median()),
        "abs_eigenvalue_quantiles": quantile_tensor(eig_abs, [0.25, 0.50, 0.75, 0.90, 0.99]),
        "complex_fraction": f64(complex_mask.float().mean()),
        "complex_count": int(complex_mask.sum().item()),
        "real_only_count": int((~complex_mask).sum().item()),
        "positive_real_fraction": f64((eig_real > zero_rel_tol * real_scale).float().mean()),
        "negative_real_fraction": f64((eig_real < -zero_rel_tol * real_scale).float().mean()),
        "near_zero_real_fraction": f64(zero_real_mask.float().mean()),
        "mean_abs_imaginary_part": f64(abs_imag.mean()),
        "max_abs_imaginary_part": f64(abs_imag.max()),
        "mean_abs_phase_deg_all": f64(torch.rad2deg(torch.abs(eig_phase)).mean()),
    }
    if complex_abs_imag.numel():
        eigen_stats.update({
            "mean_abs_imaginary_part_complex_only": f64(complex_abs_imag.mean()),
            "median_abs_imaginary_part_complex_only": f64(complex_abs_imag.median()),
            "mean_abs_phase_deg_complex_only": f64(complex_phase_abs_deg.mean()),
            "median_abs_phase_deg_complex_only": f64(complex_phase_abs_deg.median()),
        })
    else:
        eigen_stats.update({
            "mean_abs_imaginary_part_complex_only": 0.0,
            "median_abs_imaginary_part_complex_only": 0.0,
            "mean_abs_phase_deg_complex_only": 0.0,
            "median_abs_phase_deg_complex_only": 0.0,
        })

    metrics = {
        "layer": int(layer),
        "dimension": int(d),
        "frobenius_norm": f64(fro),
        "trace": f64(tr),
        "trace_over_dim": f64(tr / d),
        "top_singular_value": f64(smax),
        "smallest_singular_value": f64(smin),
        "condition_number": finite_or_none(cond),
        "log10_condition_number": finite_or_none(log10_cond),
        "stable_rank": stable_rank(s),
        "entropy_effective_rank": entropy_effective_rank(s),
        "participation_ratio_rank": participation_ratio(s),
        "identity": {
            "frobenius_distance": f64(identity_dist),
            "frobenius_distance_over_sqrt_dim": f64(identity_dist / sqrt_d),
            "frobenius_cosine_with_identity": f64(identity_cos),
        },
        "polar": {
            "orthogonal_factor_trace": f64(trace_Q),
            "orthogonal_factor_trace_over_dim": f64(trace_Q / d),
            "orthogonal_factor_distance_to_identity": f64(q_i),
            "orthogonal_factor_distance_to_identity_over_sqrt_dim": f64(q_i / sqrt_d),
            "stretch_factor_distance_to_identity": f64(h_i),
            "stretch_factor_distance_to_identity_over_sqrt_dim": f64(h_i / sqrt_d),
            "stretch_factor_mean_eigenvalue": f64(s.mean()),
            "stretch_factor_top_eigenvalue": f64(smax),
            "stretch_factor_min_eigenvalue": f64(smin),
        },
        "non_normality": {
            "commutator_frobenius_norm": f64(comm_norm),
            "normalized_commutator": f64(nonnormality),
            "definition": "||J^T J - J J^T||_F / ||J||_F^2",
        },
        "eigen": eigen_stats,
        "eigenvalues_real": eig_real.detach().cpu().float().tolist(),
        "eigenvalues_imag": eig_imag.detach().cpu().float().tolist(),
        "singular_values": s.detach().cpu().float().tolist(),
        "top_k_relative_local_singular_gaps": relative_local_gaps(s, k),
        "seconds_for_layer": float(time.time() - t0),
    }

    carry = {"J": J, "s": s, "Vh": Vh}

    del U, V, Cuv, eig
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, carry


@torch.no_grad()
def characterize_adjacent(
    layer_a: int,
    prev: Dict[str, torch.Tensor],
    layer_b: int,
    curr: Dict[str, torch.Tensor],
    top_k: int,
) -> Dict[str, Any]:
    Ja, Jb = prev["J"], curr["J"]
    sa, sb = prev["s"], curr["s"]
    Vha, Vhb = prev["Vh"], curr["Vh"]
    d = Ja.shape[0]
    k = min(top_k, d)

    mat_cos = cosine_1d(Ja, Jb)
    diff = torch.linalg.vector_norm(Jb - Ja)
    rel_change_a = diff / torch.linalg.vector_norm(Ja).clamp_min(1e-30)
    rel_change_sym = 2.0 * diff / (
        torch.linalg.vector_norm(Ja) + torch.linalg.vector_norm(Jb)
    ).clamp_min(1e-30)

    spectrum_cos = cosine_1d(sa, sb)
    tiny = torch.finfo(sa.dtype).tiny
    log_spectrum_cos = cosine_1d(torch.log(sa.clamp_min(tiny)), torch.log(sb.clamp_min(tiny)))

    # Same-ranked right SVs. Use |cos| because each SVD vector has arbitrary sign.
    same_rank_signed_full = torch.sum(Vha * Vhb, dim=1)
    same_rank_abs_full = torch.abs(same_rank_signed_full).clamp(0, 1)
    angles_full = torch.rad2deg(safe_acos(same_rank_abs_full))

    same_rank_signed_k = same_rank_signed_full[:k]
    same_rank_abs_k = same_rank_abs_full[:k]
    angles_k = angles_full[:k]

    # Top-k subspace principal angles. This ignores rank swaps/rotations within the subspace.
    Va_k = Vha[:k].T
    Vb_k = Vhb[:k].T
    cross_k = Va_k.T @ Vb_k
    principal_cos = torch.linalg.svdvals(cross_k).clamp(0, 1)
    principal_angles = torch.rad2deg(safe_acos(principal_cos))

    return {
        "layer_a": int(layer_a),
        "layer_b": int(layer_b),
        "matrix": {
            "frobenius_cosine": float(mat_cos),
            "frobenius_difference": f64(diff),
            "relative_change_vs_layer_a": f64(rel_change_a),
            "symmetric_relative_change": f64(rel_change_sym),
        },
        "singular_spectrum": {
            "cosine": float(spectrum_cos),
            "log_spectrum_cosine": float(log_spectrum_cos),
            "top_singular_value_ratio_b_over_a": f64(sb[0] / sa[0].clamp_min(1e-30)),
        },
        "right_svd_basis": {
            "top_k": int(k),
            "top_k_same_rank_mean_abs_cosine": f64(same_rank_abs_k.mean()),
            "top_k_same_rank_median_abs_cosine": f64(same_rank_abs_k.median()),
            "top_k_same_rank_mean_projective_angle_deg": f64(angles_k.mean()),
            "top_k_same_rank_median_projective_angle_deg": f64(angles_k.median()),
            "top_k_same_rank_raw_sign_flip_fraction": f64((same_rank_signed_k < 0).float().mean()),
            "full_basis_same_rank_mean_abs_cosine": f64(same_rank_abs_full.mean()),
            "full_basis_same_rank_mean_projective_angle_deg": f64(angles_full.mean()),
            "top_k_same_rank_signed_cosines": same_rank_signed_k.detach().cpu().float().tolist(),
            "top_k_same_rank_abs_cosines": same_rank_abs_k.detach().cpu().float().tolist(),
            "top_k_same_rank_projective_angles_deg": angles_k.detach().cpu().float().tolist(),
        },
        "top_k_subspace": {
            "principal_cosines": principal_cos.detach().cpu().float().tolist(),
            "principal_angles_deg": principal_angles.detach().cpu().float().tolist(),
            "mean_principal_cosine": f64(principal_cos.mean()),
            "min_principal_cosine": f64(principal_cos.min()),
            "mean_principal_angle_deg": f64(principal_angles.mean()),
            "max_principal_angle_deg": f64(principal_angles.max()),
        },
    }


def build_global_summary(layers: List[Dict[str, Any]], adjacent: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not layers:
        return {}

    summary: Dict[str, Any] = {
        "num_layers": len(layers),
        "layer_range": [layers[0]["layer"], layers[-1]["layer"]],
    }

    if adjacent:
        summary["adjacent_means"] = {
            "matrix_frobenius_cosine": float(np.mean([x["matrix"]["frobenius_cosine"] for x in adjacent])),
            "matrix_symmetric_relative_change": float(np.mean([x["matrix"]["symmetric_relative_change"] for x in adjacent])),
            "top_k_same_rank_abs_cosine": float(np.mean([x["right_svd_basis"]["top_k_same_rank_mean_abs_cosine"] for x in adjacent])),
            "top_k_basis_velocity_deg": float(np.mean([x["right_svd_basis"]["top_k_same_rank_mean_projective_angle_deg"] for x in adjacent])),
            "top_k_subspace_mean_principal_angle_deg": float(np.mean([x["top_k_subspace"]["mean_principal_angle_deg"] for x in adjacent])),
        }

    extrema_specs = {
        "highest_non_normality": (("non_normality", "normalized_commutator"), "max"),
        "highest_complex_eigen_fraction": (("eigen", "complex_fraction"), "max"),
        "highest_effective_rank": (("entropy_effective_rank",), "max"),
        "closest_to_identity": (("identity", "frobenius_distance_over_sqrt_dim"), "min"),
        "largest_polar_rotation_from_identity": (("polar", "orthogonal_factor_distance_to_identity_over_sqrt_dim"), "max"),
    }
    extrema = {}
    for name, (path, mode) in extrema_specs.items():
        vals = []
        for row in layers:
            x: Any = row
            for key in path:
                x = x[key]
            if x is not None:
                vals.append((float(x), int(row["layer"])))
        if vals:
            value, layer = min(vals) if mode == "min" else max(vals)
            extrema[name] = {"layer": layer, "value": value}
    summary["extrema"] = extrema
    return summary


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (int, np.integer, str, bool)) or obj is None:
        return obj
    return obj


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch_dtype(args.dtype)

    print(f"[info] device={device} dtype={dtype}")
    print("[info] loading trained J-Lens...")
    lens = load_lens(args)
    jacobians = get_jacobian_dict(lens)
    available = sorted(jacobians)
    layers_to_run = parse_layer_spec(args.layers, available)

    print(f"[info] available layers: {available}")
    print(f"[info] analyzing layers: {layers_to_run}")
    if not layers_to_run:
        raise SystemExit("No layers selected.")

    layer_results: List[Dict[str, Any]] = []
    adjacent_results: List[Dict[str, Any]] = []
    prev_layer: int | None = None
    prev_carry: Dict[str, torch.Tensor] | None = None
    total_t0 = time.time()

    for idx, layer in enumerate(layers_to_run, start=1):
        print(f"[{idx}/{len(layers_to_run)}] layer {layer}: SVD + eigenspectrum + diagnostics...")
        metrics, carry = characterize_layer(
            layer=layer,
            J_in=jacobians[layer],
            dtype=dtype,
            device=device,
            top_k=args.top_k,
            complex_rel_tol=args.complex_rel_tol,
            zero_rel_tol=args.zero_rel_tol,
        )
        layer_results.append(metrics)

        if prev_carry is not None and prev_layer is not None and layer == prev_layer + 1:
            pair = characterize_adjacent(prev_layer, prev_carry, layer, carry, args.top_k)
            adjacent_results.append(pair)
            print(
                "         "
                f"J-cos={pair['matrix']['frobenius_cosine']:.4f} | "
                f"top-{pair['right_svd_basis']['top_k']} basis cos="
                f"{pair['right_svd_basis']['top_k_same_rank_mean_abs_cosine']:.4f} | "
                f"velocity={pair['right_svd_basis']['top_k_same_rank_mean_projective_angle_deg']:.2f}°"
            )

        if prev_carry is not None:
            del prev_carry
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        prev_layer = layer
        prev_carry = carry

    if prev_carry is not None:
        del prev_carry

    result = {
        "format_version": 1,
        "description": "Matrix-level characterization of trained J-Lens Jacobians. SVD indices are zero-based.",
        "metadata": {
            "lens_path": args.lens_path,
            "hf_repo": None if args.lens_path else args.hf_repo,
            "hf_filename": None if args.lens_path else args.hf_filename,
            "device": str(device),
            "dtype": args.dtype,
            "top_k": int(args.top_k),
            "complex_rel_tol": float(args.complex_rel_tol),
            "zero_rel_tol": float(args.zero_rel_tol),
            "available_layers": available,
            "analyzed_layers": layers_to_run,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "total_seconds": float(time.time() - total_t0),
        },
        "interpretation_notes": {
            "svd_basis_velocity": "Mean arccos(|v_l,i dot v_l+1,i|) in degrees. Absolute cosine removes arbitrary SVD sign.",
            "non_normality": "||J^T J - J J^T||_F / ||J||_F^2. Higher means greater departure from a normal matrix.",
            "polar": "J=QH. Q is orthogonal rotation/reflection; H is PSD stretch.",
            "complex_eigenvalues": "Complex eigenvalues indicate rotation-like structure in the real linear map; they are not themselves evidence of nonlinear computation.",
            "linearity_caveat": "Direct nonlinearity claims require Jacobian variability across contexts or finite-perturbation tests.",
        },
        "summary": None,
        "layers": layer_results,
        "adjacent_pairs": adjacent_results,
    }
    result["summary"] = build_global_summary(layer_results, adjacent_results)
    result = sanitize_json(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {out}")
    print(f"[done] total time: {time.time() - total_t0:.1f}s")
    print("[done] Return this single JSON file for analysis.")


if __name__ == "__main__":
    main()
