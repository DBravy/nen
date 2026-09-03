#!/usr/bin/env python3
"""
build_g_direction_bank.py

Export the top-k eigenvectors (equivalently the SVD, since the matrix is
symmetric) of the gated-gain matrix

    G_l = S_l - Jbar_l^T Jbar_l,    S_l = E_prompt[ mean_pos J_p^T J_p ]

as an LXX.npz direction bank in the exact format scan_sparse_jlens_directions.py
and scan_unrealized_words.py accept via --directions-dir. The scanners then
gather all their usual data (activation moments, recruitment frequencies, top
contexts, unembedding neighbors, selectivity rankings) over G's eigenvectors
instead of the lens Jacobian's right singular vectors. With --directions-dir
they load V and S verbatim and recompute nothing.

Relationship to build_second_moment.py: its `analyze` subcommand ALREADY
exports scanner-compatible banks at OUT/G_bank and OUT/S_bank (S values are
sqrt-eigenvalues). If you have run analyze, you can skip this script and point
the scanners straight at OUT/G_bank. This script exists for when you want a
bank without rerunning the full analysis, or want what analyze does not do:

  * sign canonicalization matching the scanners' own convention (largest
    |coordinate| positive), so positive/negative context polarities are
    deterministic and comparable across reruns; analyze's export keeps
    eigh's arbitrary signs, so its vectors can differ from these by sign
  * a choice of S convention (--s-mode sqrt matches analyze's RMS-gain
    scale and is the default; --s-mode eig stores raw eigenvalues, the
    literal singular values of G)
  * per-layer eigen diagnostics (negative eigenvalue mass, effective rank)
    and raw eigenvalue spectra stored inside each npz
  * per-layer provenance in OUT/metadata.json

Input format (pinned to build_second_moment.py's harvest):
  second_moment.npz with keys
    S_L{ll:02d}          [d, d]  raw second moment S_l (probes are standard
                                 normal, so no scale correction is needed)
    shared_means_L{ll:02d} [n_prompts, m_shared, d]  position-averaged probe
                                 gradients J_p^T r for the shared probes
    shared_probes        [m_shared, d]
    n_prompts, config_json
  Falls back to second_moment.pt (torch) when the npz is absent, like analyze.

Jbar resolution order: --lens-npz (from build_second_moment.py export-lens,
keys J_L{ll:02d}; torch-free) > --lens-path > the lens_path recorded in the
harvest's config_json > hub download. Use the LOCAL refit from
fit_local_lens.py when the downloaded lens failed the diagnose check; the
subtraction in G is only meaningful when S and Jbar come from the same
numerics. When shared probes are present, the same consistency check analyze
performs (mean probe gradient vs Jbar^T r) is printed per layer.

A probe-sketch fallback (--g-source probes) builds a G-like matrix from
shared_means alone when no S_LXX keys exist. Caveat: shared means are
position-averaged, so the sketch estimates the across-prompt covariance of
position-averaged Jacobians and MISSES the within-prompt position variance
that S_l's G includes. It is a degraded fallback, not the same quantity.

Usage
  python build_g_direction_bank.py inspect --input sm
  python build_g_direction_bank.py build --input sm --lens-path local_lens.pt \
      --out g_bank --k 64
  python scan_sparse_jlens_directions.py --directions-dir g_bank/directions \
      --out g_scan_sparse --k 64 --layers <printed by build>
  python scan_unrealized_words.py --directions-dir g_bank/directions \
      --out g_scan_words --k 64 --layers <printed by build>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------- input handling -------------------------------


def resolve_input(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        for name in ("second_moment.npz", "second_moment.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(
            f"{p} contains neither second_moment.npz nor second_moment.pt"
        )
    if not p.exists():
        raise FileNotFoundError(p)
    return p


class Harvest:
    """Uniform view over second_moment.npz or second_moment.pt."""

    def __init__(self, path: Path):
        self.path = path
        self.S: dict[int, np.ndarray] = {}
        self.G: dict[int, np.ndarray] = {}
        self.shared_means: dict[int, np.ndarray] = {}
        self.shared_probes: np.ndarray | None = None
        self.config: dict[str, Any] = {}
        self.n_prompts: int | None = None
        self.raw_keys: list[tuple[str, tuple, str]] = []  # (key, shape, interpretation)

        if path.suffix == ".pt":
            import torch

            blob = torch.load(path, map_location="cpu", weights_only=False)
            self.S = {int(l): S.double().numpy() for l, S in blob["S"].items()}
            self.shared_means = {int(l): v.numpy() for l, v in blob["shared_means"].items()}
            self.shared_probes = blob["shared_probes"].numpy()
            self.config = dict(blob.get("config", {}))
            self.config.setdefault("lens_path", blob.get("lens_path"))
            self.n_prompts = int(blob["n_prompts"])
            for k in ("S", "shared_means", "shared_probes", "n_prompts", "config"):
                self.raw_keys.append((k, (), "torch blob field"))
            return

        z = np.load(path)
        for key in z.files:
            a = z[key]
            interp = ""
            if m := re.fullmatch(r"S_L(\d+)", key):
                self.S[int(m.group(1))] = np.asarray(a, dtype=np.float64)
                interp = f"raw second moment S_l (layer {int(m.group(1))})"
            elif m := re.fullmatch(r"G_L(\d+)", key):
                self.G[int(m.group(1))] = np.asarray(a, dtype=np.float64)
                interp = f"precomputed G (layer {int(m.group(1))})"
            elif m := re.fullmatch(r"shared_means_L(\d+)", key):
                self.shared_means[int(m.group(1))] = np.asarray(a)
                interp = f"position-averaged probe gradients (layer {int(m.group(1))})"
            elif key == "shared_probes":
                self.shared_probes = np.asarray(a)
                interp = "shared Hutchinson probes"
            elif key == "n_prompts":
                self.n_prompts = int(a)
                interp = "prompt count"
            elif key == "config_json":
                self.config = json.loads(str(a))
                interp = "harvest config (incl. lens_path)"
            self.raw_keys.append((key, tuple(np.shape(a)), interp or "unused"))

    @property
    def d_model(self) -> int:
        if self.config.get("d_model"):
            return int(self.config["d_model"])
        for src in (self.S, self.G):
            for M in src.values():
                return int(M.shape[0])
        if self.shared_probes is not None:
            return int(self.shared_probes.shape[-1])
        raise ValueError("could not determine d_model from harvest")

    @property
    def layers(self) -> list[int]:
        return sorted(set(self.S) | set(self.G) | set(self.shared_means))


# ----------------------------- Jbar -----------------------------------------


def load_jbar(args: argparse.Namespace, layers: list[int], config: dict[str, Any]) -> tuple[dict[int, np.ndarray], str] | tuple[None, str]:
    if args.lens_npz:
        z = np.load(args.lens_npz)
        out = {}
        for l in layers:
            key = f"J_L{l:02d}"
            if key in z.files:
                out[l] = np.asarray(z[key], dtype=np.float64)
        return out, f"npz:{args.lens_npz}"

    lens_path = args.lens_path or config.get("lens_path")
    if lens_path is None and not args.lens_repo:
        return None, "none"
    import jlens  # same dependency the scanners use

    if lens_path:
        lens = jlens.JacobianLens.load(lens_path)
        src = str(lens_path)
        if not args.lens_path:
            print(
                f"[lens] defaulting to the harvest's recorded lens: {src}. If your "
                f"fit_local_lens.py diagnose showed the downloaded lens mismatching "
                f"this machine's numerics, pass --lens-path with the local refit."
            )
    else:
        lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
        src = f"{args.lens_repo}/{args.lens_file}"
        print("[warn] Jbar from a hub lens; the G subtraction is only valid on matching numerics.")
    out = {l: lens.jacobians[l].double().cpu().numpy() for l in layers if l in lens.jacobians}
    print(f"[lens] Jbar from {src}: layers {sorted(out)} (n_prompts={lens.n_prompts})")
    return out, src


def consistency_check(h: Harvest, jbar: dict[int, np.ndarray], layers: list[int]) -> dict[int, float]:
    """Same check as build_second_moment.py analyze: mean probe gradient vs
    Jbar^T r. Near-1 cos and ratio mean S and Jbar share estimator and dtype,
    so G's subtraction is meaningful."""
    out: dict[int, float] = {}
    if h.shared_probes is None or not h.shared_means:
        return out
    print(f"{'layer':>5} {'cos':>7} {'|ours|/|lens|':>14}   (both near 1 => G subtraction is meaningful)")
    for l in layers:
        if l not in h.shared_means or l not in jbar:
            continue
        ours = h.shared_means[l].mean(axis=0)  # [m_shared, d]
        ref = h.shared_probes @ jbar[l]        # rows: (Jbar^T r)^T
        cos = np.array([float(o @ r / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-12)) for o, r in zip(ours, ref)])
        ratio = np.array([np.linalg.norm(o) / (np.linalg.norm(r) + 1e-12) for o, r in zip(ours, ref)])
        out[l] = float(cos.mean())
        flag = "" if cos.mean() > 0.95 and abs(ratio.mean() - 1) < 0.1 else "  <-- check lens/numerics"
        print(f"{l:>5} {cos.mean():>7.3f} {ratio.mean():>14.3f}{flag}")
    return out


# ----------------------------- G assembly -----------------------------------


def sketch_G_from_means(means: np.ndarray, probes: np.ndarray, jbar_l: np.ndarray | None) -> np.ndarray:
    """Fallback when no S_LXX keys exist. Position-averaged, so this misses
    within-prompt position variance; see module docstring."""
    d = probes.shape[-1]
    scale = d / max(float(np.mean(np.sum(probes.astype(np.float64) ** 2, axis=-1))), 1e-30)
    P, m, _ = means.shape
    X = means.reshape(P * m, d).astype(np.float64)
    M = (X.T @ X) * (scale / (P * m))
    if jbar_l is not None:
        T = jbar_l.T @ jbar_l
    else:
        gbar = means.astype(np.float64).mean(axis=0)  # [m, d]
        T = (gbar.T @ gbar) * (scale / m)
    return M - T


def assemble_G(
    h: Harvest,
    layers: list[int],
    jbar: dict[int, np.ndarray] | None,
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    G_out: dict[int, np.ndarray] = {}
    prov: dict[int, dict[str, Any]] = {}
    for l in layers:
        mode = args.g_source
        if mode == "auto":
            mode = "g" if l in h.G else "m" if l in h.S else "probes"
        info: dict[str, Any] = {"mode": mode}

        if mode == "g":
            if l not in h.G:
                raise ValueError(f"layer {l}: --g-source g but no G_L{l:02d} key")
            G = h.G[l]
            info["source"] = f"G_L{l:02d}"
        elif mode == "m":
            if l not in h.S:
                raise ValueError(f"layer {l}: --g-source m but no S_L{l:02d} key")
            if jbar is None or l not in jbar:
                raise ValueError(
                    f"layer {l}: have S_L{l:02d} but no Jbar; pass --lens-path or --lens-npz"
                )
            S = 0.5 * (h.S[l] + h.S[l].T)
            T = jbar[l].T @ jbar[l]
            G = S - T
            total, invariant = float(np.trace(S)), float(np.trace(T))
            info.update(
                source=f"S_L{l:02d} - Jbar^T Jbar",
                trace_S=total,
                trace_invariant=invariant,
                gated_fraction=1.0 - invariant / max(total, 1e-30),
            )
            if total < invariant:
                print(
                    f"[warn] L{l:02d}: trace(S) < trace(Jbar^T Jbar) "
                    f"({total:.4g} < {invariant:.4g}); the lens does not match the "
                    f"harvest's network, or estimator noise dominates."
                )
        elif mode == "probes":
            if l not in h.shared_means or h.shared_probes is None:
                raise ValueError(f"layer {l}: probe sketch needs shared_probes and shared_means_L{l:02d}")
            print(
                f"[warn] L{l:02d}: probe-sketch fallback (position-averaged); this is "
                f"not identical to the S_LXX-based G. Prefer a real harvest."
            )
            jl = jbar.get(l) if jbar else None
            G = sketch_G_from_means(h.shared_means[l], h.shared_probes, jl)
            P, m, _ = h.shared_means[l].shape
            info.update(
                source=f"shared_means_L{l:02d} (P={P}, m={m})",
                mode="probes_minus_lens" if jl is not None else "probes_covariance",
            )
            if P * m < 4 * args.k:
                print(f"[warn] L{l:02d}: only P*m={P*m} samples for k={args.k}; noisy eigenvectors.")
        else:
            raise ValueError(f"unknown --g-source {mode!r}")

        G_out[l] = 0.5 * (G + G.T)
        prov[l] = info
    return G_out, prov


# ----------------------------- eigendecomposition ---------------------------


def canonicalize_signs(V: np.ndarray) -> np.ndarray:
    """Same convention as the scanners: largest-|coordinate| of each column is
    made positive, so stored context polarities are deterministic. Note that
    build_second_moment.py analyze's G_bank export does NOT do this, so its
    vectors can differ from these by a sign flip."""
    V = V.copy()
    for i in range(V.shape[1]):
        anchor = int(np.argmax(np.abs(V[:, i])))
        if V[anchor, i] < 0:
            V[:, i] *= -1.0
    return V


def eig_bank(G: np.ndarray, k: int, s_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    d = G.shape[0]
    if k > d:
        raise ValueError(f"k={k} exceeds d_model={d}")
    w, Q = np.linalg.eigh(G)  # ascending
    w = w[::-1]
    Q = Q[:, ::-1]
    pos = np.clip(w, 0.0, None)
    neg_mass = float(np.abs(w[w < 0]).sum() / max(pos.sum(), 1e-30))
    eff_rank = float(pos.sum() ** 2 / max((pos**2).sum(), 1e-30))
    diag = {
        "lambda_max": float(w[0]),
        "lambda_min": float(w[-1]),
        "negative_eigenvalue_mass": neg_mass,
        "effective_rank": eff_rank,
        "top8_eigenvalues": [float(x) for x in w[:8]],
    }
    if float(w[k - 1]) <= 0:
        print(f"[warn] eigenvalue rank {k} is non-positive ({w[k-1]:.4g}); k exceeds the meaningful rank of G")
    Vk = canonicalize_signs(Q[:, :k])
    lam_k = pos[:k]
    S = np.sqrt(lam_k) if s_mode == "sqrt" else lam_k
    return Vk.astype(np.float32), S.astype(np.float32), w.astype(np.float32), diag


# ----------------------------- subcommands ----------------------------------


def parse_layer_arg(spec: str, available: list[int]) -> list[int]:
    if spec.strip().lower() == "all":
        return sorted(available)
    req = sorted({int(x) for x in spec.split(",") if x.strip()})
    missing = [l for l in req if l not in available]
    if missing:
        raise ValueError(f"layers {missing} not in harvest (available: {sorted(available)})")
    return req


def cmd_inspect(args: argparse.Namespace) -> None:
    path = resolve_input(args.input)
    h = Harvest(path)
    print(f"[inspect] {path}  d_model={h.d_model}  n_prompts={h.n_prompts}")
    if h.config:
        cfg = {k: h.config[k] for k in ("model", "probes", "shared_probes", "lens_path") if k in h.config}
        print(f"[config] {cfg}")
    print(f"{'key':<28} {'shape':<24} interpretation")
    for key, shape, interp in h.raw_keys:
        print(f"{key:<28} {str(shape):<24} {interp}")
    print(f"\nLayers with usable data: {h.layers}")
    print(f"Layers with S_LXX (preferred G source): {sorted(h.S)}")
    if not h.S and not h.G:
        print("No S/G matrices found; only the degraded probe-sketch fallback is possible.")


def cmd_build(args: argparse.Namespace) -> None:
    path = resolve_input(args.input)
    h = Harvest(path)
    if not h.layers:
        raise SystemExit("no per-layer arrays found in the harvest; run inspect")
    layers = parse_layer_arg(args.layers, h.layers)
    print(f"[build] input={path} d_model={h.d_model} layers={layers} k={args.k} s_mode={args.s_mode}")

    jbar, jbar_src = load_jbar(args, layers, h.config)
    consistency: dict[int, float] = {}
    if jbar:
        consistency = consistency_check(h, jbar, layers)

    G_all, prov = assemble_G(h, layers, jbar, args)

    out_dir = Path(args.out)
    ddir = out_dir / "directions"
    ddir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "input": str(path),
        "d_model": h.d_model,
        "harvest_n_prompts": h.n_prompts,
        "k": args.k,
        "s_mode": args.s_mode,
        "g_source": args.g_source,
        "jbar_source": jbar_src,
        "definition": "G = S - Jbar^T Jbar with S = E_prompt[mean_pos J_p^T J_p]; "
        "V = top-k eigenvectors (sign-canonicalized), S = sqrt-eigenvalues "
        "(s_mode=sqrt, matching build_second_moment analyze) or raw eigenvalues (s_mode=eig)",
        "layers": {},
    }

    for l in layers:
        V, S, w_full, diag = eig_bank(G_all[l], args.k, args.s_mode)
        out_path = ddir / f"L{l:02d}.npz"
        np.savez_compressed(
            out_path,
            V=V,
            S=S,
            U=V,  # symmetric matrix: left = right singular vectors
            eigvals=w_full,
            s_mode=np.array(args.s_mode),
            g_source=np.array(prov[l]["mode"]),
        )
        entry = {**prov[l], **diag, "file": str(out_path)}
        if l in consistency:
            entry["consistency_cos"] = consistency[l]
        meta["layers"][f"L{l:02d}"] = entry
        gf = prov[l].get("gated_fraction")
        gf_txt = f" gated_frac={gf:.3f}" if gf is not None else ""
        print(
            f"[bank] L{l:02d}: {prov[l]['mode']:<12}{gf_txt} lam_max={diag['lambda_max']:.4g} "
            f"eff_rank={diag['effective_rank']:.1f} neg_mass={diag['negative_eigenvalue_mass']:.3f} "
            f"-> {out_path}"
        )

    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] wrote {out_dir / 'metadata.json'}")

    layer_csv = ",".join(str(l) for l in layers)
    print("\nNext, scan the corpus over these directions (the scanners load the bank")
    print("verbatim and recompute nothing; scan_sparse gathers a superset of what")
    print("scan_unrealized_words gathers, so one scan may be enough):\n")
    print(
        f"  python scan_sparse_jlens_directions.py --directions-dir {ddir} "
        f"--out g_scan_sparse --k {args.k} --layers {layer_csv} --n-docs 2000"
    )
    print(
        f"  python scan_unrealized_words.py --directions-dir {ddir} "
        f"--out g_scan_words --k {args.k} --layers {layer_csv} --n-docs 2000"
    )
    print(
        "\n(build_second_moment.py analyze also exports OUT/G_bank in this format, "
        "without sign canonicalization or the eigvals/diagnostics extras.)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "build"):
        s = sub.add_parser(name)
        s.add_argument("--input", required=True, help="harvest dir or second_moment.npz/.pt path")
        if name == "build":
            s.add_argument("--out", default="g_bank")
            s.add_argument("--k", type=int, default=64)
            s.add_argument("--layers", default="all")
            s.add_argument(
                "--g-source",
                choices=["auto", "g", "m", "probes"],
                default="auto",
                help="auto: G_LXX key > S_LXX with lens Jbar > position-averaged probe sketch",
            )
            s.add_argument("--s-mode", choices=["sqrt", "eig"], default="sqrt",
                           help="sqrt: RMS gain, matches analyze's G_bank; eig: raw eigenvalues")
            s.add_argument("--lens-path", default=None, help="lens.pt for Jbar (use your LOCAL refit)")
            s.add_argument("--lens-npz", default=None, help="export-lens npz (J_LXX keys; torch-free)")
            s.add_argument("--lens-repo", default=None)
            s.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
            s.set_defaults(func=cmd_build)
        else:
            s.set_defaults(func=cmd_inspect)
    args = p.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
