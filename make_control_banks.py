#!/usr/bin/env python3
"""
make_control_banks.py

Generate Stage-1 control direction banks from an existing J-lens right-SV
direction bank (the OUT/directions/LXX.npz files written by
scan_sparse_jlens_directions.py). Two kinds of control:

  rotated  V_rot = V @ Q with Q ~ Haar(k).  Random *orthonormal* directions
           inside the span of the top-k right singular vectors. Isolates
           "are the eigen-axes special" from "is the subspace special".
           S is set to the exact induced gain ||J w_i|| = ||diag(S) Q[:,i]||,
           which is exact because w_i lies entirely in the top-k right
           subspace.

  random   Haar-random orthonormal k-frame in the full d_model space.
           Tests whether merely being in the top-J subspace matters.
           S is set to the exact gain ||J w_i|| if the J-lens can be loaded
           (import jlens; requires torch + the lens checkpoint, already
           cached if you ran the scanner), otherwise to 1.0 with a warning.
           S never enters the Stage-1 tail statistics either way; it only
           feeds sigma_* convenience columns.

Output banks are drop-in compatible with the scanner's --directions-dir
(arrays "V" [d_model, k] float32 and "S" [k] float32, one LXX.npz per
source layer, same filenames).

Typical usage (defaults mirror the scanner's):

    python make_control_banks.py \
        --source-dir unrealized_words_fineweb/directions \
        --out-root control_banks \
        --n-replicates 2 \
        --seed 0

producing control_banks/rotated_r0, control_banks/rotated_r1,
control_banks/random_r0, control_banks/random_r1, each with LXX.npz files
and a bank_metadata.json.

Then scan each bank with the SAME corpus arguments and seed as the J-SV
run so all conditions see identical token windows (paired design):

    python scan_sparse_jlens_directions.py --out scan_rot_r0 \
        --directions-dir control_banks/rotated_r0 \
        --k 64 --n-docs 2000 --seed 0 --skip-unembedding

RNG is deterministic per (base seed, kind, replicate, layer), so replicates
are reproducible and independent across layers.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

KIND_ID = {"rotated": 1, "random": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate rotated / random control direction banks for Stage 1.",
    )
    p.add_argument(
        "--source-dir",
        required=True,
        help="Directions dir with LXX.npz (arrays V, S[, U]) from the J-SV scan.",
    )
    p.add_argument("--out-root", required=True, help="Root directory for control banks.")
    p.add_argument(
        "--kinds",
        nargs="+",
        default=["rotated", "random"],
        choices=sorted(KIND_ID),
        help="Which control banks to generate.",
    )
    p.add_argument("--n-replicates", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Override d_model inference (only needed if V is square).",
    )
    p.add_argument(
        "--no-lens",
        action="store_true",
        help="Do not load the J-lens for exact random-bank gains; store S=1.0.",
    )
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    args = p.parse_args()
    if args.n_replicates <= 0:
        p.error("--n-replicates must be positive")
    return args


def load_source(path: Path, d_model_override: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return V as [d_model, k] float64 and S as [k] float64."""
    z = np.load(path)
    if "V" not in z or "S" not in z:
        raise ValueError(f"{path} must contain arrays named V and S")
    V = np.asarray(z["V"], dtype=np.float64)
    S = np.asarray(z["S"], dtype=np.float64).reshape(-1)
    if V.ndim != 2:
        raise ValueError(f"{path}: V must be 2D, got {V.shape}")

    if d_model_override is not None:
        if V.shape[0] == d_model_override:
            pass
        elif V.shape[1] == d_model_override:
            V = V.T
        else:
            raise ValueError(
                f"{path}: neither axis matches --d-model={d_model_override}; shape={V.shape}"
            )
    else:
        if V.shape[0] == V.shape[1]:
            raise ValueError(
                f"{path}: V is square ({V.shape}); pass --d-model to disambiguate"
            )
        if V.shape[0] < V.shape[1]:
            V = V.T  # store as [d_model, k]; d_model is the larger axis

    k = V.shape[1]
    if S.shape[0] < k:
        raise ValueError(f"{path}: S has {S.shape[0]} values but V has {k} columns")
    return V, S[:k]


def haar_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed k x k orthogonal matrix (QR with sign fix)."""
    g = rng.standard_normal((k, k))
    q, r = np.linalg.qr(g)
    d = np.sign(np.diagonal(r))
    d[d == 0] = 1.0
    return q * d


def haar_frame(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform orthonormal k-frame in R^d (columns), via reduced QR."""
    g = rng.standard_normal((d, k))
    q, r = np.linalg.qr(g)  # q: [d, k]
    dsign = np.sign(np.diagonal(r))
    dsign[dsign == 0] = 1.0
    return q * dsign


def orthonormality_defect(V: np.ndarray) -> float:
    k = V.shape[1]
    return float(np.abs(V.T @ V - np.eye(k)).max())


def find_layer_files(source_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for f in sorted(glob.glob(str(source_dir / "L*.npz"))):
        m = re.fullmatch(r"L(\d+)\.npz", Path(f).name)
        if m:
            out.append((int(m.group(1)), Path(f)))
    if not out:
        raise FileNotFoundError(f"No LXX.npz files found in {source_dir}")
    return out


class LensGains:
    """Lazy exact ||J w|| gains for full-space random directions."""

    def __init__(self, repo: str, filename: str) -> None:
        self.repo = repo
        self.filename = filename
        self._lens = None
        self.available = True

    def _load(self) -> None:
        if self._lens is not None or not self.available:
            return
        try:
            import jlens  # noqa: PLC0415

            self._lens = jlens.JacobianLens.from_pretrained(
                self.repo, filename=self.filename
            )
        except Exception as exc:  # ImportError, download failure, ...
            print(
                f"[warn] could not load J-lens for exact random-bank gains ({exc}); "
                "falling back to S=1.0",
                file=sys.stderr,
            )
            self.available = False

    def gains(self, layer: int, W: np.ndarray) -> np.ndarray | None:
        """W is [d_model, k]; returns [k] gains or None if unavailable."""
        self._load()
        if not self.available:
            return None
        import torch  # noqa: PLC0415

        try:
            J = self._lens.jacobians[layer]
        except Exception as exc:
            print(
                f"[warn] lens has no Jacobian for layer {layer} ({exc}); S=1.0",
                file=sys.stderr,
            )
            return None
        J32 = J.detach().to(dtype=torch.float32, device="cpu")
        Wt = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32))
        return torch.linalg.norm(J32 @ Wt, dim=0).numpy().astype(np.float64)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_root = Path(args.out_root)
    layer_files = find_layer_files(source_dir)
    print(f"[source] {len(layer_files)} layer files in {source_dir}")

    lens_gains = None
    if "random" in args.kinds and not args.no_lens:
        lens_gains = LensGains(args.lens_repo, args.lens_file)

    for kind in args.kinds:
        for rep in range(args.n_replicates):
            bank_dir = out_root / f"{kind}_r{rep}"
            bank_dir.mkdir(parents=True, exist_ok=True)
            meta: dict = {
                "kind": kind,
                "replicate": rep,
                "base_seed": args.seed,
                "rng_scheme": "np.random.default_rng([seed, kind_id, replicate, layer])",
                "source_dir": str(source_dir),
                "layers": [],
                "gains": None,
                "max_source_orthonormality_defect": 0.0,
                "max_output_orthonormality_defect": 0.0,
            }

            for layer, path in layer_files:
                V, S = load_source(path, args.d_model)
                d_model, k = V.shape
                src_defect = orthonormality_defect(V)
                meta["max_source_orthonormality_defect"] = max(
                    meta["max_source_orthonormality_defect"], src_defect
                )
                if src_defect > 1e-3:
                    print(
                        f"[warn] {path.name}: source V not orthonormal "
                        f"(defect {src_defect:.2e}); rotated gains are approximate",
                        file=sys.stderr,
                    )

                rng = np.random.default_rng([args.seed, KIND_ID[kind], rep, layer])

                if kind == "rotated":
                    Q = haar_orthogonal(k, rng)
                    V_new = V @ Q
                    # Exact: ||J (V Q e_i)|| = ||diag(S) Q[:, i]||.
                    S_new = np.sqrt(((S[:, None] * Q) ** 2).sum(axis=0))
                    meta["gains"] = "exact (||diag(S) Q[:,i]||, within top-k subspace)"
                else:  # random
                    V_new = haar_frame(d_model, k, rng)
                    g = None
                    if lens_gains is not None:
                        g = lens_gains.gains(layer, V_new)
                    if g is None:
                        S_new = np.ones(k, dtype=np.float64)
                        meta["gains"] = "placeholder (S=1.0); sigma_* columns meaningless"
                    else:
                        S_new = g
                        meta["gains"] = "exact (||J w|| via loaded J-lens)"

                out_defect = orthonormality_defect(V_new)
                meta["max_output_orthonormality_defect"] = max(
                    meta["max_output_orthonormality_defect"], out_defect
                )

                np.savez_compressed(
                    bank_dir / path.name,
                    V=V_new.astype(np.float32),
                    S=S_new.astype(np.float32),
                )
                meta["layers"].append(
                    {"layer": layer, "d_model": int(d_model), "k": int(k)}
                )

            with (bank_dir / "bank_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(
                f"[bank] {bank_dir}  layers={len(meta['layers'])}  "
                f"gains={meta['gains']}  "
                f"out_defect={meta['max_output_orthonormality_defect']:.2e}"
            )

    print("\nDone. Scan each bank with scan_sparse_jlens_directions.py using")
    print("--directions-dir <bank> and IDENTICAL corpus args (--seed, --n-docs,")
    print("--max-seq-len, dataset flags) so all conditions see the same tokens.")


if __name__ == "__main__":
    main()
