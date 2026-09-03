#!/usr/bin/env python3
"""
compute_name_overlap.py

Pure geometry, no forward passes: for every (subject, position-mode, band,
layer, position) in one or more ablation runs, measure how much the ablated
J-lens subspace overlaps the subject's name direction(s).

For layer l and token t the lens vector is v_t = J_l^T (gamma * u_t), exactly
as in the ablation scripts. Let Q be an orthonormal basis of the span of the
k chosen (ablated) lens vectors at a given layer/position, and n_hat the
normalized name vector (or, with variants, an orthonormal basis N of the
name span). Reported per record:

  name_in_ablation   ||Q^T n_hat||^2 : the fraction of the name direction
                     lying inside the ablated subspace (0 = orthogonal,
                     1 = fully contained). With a name span, the mean over
                     its basis vectors (= ||Q^T N||_F^2 / dim N).
  ablation_on_name   ||N^T Q||_F^2 / k : the fraction of the ablated subspace
                     lying along the name span.
  top3_overlap       the three chosen tokens whose individual lens vectors
                     have the largest |cos| with n_hat, with the cosines —
                     which associations carry the name.
  null_mean/null_sd  the same name_in_ablation statistic for random sets of
                     k vocabulary tokens at that layer (--n-null draws):
                     the anisotropy-aware chance level. The isotropic chance
                     level is k / d_model (~0.0035 for k=10, d=2880).

Needs the run dir(s) only for removed_lens_tokens.json and lens_readouts.json
(subject spans). Loads W_U and the final-norm weight directly from the model's
safetensors shards when possible (no 20B instantiation); falls back to a full
model load otherwise.

Usage:
    python compute_name_overlap.py \
        --run-dirs jspace_v3 jspace_v3_last \
        --name-variants \
        --out name_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

VARIANT_SUFFIXES = ("", "'s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="Ablation out dirs containing removed_lens_tokens.json and lens_readouts.json.")
    p.add_argument("--out", default="name_overlap.csv")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--condition", default="jspace",
                   help="Which logged condition's chosen sets to analyze.")
    p.add_argument("--name-source", choices=["self", "last"], default="self")
    p.add_argument("--name-variants", action="store_true")
    p.add_argument("--no-norm-weight", action="store_true")
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------- weights -------


def load_unembedding(model_name: str):
    """Return (W_U [vocab, d] float32 tensor, gamma [d] float32 or None, tokenizer)."""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    try:
        W, g = _load_from_safetensors(model_name)
        print("[weights] loaded lm_head + final norm from safetensors shards")
        return W, g, tok
    except Exception as exc:
        print(f"[weights] safetensors path failed ({exc}); falling back to full model load", file=sys.stderr)
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="cuda")
    except TypeError:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
    W = hf.lm_head.weight.detach().float().cpu()
    norm = hf.model.norm if hasattr(hf.model, "norm") else None
    g = norm.weight.detach().float().cpu() if norm is not None and hasattr(norm, "weight") else None
    del hf
    torch.cuda.empty_cache()
    return W, g, tok


def _load_from_safetensors(model_name: str):
    import glob
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = snapshot_download(model_name, allow_patterns=["*.safetensors.index.json", "*.safetensors", "config.json"])
    idx_files = glob.glob(os.path.join(snap, "*.safetensors.index.json"))
    want = {"lm_head.weight": None, "model.norm.weight": None, "model.embed_tokens.weight": None}
    if idx_files:
        index = json.load(open(idx_files[0]))["weight_map"]
        shards = {}
        for key in want:
            if key in index:
                shards.setdefault(index[key], []).append(key)
        for shard, keys in shards.items():
            with safe_open(os.path.join(snap, shard), framework="pt", device="cpu") as f:
                for key in keys:
                    want[key] = f.get_tensor(key)
    else:
        for st in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(st, framework="pt", device="cpu") as f:
                for key in list(want):
                    if want[key] is None and key in f.keys():
                        want[key] = f.get_tensor(key)
    W = want["lm_head.weight"] if want["lm_head.weight"] is not None else want["model.embed_tokens.weight"]
    if W is None:
        raise RuntimeError("no lm_head.weight or model.embed_tokens.weight in shards")
    g = want["model.norm.weight"]
    return W.float(), (g.float() if g is not None else None)


# ------------------------------------------------------------- geometry ------


class LensVectors:
    def __init__(self, lens, W_U: torch.Tensor, gamma: torch.Tensor | None):
        self.lens = lens
        self.W = W_U
        self.g = gamma
        self._Jt: dict[int, torch.Tensor] = {}

    def vec(self, layer: int, token_ids: list[int]) -> torch.Tensor:
        """[d, m] lens vectors v_t = J^T (gamma * u_t)."""
        if layer not in self._Jt:
            self._Jt[layer] = self.lens.jacobians[layer].detach().float().cpu()
        U = self.W[token_ids]
        if self.g is not None:
            U = U * self.g
        return (U @ self._Jt[layer]).T.contiguous()


def orth(V: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(V)
    return Q


def subspace_stats(Vc: torch.Tensor, Vn: torch.Tensor):
    """(name_in_ablation, ablation_on_name, per-chosen |cos| with the first name vector)."""
    Qc, Qn = orth(Vc), orth(Vn)
    M = Qc.T @ Qn  # [k, m]
    name_in = float((M ** 2).sum() / Qn.shape[1])
    abl_on = float((M ** 2).sum() / Qc.shape[1])
    n0 = Vn[:, 0] / Vn[:, 0].norm()
    cs = (Vc / Vc.norm(dim=0, keepdim=True)).T @ n0
    return name_in, abl_on, cs.abs()


# --------------------------------------------------------------- main --------


def main() -> None:
    args = parse_args()
    import jlens

    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    W, g, tok = load_unembedding(args.model)
    if args.no_norm_weight:
        g = None
    lv = LensVectors(lens, W, g)
    vocab = W.shape[0]
    rng = np.random.default_rng(args.seed)

    def encode_ids(text):
        try:
            return tok(text, add_special_tokens=False).input_ids
        except TypeError:
            return tok(text).input_ids

    def name_ids_for(position_tokens: list[str], pos_index: int) -> list[int]:
        base_str = position_tokens[-1] if args.name_source == "last" else position_tokens[pos_index]
        base_ids = encode_ids(base_str)
        idset = set(base_ids[:1]) if len(base_ids) == 1 else set()
        if not idset:  # decode round-trip failed; recover via single-token encode of stripped form
            idset = set(encode_ids(base_str.strip())[:1])
        if args.name_variants:
            stem = base_str.strip()
            for sfx in VARIANT_SUFFIXES:
                for lead in ("", " "):
                    e = encode_ids(lead + stem + sfx)
                    if len(e) == 1:
                        idset.add(e[0])
        return sorted(idset)

    null_cache: dict[tuple[int, int, tuple], tuple[float, float]] = {}

    def null_stats(layer: int, k: int, Vn: torch.Tensor, key):
        ck = (layer, k, key)
        if ck not in null_cache:
            vals = []
            for _ in range(args.n_null):
                ids = rng.integers(0, vocab, size=k).tolist()
                ni, _, _ = subspace_stats(lv.vec(layer, ids), Vn)
                vals.append(ni)
            null_cache[ck] = (float(np.mean(vals)), float(np.std(vals)))
        return null_cache[ck]

    rows = []
    for run in args.run_dirs:
        run = Path(run)
        REM = json.load(open(run / "removed_lens_tokens.json"))
        RD = json.load(open(run / "lens_readouts.json"))
        for subject, per_pm in REM.items():
            for pmode, per_band in per_pm.items():
                for band, per_cond in per_band.items():
                    per_rel = per_cond.get(args.condition)
                    if not per_rel:
                        continue
                    first_leaf = next(iter(per_rel.values()))
                    if not (isinstance(first_leaf, dict) and "position_tokens" in first_leaf):
                        raise SystemExit(
                            f"{run}/removed_lens_tokens.json is the old (v2/v3) format, whose position "
                            "labels are ambiguous across relations. Re-log with jspace_ablate_subject_v3_1.py "
                            "(a one-relation run suffices) and point --run-dirs there.")
                    rel_name = "birth_city" if "birth_city" in per_rel else next(iter(per_rel))
                    entry = per_rel[rel_name]
                    postok = entry["position_tokens"]
                    stoks = [postok[p] for p in sorted(postok, key=int)]
                    pos_to_idx = {p: i for i, p in enumerate(sorted(postok, key=int))}
                    for layer_s, per_pos in entry["tokens"].items():
                        layer = int(layer_s)
                        for pos_s, chosen_toks in per_pos.items():
                            if not chosen_toks or pos_s not in pos_to_idx:
                                continue
                            chosen_ids = [encode_ids(t)[0] for t in chosen_toks if len(encode_ids(t)) == 1]
                            if len(chosen_ids) < len(chosen_toks):
                                # fall back for tokens that don't round-trip: skip them
                                pass
                            if not chosen_ids:
                                continue
                            n_ids = name_ids_for(stoks, pos_to_idx[pos_s])
                            if not n_ids:
                                continue
                            Vc = lv.vec(layer, chosen_ids)
                            Vn = lv.vec(layer, n_ids)
                            name_in, abl_on, cs = subspace_stats(Vc, Vn)
                            order = torch.argsort(cs, descending=True)[:3]
                            top3 = "; ".join(f"{chosen_toks[i]}:{cs[i]:.2f}" for i in order.tolist())
                            nmu, nsd = null_stats(layer, len(chosen_ids), Vn, tuple(n_ids))
                            rows.append({
                                "run": str(run), "subject": subject, "position_mode": pmode, "band": band,
                                "layer": layer, "position": int(pos_s), "position_token": postok[pos_s], "relation_context": rel_name,
                                "k": len(chosen_ids), "name_dim": len(n_ids),
                                "name_explicitly_chosen": int(any(t.strip() == postok[pos_s].strip() for t in chosen_toks)),
                                "name_in_ablation": round(name_in, 4), "ablation_on_name": round(abl_on, 4),
                                "null_mean": round(nmu, 4), "null_sd": round(nsd, 4),
                                "excess_over_null_sd": round((name_in - nmu) / max(nsd, 1e-6), 1),
                                "top3_overlapping_removed": top3,
                            })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arr = np.array([r["name_in_ablation"] for r in rows])
    nul = np.array([r["null_mean"] for r in rows])
    print(f"\n[{len(rows)} records] name_in_ablation: median {np.median(arr):.3f}, "
          f"IQR [{np.percentile(arr,25):.3f}, {np.percentile(arr,75):.3f}], max {arr.max():.3f} "
          f"| null median {np.median(nul):.4f} | isotropic k/d ~ {rows[0]['k']/2880:.4f}")
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r["name_in_ablation"])
    print("median by layer: " + "  ".join(f"L{l}:{np.median(v):.3f}" for l, v in sorted(by_layer.items())))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
