#!/usr/bin/env python3
"""
fit_local_lens.py

Fit a Jacobian lens on the model AS IT RUNS ON THIS MACHINE (e.g. gpt-oss-20b
through MXFP4 kernels), using the exact recipe behind solarkyle/jspace-lenses:
the first N WikiText-103 prompts, 128-token windows, positions 0-15 and the
last excluded, target = last block, one-hot cotangents at every valid target
position, mean over source positions and prompts. This is just jlens.fit with
the author's arguments; it exists so that E[J^T J] (build_second_moment.py) and
Jbar are measured on the same network and their subtraction (G) is valid at
every layer.

Subcommands
  diagnose  ~10 min. One exact per-prompt Jacobian at one layer, compared to
            (a) the probe-gradient mean from an existing harvest (same model:
            expect cos ~1) and (b) the downloaded lens (different numerics:
            expect cos well below 1 at early/mid layers). Settles the cause
            before spending hours.
  fit       Hours. Checkpointed every few prompts and resumable; you can fit 50
            prompts now and rerun with --n-prompts 100 later to extend.
  compare   Seconds. Local lens vs downloaded lens: Frobenius cosine per layer,
            top-k subspace overlap, and for named axes (e.g. L07_SV39) the
            closest local singular vector and its rank.

Usage
  python fit_local_lens.py diagnose --harvest sm --layer 8
  python fit_local_lens.py fit --out local_lens.pt --n-prompts 100 --dim-batch 8
  python fit_local_lens.py compare --local local_lens.pt --axes L07_SV39 L08_SV39 L14_SV01 L08_SV02
  python build_second_moment.py analyze --out sm --lens-path local_lens.pt --bank scan_svd_r0/directions --k 64

Cost of fit: ceil(2880 / dim_batch) backward passes per prompt. At dim_batch 8
that is 360 passes, ~5-8 min per prompt on an L4 -> 10-14 h for 100 prompts.
If dim_batch 8 runs out of memory, use 4 (twice as long).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"


def load_model(model_id: str):
    import jlens
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model = jlens.from_hf(hf, tok)
    qc = getattr(hf.config, "quantization_config", None)
    print(f"[model] {model_id}: n_layers={model.n_layers} d_model={model.d_model} dtype={next(hf.parameters()).dtype} "
          f"quantization={'none' if qc is None else type(qc).__name__}")
    return model


def load_download(repo: str, filename: str, path: str | None):
    import jlens
    from huggingface_hub import hf_hub_download

    p = path or hf_hub_download(repo, filename)
    return jlens.JacobianLens.load(p), p


def cos(a, b) -> float:
    import torch

    a = a.flatten().double()
    b = b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


# ----------------------------------------------------------------------------- diagnose


def cmd_diagnose(args: argparse.Namespace) -> None:
    import torch
    from jlens.examples import load_wikitext_prompts
    from jlens.fitting import jacobian_for_prompt

    layer = int(args.layer)
    z = np.load(Path(args.harvest) / "second_moment.npz")
    probes = torch.tensor(z["shared_probes"])  # [m_shared, d]
    means = torch.tensor(z[f"shared_means_L{layer:02d}"])  # [n_prompts, m_shared, d]
    prompt_idx = int(args.prompt_index)
    prompts = load_wikitext_prompts(prompt_idx + 1)
    model = load_model(args.model)
    download, dpath = load_download(args.lens_repo, args.lens_file, args.lens_path)

    print(f"[diagnose] exact Jacobian for prompt {prompt_idx} at layer {layer} (dim_batch={args.dim_batch}) ...")
    J, seq_len, n_valid = jacobian_for_prompt(
        model, prompts[prompt_idx], [layer], dim_batch=args.dim_batch, max_seq_len=128
    )
    Jl = J[layer].double()  # rows = output dims; J^T r = gradient of r.h_final wrt h_layer
    Jd = download.jacobians[layer].double()
    print(f"[diagnose] seq_len={seq_len} valid_positions={n_valid}")
    c_probe, c_down, ratio = [], [], []
    for k in range(min(int(args.n_probes), probes.shape[0])):
        r = probes[k].double()
        exact = Jl.T @ r
        ours = means[prompt_idx, k].double()
        down = Jd.T @ r
        c_probe.append(cos(ours, exact))
        c_down.append(cos(exact, down))
        ratio.append(float(exact.norm() / (down.norm() + 1e-30)))
    print()
    print(f"probe-mean vs exact per-prompt Jacobian (SAME model):        cos = {np.mean(c_probe):.3f}  "
          f"(min {np.min(c_probe):.3f})   <- near 1 means the probe estimator is correct")
    print(f"exact per-prompt Jacobian vs DOWNLOADED lens (100-prompt mean): cos = {np.mean(c_down):.3f}  "
          f"|exact|/|download| = {np.mean(ratio):.2f}")
    print(f"whole-matrix Frobenius cosine, exact vs download:               {cos(Jl, Jd):.3f}")
    print()
    print("Reading: a single prompt against a 100-prompt mean should give a moderate cosine (~0.5-0.8 at mid")
    print("layers, as the per-prompt harvest checks showed). If the first number is ~1 and the whole-matrix")
    print("cosine against the download is well below what the harvest's pooled check would need (~0.95),")
    print("the download describes a numerically different network and a local refit is required.")


# ----------------------------------------------------------------------------- fit


def cmd_fit(args: argparse.Namespace) -> None:
    import torch
    import jlens
    from jlens.examples import load_wikitext_prompts

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model = load_model(args.model)
    prompts = load_wikitext_prompts(args.prompt_offset + args.n_prompts)[args.prompt_offset :]
    print(f"[fit] {len(prompts)} WikiText-103 prompts (offset {args.prompt_offset}), dim_batch={args.dim_batch}, "
          f"max_seq_len=128, skip_first=16, target=last block (reference recipe)")
    ckpt = args.checkpoint or (str(Path(args.out).with_suffix("")) + "_ckpt.pt")
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        checkpoint_path=ckpt,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    lens.save(args.out, dtype=torch.float32)
    print(f"[fit] saved {args.out} (n_prompts={lens.n_prompts}, layers={lens.source_layers}, fp32)")
    print("Next: python build_second_moment.py analyze --out sm --lens-path", args.out, "--bank scan_svd_r0/directions --k 64")


# ----------------------------------------------------------------------------- compare


def parse_axis(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"L(\d+)_SV(\d+)", name.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad axis {name!r}")
    return int(m.group(1)), int(m.group(2))


def cmd_compare(args: argparse.Namespace) -> None:
    import jlens

    local = jlens.JacobianLens.load(args.local)
    download, dpath = load_download(args.lens_repo, args.lens_file, args.lens_path)
    layers = sorted(set(local.source_layers) & set(download.source_layers))
    print(f"[compare] local={args.local} (n_prompts={local.n_prompts})  download={dpath} (n_prompts={download.n_prompts})")
    print(f"{'layer':>5} {'frob_cos':>9} {'top8':>7} {'top32':>7} {'top64':>7}   (subspace overlap of right singular vectors)")
    svds = {}
    for l in layers:
        A = local.jacobians[l].double().numpy()
        B = download.jacobians[l].double().numpy()
        fc = float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
        _, sa, Va = np.linalg.svd(A, full_matrices=False)
        _, sb, Vb = np.linalg.svd(B, full_matrices=False)
        svds[l] = (Va, sa, Vb, sb)
        ov = []
        for k in (8, 32, 64):
            P = Va[:k] @ Vb[:k].T
            ov.append(float((P ** 2).sum(axis=0).mean()))
        print(f"{l:>5} {fc:>9.3f} {ov[0]:>7.3f} {ov[1]:>7.3f} {ov[2]:>7.3f}")

    if args.axes:
        print()
        print("Named downloaded axes -> closest local singular vector:")
        print(f"{'axis':>10} {'best local':>11} {'|cos|':>6} {'sigma_dl':>9} {'sigma_local':>12}")
        for name in args.axes:
            l, r = parse_axis(name)
            if l not in svds:
                print(f"{name:>10}  layer not in both lenses")
                continue
            Va, sa, Vb, sb = svds[l]
            v = Vb[r - 1]
            c = np.abs(Va @ v)
            j = int(np.argmax(c))
            print(f"{name:>10} {'SV%02d' % (j + 1):>11} {c[j]:>6.3f} {sb[r - 1]:>9.1f} {sa[j]:>12.1f}")


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("diagnose", "fit", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
        s.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
        s.add_argument("--lens-path", default=None, help="local copy of the downloaded lens.pt")
        if name == "diagnose":
            s.add_argument("--harvest", required=True, help="build_second_moment harvest dir (uses its shared probes)")
            s.add_argument("--layer", type=int, default=8)
            s.add_argument("--prompt-index", type=int, default=0)
            s.add_argument("--n-probes", type=int, default=8)
            s.add_argument("--dim-batch", type=int, default=4)
            s.set_defaults(func=cmd_diagnose)
        elif name == "fit":
            s.add_argument("--out", default="local_lens.pt")
            s.add_argument("--n-prompts", type=int, default=100)
            s.add_argument("--prompt-offset", type=int, default=0)
            s.add_argument("--dim-batch", type=int, default=8)
            s.add_argument("--checkpoint", default=None)
            s.add_argument("--checkpoint-every", type=int, default=2)
            s.add_argument("--no-resume", action="store_true")
            s.set_defaults(func=cmd_fit)
        else:
            s.add_argument("--local", required=True)
            s.add_argument("--axes", nargs="*", default=[], help="e.g. L07_SV39 L08_SV39")
            s.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
