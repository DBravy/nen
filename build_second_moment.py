#!/usr/bin/env python3
"""
build_second_moment.py

Build the second-moment Jacobian  S_l = E[J_c^T J_c]  and the gated-gain
matrix  G_l = S_l - Jbar_l^T Jbar_l  for every fitted layer of a model, using
EXACTLY the conventions of the jlens reference fit (which is how the
solarkyle/jspace-lenses lens.pt files were produced):

  * prompts:   jlens.examples.load_wikitext_prompts(100)  -> the same first
               100 WikiText-103 records (>= 600 chars) the lens was fit on
  * encoding:  model.encode(prompt, max_length=128)  (jlens default)
  * positions: jlens.fitting.valid_position_mask(seq_len, skip_first=16)
  * target:    the last block (n_layers - 1); sources: all layers below it
  * estimator: cotangent placed at EVERY valid target position; the
               gradient at source position p is  J_p^T r  where
               J_p = sum_{p' >= p} dh_T[p'] / dh_l[p]  (future-summed), and
               Jbar = E_prompt[ mean_p J_p ]  is the lens matrix.

The only difference from the fit: instead of one-hot cotangents (rows of J),
we backprop RANDOM Gaussian cotangents r (Hutchinson probes). For each probe,
E_r[(J_p^T r)(J_p^T r)^T] = J_p^T J_p, so accumulating outer products of the
probe gradients over positions, probes and prompts estimates
S_l = E_prompt[ mean_p J_p^T J_p ]  with the lens's own weighting, at a cost
of ceil(m / batch) backward passes per prompt instead of ceil(d_model / batch).
With 100 prompts and 32 probes each that is ~800 backward passes, about 4% of
a lens fit.

Jbar^T Jbar is taken from the downloaded lens (fit with the full one-hot
basis, so it is far more accurate than a probe estimate). A subset of probes
is SHARED across prompts; their mean gradients estimate Jbar^T r directly
and are compared against the lens's own Jbar^T r as a consistency check
(same conventions + same model dtype -> cosine near 1). If that check fails,
G would mix two different estimators and its subtraction is not meaningful.

Subcommands
  harvest  (GPU)  accumulate S_l; checkpointed per prompt; --smoke tests one
                  prompt first (memory, time, gradient sanity, lens check).
  analyze  (CPU)  eigendecompose S_l and G_l, per-layer gated fraction of
                  transport, consistency diagnostics, gatedness ratio of any
                  existing direction bank vs rotated/random nulls, and export
                  top-k eigenvector banks in the scanner's LXX.npz format.

Usage
  python build_second_moment.py harvest --out sm --smoke
  python build_second_moment.py harvest --out sm --probes 32 --probe-batch 4
  python build_second_moment.py analyze --out sm --bank scan_svd_r0/directions --k 64

Feasibility note: the reference lens was fit in bf16 on an A100-80GB. Whether
autograd runs through the MXFP4 expert kernels on a 24 GB card is the open
question; --smoke answers it in one prompt. If it fails, this job is small
enough (hundreds of backward passes) to run on a rented bf16 GPU in well under
an hour.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"


# =============================================================================
# harvest
# =============================================================================


def _atomic_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cmd_harvest(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "second_moment_ckpt.pt"

    print("[1/4] Loading J-Lens (for conventions + consistency check)...")
    import jlens
    from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask
    from jlens.examples import load_wikitext_prompts
    from huggingface_hub import hf_hub_download

    lens_path = args.lens_path or hf_hub_download(args.lens_repo, args.lens_file)
    lens = jlens.JacobianLens.load(lens_path)
    d_model = int(lens.d_model)
    source_layers = list(lens.source_layers) if args.layers is None else [int(x) for x in args.layers]
    for l in source_layers:
        if l not in lens.jacobians:
            raise SystemExit(f"layer {l} not in lens {lens.source_layers}")
    check_layers = sorted(set(source_layers) & set(args.check_layers or source_layers))
    J_check = {l: lens.jacobians[l].float().cpu() for l in check_layers}
    print(f"[lens] {lens_path}: d_model={d_model} fitted_layers={lens.source_layers} n_prompts={lens.n_prompts}")
    del lens

    print("[2/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model = jlens.from_hf(hf_model, tokenizer)
    if model.d_model != d_model:
        raise SystemExit(f"model d_model={model.d_model} != lens d_model={d_model}")
    n_layers = model.n_layers
    target_layer = n_layers - 1 if args.target_layer is None else int(args.target_layer)
    if max(source_layers) >= target_layer:
        raise SystemExit("source layers must be below the target layer")
    print(f"[model] n_layers={n_layers} target_layer={target_layer} sources={source_layers}")

    print("[3/4] Loading prompts (jlens.examples.load_wikitext_prompts)...")
    prompts = load_wikitext_prompts(args.prompt_offset + args.n_prompts)[args.prompt_offset :]
    print(f"[prompts] {len(prompts)} WikiText-103 prompts (offset {args.prompt_offset})")

    m = int(args.probes)
    B = int(args.probe_batch)
    K = math.ceil(m / B)
    m_shared = min(int(args.shared_probes), m)
    acc_device = torch.device(args.accumulate_on)

    # Shared probes: fixed across prompts (seeded), used for the Jbar consistency
    # check and prompt-level statistics. Remaining probes are fresh per prompt.
    g_shared = torch.Generator(device="cpu").manual_seed(args.seed)
    shared_probes = torch.randn(m_shared, d_model, generator=g_shared)  # [m_shared, d]

    # ---- state ------------------------------------------------------------------
    def fresh_state() -> dict:
        return {
            "S": {l: torch.zeros(d_model, d_model, dtype=torch.float64, device=acc_device) for l in source_layers},
            "n_done": 0,
            "next_idx": 0,
            "shared_means": {l: [] for l in source_layers},  # per prompt: [m_shared, d] fp32 cpu
            "n_valid": [],
            "seq_lens": [],
            "config": {
                "model": args.model, "target_layer": target_layer, "source_layers": source_layers,
                "max_seq_len": args.max_seq_len, "skip_first": SKIP_FIRST_N_POSITIONS,
                "probes": m, "probe_batch": B, "shared_probes": m_shared, "seed": args.seed,
                "prompt_offset": args.prompt_offset, "n_prompts": args.n_prompts, "d_model": d_model,
            },
        }

    state = None
    if args.resume and ckpt_path.exists() and not args.smoke:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["config"] != fresh_state()["config"]:
            raise SystemExit(f"checkpoint config differs from current args; delete {ckpt_path} or match args")
        state["S"] = {l: S.to(acc_device) for l, S in state["S"].items()}
        print(f"[resume] {state['n_done']} prompts done, resuming at index {state['next_idx']}")
    if state is None:
        state = fresh_state()

    # ---- main loop ----------------------------------------------------------------
    print("[4/4] Harvesting probe gradients...")
    t0 = time.time()
    n_prompts_to_do = 1 if args.smoke else len(prompts)
    K_eff = 1 if args.smoke else K

    for idx in range(state["next_idx"], n_prompts_to_do):
        prompt = prompts[idx]
        input_ids = model.encode(prompt, max_length=args.max_seq_len)
        seq_len = int(input_ids.shape[1])
        try:
            pmask = valid_position_mask(seq_len, skip_first=SKIP_FIRST_N_POSITIONS)
        except ValueError as exc:
            print(f"[skip] prompt {idx}: {exc}")
            state["next_idx"] = idx + 1
            continue
        valid = pmask.nonzero(as_tuple=True)[0]
        n_valid = int(valid.numel())

        # Probe set for this prompt: shared first, then fresh (seeded by prompt index).
        g_local = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + idx + 1)
        fresh = torch.randn(m - m_shared, d_model, generator=g_local) if m > m_shared else torch.zeros(0, d_model)
        probes = torch.cat([shared_probes, fresh], dim=0)  # [m, d]

        S_prompt = {l: torch.zeros(d_model, d_model, dtype=torch.float32, device=acc_device) for l in source_layers}
        shared_mean = {l: torch.zeros(m_shared, d_model, dtype=torch.float32) for l in source_layers}
        n_probes_used = 0

        t_prompt = time.time()
        with jlens.ActivationRecorder(
            model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)
        ) as recorder, torch.enable_grad():
            replicated = input_ids.expand(B, -1)
            model.forward(replicated)
            target_act = recorder.activations[target_layer]  # [B, seq, d]
            source_acts = [recorder.activations[l] for l in source_layers]
            valid_dev = valid.to(target_act.device)
            cotangent = torch.zeros_like(target_act)

            for k in range(K_eff):
                p0, p1 = k * B, min((k + 1) * B, m)
                nb = p1 - p0
                if nb <= 0:
                    break
                cotangent.zero_()
                r = probes[p0:p1].to(device=target_act.device, dtype=target_act.dtype)  # [nb, d]
                cotangent[:nb, valid_dev, :] = r[:, None, :]
                try:
                    grads = torch.autograd.grad(
                        outputs=target_act, inputs=source_acts, grad_outputs=cotangent,
                        retain_graph=(k < K_eff - 1),
                    )
                except Exception as exc:
                    raise SystemExit(
                        "autograd failed through the model. If this is an MXFP4/kernel error, the quantized "
                        "expert kernels have no backward; run this job on a bf16-capable GPU (the reference "
                        f"lens was fit in bf16 on an A100-80GB). Original error: {exc}"
                    )
                for l, grad in zip(source_layers, grads, strict=True):
                    pos = valid.to(grad.device)
                    gp = grad[:nb, pos, :].float()  # [nb, n_valid, d]  = J_p^T r_b
                    # second moment: mean over positions, sum over probes (normalized later)
                    flat = gp.reshape(nb * n_valid, d_model).to(acc_device)
                    S_prompt[l].addmm_(flat.T, flat, beta=1.0, alpha=1.0 / n_valid)
                    # shared-probe means over positions
                    for b in range(nb):
                        gi = p0 + b
                        if gi < m_shared:
                            shared_mean[l][gi] = gp[b].mean(dim=0).cpu()
                    if not torch.isfinite(gp).all():
                        raise SystemExit(f"non-finite gradient at layer {l}, prompt {idx}")
                n_probes_used += nb
                del grads

        # fold into running mean over prompts (equal weight per prompt, like the fit)
        for l in source_layers:
            state["S"][l] += (S_prompt[l].double() / n_probes_used)
            state["shared_means"][l].append(shared_mean[l])
        state["n_valid"].append(n_valid)
        state["seq_lens"].append(seq_len)
        state["n_done"] += 1
        state["next_idx"] = idx + 1
        dt = time.time() - t_prompt

        # consistency check against the lens on shared probes
        if check_layers and m_shared > 0:
            msgs = []
            for l in check_layers:
                ours = shared_mean[l][0]  # [d]   estimate of Jbar_prompt^T r_0 (this prompt only)
                ref = J_check[l].T @ shared_probes[0]
                cos = float(torch.dot(ours, ref) / (ours.norm() * ref.norm() + 1e-12))
                msgs.append(f"L{l:02d} cos={cos:+.3f} |ours|/|lens|={float(ours.norm() / (ref.norm() + 1e-12)):.2f}")
            note = "(single-prompt vs 100-prompt mean; expect cos well above 0 but below 1)"
            print(f"[check] {' '.join(msgs)} {note}")

        mem = f" peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if torch.cuda.is_available() else ""
        print(f"[harvest] prompt {idx + 1}/{n_prompts_to_do} seq={seq_len} valid={n_valid} probes={n_probes_used} "
              f"backward_passes={K_eff} {dt:.1f}s{mem} elapsed={time.time() - t0:.0f}s")

        if args.smoke:
            print("\n[smoke] one prompt succeeded. Estimated full run: "
                  f"{len(prompts)} prompts x {K} backward passes x ~{dt / K_eff:.1f}s = "
                  f"{len(prompts) * K * dt / K_eff / 3600:.2f} h")
            return
        if (idx + 1) % args.checkpoint_every == 0 or idx + 1 == n_prompts_to_do:
            save_state = dict(state)
            save_state["S"] = {l: S.cpu() for l, S in state["S"].items()}
            _atomic_save(save_state, ckpt_path)
            print(f"[checkpoint] {ckpt_path} ({state['n_done']} prompts)")

    # ---- finalize --------------------------------------------------------------
    n = state["n_done"]
    S_final = {l: (state["S"][l] / n).cpu().float() for l in source_layers}
    shared = {l: torch.stack(state["shared_means"][l]) for l in source_layers}  # [n, m_shared, d]
    _atomic_save(
        {"S": S_final, "shared_means": shared, "shared_probes": shared_probes, "n_prompts": n,
         "n_valid": state["n_valid"], "config": state["config"], "lens_path": str(lens_path)},
        out / "second_moment.pt",
    )
    npz = {f"S_L{l:02d}": S_final[l].numpy() for l in source_layers}
    npz.update({f"shared_means_L{l:02d}": shared[l].numpy() for l in source_layers})
    npz["shared_probes"] = shared_probes.numpy()
    npz["n_prompts"] = np.asarray(n)
    npz["config_json"] = np.asarray(json.dumps({**state["config"], "lens_path": str(lens_path)}))
    np.savez_compressed(out / "second_moment.npz", **npz)
    print(f"\n[done] {n} prompts -> {out / 'second_moment.pt'} and second_moment.npz")
    print("Next: python build_second_moment.py analyze --out", out, "--bank <directions dir> --k 64")


# =============================================================================
# analyze
# =============================================================================


def haar_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((k, k)))
    d = np.sign(np.diagonal(r))
    d[d == 0] = 1.0
    return q * d


def cmd_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    npz_path = out / "second_moment.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        cfg = json.loads(str(z["config_json"]))
        import re as _re

        S_all = {int(m.group(1)): np.asarray(z[k], dtype=np.float64)
                 for k in z.files if (m := _re.fullmatch(r"S_L(\d+)", k))}
        shared_means = {int(m.group(1)): np.asarray(z[k])
                        for k in z.files if (m := _re.fullmatch(r"shared_means_L(\d+)", k))}
        shared_probes = np.asarray(z["shared_probes"])
        n_prompts = int(z["n_prompts"])
    else:
        import torch

        blob = torch.load(out / "second_moment.pt", map_location="cpu", weights_only=False)
        S_all = {int(l): S.double().numpy() for l, S in blob["S"].items()}
        shared_means = {int(l): v.numpy() for l, v in blob["shared_means"].items()}
        shared_probes = blob["shared_probes"].numpy()
        cfg = {**blob["config"], "lens_path": blob.get("lens_path")}
        n_prompts = int(blob["n_prompts"])
    layers = sorted(S_all)
    d = cfg["d_model"]
    rng = np.random.default_rng(args.seed)

    if args.lens_npz:
        zl = np.load(args.lens_npz)
        Jbar = {l: np.asarray(zl[f"J_L{l:02d}"], dtype=np.float64) for l in layers}
    else:
        import jlens
        from huggingface_hub import hf_hub_download

        lens_path = args.lens_path or cfg.get("lens_path") or hf_hub_download(args.lens_repo, args.lens_file)
        lens = jlens.JacobianLens.load(lens_path)
        Jbar = {l: lens.jacobians[l].double().numpy() for l in layers}
    blob = {"n_prompts": n_prompts}

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"Second-moment analysis: {cfg['model']}  prompts={blob['n_prompts']}  probes/prompt={cfg['probes']}  layers={layers}")
    emit("=" * 78)

    # ---- 1. consistency: shared-probe means vs lens ----------------------------
    emit()
    emit("Consistency check (mean over prompts of our probe gradient vs lens Jbar^T r, per shared probe):")
    emit(f"{'layer':>5} {'cos':>7} {'|ours|/|lens|':>14}   (cos near 1 and ratio near 1 => same estimator & dtype; G is meaningful)")
    consistency: dict[int, float] = {}
    for l in layers:
        ours = shared_means[l].mean(axis=0)  # [m_shared, d]
        ref = shared_probes @ Jbar[l]  # rows: r^T Jbar = (Jbar^T r)^T
        cos = np.array([float(o @ r_ / (np.linalg.norm(o) * np.linalg.norm(r_) + 1e-12)) for o, r_ in zip(ours, ref)])
        ratio = np.array([np.linalg.norm(o) / (np.linalg.norm(r_) + 1e-12) for o, r_ in zip(ours, ref)])
        consistency[l] = float(cos.mean())
        emit(f"{l:>5} {cos.mean():>7.3f} {ratio.mean():>14.3f}")

    # ---- 2. per-layer decomposition ----------------------------------------------
    emit()
    emit("Per-layer transport decomposition: trace(S) = total gain, trace(Jbar^T Jbar) = invariant gain,")
    emit("gated fraction = 1 - invariant/total. neg_mass = |sum of negative eigenvalues of G| / trace(G+),")
    emit("a symptom of estimator noise or inconsistency (should be small).")
    emit(f"{'layer':>5} {'total':>12} {'invariant':>12} {'gated_frac':>11} {'neg_mass':>9} {'top eig S':>11} {'top eig G':>11}")
    banks_S: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    banks_G: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_layer_rows = []
    for l in layers:
        S = 0.5 * (S_all[l] + S_all[l].T)
        JJ = Jbar[l].T @ Jbar[l]
        G = S - JJ
        G = 0.5 * (G + G.T)
        wS, VS = np.linalg.eigh(S)
        wG, VG = np.linalg.eigh(G)
        wS, VS = wS[::-1], VS[:, ::-1]
        wG, VG = wG[::-1], VG[:, ::-1]
        total, invariant = float(np.trace(S)), float(np.trace(JJ))
        neg = float(-wG[wG < 0].sum())
        pos = float(wG[wG > 0].sum())
        gated_frac = 1.0 - invariant / max(total, 1e-30)
        emit(f"{l:>5} {total:>12.4g} {invariant:>12.4g} {gated_frac:>11.3f} {neg / max(pos, 1e-30):>9.3f} {wS[0]:>11.4g} {wG[0]:>11.4g}")
        banks_S[l] = (VS[:, : args.k].astype(np.float32), np.sqrt(np.clip(wS[: args.k], 0, None)).astype(np.float32))
        banks_G[l] = (VG[:, : args.k].astype(np.float32), np.sqrt(np.clip(wG[: args.k], 0, None)).astype(np.float32))
        per_layer_rows.append({"layer": l, "total_gain": total, "invariant_gain": invariant,
                               "gated_fraction": gated_frac, "neg_mass": neg / max(pos, 1e-30),
                               "consistency_cos": consistency[l]})

    # ---- 3. overlap between S's and Jbar's top subspaces --------------------------
    emit()
    emit(f"Top-{args.k} subspace overlap: mean squared cosine of Jbar's top-k right singular vectors within")
    emit("S's top-k eigenspace (1 = identical subspace) and within G's top-k eigenspace (0 = G found new directions).")
    emit(f"{'layer':>5} {'Jbar in S':>10} {'Jbar in G':>10}")
    for l in layers:
        _, _, Vt = np.linalg.svd(Jbar[l], full_matrices=False)
        VJ = Vt[: args.k].T  # [d, k]
        for name, (VB, _) in (("S", banks_S[l]), ("G", banks_G[l])):
            proj = VB.T @ VJ  # [k, k]
            overlap = float((proj ** 2).sum(axis=0).mean())
            if name == "S":
                oS = overlap
            else:
                oG = overlap
        emit(f"{l:>5} {oS:>10.3f} {oG:>10.3f}")

    # ---- 4. gatedness of an existing bank vs nulls ---------------------------------
    if args.bank:
        emit()
        emit("Gatedness ratio  (f^T S f - |Jbar f|^2) / |Jbar f|^2  for bank axes, rotated axes, random directions.")
        emit("High = effect large but inconsistent across contexts. Bank axis values listed per layer as ranks.")
        bank_dir = Path(args.bank)
        rows_out = []
        for l in layers:
            p = bank_dir / f"L{l:02d}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            V = np.asarray(z["V"], dtype=np.float64)
            if V.shape[0] < V.shape[1]:
                V = V.T
            V = V[:, : args.k]
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            S = S_all[l]
            J = Jbar[l]

            def ratio(F: np.ndarray) -> np.ndarray:
                tot = np.einsum("ij,jk,ki->i", F.T, S, F)
                inv = np.sum((J @ F) ** 2, axis=0)
                return (tot - inv) / np.maximum(inv, 1e-30)

            r_bank = ratio(V)
            r_rot = ratio(V @ haar_orthogonal(V.shape[1], rng))
            R = rng.standard_normal((d, args.k))
            R /= np.linalg.norm(R, axis=0, keepdims=True)
            r_rand = ratio(R)
            emit(f"  L{l:02d}: bank median={np.median(r_bank):.3f} (min {r_bank.min():.3f}, max {r_bank.max():.3f}) | "
                 f"rotated median={np.median(r_rot):.3f} | random median={np.median(r_rand):.3f}, 95th={np.percentile(r_rand, 95):.3f}")
            for j in range(V.shape[1]):
                rows_out.append({"layer": l, "sv_rank_1based": j + 1, "gatedness_ratio": float(r_bank[j]),
                                 "total_gain": float(np.einsum("i,ij,j", V[:, j], S, V[:, j])),
                                 "invariant_gain": float(np.sum((J @ V[:, j]) ** 2))})
        if rows_out:
            import csv

            with (out / "bank_gatedness.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
            emit(f"  -> per-axis values in {out / 'bank_gatedness.csv'}")

    # ---- 5. export banks ------------------------------------------------------------
    for name, banks in (("S_bank", banks_S), ("G_bank", banks_G)):
        bdir = out / name
        bdir.mkdir(exist_ok=True)
        for l, (V, s) in banks.items():
            np.savez_compressed(bdir / f"L{l:02d}.npz", V=V, S=s)
    emit()
    emit(f"Exported top-{args.k} eigenvector banks: {out / 'S_bank'} (total gain) and {out / 'G_bank'} (gated gain),")
    emit("scanner-compatible (--directions-dir). S values are sqrt(eigenvalue), i.e. RMS gain.")

    import csv

    with (out / "per_layer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer_rows[0].keys()))
        w.writeheader()
        w.writerows(per_layer_rows)
    (out / "second_moment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {out / 'second_moment_report.txt'}, per_layer.csv")


def cmd_export_lens(args: argparse.Namespace) -> None:
    import jlens
    from huggingface_hub import hf_hub_download

    lens_path = args.lens_path or hf_hub_download(args.lens_repo, args.lens_file)
    lens = jlens.JacobianLens.load(lens_path)
    np.savez_compressed(
        args.out_npz,
        **{f"J_L{l:02d}": lens.jacobians[l].float().numpy() for l in lens.source_layers},
        d_model=np.asarray(lens.d_model), n_prompts=np.asarray(lens.n_prompts),
    )
    print(f"[export] {len(lens.source_layers)} layers -> {args.out_npz}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="GPU: accumulate E[J^T J] via random-probe backprop")
    h.add_argument("--out", required=True)
    h.add_argument("--model", default=DEFAULT_MODEL)
    h.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    h.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    h.add_argument("--lens-path", default=None, help="local lens.pt (skips hub download)")
    h.add_argument("--n-prompts", type=int, default=100, help="the lens used the first 100")
    h.add_argument("--prompt-offset", type=int, default=0, help="e.g. 100 for a held-out prompt set")
    h.add_argument("--max-seq-len", type=int, default=128, help="jlens default")
    h.add_argument("--layers", nargs="*", default=None, help="subset of source layers (default: all fitted)")
    h.add_argument("--target-layer", type=int, default=None, help="default: last block, as in the fit")
    h.add_argument("--probes", type=int, default=32, help="Hutchinson probes per prompt")
    h.add_argument("--probe-batch", type=int, default=4, help="probes per backward pass (prompt replicated this many times)")
    h.add_argument("--shared-probes", type=int, default=8, help="probes shared across prompts (lens consistency check)")
    h.add_argument("--check-layers", nargs="*", type=int, default=[8, 16], help="layers printed in the per-prompt check")
    h.add_argument("--accumulate-on", default="cuda", choices=["cuda", "cpu"])
    h.add_argument("--checkpoint-every", type=int, default=5)
    h.add_argument("--no-resume", dest="resume", action="store_false")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--smoke", action="store_true", help="one prompt, one backward pass, then exit")
    h.set_defaults(func=cmd_harvest)

    a = sub.add_parser("analyze", help="CPU: eigen-analysis, G, gatedness, bank export")
    a.add_argument("--out", required=True)
    a.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    a.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    a.add_argument("--lens-path", default=None)
    a.add_argument("--lens-npz", default=None, help="lens exported with the export-lens subcommand (no torch needed)")
    a.add_argument("--bank", default=None, help="existing directions dir (e.g. scan_svd_r0/directions) for gatedness ratios")
    a.add_argument("--k", type=int, default=64)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("export-lens", help="convert lens.pt to an npz of Jbar matrices (for torch-free analysis)")
    e.add_argument("--out-npz", required=True)
    e.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    e.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    e.add_argument("--lens-path", default=None)
    e.set_defaults(func=cmd_export_lens)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
