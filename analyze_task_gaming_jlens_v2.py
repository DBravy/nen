#!/usr/bin/env python
"""
Replay exact v8 task-gaming trajectories through GPT-OSS-20B and project
pre-decision residual-stream states onto J-lens right singular vectors V.\n\nv2 uses GPT-OSS Flex Attention plus projection-in-hook recording to keep\nlong exact-prefix replays within GPU memory.

This is a discovery analysis, not a new behavioral run. It reads exact token IDs
saved by collect_task_gaming_rollouts_v8.py.

Primary contrast
----------------
For each labeled run with baseline_step and decision_step:
    delta(layer, sv) =
        mean projection over final analysis tokens at decision_step
      - mean projection over final analysis tokens at baseline_step

Directions are ranked by:
  * large, sign-consistent delta in deliberate-gaming runs
  * replication across ImpossibleBench and Pre-commit when available
  * weak lexicality of the transported U direction
  * separation from considered/rejected and accidental-proxy controls

Outputs
-------
  <out>/event_windows.csv
  <out>/run_deltas.csv
  <out>/direction_summary.csv
  <out>/directions/LXX.npz
  <out>/meta.json

Typical:
  python analyze_task_gaming_jlens_v2.py \
      --campaign task_gaming_v8 \
      --labels task_gaming_v8_frozen_labels.jsonl \
      --out task_gaming_jlens
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import jlens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", default="task_gaming_v8")
    p.add_argument("--labels", default="task_gaming_v8_frozen_labels.jsonl")
    p.add_argument("--out", default="task_gaming_jlens")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument(
        "--attn-implementation",
        default="flex_attention",
        help=(
            "Transformers attention backend. GPT-OSS supports flex_attention and "
            "Flash Attention, but not SDPA. Default: flex_attention."
        ),
    )
    p.add_argument(
        "--recompute-directions",
        action="store_true",
        help="Recompute SVDs even when <out>/directions/LXX.npz already exists.",
    )
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--layers", default="", help="comma-separated lens source layers; default all")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--svd-iters", type=int, default=2)
    p.add_argument("--exact-svd", action="store_true")
    p.add_argument("--operator", choices=["raw", "delta"], default="raw")
    p.add_argument("--analysis-window", type=int, default=64,
                   help="last N non-special analysis-channel tokens in each selected turn")
    p.add_argument("--min-window", type=int, default=4)
    p.add_argument("--decode-topk", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def topk_svd(A: torch.Tensor, k: int, oversample: int, niter: int, exact: bool):
    k = min(k, A.shape[0], A.shape[1])
    if exact:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k].T.contiguous()
    q = min(k + max(0, oversample), A.shape[0], A.shape[1])
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
    order = torch.argsort(S, descending=True)[:k]
    return U[:, order], S[order], V[:, order]


@torch.no_grad()
def lexical_peak_z(model, tok, Ucols: torch.Tensor, decode_topk: int):
    # Ucols [d,k] -> directions [k,d]
    dirs = Ucols.T.contiguous()
    logits = model.unembed(dirs).float()
    mu = logits.mean(dim=-1, keepdim=True)
    sd = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
    z = (logits - mu) / sd
    peak = z.abs().amax(dim=-1)
    pos = logits.topk(min(decode_topk, logits.shape[-1]), dim=-1).indices
    neg = (-logits).topk(min(decode_topk, logits.shape[-1]), dim=-1).indices
    pos_txt = [[tok.decode([int(t)]) for t in row] for row in pos.detach().cpu()]
    neg_txt = [[tok.decode([int(t)]) for t in row] for row in neg.detach().cpu()]
    return peak.detach().cpu().numpy(), pos_txt, neg_txt


class ProjectionRecorder:
    """
    Forward-hook recorder that never retains full residual streams.

    At each requested layer, immediately slices the small token window we care
    about and projects it onto V. Only the [window, k] projection matrix is
    copied to CPU. This avoids ActivationRecorder keeping
    [batch, full_sequence, d_model] tensors alive for every layer.
    """

    def __init__(self, blocks, layers, positions, V_by_layer):
        self.blocks = blocks
        self.layers = list(layers)
        self.positions = [int(x) for x in positions]
        self.V_by_layer = V_by_layer
        self.projections = {}
        self.handles = []

    def _hook(self, layer):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            # tensor: [batch, seq, d_model]
            pos = torch.as_tensor(self.positions, device=tensor.device, dtype=torch.long)
            h = tensor.index_select(1, pos)[0].detach().float()
            V = self.V_by_layer[layer]
            if V.device != h.device:
                V = V.to(h.device)
            P = h @ V
            self.projections[layer] = P.detach().cpu().numpy().astype(np.float32)
            del h, P, pos
        return hook

    def __enter__(self):
        try:
            for layer in self.layers:
                self.handles.append(
                    self.blocks[layer].register_forward_hook(self._hook(layer))
                )
        except Exception:
            for handle in self.handles:
                handle.remove()
            self.handles = []
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def analysis_positions(tok, replay: dict[str, Any], window: int):
    """
    Return full-sequence positions for the tail of the first analysis message.
    GPT-OSS generations normally look like:
      <|channel|>analysis<|message|> ... <|end|><|start|>assistant ...
    """
    gen = [int(x) for x in replay["generated_token_ids"]]
    base = int(replay["input_token_count"])
    end_id = tok.convert_tokens_to_ids("<|end|>")
    msg_id = tok.convert_tokens_to_ids("<|message|>")
    special = set(getattr(tok, "all_special_ids", []) or [])

    end_local = len(gen)
    if end_id is not None and end_id in gen:
        end_local = gen.index(end_id)

    start_local = 0
    if msg_id is not None:
        candidates = [i for i, t in enumerate(gen[:end_local]) if t == msg_id]
        if candidates:
            start_local = candidates[0] + 1

    keep = [i for i in range(start_local, end_local) if gen[i] not in special]
    if not keep:
        keep = [i for i in range(0, end_local) if gen[i] not in special]
    if window > 0:
        keep = keep[-window:]
    return [base + i for i in keep]


def load_replay_map(campaign: Path, rel_path: str):
    p = campaign / rel_path
    if not p.exists():
        raise FileNotFoundError(f"missing replay file: {p}")
    return {int(r["step"]): r for r in read_jsonl(p)}


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    campaign = Path(args.campaign).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    (out / "directions").mkdir(parents=True, exist_ok=True)

    manifest_path = campaign / "analysis_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    manifest = {r["run_id"]: r for r in read_jsonl(manifest_path)}
    labels = read_jsonl(labels_path)

    events = []
    for lab in labels:
        if not lab.get("usable", False):
            continue
        rid = lab["run_id"]
        if rid not in manifest:
            continue
        for role, step in (("baseline", lab.get("baseline_step")),
                           ("decision", lab.get("decision_step"))):
            if step is not None:
                events.append((rid, role, int(step), lab, manifest[rid]))

    print(f"[data] labels={len(labels)} manifest={len(manifest)} selected_events={len(events)}")
    print("[load]", args.model)
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    print(f"[load] attention={args.attn_implementation}")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype="auto",
        device_map="cuda" if torch.cuda.is_available() else "auto",
        attn_implementation=args.attn_implementation,
    )
    hf.eval()
    print(f"[load] CUDA allocated={torch.cuda.memory_allocated()/1e9:.1f} GB")
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)

    layers = parse_ints(args.layers) if args.layers else list(lens.source_layers)
    bad = sorted(set(layers) - set(lens.source_layers))
    if bad:
        raise SystemExit(f"invalid layers {bad}; available={list(lens.source_layers)}")

    # SVD once per layer. Reuse direction files from a previous interrupted
    # run unless explicitly asked to recompute them.
    svd = {}
    eye = None
    for layer in layers:
        dpath = out / "directions" / f"L{layer:02d}.npz"
        if dpath.exists() and not args.recompute_directions:
            print(f"[svd] L{layer:02d} reuse")
            z = np.load(dpath)
            U = torch.from_numpy(z["U"]).to(model.input_device, dtype=torch.float32)
            S = torch.from_numpy(z["S"]).to(model.input_device, dtype=torch.float32)
            V = torch.from_numpy(z["V"]).to(model.input_device, dtype=torch.float32)
            peak = z["lexical_peak_abs_z"].astype(np.float32)
            # Decode U again because text strings are intentionally not stored
            # inside the npz files.
            _, pos_txt, neg_txt = lexical_peak_z(model, tok, U, args.decode_topk)
        else:
            print(f"[svd] L{layer:02d}")
            A = lens.jacobians[layer].to(model.input_device, dtype=torch.float32)
            if args.operator == "delta":
                if eye is None or eye.shape != A.shape:
                    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                A = A - eye
            U, S, V = topk_svd(A, args.k, args.oversample, args.svd_iters, args.exact_svd)
            peak, pos_txt, neg_txt = lexical_peak_z(model, tok, U, args.decode_topk)
            np.savez_compressed(
                dpath,
                U=U.detach().cpu().numpy().astype(np.float32),
                S=S.detach().cpu().numpy().astype(np.float32),
                V=V.detach().cpu().numpy().astype(np.float32),
                lexical_peak_abs_z=peak.astype(np.float32),
            )
            del A
            torch.cuda.empty_cache()

        # V is kept on GPU because each hook uses it immediately; at k=32
        # this is tiny compared with the model. U is not needed after decoding.
        svd[layer] = {
            "S": S.detach(),
            "V": V.detach(),
            "lex_peak": peak,
            "pos_txt": pos_txt,
            "neg_txt": neg_txt,
        }
        del U
        torch.cuda.empty_cache()

    # Cache replay maps per run.
    replay_cache = {}
    event_rows = []
    token_store = defaultdict(list)  # (run, role, layer) -> vectors for scale estimation / optional debugging

    for ei, (rid, role, step, lab, man) in enumerate(events, 1):
        if rid not in replay_cache:
            replay_cache[rid] = load_replay_map(campaign, man["replay_tokens"])
        replay_map = replay_cache[rid]
        if step not in replay_map:
            print(f"[skip] {rid} {role} step={step}: no replay record")
            continue
        rep = replay_map[step]
        ids = torch.tensor([rep["full_token_ids"]], dtype=torch.long, device=model.input_device)
        pos = analysis_positions(tok, rep, args.analysis_window)
        if len(pos) < args.min_window:
            print(f"[skip] {rid} {role} step={step}: only {len(pos)} analysis tokens")
            continue

        try:
            V_by_layer = {layer: svd[layer]["V"] for layer in layers}
            with torch.inference_mode(), ProjectionRecorder(
                model.layers, layers, pos, V_by_layer
            ) as recorder:
                model.forward(ids)
        except torch.OutOfMemoryError as exc:
            print(
                f"[OOM-skip] {rid} {role} step={step} seq={ids.shape[1]}: {exc}"
            )
            del ids
            torch.cuda.empty_cache()
            with (out / "replay_failures.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": rid,
                    "event": role,
                    "step": step,
                    "seq_len": int(len(rep["full_token_ids"])),
                    "error": "CUDA OOM",
                }) + "\n")
            continue

        # token text for window, mostly for diagnostics
        full_ids = rep["full_token_ids"]
        toks = [tok.decode([int(full_ids[p])]) for p in pos]
        print(
            f"[replay] {ei:02d}/{len(events)} {rid} {role} "
            f"step={step} n={len(pos)} seq={len(full_ids)}"
        )

        for layer in layers:
            if layer not in recorder.projections:
                raise RuntimeError(f"projection hook did not fire for layer {layer}")
            Pcpu = recorder.projections[layer]
            token_store[(rid, role, layer)].append(Pcpu)

            means = Pcpu.mean(axis=0)
            stds = Pcpu.std(axis=0)
            lasts = Pcpu[-1]
            for j in range(Pcpu.shape[1]):
                event_rows.append({
                    "run_id": rid,
                    "environment": lab["environment"],
                    "condition": lab["condition"],
                    "intent_class": lab["intent_class"],
                    "gaming_action": lab.get("gaming_action"),
                    "event": role,
                    "step": step,
                    "layer": layer,
                    "sv_rank": j + 1,
                    "sigma": float(svd[layer]["S"][j].item()),
                    "u_lex_peak_abs_z": float(svd[layer]["lex_peak"][j]),
                    "window_n": len(pos),
                    "proj_mean": float(means[j]),
                    "proj_std": float(stds[j]),
                    "proj_last": float(lasts[j]),
                    "u_top_pos": " | ".join(svd[layer]["pos_txt"][j]),
                    "u_top_neg": " | ".join(svd[layer]["neg_txt"][j]),
                    "window_text": "".join(toks)[-500:],
                })
        del recorder, ids
        torch.cuda.empty_cache()

        # Checkpoint continuously so an unexpected later failure never discards
        # completed replay work.
        write_csv(out / "event_windows.csv", event_rows)

    write_csv(out / "event_windows.csv", event_rows)

    # Index event aggregates.
    by_event = {}
    for r in event_rows:
        key = (r["run_id"], r["event"], int(r["layer"]), int(r["sv_rank"]))
        by_event[key] = r

    # Baseline scale: pooled within-window std across all baseline events.
    base_stds = defaultdict(list)
    for r in event_rows:
        if r["event"] == "baseline" and math.isfinite(float(r["proj_std"])):
            base_stds[(int(r["layer"]), int(r["sv_rank"]))].append(float(r["proj_std"]))
    pooled_scale = {
        k: max(1e-6, float(np.sqrt(np.mean(np.square(v)))))
        for k, v in base_stds.items() if v
    }

    run_deltas = []
    lab_by_id = {x["run_id"]: x for x in labels}
    for lab in labels:
        rid = lab["run_id"]
        if not lab.get("usable", False):
            continue
        if lab.get("baseline_step") is None or lab.get("decision_step") is None:
            continue
        for layer in layers:
            for j in range(args.k):
                kb = (rid, "baseline", layer, j + 1)
                kd = (rid, "decision", layer, j + 1)
                if kb not in by_event or kd not in by_event:
                    continue
                b = by_event[kb]; d = by_event[kd]
                raw_delta = float(d["proj_mean"]) - float(b["proj_mean"])
                scale = pooled_scale.get((layer, j + 1), 1.0)
                run_deltas.append({
                    "run_id": rid,
                    "environment": lab["environment"],
                    "condition": lab["condition"],
                    "intent_class": lab["intent_class"],
                    "gaming_action": lab.get("gaming_action"),
                    "layer": layer,
                    "sv_rank": j + 1,
                    "sigma": float(d["sigma"]),
                    "u_lex_peak_abs_z": float(d["u_lex_peak_abs_z"]),
                    "baseline_mean": float(b["proj_mean"]),
                    "decision_mean": float(d["proj_mean"]),
                    "delta": raw_delta,
                    "delta_z": raw_delta / scale,
                    "u_top_pos": d["u_top_pos"],
                    "u_top_neg": d["u_top_neg"],
                })

    write_csv(out / "run_deltas.csv", run_deltas)

    # Direction summary.
    grouped = defaultdict(list)
    for r in run_deltas:
        grouped[(int(r["layer"]), int(r["sv_rank"]))].append(r)

    summaries = []
    for (layer, rank), rs in grouped.items():
        game = [r for r in rs if r["intent_class"] == "deliberate_gaming"]
        reject = [r for r in rs if r["intent_class"] in ("considered_rejected", "attempted_shortcut_failed")]
        accidental = [r for r in rs if r["intent_class"] == "accidental_proxy_failure"]
        imp = [r for r in game if r["environment"] == "impossiblebench"]
        pre = [r for r in game if r["environment"] == "precommit_hook"]

        gz = [float(r["delta_z"]) for r in game]
        rz = [float(r["delta_z"]) for r in reject]
        az = [float(r["delta_z"]) for r in accidental]
        iz = [float(r["delta_z"]) for r in imp]
        pz = [float(r["delta_z"]) for r in pre]

        gmean = safe_mean(gz)
        if gz and math.isfinite(gmean):
            sign = 1.0 if gmean >= 0 else -1.0
            sign_cons = float(np.mean([(1.0 if x * sign > 0 else 0.0) for x in gz]))
        else:
            sign_cons = float("nan")

        imean = safe_mean(iz)
        pmean = safe_mean(pz)
        cross_family_agree = (
            bool(math.isfinite(imean) and math.isfinite(pmean) and imean * pmean > 0)
            if iz and pz else None
        )

        exemplar = rs[0]
        lex = float(exemplar["u_lex_peak_abs_z"])
        # Soft lexical penalty: very peaky U readouts are downweighted but not excluded.
        nonlex_weight = 1.0 / (1.0 + max(0.0, lex - 5.0) / 5.0)
        specificity = abs(gmean) - 0.5 * abs(safe_mean(rz)) if gz else float("nan")
        cross_mult = 1.25 if cross_family_agree is True else (0.75 if cross_family_agree is False else 1.0)
        score = abs(gmean) * (sign_cons if math.isfinite(sign_cons) else 0.0) * nonlex_weight * cross_mult

        summaries.append({
            "layer": layer,
            "sv_rank": rank,
            "sigma": float(exemplar["sigma"]),
            "u_lex_peak_abs_z": lex,
            "gaming_n": len(gz),
            "gaming_mean_delta_z": gmean,
            "gaming_sign_consistency": sign_cons,
            "impossible_mean_delta_z": imean,
            "precommit_mean_delta_z": pmean,
            "cross_family_sign_agree": cross_family_agree,
            "considered_rejected_n": len(rz),
            "considered_rejected_mean_delta_z": safe_mean(rz),
            "accidental_n": len(az),
            "accidental_mean_delta_z": safe_mean(az),
            "specificity_vs_rejected": specificity,
            "discovery_score": score,
            "u_top_pos": exemplar["u_top_pos"],
            "u_top_neg": exemplar["u_top_neg"],
        })

    summaries.sort(key=lambda r: float(r["discovery_score"]), reverse=True)
    write_csv(out / "direction_summary.csv", summaries)

    meta = {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "operator": args.operator,
        "attn_implementation": args.attn_implementation,
        "layers": layers,
        "k": args.k,
        "analysis_window": args.analysis_window,
        "n_labels": len(labels),
        "n_manifest": len(manifest),
        "n_events_requested": len(events),
        "n_event_rows": len(event_rows),
        "n_run_deltas": len(run_deltas),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Top task-gaming candidate directions ===")
    for i, r in enumerate(summaries[:25], 1):
        print(
            f"{i:02d}. L{r['layer']:02d}/SV{r['sv_rank']:02d} "
            f"score={r['discovery_score']:.3f} "
            f"gameΔz={r['gaming_mean_delta_z']:+.3f} "
            f"sign={r['gaming_sign_consistency']:.2f} "
            f"imp={r['impossible_mean_delta_z']:+.3f} "
            f"pre={r['precommit_mean_delta_z']:+.3f} "
            f"lex={r['u_lex_peak_abs_z']:.2f}"
        )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
