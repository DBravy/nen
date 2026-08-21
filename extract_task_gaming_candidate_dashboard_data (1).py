#!/usr/bin/env python
"""
Extract dashboard-ready token-level J-lens data for three task-gaming candidates:

  L17 / SV28
  L18 / SV30
  L22 / SV32

This does NOT generate new behavior. It replays the exact token sequences saved
by collect_task_gaming_rollouts_v8.py.

For every completed rollout and every assistant turn, the script exports:

1) Token-level activation on each candidate source direction v:
       a_t = h_t · v

2) The implied singular-direction contribution scale:
       a_t * sigma
   because J v = sigma u.

3) The FULL J-lens readout at layers 17, 18, and 22 for each assistant message
   token:
       readout_t = unembed( h_t J_l^T )
   saved as top-k and bottom-k vocabulary items.

4) Static readouts for the three singular directions themselves:
       source-side:       unembed(v)
       J-transported:     unembed(J v) = unembed(sigma u)
   with a deeper top/bottom vocabulary list.

Outputs
-------
<out>/
  direction_readouts.json
      Static decode of L17/SV28, L18/SV30, L22/SV32.

  token_data.jsonl
      One nested JSON object per assistant-generated message token.
      Contains all three activations and all three full J-lens readouts.

  activations.csv
      Flat plotting-friendly table with one row per token and the three
      candidate activations.

  turns.jsonl
      One row per replayed assistant turn, including exact generated text,
      token counts, and token_data row range.

  trials.jsonl
      Trial metadata + frozen behavioral labels + original campaign paths.

  activation_stats.json
      Global raw activation mean/std/min/max for each candidate.

  missing_trials.json
      Frozen-label runs absent from the v8 manifest (e.g. an OOM rollout).

  meta.json

Typical
-------
python extract_task_gaming_candidate_dashboard_data.py \
    --campaign task_gaming_v8 \
    --labels task_gaming_v8_frozen_labels.jsonl \
    --directions-dir task_gaming_jlens/directions \
    --out task_gaming_candidate_data

The full J transport/unembed is GPU-batched for speed. The output is intentionally redundant enough that a dashboard can be built
without loading GPT-OSS or J-lens again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import jlens


CANDIDATES = [
    {"name": "L17_SV28", "layer": 17, "sv_rank": 28},
    {"name": "L18_SV30", "layer": 18, "sv_rank": 30},
    {"name": "L22_SV32", "layer": 22, "sv_rank": 32},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", default="task_gaming_v8")
    p.add_argument("--labels", default="task_gaming_v8_frozen_labels.jsonl")
    p.add_argument("--directions-dir", default="task_gaming_jlens/directions")
    p.add_argument("--out", default="task_gaming_candidate_data")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument(
        "--attn-implementation",
        default="flex_attention",
        help="Use flex_attention for GPT-OSS long-prefix replay.",
    )
    p.add_argument(
        "--jlens-topk",
        type=int,
        default=10,
        help="Top/bottom vocabulary items saved for each full per-token J-lens readout.",
    )
    p.add_argument(
        "--direction-topk",
        type=int,
        default=80,
        help="Top/bottom vocabulary items saved for each static singular-direction readout.",
    )
    p.add_argument(
        "--readout-batch-size",
        type=int,
        default=32,
        help="Batch size for vocabulary unembedding; lower if GPU memory is tight.",
    )
    p.add_argument(
        "--channels",
        default="analysis,commentary,final",
        help="Comma-separated Harmony message channels to export. Default exports all content channels.",
    )
    p.add_argument(
        "--include-special",
        action="store_true",
        help="Also export Harmony special tokens. Normally these are omitted.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted extraction by skipping turns already present in turns.jsonl.",
    )
    return p.parse_args()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, obj: dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_token_text(tok, token_id: int) -> str:
    return tok.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def jsonable_float(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def content_token_positions(
    tok,
    replay: dict[str, Any],
    allowed_channels: set[str],
    include_special: bool,
):
    """
    Parse Harmony generated tokens into message-content positions.

    Returns rows with:
      full_position, generated_local_index, message_index, channel

    A generation can contain multiple assistant messages, e.g.
      analysis -> commentary tool call.
    """
    gen = [int(x) for x in replay["generated_token_ids"]]
    base = int(replay["input_token_count"])

    chan_id = tok.convert_tokens_to_ids("<|channel|>")
    msg_id = tok.convert_tokens_to_ids("<|message|>")
    end_ids = {
        x
        for x in [
            tok.convert_tokens_to_ids("<|end|>"),
            tok.convert_tokens_to_ids("<|call|>"),
            tok.convert_tokens_to_ids("<|return|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        ]
        if isinstance(x, int) and x >= 0
    }
    special = set(getattr(tok, "all_special_ids", []) or [])

    rows = []
    message_index = 0
    i = 0

    while i < len(gen):
        if gen[i] != chan_id:
            i += 1
            continue

        # Header runs from <|channel|> to <|message|>.
        j = i + 1
        while j < len(gen) and gen[j] != msg_id and gen[j] not in end_ids:
            j += 1
        if j >= len(gen) or gen[j] != msg_id:
            i += 1
            continue

        header = tok.decode(
            gen[i + 1 : j],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        channel = header.split()[0].strip().lower() if header else "unknown"

        k = j + 1
        while k < len(gen) and gen[k] not in end_ids:
            k += 1

        if not allowed_channels or channel in allowed_channels:
            for local_idx in range(j + 1, k):
                tid = gen[local_idx]
                if (not include_special) and tid in special:
                    continue
                rows.append(
                    {
                        "full_position": base + local_idx,
                        "generated_local_index": local_idx,
                        "message_index": message_index,
                        "channel": channel,
                    }
                )

        message_index += 1
        i = k + 1

    # Fallback for an unusual generation without Harmony channel markers.
    if not rows:
        for local_idx, tid in enumerate(gen):
            if (not include_special) and tid in special:
                continue
            rows.append(
                {
                    "full_position": base + local_idx,
                    "generated_local_index": local_idx,
                    "message_index": 0,
                    "channel": "unknown",
                }
            )
    return rows


class HiddenWindowRecorder:
    """
    Capture only selected token positions from selected layers and move them to CPU
    immediately. We do not retain whole-sequence residual streams.
    """

    def __init__(self, blocks, layers, positions):
        self.blocks = blocks
        self.layers = list(layers)
        self.positions = [int(x) for x in positions]
        self.hidden = {}
        self.handles = []

    def _hook(self, layer):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            pos = torch.as_tensor(
                self.positions, device=tensor.device, dtype=torch.long
            )
            h = tensor.index_select(1, pos)[0].detach()
            self.hidden[layer] = h.to("cpu", dtype=torch.float32)
            del h, pos
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


@torch.inference_mode()
def vocab_extremes(model, tok, residuals: torch.Tensor, k: int, batch_size: int):
    """
    residuals: CPU float32 [n,d].
    Returns one record per residual vector with top and bottom vocab scores.

    The z-score is per residual vector across the vocabulary.
    """
    out = []
    device = model.input_device
    n = residuals.shape[0]

    for start in range(0, n, batch_size):
        x = residuals[start : start + batch_size].to(device)
        logits = model.unembed(x).float()

        mu = logits.mean(dim=-1, keepdim=True)
        sd = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
        z = (logits - mu) / sd

        kk = min(k, logits.shape[-1])
        top_vals, top_ids = torch.topk(logits, kk, dim=-1)
        bot_vals_neg, bot_ids = torch.topk(-logits, kk, dim=-1)
        bot_vals = -bot_vals_neg

        top_z = torch.gather(z, 1, top_ids)
        bot_z = torch.gather(z, 1, bot_ids)

        top_vals = top_vals.detach().cpu()
        bot_vals = bot_vals.detach().cpu()
        top_ids = top_ids.detach().cpu()
        bot_ids = bot_ids.detach().cpu()
        top_z = top_z.detach().cpu()
        bot_z = bot_z.detach().cpu()

        for r in range(top_ids.shape[0]):
            top = []
            bottom = []
            for q in range(kk):
                tid = int(top_ids[r, q])
                top.append(
                    {
                        "token_id": tid,
                        "token": safe_token_text(tok, tid),
                        "token_string": tok.convert_ids_to_tokens(tid),
                        "logit": float(top_vals[r, q]),
                        "z": float(top_z[r, q]),
                    }
                )
                tid = int(bot_ids[r, q])
                bottom.append(
                    {
                        "token_id": tid,
                        "token": safe_token_text(tok, tid),
                        "token_string": tok.convert_ids_to_tokens(tid),
                        "logit": float(bot_vals[r, q]),
                        "z": float(bot_z[r, q]),
                    }
                )
            out.append({"top": top, "bottom": bottom})

        del x, logits, z
        torch.cuda.empty_cache()

    return out


@torch.inference_mode()
def jlens_vocab_extremes(
    model,
    tok,
    source_hidden_cpu: torch.Tensor,
    J_gpu: torch.Tensor,
    k: int,
    batch_size: int,
):
    """
    Compute full J-lens vocabulary readouts without ever materializing all
    transported states or vocab logits at once.

      source_hidden_cpu [n,d]
      transported = h @ J.T
      logits = unembed(transported)
    """
    out = []
    device = model.input_device
    n = source_hidden_cpu.shape[0]

    for start in range(0, n, batch_size):
        h = source_hidden_cpu[start : start + batch_size].to(device)
        transported = h @ J_gpu.T
        logits = model.unembed(transported).float()

        mu = logits.mean(dim=-1, keepdim=True)
        sd = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
        z = (logits - mu) / sd

        kk = min(k, logits.shape[-1])
        top_vals, top_ids = torch.topk(logits, kk, dim=-1)
        bot_vals_neg, bot_ids = torch.topk(-logits, kk, dim=-1)
        bot_vals = -bot_vals_neg

        top_z = torch.gather(z, 1, top_ids)
        bot_z = torch.gather(z, 1, bot_ids)

        top_vals = top_vals.detach().cpu()
        bot_vals = bot_vals.detach().cpu()
        top_ids = top_ids.detach().cpu()
        bot_ids = bot_ids.detach().cpu()
        top_z = top_z.detach().cpu()
        bot_z = bot_z.detach().cpu()

        for r in range(top_ids.shape[0]):
            top = []
            bottom = []
            for q in range(kk):
                tid = int(top_ids[r, q])
                top.append(
                    {
                        "token_id": tid,
                        "token": safe_token_text(tok, tid),
                        "token_string": tok.convert_ids_to_tokens(tid),
                        "logit": float(top_vals[r, q]),
                        "z": float(top_z[r, q]),
                    }
                )
                tid = int(bot_ids[r, q])
                bottom.append(
                    {
                        "token_id": tid,
                        "token": safe_token_text(tok, tid),
                        "token_string": tok.convert_ids_to_tokens(tid),
                        "logit": float(bot_vals[r, q]),
                        "z": float(bot_z[r, q]),
                    }
                )
            out.append({"top": top, "bottom": bottom})

        del h, transported, logits, z
        torch.cuda.empty_cache()

    return out


@torch.inference_mode()
def static_readout(model, tok, vec_cpu: torch.Tensor, k: int):
    return vocab_extremes(model, tok, vec_cpu[None, :], k, 1)[0]


def load_candidates(directions_dir: Path, lens, model):
    """
    Load U/S/V from the SVD files created by analyze_task_gaming_jlens_v2.py.
    Also load the three J matrices.
    """
    result = {}
    J_by_layer = {}

    for spec in CANDIDATES:
        layer = spec["layer"]
        rank = spec["sv_rank"]
        name = spec["name"]

        p = directions_dir / f"L{layer:02d}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Point --directions-dir at task_gaming_jlens/directions."
            )
        z = np.load(p)
        if rank > z["V"].shape[1]:
            raise ValueError(
                f"{name} asks for SV{rank}, but {p} only contains {z['V'].shape[1]} vectors."
            )

        j = rank - 1
        U = torch.from_numpy(z["U"][:, j].astype(np.float32))
        V = torch.from_numpy(z["V"][:, j].astype(np.float32))
        sigma = float(z["S"][j])

        J = lens.jacobians[layer].detach().to("cpu", dtype=torch.float32)
        transported = V @ J.T
        target = sigma * U

        denom = float(torch.linalg.vector_norm(transported) * torch.linalg.vector_norm(target))
        cos = (
            float(torch.dot(transported, target) / denom)
            if denom > 0
            else float("nan")
        )
        relerr = float(
            torch.linalg.vector_norm(transported - target)
            / torch.linalg.vector_norm(target).clamp_min(1e-12)
        )

        result[name] = {
            "name": name,
            "layer": layer,
            "sv_rank": rank,
            "sigma": sigma,
            "U": U,
            "V": V,
            "Jv": transported,
            "svd_identity_cosine": cos,
            "svd_identity_relative_error": relerr,
        }
        # Keep only three J matrices resident on GPU (~small relative to model)
        # so every token readout can use fast batched transport.
        J_by_layer[layer] = J.to(model.input_device, dtype=torch.float32)
        del J

    return result, J_by_layer


def make_static_direction_readouts(
    candidates: dict[str, dict[str, Any]],
    model,
    tok,
    topk: int,
):
    rows = {}
    for name, c in candidates.items():
        print(f"[direction] {name} sigma={c['sigma']:.4f}")
        jv = c["Jv"]
        v = c["V"]
        u = c["U"]

        rows[name] = {
            "name": name,
            "layer": c["layer"],
            "sv_rank": c["sv_rank"],
            "sigma": c["sigma"],
            "svd_identity_cosine_Jv_vs_sigmaU": c["svd_identity_cosine"],
            "svd_identity_relative_error": c["svd_identity_relative_error"],
            "j_transport_readout": {
                "definition": "unembed(J @ v) = unembed(sigma * u)",
                **static_readout(model, tok, jv, topk),
            },
            "u_readout": {
                "definition": "unembed(u); ranking should match Jv up to positive scaling/final normalization",
                **static_readout(model, tok, u, topk),
            },
            "source_v_unembed": {
                "definition": "unembed(v); source-side comparison only, not the J-lens transport",
                **static_readout(model, tok, v, topk),
            },
        }
    return rows


def init_outputs(out: Path, resume: bool):
    out.mkdir(parents=True, exist_ok=True)
    token_json = out / "token_data.jsonl"
    act_csv = out / "activations.csv"
    turns = out / "turns.jsonl"

    if not resume:
        for p in [token_json, act_csv, turns, out / "replay_failures.jsonl"]:
            p.write_text("", encoding="utf-8")
    return token_json, act_csv, turns


def existing_turns(turns_path: Path):
    done = set()
    for r in read_jsonl(turns_path):
        done.add((r["run_id"], int(r["step"])))
    return done


def main():
    args = parse_args()

    campaign = Path(args.campaign).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    directions_dir = Path(args.directions_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    token_json_path, activations_csv_path, turns_path = init_outputs(out, args.resume)

    manifest_path = campaign / "analysis_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")

    manifest_rows = read_jsonl(manifest_path)
    manifest = {r["run_id"]: r for r in manifest_rows}
    labels = read_jsonl(labels_path)
    label_by_id = {r["run_id"]: r for r in labels}

    missing = [
        {
            "run_id": r["run_id"],
            "intent_class": r.get("intent_class"),
            "notes": r.get("notes"),
        }
        for r in labels
        if r["run_id"] not in manifest
    ]
    (out / "missing_trials.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[data] manifest={len(manifest)} labels={len(labels)} missing={len(missing)}")
    print(f"[load] {args.model} attention={args.attn_implementation}")

    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype="auto",
        device_map="cuda" if torch.cuda.is_available() else "auto",
        attn_implementation=args.attn_implementation,
    )
    hf.eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo, filename=args.lens_file
    )
    print(f"[load] CUDA allocated={torch.cuda.memory_allocated()/1e9:.1f} GB")

    candidates, J_by_layer = load_candidates(directions_dir, lens, model)
    layers = sorted(J_by_layer)

    # Deep static readouts first.
    direction_readouts = make_static_direction_readouts(
        candidates, model, tok, args.direction_topk
    )
    (out / "direction_readouts.json").write_text(
        json.dumps(direction_readouts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Human-readable static summary.
    lines = []
    for name, r in direction_readouts.items():
        lines.append(
            f"{name}  sigma={r['sigma']:.6f}  "
            f"cos(Jv,sigmaU)={r['svd_identity_cosine_Jv_vs_sigmaU']:.6f}"
        )
        lines.append("  Jv TOP:")
        for x in r["j_transport_readout"]["top"][:20]:
            lines.append(f"    {x['z']:+7.3f}  {x['logit']:+10.4f}  {x['token']!r}")
        lines.append("  Jv BOTTOM:")
        for x in r["j_transport_readout"]["bottom"][:20]:
            lines.append(f"    {x['z']:+7.3f}  {x['logit']:+10.4f}  {x['token']!r}")
        lines.append("")
    (out / "direction_readouts.txt").write_text("\n".join(lines), encoding="utf-8")

    # Trial metadata for dashboard joins.
    trial_rows = []
    for man in manifest_rows:
        lab = label_by_id.get(man["run_id"], {})
        trial_rows.append(
            {
                **man,
                "intent_class": lab.get("intent_class"),
                "usable": lab.get("usable"),
                "gaming_considered": lab.get("gaming_considered"),
                "gaming_attempted": lab.get("gaming_attempted"),
                "gaming_action": lab.get("gaming_action"),
                "accidental_shortcut": lab.get("accidental_shortcut"),
                "baseline_step": lab.get("baseline_step"),
                "decision_step": lab.get("decision_step"),
                "label_confidence": lab.get("confidence"),
                "label_notes": lab.get("notes"),
            }
        )
    with (out / "trials.jsonl").open("w", encoding="utf-8") as f:
        for r in trial_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    allowed_channels = {
        x.strip().lower() for x in args.channels.split(",") if x.strip()
    }

    done = existing_turns(turns_path) if args.resume else set()

    # CSV header.
    act_fields = [
        "row_id",
        "run_id",
        "environment",
        "condition",
        "intent_class",
        "gaming_considered",
        "gaming_attempted",
        "gaming_action",
        "accidental_shortcut",
        "step",
        "is_baseline_step",
        "is_decision_step",
        "message_index",
        "channel",
        "token_index_in_turn",
        "generated_local_index",
        "full_position",
        "token_id",
        "token",
        "token_string",
    ]
    for spec in CANDIDATES:
        act_fields += [
            f"{spec['name']}_activation",
            f"{spec['name']}_sigma",
            f"{spec['name']}_contribution_scale",
        ]

    csv_exists_with_header = (
        args.resume
        and activations_csv_path.exists()
        and activations_csv_path.stat().st_size > 0
    )
    act_file = activations_csv_path.open("a", newline="", encoding="utf-8")
    act_writer = csv.DictWriter(act_file, fieldnames=act_fields)
    if not csv_exists_with_header:
        act_writer.writeheader()
        act_file.flush()

    # Global row_id can resume from current CSV line count.
    if args.resume and activations_csv_path.stat().st_size > 0:
        with activations_csv_path.open("r", encoding="utf-8") as f:
            row_id = max(0, sum(1 for _ in f) - 1)
    else:
        row_id = 0

    all_activation_values = {c["name"]: [] for c in CANDIDATES}
    failures = 0
    processed = 0
    total_turns = sum(
        len(read_jsonl(campaign / m["replay_tokens"])) for m in manifest_rows
    )

    # Pre-index candidate by layer.
    candidate_by_layer = {c["layer"]: c for c in CANDIDATES}

    try:
        for man in manifest_rows:
            rid = man["run_id"]
            lab = label_by_id.get(rid, {})
            replay_path = campaign / man["replay_tokens"]
            replays = read_jsonl(replay_path)

            for rep in replays:
                step = int(rep["step"])
                processed += 1
                if (rid, step) in done:
                    print(f"[resume-skip] {processed:03d}/{total_turns} {rid} step={step}")
                    continue

                token_pos_rows = content_token_positions(
                    tok,
                    rep,
                    allowed_channels=allowed_channels,
                    include_special=args.include_special,
                )
                if not token_pos_rows:
                    print(f"[skip] {rid} step={step}: no selected message tokens")
                    continue

                positions = [r["full_position"] for r in token_pos_rows]
                ids = torch.tensor(
                    [rep["full_token_ids"]],
                    dtype=torch.long,
                    device=model.input_device,
                )

                try:
                    with torch.inference_mode(), HiddenWindowRecorder(
                        model.layers, layers, positions
                    ) as recorder:
                        model.forward(ids)
                except torch.OutOfMemoryError as exc:
                    failures += 1
                    print(
                        f"[OOM-skip] {rid} step={step} seq={len(rep['full_token_ids'])}: {exc}"
                    )
                    append_jsonl(
                        out / "replay_failures.jsonl",
                        {
                            "run_id": rid,
                            "step": step,
                            "seq_len": len(rep["full_token_ids"]),
                            "error": "CUDA OOM",
                        },
                    )
                    del ids
                    torch.cuda.empty_cache()
                    continue

                print(
                    f"[replay] {processed:03d}/{total_turns} "
                    f"{rid} step={step} tokens={len(positions)} "
                    f"seq={len(rep['full_token_ids'])}"
                )

                # Candidate activations + full J-lens readouts per selected layer.
                activations_by_name = {}
                jlens_by_layer = {}

                for layer in layers:
                    if layer not in recorder.hidden:
                        raise RuntimeError(f"Layer {layer} hook did not fire.")

                    h_cpu = recorder.hidden[layer]  # [n,d] CPU float32
                    spec = candidate_by_layer[layer]
                    cand = candidates[spec["name"]]
                    v = cand["V"]
                    activation = h_cpu @ v
                    activations_by_name[spec["name"]] = activation.numpy()

                    # Full J-lens transport + unembed in small GPU batches:
                    # row-vector convention h @ J^T.
                    jlens_by_layer[layer] = jlens_vocab_extremes(
                        model,
                        tok,
                        h_cpu,
                        J_by_layer[layer],
                        args.jlens_topk,
                        args.readout_batch_size,
                    )

                # Turn-level exact generation.
                gen_ids = [int(x) for x in rep["generated_token_ids"]]
                turn_start_row = row_id

                for token_index_in_turn, posinfo in enumerate(token_pos_rows):
                    row_id += 1
                    local_idx = posinfo["generated_local_index"]
                    tid = int(gen_ids[local_idx])
                    tok_text = safe_token_text(tok, tid)
                    tok_string = tok.convert_ids_to_tokens(tid)

                    static_meta = {
                        "row_id": row_id,
                        "run_id": rid,
                        "environment": man["environment"],
                        "condition": man["condition"],
                        "intent_class": lab.get("intent_class"),
                        "gaming_considered": lab.get("gaming_considered"),
                        "gaming_attempted": lab.get("gaming_attempted"),
                        "gaming_action": lab.get("gaming_action"),
                        "accidental_shortcut": lab.get("accidental_shortcut"),
                        "step": step,
                        "is_baseline_step": step == lab.get("baseline_step"),
                        "is_decision_step": step == lab.get("decision_step"),
                        "message_index": posinfo["message_index"],
                        "channel": posinfo["channel"],
                        "token_index_in_turn": token_index_in_turn,
                        "generated_local_index": local_idx,
                        "full_position": posinfo["full_position"],
                        "token_id": tid,
                        "token": tok_text,
                        "token_string": tok_string,
                    }

                    candidate_json = {}
                    csv_row = dict(static_meta)

                    for spec in CANDIDATES:
                        name = spec["name"]
                        a = float(activations_by_name[name][token_index_in_turn])
                        sigma = float(candidates[name]["sigma"])
                        scale = a * sigma
                        all_activation_values[name].append(a)

                        candidate_json[name] = {
                            "layer": spec["layer"],
                            "sv_rank": spec["sv_rank"],
                            "activation": a,
                            "sigma": sigma,
                            "direction_contribution_scale": scale,
                            "direction_readout_orientation": (
                                "positive_Jv" if scale >= 0 else "negative_Jv"
                            ),
                        }
                        csv_row[f"{name}_activation"] = a
                        csv_row[f"{name}_sigma"] = sigma
                        csv_row[f"{name}_contribution_scale"] = scale

                    jlens_json = {}
                    for layer in layers:
                        jlens_json[f"L{layer:02d}"] = {
                            "layer": layer,
                            "definition": "unembed(h_layer_token @ J_layer.T)",
                            **jlens_by_layer[layer][token_index_in_turn],
                        }

                    token_obj = {
                        **static_meta,
                        "candidates": candidate_json,
                        "jlens": jlens_json,
                    }
                    append_jsonl(token_json_path, token_obj)
                    act_writer.writerow(csv_row)

                act_file.flush()

                append_jsonl(
                    turns_path,
                    {
                        "run_id": rid,
                        "environment": man["environment"],
                        "condition": man["condition"],
                        "intent_class": lab.get("intent_class"),
                        "step": step,
                        "input_token_count": int(rep["input_token_count"]),
                        "generated_token_count": int(rep["generated_token_count"]),
                        "full_token_count": len(rep["full_token_ids"]),
                        "selected_message_token_count": len(token_pos_rows),
                        "first_row_id": turn_start_row + 1,
                        "last_row_id": row_id,
                        "generated_text": tok.decode(
                            gen_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    },
                )

                del recorder, ids
                torch.cuda.empty_cache()

    finally:
        act_file.close()

    # Stats can be recomputed from CSV so resume mode remains correct.
    stat_values = {c["name"]: [] for c in CANDIDATES}
    if activations_csv_path.exists() and activations_csv_path.stat().st_size > 0:
        with activations_csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for spec in CANDIDATES:
                    name = spec["name"]
                    try:
                        stat_values[name].append(float(r[f"{name}_activation"]))
                    except Exception:
                        pass

    stats = {}
    for spec in CANDIDATES:
        name = spec["name"]
        x = np.asarray(stat_values[name], dtype=np.float64)
        stats[name] = {
            "layer": spec["layer"],
            "sv_rank": spec["sv_rank"],
            "n": int(x.size),
            "mean": jsonable_float(np.mean(x)) if x.size else None,
            "std": jsonable_float(np.std(x)) if x.size else None,
            "min": jsonable_float(np.min(x)) if x.size else None,
            "max": jsonable_float(np.max(x)) if x.size else None,
            "p01": jsonable_float(np.quantile(x, 0.01)) if x.size else None,
            "p05": jsonable_float(np.quantile(x, 0.05)) if x.size else None,
            "p50": jsonable_float(np.quantile(x, 0.50)) if x.size else None,
            "p95": jsonable_float(np.quantile(x, 0.95)) if x.size else None,
            "p99": jsonable_float(np.quantile(x, 0.99)) if x.size else None,
        }
    (out / "activation_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    meta = {
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "attn_implementation": args.attn_implementation,
        "campaign": str(campaign),
        "directions_dir": str(directions_dir),
        "candidates": CANDIDATES,
        "jlens_topk": args.jlens_topk,
        "direction_topk": args.direction_topk,
        "channels": sorted(allowed_channels),
        "manifest_trials": len(manifest),
        "missing_labeled_trials": len(missing),
        "replay_failures": failures,
        "token_rows": row_id,
        "definitions": {
            "candidate_activation": "h_layer_token dot v",
            "direction_contribution_scale": "(h dot v) * sigma",
            "full_jlens_readout": "unembed(h_layer_token @ J_layer.T)",
            "direction_jlens_readout": "unembed(v @ J_layer.T) = unembed(sigma * u)",
        },
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\n=== Static candidate J-lens readouts ===")
    print((out / "direction_readouts.txt").read_text(encoding="utf-8"))
    print(f"[done] token rows={row_id} failures={failures}")
    print(f"[done] outputs={out}")


if __name__ == "__main__":
    main()
