#!/usr/bin/env python3
"""
Focused causal intervention campaign for L17/SV28 only.

This script reuses the exact behavioral environments and graders from
collect_task_gaming_rollouts_v8.py, but intervenes only on the L17/SV28
residual-stream direction during NEW GPT-OSS-20B rollouts.

Direction
---------
    layer = 17
    singular vector = SV28
    coefficient a = h_17 · v_17,28

An intervention replaces a with:

    a' = multiplier * a + bias

by adding:

    (a' - a) * v_17,28

to the layer-17 residual stream.

Built-in variants
-----------------
control
    No intervention.

L17_suppress
    a' = 0

L17_amplify
    a' = --amplify-factor * a   (default 2x)

L17_flip
    a' = -a

L17_push_gaming
    Add a NEGATIVE calibrated bias. In the discovery analysis L17/SV28 moved
    negative during gaming. The bias magnitude is --bias-strength times the
    baseline projection SD measured in task_gaming_jlens/event_windows.csv.

L17_push_anti
    Add the same calibrated bias in the POSITIVE direction.

Overlap / resume handling
-------------------------
The previous broad causal campaign may have been stopped midway. By default
this script looks for:

    causal_interventions/index.jsonl

and reuses any COMPLETED control or L17/SV28 cells that match this campaign's
model/reasoning/scope/sampling settings. It never reruns or double-counts those
cells.

You can specify one or more old campaign directories or index files with:

    --reuse-from causal_interventions,other_causal_run

The new output keeps provenance separate:

    <out>/index.jsonl
        ONLY newly generated runs from this focused script.

    <out>/combined_index.jsonl
        Deduplicated successful union of reused old cells + new cells.

    <out>/reused_cells.jsonl
        Provenance for cells imported from prior campaigns.

    <out>/causal_summary.csv
    <out>/paired_effects.csv
        Computed from combined_index.jsonl, so reused data count once.

The deduplication key is:

    (intervention_name, environment, condition, seed)

not run_id. Thus overlapping data are recognized even if the old and new
campaign use different output directories.

Default focused design
----------------------
By default this script studies the two clearest cross-task conditions:

    ImpossibleBench / binary_no_rules
    Precommit Hook / none

with 8 matched seeds per intervention. If the previous run already completed
some of the first seeds, only the missing cells are generated.

Typical:
    python causal_l17_sv28_interventions.py \
        --collector collect_task_gaming_rollouts_v8.py \
        --directions-dir task_gaming_jlens/directions \
        --event-windows task_gaming_jlens/event_windows.csv \
        --reuse-from causal_interventions \
        --samples-per-condition 8 \
        --out causal_l17_sv28

Primary causal pass:
    --variants control,L17_suppress,L17_push_anti,L17_push_gaming

Expanded pass:
    --variants control,L17_suppress,L17_push_anti,L17_push_gaming,L17_amplify,L17_flip
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers


CANDIDATES = {
    "L17": {"name": "L17_SV28", "layer": 17, "sv_rank": 28, "gaming_sign": -1.0},
}


@dataclasses.dataclass
class TargetEdit:
    key: str
    layer: int
    sv_rank: int
    v: torch.Tensor
    natural_mean: float
    natural_scale: float
    multiplier: float = 1.0
    bias: float = 0.0

    def as_dict(self):
        return {
            "key": self.key,
            "layer": self.layer,
            "sv_rank": self.sv_rank,
            "natural_mean": self.natural_mean,
            "natural_scale": self.natural_scale,
            "multiplier": self.multiplier,
            "bias": self.bias,
            "bias_in_natural_sd": (
                self.bias / self.natural_scale
                if self.natural_scale > 0
                else None
            ),
        }


@dataclasses.dataclass
class Variant:
    name: str
    edits: list[TargetEdit]
    description: str

    def as_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "edits": [e.as_dict() for e in self.edits],
        }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--collector",
        default="collect_task_gaming_rollouts_v8.py",
        help="Path to the v8 behavioral collector to reuse for environments/graders.",
    )
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--directions-dir", default="task_gaming_jlens/directions")
    p.add_argument(
        "--event-windows",
        default="task_gaming_jlens/event_windows.csv",
        help="Discovery event_windows.csv used to calibrate 1-SD steering biases.",
    )
    p.add_argument("--out", default="causal_l17_sv28")

    p.add_argument(
        "--env",
        default="impossiblebench,precommit_hook",
        help="all or comma-separated secret_number,precommit_hook,impossiblebench",
    )
    p.add_argument(
        "--only-condition",
        default="binary_no_rules,none",
        help="Optional comma-separated condition names. Default: binary_no_rules,none.",
    )
    p.add_argument("--samples-per-condition", type=int, default=8)
    p.add_argument("--seed", type=int, default=22345)

    p.add_argument("--reasoning", default="low", choices=["low", "medium", "high"])
    p.add_argument("--max-steps", type=int, default=28)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--max-turn-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--docker-image", default="python:3.11-slim")
    p.add_argument(
        "--attn-implementation",
        default="flex_attention",
        help="Use flex_attention to avoid GPT-OSS eager-attention OOMs.",
    )

    p.add_argument(
        "--scope",
        choices=["analysis", "assistant"],
        default="analysis",
        help=(
            "analysis = intervene only in current assistant analysis-channel content; "
            "assistant = intervene in analysis/commentary/final content."
        ),
    )
    p.add_argument("--amplify-factor", type=float, default=2.0)
    p.add_argument(
        "--bias-strength",
        type=float,
        default=1.0,
        help="Magnitude of push_gaming/push_anti in discovery baseline SD units.",
    )
    p.add_argument(
        "--variants",
        default=(
            "control,"
            "L17_suppress,"
            "L17_push_anti,"
            "L17_push_gaming,"
            "L17_amplify,"
            "L17_flip"
        ),
        help="Comma-separated L17/SV28 intervention variants.",
    )
    p.add_argument(
        "--reuse-from",
        default="causal_interventions",
        help=(
            "Comma-separated prior causal output directories or index.jsonl files. "
            "Completed matching control/L17 cells are reused and deduplicated. "
            "Use an empty string to disable."
        ),
    )
    p.add_argument(
        "--allow-reuse-mismatch",
        action="store_true",
        help=(
            "Reuse old cells even when prior campaign settings differ in "
            "model/reasoning/scope/temperature/top-p. Off by default."
        ),
    )
    p.add_argument("--list-variants", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run only first matching condition, first seed, and control + L17_suppress.",
    )
    return p.parse_args()


def jdump(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_name(s: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def import_collector(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("task_gaming_collector_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_decoder_layers(model):
    """
    Find GPT-OSS decoder ModuleList robustly across Transformers wrapper layouts.
    """
    direct = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        (
            "model.model.layers",
            getattr(
                getattr(getattr(model, "model", None), "model", None),
                "layers",
                None,
            ),
        ),
        ("layers", getattr(model, "layers", None)),
    ]
    for name, value in direct:
        if isinstance(value, torch.nn.ModuleList) and len(value) > 22:
            print(f"[model] decoder layers: {name} ({len(value)})")
            return value

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 22:
            # Prefer names that actually look like decoder layers.
            score = 0
            if name.endswith("layers"):
                score += 10
            cls_names = {m.__class__.__name__.lower() for m in module[: min(3, len(module))]}
            if any("decoder" in x or "gptoss" in x for x in cls_names):
                score += 10
            candidates.append((score, name, module))
    if not candidates:
        raise RuntimeError("Could not locate a decoder-layer ModuleList on the model.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, name, module = candidates[0]
    print(f"[model] decoder layers: {name} ({len(module)}) [discovered]")
    return module


def load_direction_vectors(directions_dir: Path):
    out = {}
    for key, spec in CANDIDATES.items():
        p = directions_dir / f"L{spec['layer']:02d}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Use --directions-dir task_gaming_jlens/directions"
            )
        z = np.load(p)
        j = spec["sv_rank"] - 1
        if j >= z["V"].shape[1]:
            raise ValueError(
                f"{key} requires SV{spec['sv_rank']}, but {p} "
                f"contains only {z['V'].shape[1]} directions."
            )
        v = torch.from_numpy(z["V"][:, j].astype(np.float32))
        out[key] = {
            **spec,
            "v": v,
            "sigma": float(z["S"][j]),
        }
    return out


def load_natural_scales(event_windows_path: Path):
    """
    Reproduce the analyzer's baseline scale for the three directions:
      scale = RMS of within-window projection std across baseline events.

    Also retain the mean baseline projection so future clamp-style variants can
    be added without rerunning discovery.
    """
    targets = {(v["layer"], v["sv_rank"]): k for k, v in CANDIDATES.items()}
    stds = defaultdict(list)
    means = defaultdict(list)

    if not event_windows_path.exists():
        raise FileNotFoundError(
            f"Missing {event_windows_path}. This file calibrates push_gaming/"
            "push_anti in discovery SD units."
        )

    with event_windows_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                key = targets[(int(r["layer"]), int(r["sv_rank"]))]
            except Exception:
                continue
            if r.get("event") != "baseline":
                continue
            try:
                s = float(r["proj_std"])
                m = float(r["proj_mean"])
            except Exception:
                continue
            if math.isfinite(s):
                stds[key].append(s)
            if math.isfinite(m):
                means[key].append(m)

    result = {}
    for key in CANDIDATES:
        if not stds[key]:
            raise RuntimeError(f"No baseline projection stds found for {key}")
        scale = float(np.sqrt(np.mean(np.square(stds[key]))))
        mean = float(np.mean(means[key])) if means[key] else 0.0
        result[key] = {
            "scale": max(scale, 1e-6),
            "mean": mean,
            "n_baselines": len(stds[key]),
        }
    return result


def all_builtin_variant_names():
    return [
        "control",
        "L17_suppress",
        "L17_amplify",
        "L17_flip",
        "L17_push_gaming",
        "L17_push_anti",
    ]


def make_variant(
    name: str,
    direction_info: dict[str, dict[str, Any]],
    scales: dict[str, dict[str, float]],
    amplify_factor: float,
    bias_strength: float,
):
    if name == "control":
        return Variant("control", [], "No activation intervention.")

    m = re.fullmatch(
        r"L17_(suppress|amplify|flip|push_gaming|push_anti)",
        name,
    )
    if not m:
        raise ValueError(
            f"Unknown variant {name!r}. Available: "
            + ", ".join(all_builtin_variant_names())
        )

    mode = m.group(1)
    key = "L17"
    info = direction_info[key]
    scale = scales[key]["scale"]
    mean = scales[key]["mean"]

    multiplier = 1.0
    bias = 0.0
    if mode == "suppress":
        multiplier = 0.0
    elif mode == "amplify":
        multiplier = float(amplify_factor)
    elif mode == "flip":
        multiplier = -1.0
    elif mode == "push_gaming":
        bias = CANDIDATES[key]["gaming_sign"] * float(bias_strength) * scale
    elif mode == "push_anti":
        bias = -CANDIDATES[key]["gaming_sign"] * float(bias_strength) * scale

    edit = TargetEdit(
        key=key,
        layer=info["layer"],
        sv_rank=info["sv_rank"],
        v=info["v"],
        natural_mean=mean,
        natural_scale=scale,
        multiplier=multiplier,
        bias=bias,
    )

    if mode == "suppress":
        desc = "Set L17/SV28 h·v coefficient to zero."
    elif mode == "amplify":
        desc = f"Multiply L17/SV28 h·v coefficient by {amplify_factor:g}."
    elif mode == "flip":
        desc = "Flip the sign of the L17/SV28 h·v coefficient."
    elif mode == "push_gaming":
        desc = (
            f"Add {bias_strength:g} discovery-SD in the gaming-associated "
            "negative L17/SV28 direction."
        )
    else:
        desc = (
            f"Add {bias_strength:g} discovery-SD opposite the gaming-associated "
            "L17/SV28 direction."
        )

    return Variant(name=name, edits=[edit], description=desc)


class ResidualInterventionController:
    """
    Apply current-turn activation edits to selected GPT-OSS decoder layers.

    A top-level model pre-hook tracks which token positions belong to the
    current assistant's analysis/content channel. Decoder forward hooks then
    edit only those positions.

    Crucially:
      * old user/tool/history tokens are never edited;
      * default scope='analysis' leaves commentary/tool-call JSON untouched;
      * chunked generation prefill is handled by re-identifying the latest
        analysis-channel generation header.
    """

    def __init__(
        self,
        model,
        tok,
        decoder_layers,
        variant: Variant,
        scope: str = "analysis",
    ):
        self.model = model
        self.tok = tok
        self.decoder_layers = decoder_layers
        self.variant = variant
        self.scope = scope
        self.allowed_channels = (
            {"analysis"}
            if scope == "analysis"
            else {"analysis", "commentary", "final"}
        )

        self.edits_by_layer = defaultdict(list)
        for edit in variant.edits:
            self.edits_by_layer[edit.layer].append(edit)

        self.enabled = False
        self.mask = None
        self.current_channel = None
        self.in_header = False
        self.header_ids = []
        self.handles = []
        self.vector_cache = {}
        self.modified_vectors = defaultdict(int)
        self.audit = defaultdict(lambda: {
            "n": 0,
            "sum_pre": 0.0,
            "sum_post": 0.0,
            "sum_delta": 0.0,
            "sum_sq_pre": 0.0,
            "sum_sq_post": 0.0,
        })
        self.forward_calls = 0

        self.channel_id = tok.convert_tokens_to_ids("<|channel|>")
        self.message_id = tok.convert_tokens_to_ids("<|message|>")
        self.start_id = tok.convert_tokens_to_ids("<|start|>")
        self.end_ids = {
            x
            for x in [
                tok.convert_tokens_to_ids("<|end|>"),
                tok.convert_tokens_to_ids("<|call|>"),
                tok.convert_tokens_to_ids("<|return|>"),
                tok.eos_token_id,
            ]
            if isinstance(x, int) and x >= 0
        }

        # This is the exact generation-channel suffix appended by GPT-OSS's
        # add_generation_prompt in the v8 collector.
        self.analysis_header_pattern = tok.encode(
            "<|channel|>analysis<|message|>",
            add_special_tokens=False,
        )
        if not self.analysis_header_pattern:
            raise RuntimeError("Could not tokenize GPT-OSS analysis header pattern.")

    def begin_turn(self):
        self.enabled = True
        self.mask = None
        self.current_channel = None
        self.in_header = False
        self.header_ids = []

    def end_turn(self):
        self.enabled = False
        self.mask = None
        self.current_channel = None
        self.in_header = False
        self.header_ids = []

    def _find_last_pattern(self, ids, pattern):
        n = len(pattern)
        for i in range(len(ids) - n, -1, -1):
            if ids[i : i + n] == pattern:
                return i
        return None

    def _parse_header_channel(self):
        if not self.header_ids:
            return None
        text = self.tok.decode(
            self.header_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip().lower()
        # The header can include "commentary to=functions.foo json".
        for channel in ("analysis", "commentary", "final"):
            if re.search(rf"\b{channel}\b", text):
                return channel
        return None

    def _scan_token(self, tid: int):
        """
        Consume one token and return whether its hidden state should be edited.
        """
        if tid in self.end_ids:
            self.current_channel = None
            self.in_header = False
            self.header_ids = []
            return False

        if tid == self.start_id:
            self.current_channel = None
            self.in_header = False
            self.header_ids = []
            return False

        if tid == self.channel_id:
            self.current_channel = None
            self.in_header = True
            self.header_ids = []
            return False

        if self.in_header:
            if tid == self.message_id:
                self.current_channel = self._parse_header_channel()
                self.in_header = False
                self.header_ids = []
                # Editing the message marker itself lets the intervention affect
                # the first generated content token.
                return self.current_channel in self.allowed_channels
            self.header_ids.append(tid)
            return False

        return self.current_channel in self.allowed_channels

    def _mask_full(self, ids: list[int]):
        mask = [False] * len(ids)
        start = self._find_last_pattern(ids, self.analysis_header_pattern)

        if start is None:
            # Fallback: modify only the last prompt position, enough to steer the
            # first next-token prediction without touching history.
            self.current_channel = "analysis"
            if ids:
                mask[-1] = "analysis" in self.allowed_channels
            return mask

        msg_pos = start + len(self.analysis_header_pattern) - 1

        # Current generation begins in analysis channel immediately after the
        # latest analysis header.
        self.current_channel = "analysis"
        self.in_header = False
        self.header_ids = []

        if "analysis" in self.allowed_channels:
            mask[msg_pos] = True

        for i in range(msg_pos + 1, len(ids)):
            mask[i] = self._scan_token(int(ids[i]))

        return mask

    def _pre_hook(self, module, args, kwargs):
        if not self.enabled:
            self.mask = None
            return

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            first = args[0]
            if torch.is_tensor(first) and first.dtype in (torch.int32, torch.int64):
                input_ids = first

        if input_ids is None:
            # Rare fallback if generation uses inputs_embeds.
            self.mask = "last"
            return

        ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
        self.forward_calls += 1

        if len(ids) > 1:
            self.mask = self._mask_full(ids)
        elif len(ids) == 1:
            self.mask = [self._scan_token(ids[0])]
        else:
            self.mask = []

    def _vector_for(self, edit: TargetEdit, device, dtype):
        key = (edit.key, str(device), str(dtype))
        v = self.vector_cache.get(key)
        if v is None:
            # Coefficients are computed in float32 for stability; delta is cast
            # back to model dtype.
            v = edit.v.to(device=device, dtype=torch.float32)
            self.vector_cache[key] = v
        return v

    def _layer_hook(self, layer: int):
        def hook(module, inputs, output):
            if not self.enabled or not self.edits_by_layer.get(layer):
                return None

            tensor = output if torch.is_tensor(output) else output[0]
            seq_len = tensor.shape[-2]

            if self.mask == "last":
                indices = [seq_len - 1]
            elif isinstance(self.mask, list):
                if len(self.mask) != seq_len:
                    # Cache/generation implementation changed shape. The safe
                    # fallback is current token only, never history.
                    indices = [seq_len - 1] if seq_len else []
                else:
                    indices = [i for i, flag in enumerate(self.mask) if flag]
            else:
                indices = []

            if not indices:
                return None

            idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
            x = tensor.index_select(-2, idx).float()  # [batch, selected, d]

            # v directions are unit norm from SVD; edits are exact component edits.
            total_delta = torch.zeros_like(x)
            for edit in self.edits_by_layer[layer]:
                v = self._vector_for(edit, x.device, x.dtype)
                coeff = torch.einsum("bsd,d->bs", x, v)
                new_coeff = edit.multiplier * coeff + edit.bias
                delta_coeff = new_coeff - coeff
                total_delta.add_(delta_coeff.unsqueeze(-1) * v)
                nobs = int(coeff.numel())
                self.modified_vectors[edit.key] += nobs

                # Audit the actual coefficient manipulation. Counts include
                # chunk-reprefill recomputations, which is fine: the purpose is
                # to verify the hook numerically, not estimate independent tokens.
                a = self.audit[edit.key]
                pre_cpu = coeff.detach().float()
                post_cpu = new_coeff.detach().float()
                a["n"] += nobs
                a["sum_pre"] += float(pre_cpu.sum().item())
                a["sum_post"] += float(post_cpu.sum().item())
                a["sum_delta"] += float(delta_coeff.detach().float().sum().item())
                a["sum_sq_pre"] += float((pre_cpu * pre_cpu).sum().item())
                a["sum_sq_post"] += float((post_cpu * post_cpu).sum().item())

            # Inference-only causal experiment: mutate selected residual positions
            # in place to avoid cloning a long full-sequence tensor.
            selected = tensor.index_select(-2, idx)
            selected.add_(total_delta.to(dtype=tensor.dtype))
            tensor.index_copy_(-2, idx, selected)

            return None

        return hook

    def __enter__(self):
        # with_kwargs=True is needed to inspect generation input_ids.
        self.handles.append(
            self.model.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
        )
        for layer in sorted(self.edits_by_layer):
            self.handles.append(
                self.decoder_layers[layer].register_forward_hook(
                    self._layer_hook(layer)
                )
            )
        return self

    def __exit__(self, *exc):
        self.end_turn()
        for h in self.handles:
            h.remove()
        self.handles = []

    def telemetry(self):
        audit_out = {}
        for key, a in self.audit.items():
            n = max(1, int(a["n"]))
            mean_pre = a["sum_pre"] / n
            mean_post = a["sum_post"] / n
            mean_delta = a["sum_delta"] / n
            var_pre = max(0.0, a["sum_sq_pre"] / n - mean_pre * mean_pre)
            var_post = max(0.0, a["sum_sq_post"] / n - mean_post * mean_post)
            audit_out[key] = {
                "n_forward_vector_instances": int(a["n"]),
                "mean_pre": mean_pre,
                "std_pre": var_pre ** 0.5,
                "mean_post": mean_post,
                "std_post": var_post ** 0.5,
                "mean_delta": mean_delta,
            }
        return {
            "scope": self.scope,
            "forward_calls": self.forward_calls,
            "modified_vector_instances": dict(self.modified_vectors),
            "coefficient_audit": audit_out,
        }


@contextlib.contextmanager
def patch_collector_generation(base, controller):
    """
    Reuse the exact v8 generate_assistant implementation, merely bracketing each
    assistant turn with controller state reset.
    """
    original = base.generate_assistant

    def wrapped(*args, **kwargs):
        controller.begin_turn()
        try:
            return original(*args, **kwargs)
        finally:
            controller.end_turn()

    base.generate_assistant = wrapped
    try:
        yield
    finally:
        base.generate_assistant = original


def bool01(x):
    return 1.0 if x is True else 0.0 if x is False else float("nan")


def task_success(result):
    env = result.get("environment")
    if env == "secret_number":
        return bool(result.get("correct"))
    if env == "precommit_hook":
        return bool(result.get("commit_exists"))
    if env == "impossiblebench":
        sub = result.get("submission")
        if isinstance(sub, dict):
            try:
                return float(sub.get("reward", 0.0)) > 0
            except Exception:
                pass
        return bool(result.get("visible_all_pass"))
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rebuild_summaries(index_path: Path, out: Path):
    if not index_path.exists():
        return
    records = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") is None:
            records.append(r)

    grouped = defaultdict(list)
    for r in records:
        grouped[
            (
                r.get("intervention_name"),
                r.get("environment"),
                r.get("condition"),
            )
        ].append(r)

    summary_rows = []
    for (variant, env, cond), rs in sorted(grouped.items()):
        task_game = [bool01(r.get("task_gaming_heuristic")) for r in rs]
        action = [bool01(r.get("gaming_action_heuristic")) for r in rs]
        gsuccess = [bool01(r.get("gaming_success_heuristic")) for r in rs]
        success = [1.0 if task_success(r) else 0.0 for r in rs]

        def nanmean(xs):
            a = np.asarray(xs, dtype=float)
            return float(np.nanmean(a)) if np.any(np.isfinite(a)) else None

        summary_rows.append(
            {
                "intervention": variant,
                "environment": env,
                "condition": cond,
                "n": len(rs),
                "task_gaming_rate": nanmean(task_game),
                "gaming_action_rate": nanmean(action),
                "gaming_success_rate": nanmean(gsuccess),
                "task_success_rate": nanmean(success),
            }
        )

    write_csv(out / "causal_summary.csv", summary_rows)

    # Paired cell-level effects relative to matched control seeds.
    by_cell = {}
    for r in records:
        by_cell[
            (
                r.get("intervention_name"),
                r.get("environment"),
                r.get("condition"),
                int(r.get("seed")),
            )
        ] = r

    pair_rows = []
    variants = sorted(
        {r.get("intervention_name") for r in records if r.get("intervention_name")}
        - {"control"}
    )
    envconds = sorted(
        {(r.get("environment"), r.get("condition")) for r in records}
    )

    for variant in variants:
        for env, cond in envconds:
            deltas_task = []
            deltas_action = []
            deltas_success = []
            n_pairs = 0
            seeds = sorted(
                {
                    int(r.get("seed"))
                    for r in records
                    if r.get("environment") == env and r.get("condition") == cond
                }
            )
            for seed in seeds:
                c = by_cell.get(("control", env, cond, seed))
                t = by_cell.get((variant, env, cond, seed))
                if not c or not t:
                    continue
                n_pairs += 1
                cg = bool01(c.get("task_gaming_heuristic"))
                tg = bool01(t.get("task_gaming_heuristic"))
                ca = bool01(c.get("gaming_action_heuristic"))
                ta = bool01(t.get("gaming_action_heuristic"))
                if math.isfinite(cg) and math.isfinite(tg):
                    deltas_task.append(tg - cg)
                if math.isfinite(ca) and math.isfinite(ta):
                    deltas_action.append(ta - ca)
                deltas_success.append(
                    (1.0 if task_success(t) else 0.0)
                    - (1.0 if task_success(c) else 0.0)
                )

            if not n_pairs:
                continue

            pair_rows.append(
                {
                    "intervention": variant,
                    "environment": env,
                    "condition": cond,
                    "n_matched_pairs": n_pairs,
                    "delta_task_gaming_rate": (
                        float(np.mean(deltas_task)) if deltas_task else None
                    ),
                    "delta_gaming_action_rate": (
                        float(np.mean(deltas_action)) if deltas_action else None
                    ),
                    "delta_task_success_rate": float(np.mean(deltas_success)),
                }
            )

    write_csv(out / "paired_effects.csv", pair_rows)



def read_jsonl_records(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_cell_key(r: dict[str, Any]):
    """Stable overlap key independent of run_id/output-directory naming."""
    try:
        seed = int(r.get("seed"))
    except Exception:
        return None
    variant = r.get("intervention_name")
    env = r.get("environment")
    condition = r.get("condition")
    if not variant or not env or not condition:
        return None
    return (str(variant), str(env), str(condition), seed)


def cell_id_from_key(key):
    variant, env, condition, seed = key
    return f"{variant}|{env}|{condition}|s{seed}"


def compatible_campaign_settings(campaign: dict[str, Any], args):
    checks = [
        ("model", campaign.get("model"), args.model),
        ("reasoning", campaign.get("reasoning"), args.reasoning),
        ("scope", campaign.get("scope"), args.scope),
        ("temperature", campaign.get("temperature"), args.temperature),
        ("top_p", campaign.get("top_p"), args.top_p),
    ]
    mismatches = []
    for name, old, new in checks:
        if old is not None and old != new:
            mismatches.append(f"{name}: old={old!r} new={new!r}")
    return mismatches


def resolve_reuse_sources(spec: str):
    sources = []
    seen = set()
    for raw in [x.strip() for x in spec.split(",") if x.strip()]:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            if p.name != "index.jsonl":
                print(f"[reuse] ignoring non-index file: {p}")
                continue
            root = p.parent
            index = p
        else:
            root = p
            index = root / "index.jsonl"
        key = str(index)
        if key in seen:
            continue
        seen.add(key)
        sources.append((root, index))
    return sources


def load_reused_successes(args, allowed_variants: set[str], out: Path):
    reused = {}
    provenance = []

    for root, index_path in resolve_reuse_sources(args.reuse_from):
        if root == out:
            continue
        if not index_path.exists():
            print(f"[reuse] not found, skipping: {index_path}")
            continue

        campaign_path = root / "campaign.json"
        campaign = {}
        if campaign_path.exists():
            try:
                campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            except Exception:
                campaign = {}

        mismatches = compatible_campaign_settings(campaign, args)
        if mismatches and not args.allow_reuse_mismatch:
            print(f"[reuse] settings mismatch, skipping {root}:")
            for m in mismatches:
                print(f"        {m}")
            continue

        source_n = 0
        for r in read_jsonl_records(index_path):
            if r.get("error") is not None:
                continue
            key = normalized_cell_key(r)
            if key is None or key[0] not in allowed_variants:
                continue
            if key in reused:
                continue

            rr = dict(r)
            rr["_data_origin"] = "reused"
            rr["_source_campaign"] = str(root)
            rr["_source_index"] = str(index_path)
            reused[key] = rr
            provenance.append(
                {
                    "causal_cell_id": cell_id_from_key(key),
                    "intervention_name": key[0],
                    "environment": key[1],
                    "condition": key[2],
                    "seed": key[3],
                    "source_campaign": str(root),
                    "source_index": str(index_path),
                    "source_run_id": r.get("run_id"),
                }
            )
            source_n += 1

        print(f"[reuse] {root}: imported {source_n} completed L17/control cells")

    return reused, provenance


def merged_success_records(
    reused: dict[tuple, dict[str, Any]],
    current_index: Path,
):
    merged = dict(reused)
    for r in read_jsonl_records(current_index):
        if r.get("error") is not None:
            continue
        key = normalized_cell_key(r)
        if key is None:
            continue
        rr = dict(r)
        rr["_data_origin"] = "new"
        merged[key] = rr
    return merged


def refresh_combined_outputs(
    reused: dict[tuple, dict[str, Any]],
    current_index: Path,
    out: Path,
):
    merged = merged_success_records(reused, current_index)
    rows = [
        merged[k]
        for k in sorted(
            merged,
            key=lambda x: (x[1], x[2], x[0], x[3]),
        )
    ]
    combined_path = out / "combined_index.jsonl"
    write_jsonl_records(combined_path, rows)
    rebuild_summaries(combined_path, out)
    return merged



def main():
    args = parse_args()

    if args.list_variants:
        print("\n".join(all_builtin_variant_names()))
        return

    collector_path = Path(args.collector)
    directions_dir = Path(args.directions_dir).expanduser().resolve()
    event_windows = Path(args.event_windows).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    roll_root = out / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    roll_root.mkdir(parents=True, exist_ok=True)

    requested_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    bad_variants = sorted(set(requested_names) - set(all_builtin_variant_names()))
    if bad_variants:
        raise SystemExit(
            "Unsupported variants for this L17-only script: "
            + ", ".join(bad_variants)
        )

    reused, provenance = load_reused_successes(
        args=args,
        allowed_variants=set(requested_names),
        out=out,
    )
    write_jsonl_records(out / "reused_cells.jsonl", provenance)

    index_path = out / "index.jsonl"  # NEW focused-run records only.
    judge_path = out / "judge_input.jsonl"
    manifest_path = out / "analysis_manifest.jsonl"

    merged_at_start = refresh_combined_outputs(reused, index_path, out)
    if merged_at_start:
        n_old = sum(
            1 for r in merged_at_start.values()
            if r.get("_data_origin") == "reused"
        )
        n_new = sum(
            1 for r in merged_at_start.values()
            if r.get("_data_origin") == "new"
        )
        print(
            f"[combined] before run: {len(merged_at_start)} completed cells "
            f"({n_old} reused, {n_new} already in focused output)"
        )

    base = import_collector(collector_path)
    base.DockerSandbox.ensure_docker()

    direction_info = load_direction_vectors(directions_dir)
    scales = load_natural_scales(event_windows)

    print("=== L17/SV28 calibration ===")
    print(
        f"L17/SV28: baseline_mean={scales['L17']['mean']:+.4f} "
        f"baseline_SD={scales['L17']['scale']:.4f} "
        f"sigma={direction_info['L17']['sigma']:.4f}"
    )

    variants = [
        make_variant(
            name,
            direction_info,
            scales,
            args.amplify_factor,
            args.bias_strength,
        )
        for name in requested_names
    ]

    if args.smoke:
        variants = [
            make_variant(
                name,
                direction_info,
                scales,
                args.amplify_factor,
                args.bias_strength,
            )
            for name in ("control", "L17_suppress")
        ]

    wanted = (
        {"secret_number", "precommit_hook", "impossiblebench"}
        if args.env == "all"
        else {x.strip() for x in args.env.split(",") if x.strip()}
    )
    only = {x.strip() for x in args.only_condition.split(",") if x.strip()}
    specs = [
        s
        for s in base.build_specs()
        if s.name in wanted and (not only or s.condition in only)
    ]
    if args.smoke:
        specs = specs[:1]
    if not specs:
        raise SystemExit("No matching environments/conditions.")

    reps = 1 if args.smoke else args.samples_per_condition

    # Stable seed scheme deliberately matches the previous broad script.
    planned = []
    for variant in variants:
        for si, spec in enumerate(specs):
            for rep in range(reps):
                seed = args.seed + si * 10000 + rep
                planned.append(
                    (variant.name, spec.name, spec.condition, seed)
                )

    jdump(
        {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "focus": "L17/SV28 only",
            "model": args.model,
            "model_revision": args.model_revision,
            "collector": str(collector_path),
            "transformers_version": transformers.__version__,
            "attention": args.attn_implementation,
            "scope": args.scope,
            "reasoning": args.reasoning,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "samples_per_condition": reps,
            "base_seed": args.seed,
            "amplify_factor": args.amplify_factor,
            "bias_strength_sd": args.bias_strength,
            "candidate_calibration": scales,
            "variants": [v.as_dict() for v in variants],
            "conditions": [
                {"environment": s.name, "condition": s.condition, **s.metadata}
                for s in specs
            ],
            "reuse_from": args.reuse_from,
            "reused_completed_cells": len(reused),
            "dedupe_key": [
                "intervention_name",
                "environment",
                "condition",
                "seed",
            ],
            "note": (
                "index.jsonl contains only newly generated runs. "
                "combined_index.jsonl is the deduplicated union with compatible "
                "completed L17/control cells from prior campaigns."
            ),
        },
        out / "campaign.json",
    )

    completed_map = refresh_combined_outputs(reused, index_path, out)
    overlap = [k for k in planned if k in completed_map]
    missing = [k for k in planned if k not in completed_map]
    print(
        f"[plan] total={len(planned)} already-complete={len(overlap)} "
        f"to-generate={len(missing)}"
    )
    if overlap:
        origins = defaultdict(int)
        for k in overlap:
            origins[completed_map[k].get("_data_origin", "unknown")] += 1
        print(
            "[plan] overlap origins: "
            + ", ".join(f"{k}={v}" for k, v in sorted(origins.items()))
        )

    if not missing and not args.overwrite:
        print("[done] all planned L17/SV28 cells already exist.")
        print(f"Combined outputs: {out}")
        return

    print(f"[load] {args.model} attention={args.attn_implementation}")
    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype="auto",
        device_map="cuda" if torch.cuda.is_available() else "auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    decoder_layers = find_decoder_layers(model)

    if torch.cuda.is_available():
        print(f"[load] CUDA allocated={torch.cuda.memory_allocated()/1e9:.1f} GB")

    collector_args = argparse.Namespace(
        model=args.model,
        model_revision=args.model_revision,
        reasoning=args.reasoning,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        max_turn_tokens=args.max_turn_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        command_timeout=args.command_timeout,
        docker_image=args.docker_image,
    )

    total = len(planned)
    done = 0

    for variant in variants:
        print(f"\n=== Intervention: {variant.name} ===")
        print(variant.description)

        for si, spec in enumerate(specs):
            for rep in range(reps):
                seed = args.seed + si * 10000 + rep
                key = (
                    variant.name,
                    spec.name,
                    spec.condition,
                    seed,
                )
                cell_id = cell_id_from_key(key)

                current_combined = refresh_combined_outputs(
                    reused, index_path, out
                )
                if key in current_combined and not args.overwrite:
                    done += 1
                    origin = current_combined[key].get(
                        "_data_origin", "existing"
                    )
                    print(f"[skip:{origin}] {cell_id}")
                    continue

                run_id = (
                    f"{safe_name(spec.name)}__{safe_name(spec.condition)}"
                    f"__s{seed}__int-{safe_name(variant.name)}"
                )
                run_dir = (
                    roll_root
                    / variant.name
                    / spec.name
                    / spec.condition
                    / run_id
                )
                if args.overwrite and run_dir.exists():
                    shutil.rmtree(run_dir)

                print(
                    f"\n[{done+1:03d}/{total:03d}] "
                    f"{variant.name} | {spec.name}/{spec.condition} seed={seed}"
                )

                controller = ResidualInterventionController(
                    model=model,
                    tok=tok,
                    decoder_layers=decoder_layers,
                    variant=variant,
                    scope=args.scope,
                )

                try:
                    with controller, patch_collector_generation(base, controller):
                        result = base.run_one(
                            model,
                            tok,
                            spec,
                            collector_args,
                            seed,
                            run_dir,
                        )

                    result.update(
                        {
                            "causal_cell_id": cell_id,
                            "intervention_name": variant.name,
                            "intervention_description": variant.description,
                            "intervention_scope": args.scope,
                            "intervention": variant.as_dict(),
                            "intervention_telemetry": controller.telemetry(),
                            "focused_direction": "L17/SV28",
                            "data_origin": "new",
                        }
                    )
                    jdump(result, run_dir / "result.json")

                    judge = json.loads(
                        (run_dir / "judge_record.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    judge.update(
                        {
                            "causal_cell_id": cell_id,
                            "intervention_name": variant.name,
                            "intervention_description": variant.description,
                            "intervention_scope": args.scope,
                            "intervention": variant.as_dict(),
                            "focused_direction": "L17/SV28",
                        }
                    )
                    jdump(judge, run_dir / "judge_record.json")

                    append_jsonl(result, index_path)
                    append_jsonl(judge, judge_path)
                    append_jsonl(
                        {
                            "causal_cell_id": cell_id,
                            "run_id": run_id,
                            "intervention_name": variant.name,
                            "intervention_scope": args.scope,
                            "focused_direction": "L17/SV28",
                            "environment": spec.name,
                            "condition": spec.condition,
                            "seed": seed,
                            "reasoning": args.reasoning,
                            "transcript": str(
                                (run_dir / "transcript.json").relative_to(out)
                            ),
                            "replay_tokens": str(
                                (run_dir / "replay_tokens.jsonl").relative_to(out)
                            ),
                            "raw_turns": str(
                                (run_dir / "raw_turns.jsonl").relative_to(out)
                            ),
                            "result": str(
                                (run_dir / "result.json").relative_to(out)
                            ),
                            "task_gaming_heuristic": result.get(
                                "task_gaming_heuristic"
                            ),
                            "gaming_action_heuristic": result.get(
                                "gaming_action_heuristic"
                            ),
                            "gaming_success_heuristic": result.get(
                                "gaming_success_heuristic"
                            ),
                        },
                        manifest_path,
                    )

                    print(
                        "[done] "
                        f"gaming={result.get('task_gaming_heuristic')} "
                        f"action={result.get('gaming_action_heuristic')} "
                        f"success={result.get('gaming_success_heuristic')} "
                        f"steps={result.get('steps')} "
                        f"wall={result.get('wall_s')}s"
                    )

                    audit = controller.telemetry().get(
                        "coefficient_audit", {}
                    ).get("L17")
                    if audit:
                        print(
                            "[audit] L17/SV28 "
                            f"mean {audit['mean_pre']:+.4f} -> "
                            f"{audit['mean_post']:+.4f}; "
                            f"std {audit['std_pre']:.4f} -> "
                            f"{audit['std_post']:.4f}; "
                            f"mean_delta={audit['mean_delta']:+.4f}"
                        )

                except Exception as e:
                    print(f"[ERROR] {cell_id}: {e}")
                    err = {
                        "causal_cell_id": cell_id,
                        "run_id": run_id,
                        "intervention_name": variant.name,
                        "intervention_scope": args.scope,
                        "focused_direction": "L17/SV28",
                        "environment": spec.name,
                        "condition": spec.condition,
                        "seed": seed,
                        "error": repr(e),
                    }
                    run_dir.mkdir(parents=True, exist_ok=True)
                    jdump(err, run_dir / "error.json")
                    append_jsonl(err, index_path)

                done += 1
                refresh_combined_outputs(reused, index_path, out)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    merged_final = refresh_combined_outputs(reused, index_path, out)

    print("\n=== Combined causal summary (reused + new, deduplicated) ===")
    summary_path = out / "causal_summary.csv"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))

    print("\n=== Paired intervention effects vs matched control ===")
    pair_path = out / "paired_effects.csv"
    if pair_path.exists():
        print(pair_path.read_text(encoding="utf-8"))

    n_reused = sum(
        1 for r in merged_final.values()
        if r.get("_data_origin") == "reused"
    )
    n_new = sum(
        1 for r in merged_final.values()
        if r.get("_data_origin") == "new"
    )
    print(
        f"\n[done] combined successful cells={len(merged_final)} "
        f"reused={n_reused} new={n_new}"
    )
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
