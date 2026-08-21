#!/usr/bin/env python3
"""
Collect targeted GPT-OSS-20B residual-stream windows around annotated
subjective-progress events and their matched controls.

This is the collection pass that follows analyze_subjective_progress_warms.py.
It does NOT generate new behavior. It replays the exact token sequences saved
by collect_task_gaming_rollouts_v8.py.

Why this exists
---------------
The earlier W-arm analysis can only test the three preselected directions that
were present in task_gaming_candidate_data/activations.csv. Here we collect the
information needed for a real discovery search:

  1. Full residual vectors at every available source layer, but ONLY in a small
     token window around each annotated subjective-progress event and its
     matched controls.

  2. Projection of those same residual vectors onto a broad bank of existing
     J-lens right singular vectors V (default: the first 32 per layer from
     task_gaming_jlens/directions/LXX.npz).

The output therefore supports two downstream analyses:

  A. Broad SV screen:
       a[layer, sv, t] = h[layer, t] @ V[layer, sv]

  B. Direct progress-direction discovery, without assuming the signal is an SV:
       d_i[layer] = mean(h_after) - mean(h_before)
     followed by event-minus-control and correct/incorrect replication tests.

The script is designed for an L4-class memory constraint:

  * exact saved trajectories are replayed rather than regenerated;
  * one forward pass is used per (run_id, step);
  * hooks copy only requested token positions to CPU immediately;
  * whole-sequence residual streams are never retained;
  * J-lens Jacobians are NOT loaded on GPU in this pass;
  * residual files default to float16 on disk.

Prerequisite
------------
Run analyze_subjective_progress_warms.py first so that its arm-independent
matched controls exist:

    subjective_progress_warm_analysis/control_positions.jsonl

Typical usage
-------------
    python3 collect_subjective_progress_residuals.py \
        --campaign task_gaming_v8 \
        --events task_gaming_v8_subjective_progress/events.jsonl \
        --controls subjective_progress_warm_analysis/control_positions.jsonl \
        --directions-dir task_gaming_jlens/directions \
        --out subjective_progress_residual_data \
        --window 16 \
        --sv-k 32

Resume after interruption:

    python3 collect_subjective_progress_residuals.py ... --resume

To use 64 SVs/layer, the LXX.npz direction files themselves must contain at
least 64 columns. Recompute those first with analyze_task_gaming_jlens_v2.py
using --k 64 --recompute-directions, then run this script with --sv-k 64.

Outputs
-------
<out>/
  meta.json
  direction_bank.npz
      layers [L], singular_values [L,K], available_ranks [L]

  windows.jsonl
      one metadata row per event/control window

  windows/*.npz
      one fixed-width local window per event/control containing:
        layers                  [L]
        relative_tokens         [T]       (-window ... +window)
        valid_mask              [T]
        generated_local_indices [T]       -1 for missing
        full_positions          [T]       -1 for missing
        token_ids               [T]       -1 for missing
        residuals               [L,T,D]   float16 by default, NaN when missing
        sv_activations          [L,T,K]   float32, NaN when missing

  replay_failures.jsonl
      OOM/alignment failures that did not abort the whole run

Important semantics
-------------------
* t=0 is the saved final-event-token anchor used by the annotation pipeline.
* Window offsets are true generated-token offsets, matching the previous
  analyzer. A position is included only if it remains in the anchor's same
  Harmony channel and message index. Special/boundary positions therefore show
  up as missing rather than silently shifting the window.
* The hooked vector is the output of model.layers[layer] at that token, matching
  the earlier candidate extractor's h_layer_token convention.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers

try:
    import jlens
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "This script expects the same `jlens` package used by the previous "
        "task-gaming extraction scripts."
    ) from exc


SCHEMA_VERSION = "1.0"


# -----------------------------------------------------------------------------
# CLI / small utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--campaign", type=Path, default=Path("task_gaming_v8"))
    p.add_argument(
        "--events",
        type=Path,
        default=Path("task_gaming_v8_subjective_progress/events.jsonl"),
    )
    p.add_argument(
        "--controls",
        type=Path,
        default=Path("subjective_progress_warm_analysis/control_positions.jsonl"),
        help=(
            "Arm-independent matched controls produced by "
            "analyze_subjective_progress_warms.py."
        ),
    )
    p.add_argument(
        "--directions-dir",
        type=Path,
        default=Path("task_gaming_jlens/directions"),
        help="Directory containing LXX.npz files with V/S arrays.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("subjective_progress_residual_data")
    )

    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument(
        "--attn-implementation",
        default="flex_attention",
        help="GPT-OSS long-prefix replay attention backend.",
    )

    p.add_argument(
        "--window",
        type=int,
        default=16,
        help="Collect true generated-token offsets -W...+W around every anchor.",
    )
    p.add_argument(
        "--layers",
        default="",
        help=(
            "Comma-separated layer indices. Empty = every LXX.npz in "
            "--directions-dir."
        ),
    )
    p.add_argument(
        "--sv-k",
        type=int,
        default=32,
        help="Number of leading right singular vectors to project per layer; 0 disables SV projection.",
    )
    p.add_argument(
        "--max-controls-per-event",
        type=int,
        default=0,
        help="0 keeps every supplied matched control; otherwise keep the first N per event.",
    )
    p.add_argument(
        "--residual-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk dtype for full residual vectors.",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed for each window (smaller but slower).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume, skipping window_ids already recorded in windows.jsonl.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out first. Mutually exclusive with --resume.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first missing replay/alignment problem instead of logging and continuing.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window < 1:
        raise SystemExit("--window must be >= 1")
    if args.sv_k < 0:
        raise SystemExit("--sv-k must be >= 0")
    if args.max_controls_per_event < 0:
        raise SystemExit("--max-controls-per-event must be >= 0")
    if args.resume and args.overwrite:
        raise SystemExit("Choose at most one of --resume and --overwrite")


def parse_ints(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(dict(obj)), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_.")
    return text or "window"


def safe_token_text(tok, token_id: int) -> str:
    return tok.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


# -----------------------------------------------------------------------------
# Harmony token mapping — intentionally mirrors the old extractor.
# -----------------------------------------------------------------------------


def content_token_positions(tok, replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return non-special message-content positions for a saved generation.

    Each record has full_position, generated_local_index, message_index, channel.
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

    rows: list[dict[str, Any]] = []
    message_index = 0
    i = 0
    while i < len(gen):
        if gen[i] != chan_id:
            i += 1
            continue

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

        for local_idx in range(j + 1, k):
            tid = gen[local_idx]
            if tid in special:
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

    # Preserve the previous extractor's fallback for unusual non-Harmony output.
    if not rows:
        for local_idx, tid in enumerate(gen):
            if tid in special:
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


# -----------------------------------------------------------------------------
# Anchor/window descriptions
# -----------------------------------------------------------------------------


@dataclass
class AnchorSpec:
    window_id: str
    role: str  # event | control
    event_id: str
    run_id: str
    step: int
    anchor_generated_index: int
    objective_status: str | None = None
    environment: str | None = None
    condition: str | None = None
    event_type: str | None = None
    confidence: Any = None
    control_index: int | None = None
    expected_channel: str | None = None
    expected_message_index: int | None = None
    expected_token_id: int | None = None
    control_match_tier: str | None = None
    control_scope: str | None = None
    control_lexical_match: str | None = None

    # Filled after replay-token resolution.
    anchor_channel: str | None = None
    anchor_message_index: int | None = None
    anchor_full_position: int | None = None
    anchor_token_id: int | None = None
    relative_tokens: list[int] = field(default_factory=list)
    valid_mask: list[bool] = field(default_factory=list)
    generated_local_indices: list[int] = field(default_factory=list)
    full_positions: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str | None] = field(default_factory=list)


def load_anchor_specs(
    events_path: Path,
    controls_path: Path,
    max_controls_per_event: int,
) -> tuple[list[AnchorSpec], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events = read_jsonl(events_path)
    if not events:
        raise SystemExit(f"No events found in {events_path}")
    controls = read_jsonl(controls_path)
    if not controls:
        raise SystemExit(
            f"No controls found in {controls_path}. Run analyze_subjective_progress_warms.py first."
        )

    event_by_id = {str(e["event_id"]): e for e in events if e.get("event_id")}
    specs: list[AnchorSpec] = []
    excluded: list[dict[str, Any]] = []

    # Use mapped events. The three partial-OOM annotations intentionally have no
    # old W-arm row and generally no completed campaign manifest replay.
    for e in events:
        event_id = str(e.get("event_id", ""))
        if not event_id:
            excluded.append({**e, "collection_exclusion_reason": "missing_event_id"})
            continue
        anchor = e.get("anchor_generated_position")
        if anchor is None:
            excluded.append(
                {**e, "collection_exclusion_reason": "missing_anchor_generated_position"}
            )
            continue
        if e.get("w_arm_alignment_available") is False:
            excluded.append(
                {**e, "collection_exclusion_reason": "w_arm_alignment_unavailable"}
            )
            continue
        specs.append(
            AnchorSpec(
                window_id=f"{event_id}__event",
                role="event",
                event_id=event_id,
                run_id=str(e["run_id"]),
                step=int(e["step"]),
                anchor_generated_index=int(anchor),
                objective_status=e.get("objective_status"),
                environment=e.get("environment"),
                condition=e.get("condition"),
                event_type=e.get("event_type"),
                confidence=e.get("confidence"),
                expected_channel="analysis",
            )
        )

    controls_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in controls:
        controls_by_event[str(c.get("event_id", ""))].append(c)

    for event_id, rows in controls_by_event.items():
        event = event_by_id.get(event_id)
        if event is None:
            excluded.append(
                {**rows[0], "collection_exclusion_reason": "control_for_unknown_event"}
            )
            continue
        rows = sorted(rows, key=lambda r: int(r.get("control_index", 0)))
        if max_controls_per_event:
            rows = rows[:max_controls_per_event]
        for c in rows:
            ci = int(c.get("control_index", 0))
            specs.append(
                AnchorSpec(
                    window_id=f"{event_id}__control_{ci:02d}",
                    role="control",
                    event_id=event_id,
                    run_id=str(c["run_id"]),
                    step=int(c["step"]),
                    anchor_generated_index=int(c["generated_local_index"]),
                    objective_status=event.get("objective_status"),
                    environment=event.get("environment"),
                    condition=event.get("condition"),
                    event_type=event.get("event_type"),
                    confidence=event.get("confidence"),
                    control_index=ci,
                    expected_channel=c.get("channel"),
                    expected_message_index=(
                        int(c["message_index"]) if c.get("message_index") is not None else None
                    ),
                    expected_token_id=(
                        int(c["token_id"]) if c.get("token_id") is not None else None
                    ),
                    control_match_tier=c.get("match_tier"),
                    control_scope=c.get("scope"),
                    control_lexical_match=c.get("lexical_match"),
                )
            )

    specs.sort(
        key=lambda s: (
            s.run_id,
            s.step,
            s.anchor_generated_index,
            0 if s.role == "event" else 1,
            s.control_index or 0,
        )
    )
    return specs, excluded, event_by_id


def load_replay_map(campaign: Path, manifest_row: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    path = campaign / str(manifest_row["replay_tokens"])
    if not path.exists():
        raise FileNotFoundError(f"Missing replay file {path}")
    return {int(r["step"]): r for r in read_jsonl(path)}


def resolve_window(tok, replay: Mapping[str, Any], spec: AnchorSpec, window: int) -> None:
    pos_rows = content_token_positions(tok, replay)
    by_generated = {int(r["generated_local_index"]): r for r in pos_rows}
    anchor_row = by_generated.get(spec.anchor_generated_index)
    if anchor_row is None:
        raise ValueError(
            f"{spec.window_id}: anchor generated_local_index={spec.anchor_generated_index} "
            "is not a non-special message-content token"
        )

    channel = str(anchor_row["channel"])
    message_index = int(anchor_row["message_index"])
    if spec.expected_channel is not None and channel != str(spec.expected_channel):
        raise ValueError(
            f"{spec.window_id}: channel mismatch resolved={channel!r} "
            f"expected={spec.expected_channel!r}"
        )
    if (
        spec.expected_message_index is not None
        and message_index != int(spec.expected_message_index)
    ):
        raise ValueError(
            f"{spec.window_id}: message_index mismatch resolved={message_index} "
            f"expected={spec.expected_message_index}"
        )

    gen_ids = [int(x) for x in replay["generated_token_ids"]]
    anchor_tid = gen_ids[spec.anchor_generated_index]
    if spec.expected_token_id is not None and anchor_tid != spec.expected_token_id:
        raise ValueError(
            f"{spec.window_id}: token_id mismatch resolved={anchor_tid} "
            f"expected={spec.expected_token_id}"
        )

    spec.anchor_channel = channel
    spec.anchor_message_index = message_index
    spec.anchor_full_position = int(anchor_row["full_position"])
    spec.anchor_token_id = int(anchor_tid)

    rels = list(range(-window, window + 1))
    spec.relative_tokens = rels
    spec.valid_mask = []
    spec.generated_local_indices = []
    spec.full_positions = []
    spec.token_ids = []
    spec.tokens = []

    for rel in rels:
        gi = spec.anchor_generated_index + rel
        row = by_generated.get(gi)
        valid = (
            row is not None
            and str(row["channel"]) == channel
            and int(row["message_index"]) == message_index
            and 0 <= gi < len(gen_ids)
        )
        spec.valid_mask.append(bool(valid))
        if valid:
            tid = int(gen_ids[gi])
            spec.generated_local_indices.append(gi)
            spec.full_positions.append(int(row["full_position"]))
            spec.token_ids.append(tid)
            spec.tokens.append(safe_token_text(tok, tid))
        else:
            spec.generated_local_indices.append(-1)
            spec.full_positions.append(-1)
            spec.token_ids.append(-1)
            spec.tokens.append(None)


# -----------------------------------------------------------------------------
# Direction bank
# -----------------------------------------------------------------------------


def infer_direction_layers(directions_dir: Path) -> list[int]:
    layers = []
    for path in directions_dir.glob("L*.npz"):
        m = re.fullmatch(r"L(\d+)\.npz", path.name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def load_direction_bank(
    directions_dir: Path, requested_layers: Sequence[int], sv_k: int
) -> tuple[list[int], int, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Return layers, common_k, V_by_layer, singular_values, available_ranks."""
    layers = list(requested_layers)
    if not layers:
        layers = infer_direction_layers(directions_dir)
    if not layers:
        raise SystemExit(f"No LXX.npz direction files found in {directions_dir}")

    V_raw: dict[int, np.ndarray] = {}
    S_raw: dict[int, np.ndarray] = {}
    available: list[int] = []
    hidden_dims: set[int] = set()

    for layer in layers:
        path = directions_dir / f"L{layer:02d}.npz"
        if not path.exists():
            raise SystemExit(f"Missing direction file {path}")
        with np.load(path) as z:
            if "V" not in z or "S" not in z:
                raise SystemExit(f"{path} must contain V and S arrays")
            V = np.asarray(z["V"], dtype=np.float32)
            S = np.asarray(z["S"], dtype=np.float32).reshape(-1)
        if V.ndim != 2:
            raise SystemExit(f"{path}: expected V[d,k], got {V.shape}")
        k_available = min(V.shape[1], S.shape[0])
        hidden_dims.add(int(V.shape[0]))
        available.append(int(k_available))
        V_raw[layer] = V[:, :k_available]
        S_raw[layer] = S[:k_available]

    if len(hidden_dims) != 1:
        raise SystemExit(f"Direction files disagree on hidden size: {sorted(hidden_dims)}")

    if sv_k == 0:
        common_k = 0
    else:
        common_k = min(int(sv_k), min(available))
        if common_k < sv_k:
            print(
                f"[directions] requested sv-k={sv_k}, but the smallest bank has "
                f"{min(available)} ranks; using common_k={common_k}",
                file=sys.stderr,
            )

    V_by_layer = {layer: V_raw[layer][:, :common_k] for layer in layers}
    S_matrix = np.stack([S_raw[layer][:common_k] for layer in layers], axis=0)
    return (
        layers,
        common_k,
        V_by_layer,
        S_matrix.astype(np.float32),
        np.asarray(available, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Memory-safe selected-position residual recorder
# -----------------------------------------------------------------------------


class HiddenWindowRecorder:
    """Capture selected token positions from selected transformer layers.

    Captured tensors are copied to CPU float32 immediately; full sequence
    residuals are never retained by this recorder.
    """

    def __init__(self, blocks, layers: Sequence[int], positions: Sequence[int]):
        self.blocks = blocks
        self.layers = list(layers)
        self.positions = [int(x) for x in positions]
        self.hidden: dict[int, torch.Tensor] = {}
        self.handles = []

    def _hook(self, layer: int):
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
            for h in self.handles:
                h.remove()
            self.handles = []
            raise
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------


def save_window_npz(
    path: Path,
    spec: AnchorSpec,
    layers: Sequence[int],
    layer_hidden: Mapping[int, torch.Tensor],
    union_pos_to_row: Mapping[int, int],
    V_by_layer: Mapping[int, np.ndarray],
    common_k: int,
    residual_dtype: str,
    compress: bool,
) -> tuple[int, int]:
    T = len(spec.relative_tokens)
    L = len(layers)

    # Infer D from first captured layer.
    first = layer_hidden[layers[0]]
    D = int(first.shape[1])
    disk_dtype = np.float16 if residual_dtype == "float16" else np.float32

    residuals = np.full((L, T, D), np.nan, dtype=disk_dtype)
    sv_activations = np.full((L, T, common_k), np.nan, dtype=np.float32)

    valid_ts = [i for i, ok in enumerate(spec.valid_mask) if ok]
    for li, layer in enumerate(layers):
        h_cpu = layer_hidden[layer]
        if h_cpu.shape[1] != D:
            raise RuntimeError(
                f"Layer {layer} hidden size {h_cpu.shape[1]} != expected {D}"
            )

        # Gather the rows for this specific window from the per-turn union.
        if valid_ts:
            union_rows = [union_pos_to_row[spec.full_positions[ti]] for ti in valid_ts]
            h = h_cpu[union_rows].numpy().astype(np.float32, copy=False)
            residuals[li, valid_ts, :] = h.astype(disk_dtype, copy=False)
            if common_k:
                sv_activations[li, valid_ts, :] = h @ V_by_layer[layer]

    payload = {
        "layers": np.asarray(layers, dtype=np.int16),
        "relative_tokens": np.asarray(spec.relative_tokens, dtype=np.int16),
        "valid_mask": np.asarray(spec.valid_mask, dtype=np.bool_),
        "generated_local_indices": np.asarray(
            spec.generated_local_indices, dtype=np.int32
        ),
        "full_positions": np.asarray(spec.full_positions, dtype=np.int32),
        "token_ids": np.asarray(spec.token_ids, dtype=np.int32),
        "residuals": residuals,
        "sv_activations": sv_activations,
    }
    saver = np.savez_compressed if compress else np.savez
    saver(path, **payload)
    return D, int(np.sum(payload["valid_mask"]))


def window_metadata(
    spec: AnchorSpec,
    npz_path: Path,
    out: Path,
    hidden_size: int,
    valid_tokens: int,
    layers: Sequence[int],
    common_k: int,
    residual_dtype: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": spec.window_id,
        "role": spec.role,
        "event_id": spec.event_id,
        "control_index": spec.control_index,
        "run_id": spec.run_id,
        "environment": spec.environment,
        "condition": spec.condition,
        "step": spec.step,
        "objective_status": spec.objective_status,
        "event_type": spec.event_type,
        "confidence": spec.confidence,
        "anchor_generated_local_index": spec.anchor_generated_index,
        "anchor_full_position": spec.anchor_full_position,
        "anchor_channel": spec.anchor_channel,
        "anchor_message_index": spec.anchor_message_index,
        "anchor_token_id": spec.anchor_token_id,
        "anchor_token": (
            spec.tokens[spec.relative_tokens.index(0)] if 0 in spec.relative_tokens else None
        ),
        "control_match_tier": spec.control_match_tier,
        "control_scope": spec.control_scope,
        "control_lexical_match": spec.control_lexical_match,
        "window": max(abs(min(spec.relative_tokens)), abs(max(spec.relative_tokens))),
        "relative_tokens": spec.relative_tokens,
        "valid_mask": spec.valid_mask,
        "generated_local_indices": spec.generated_local_indices,
        "full_positions": spec.full_positions,
        "token_ids": spec.token_ids,
        "tokens": spec.tokens,
        "valid_token_count": valid_tokens,
        "hidden_size": hidden_size,
        "layers": list(layers),
        "sv_k": common_k,
        "residual_dtype": residual_dtype,
        "npz": str(npz_path.relative_to(out)),
        "anchor_semantics": "post-layer state after anchor token, used downstream for next-token computation",
        "relative_coordinate": "generated_local_index",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)

    campaign = args.campaign.expanduser().resolve()
    events_path = args.events.expanduser().resolve()
    controls_path = args.controls.expanduser().resolve()
    directions_dir = args.directions_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and not args.resume and any(out.iterdir()):
        raise SystemExit(
            f"Output directory {out} is non-empty. Use --resume or --overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)
    windows_dir = out / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    index_path = out / "windows.jsonl"
    failure_path = out / "replay_failures.jsonl"
    excluded_path = out / "excluded_anchors.jsonl"
    if not args.resume:
        index_path.write_text("", encoding="utf-8")
        failure_path.write_text("", encoding="utf-8")
        excluded_path.write_text("", encoding="utf-8")

    manifest_path = campaign / "analysis_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing campaign manifest: {manifest_path}")
    manifest_rows = read_jsonl(manifest_path)
    manifest = {str(r["run_id"]): r for r in manifest_rows}

    specs, initially_excluded, event_by_id = load_anchor_specs(
        events_path, controls_path, args.max_controls_per_event
    )
    for row in initially_excluded:
        append_jsonl(excluded_path, row)

    done_ids: set[str] = set()
    if args.resume:
        done_ids = {
            str(r["window_id"])
            for r in read_jsonl(index_path)
            if r.get("window_id")
            and (out / str(r.get("npz", ""))).exists()
        }
        if done_ids:
            print(f"[resume] {len(done_ids)} completed windows")

    requested_layers = parse_ints(args.layers)
    layers, common_k, V_by_layer, singular_values, available_ranks = load_direction_bank(
        directions_dir, requested_layers, args.sv_k
    )
    print(
        f"[directions] layers={len(layers)} ({layers[0]}..{layers[-1]}) "
        f"common_sv_k={common_k}"
    )

    # Save the scalar direction metadata once. V itself remains in the existing
    # directions/LXX.npz files and is not duplicated into every window.
    np.savez(
        out / "direction_bank.npz",
        layers=np.asarray(layers, dtype=np.int16),
        singular_values=singular_values,
        available_ranks=available_ranks,
    )

    print(f"[load] tokenizer {args.model}")
    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )

    # Resolve every local window BEFORE loading the 20B model. This catches bad
    # paths/indices cheaply and lets us group all requested positions per turn.
    replay_cache: dict[str, dict[int, dict[str, Any]]] = {}
    resolved_specs: list[AnchorSpec] = []
    resolution_failures = 0

    for spec in specs:
        if spec.window_id in done_ids:
            continue
        try:
            man = manifest.get(spec.run_id)
            if man is None:
                raise KeyError(f"run_id absent from analysis_manifest.jsonl: {spec.run_id}")
            if spec.run_id not in replay_cache:
                replay_cache[spec.run_id] = load_replay_map(campaign, man)
            rep = replay_cache[spec.run_id].get(spec.step)
            if rep is None:
                raise KeyError(f"saved replay has no step={spec.step}")
            resolve_window(tok, rep, spec, args.window)
            resolved_specs.append(spec)
        except Exception as exc:
            resolution_failures += 1
            row = {
                "schema_version": SCHEMA_VERSION,
                "stage": "resolve",
                "window_id": spec.window_id,
                "role": spec.role,
                "event_id": spec.event_id,
                "run_id": spec.run_id,
                "step": spec.step,
                "error": repr(exc),
            }
            append_jsonl(failure_path, row)
            print(f"[resolve-fail] {spec.window_id}: {exc}", file=sys.stderr)
            if args.strict:
                raise

    groups: dict[tuple[str, int], list[AnchorSpec]] = defaultdict(list)
    for spec in resolved_specs:
        groups[(spec.run_id, spec.step)].append(spec)

    print(
        f"[data] requested_windows={len(specs)} pending_resolved={len(resolved_specs)} "
        f"turns_to_replay={len(groups)} resolve_failures={resolution_failures}"
    )
    if not groups:
        print("[done] nothing left to collect")
        return

    print(
        f"[load] {args.model} attention={args.attn_implementation} "
        f"transformers={transformers.__version__}"
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

    if max(layers) >= len(model.layers):
        raise SystemExit(
            f"Requested layer {max(layers)} but model exposes only {len(model.layers)} blocks"
        )

    print(
        f"[load] model blocks={len(model.layers)} input_device={model.input_device} "
        f"CUDA_allocated={torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # Validate V hidden dimension against the actual model on first successful turn.
    direction_hidden_size = int(next(iter(V_by_layer.values())).shape[0]) if common_k else None

    replay_failures = 0
    windows_written = 0
    hidden_size_seen: int | None = None

    ordered_groups = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for gi, ((run_id, step), group_specs) in enumerate(ordered_groups, 1):
        rep = replay_cache[run_id][step]
        union_positions = sorted(
            {
                int(pos)
                for spec in group_specs
                for pos, valid in zip(spec.full_positions, spec.valid_mask)
                if valid and pos >= 0
            }
        )
        if not union_positions:
            print(f"[skip] {run_id} step={step}: no valid selected positions")
            continue

        ids = torch.tensor(
            [rep["full_token_ids"]], dtype=torch.long, device=model.input_device
        )
        print(
            f"[replay {gi:03d}/{len(ordered_groups):03d}] {run_id} step={step} "
            f"seq={ids.shape[1]} selected_positions={len(union_positions)} "
            f"windows={len(group_specs)}"
        )

        try:
            with torch.inference_mode(), HiddenWindowRecorder(
                model.layers, layers, union_positions
            ) as recorder:
                model.forward(ids)
        except torch.OutOfMemoryError as exc:
            replay_failures += 1
            row = {
                "schema_version": SCHEMA_VERSION,
                "stage": "forward",
                "run_id": run_id,
                "step": step,
                "seq_len": int(ids.shape[1]),
                "selected_positions": len(union_positions),
                "window_ids": [s.window_id for s in group_specs],
                "error": "CUDA OOM",
                "detail": str(exc),
            }
            append_jsonl(failure_path, row)
            print(f"[OOM-skip] {run_id} step={step}: {exc}", file=sys.stderr)
            del ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.strict:
                raise
            continue
        except Exception as exc:
            replay_failures += 1
            append_jsonl(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "forward",
                    "run_id": run_id,
                    "step": step,
                    "seq_len": int(ids.shape[1]),
                    "selected_positions": len(union_positions),
                    "window_ids": [s.window_id for s in group_specs],
                    "error": repr(exc),
                },
            )
            print(f"[forward-fail] {run_id} step={step}: {exc}", file=sys.stderr)
            del ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.strict:
                raise
            continue

        missing_layers = [layer for layer in layers if layer not in recorder.hidden]
        if missing_layers:
            raise RuntimeError(f"Hooks did not fire for layers: {missing_layers}")

        d_this = int(recorder.hidden[layers[0]].shape[1])
        if hidden_size_seen is None:
            hidden_size_seen = d_this
            if direction_hidden_size is not None and d_this != direction_hidden_size:
                raise RuntimeError(
                    f"Residual hidden size={d_this}, direction V hidden size={direction_hidden_size}"
                )
        elif d_this != hidden_size_seen:
            raise RuntimeError(
                f"Hidden size changed from {hidden_size_seen} to {d_this} on {run_id}/step={step}"
            )

        union_pos_to_row = {pos: i for i, pos in enumerate(union_positions)}

        for spec in group_specs:
            filename = safe_name(spec.window_id) + ".npz"
            path = windows_dir / filename
            hidden_size, valid_tokens = save_window_npz(
                path,
                spec,
                layers,
                recorder.hidden,
                union_pos_to_row,
                V_by_layer,
                common_k,
                args.residual_dtype,
                args.compress,
            )
            meta_row = window_metadata(
                spec,
                path,
                out,
                hidden_size,
                valid_tokens,
                layers,
                common_k,
                args.residual_dtype,
            )
            append_jsonl(index_path, meta_row)
            windows_written += 1

        del recorder, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_index = read_jsonl(index_path)
    role_counts: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    event_ids_done: set[str] = set()
    for r in all_index:
        role_counts[str(r.get("role"))] += 1
        if r.get("objective_status"):
            status_counts[str(r["objective_status"])] += 1
        if r.get("role") == "event":
            event_ids_done.add(str(r.get("event_id")))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "model_revision": args.model_revision,
        "attn_implementation": args.attn_implementation,
        "campaign": str(campaign),
        "events": str(events_path),
        "controls": str(controls_path),
        "directions_dir": str(directions_dir),
        "window": args.window,
        "layers": layers,
        "n_layers": len(layers),
        "sv_k_requested": args.sv_k,
        "sv_k_collected": common_k,
        "available_sv_ranks_by_layer": {
            str(layer): int(n) for layer, n in zip(layers, available_ranks)
        },
        "residual_dtype": args.residual_dtype,
        "compressed": bool(args.compress),
        "hidden_size": hidden_size_seen,
        "window_rows": len(all_index),
        "event_windows": int(role_counts.get("event", 0)),
        "control_windows": int(role_counts.get("control", 0)),
        "completed_event_ids": len(event_ids_done),
        "objective_status_window_counts": dict(sorted(status_counts.items())),
        "resolution_failures_this_invocation": resolution_failures,
        "forward_failures_this_invocation": replay_failures,
        "windows_written_this_invocation": windows_written,
        "definitions": {
            "residual": "output of model.layers[layer] at selected token position",
            "sv_activation": "residual @ V[:, sv_rank-1]",
            "t0": "final generated token overlapping the annotated event/control anchor",
            "window_support": "same Harmony channel and message_index as anchor; missing offsets retained",
        },
        "downstream_intended_analyses": [
            "broad event-locked SV screen across layers and ranks",
            "direct event-minus-control residual transition direction discovery",
            "correct-vs-incorrect subjective-progress replication",
            "held-out-rollout validation before causal steering",
        ],
    }
    write_json(out / "meta.json", meta)

    print("\n=== collection summary ===")
    print(f"windows in index : {len(all_index)}")
    print(f"event windows    : {role_counts.get('event', 0)}")
    print(f"control windows  : {role_counts.get('control', 0)}")
    print(f"event ids        : {len(event_ids_done)}")
    print(f"layers           : {len(layers)}")
    print(f"SVs/layer        : {common_k}")
    print(f"hidden size      : {hidden_size_seen}")
    print(f"forward failures : {replay_failures}")
    print(f"outputs          : {out}")


if __name__ == "__main__":
    main()
