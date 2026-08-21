#!/usr/bin/env python3
"""Analyze W-arm activity around annotated subjective-progress events.

This is the descriptive, event-locked analysis that should precede any causal
steering experiment.  It aligns every mapped annotation at t=0, normalizes
each W-arm within rollout, samples lexically matched controls from the same
rollout, and performs inference on paired event-minus-control effects with the
rollout as the independent unit.

The default inputs are the artifacts produced by
``build_subjective_progress_annotations.py`` and
``extract_task_gaming_candidate_dashboard_data.py``.  The compact
``activations.csv`` is preferred, but ``token_data.jsonl`` is also accepted.

Typical use:

    python3 analyze_subjective_progress_warms.py

Important semantics:

* t=0 is the final generated token overlapping the annotated event.
* Its activation is the residual state after that token, used to predict the
  next token.
* Windows use true generated-token positions and stay inside the anchor's
  assistant message, channel, and step. Missing edges or omitted special-token
  positions are retained as nulls rather than causing event deletion.
* Controls are chosen once per event (independently of arm), always from the
  same rollout and reasoning-message index, outside every event's exclusion
  window.
* Scalar metrics require every requested offset; truncated metrics are marked
  missing rather than averaged over unequal windows.
* Bootstrap and sign-flip inference first averages paired effects within each
  rollout, preventing events or controls from acting as pseudo-replicates.

Outputs include JSON/JSONL and CSV measurements, grouped statistics, ranked
arms, a Markdown report, and dependency-free SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS = ROOT / "task_gaming_v8_subjective_progress" / "events.jsonl"
DEFAULT_ACTIVATIONS = ROOT / "task_gaming_candidate_data" / "activations.csv"
DEFAULT_OUTPUT = ROOT / "subjective_progress_warm_analysis"

SCHEMA_VERSION = "1.0"
PRIMARY_METRICS = ("delta", "peak_response", "trough_response", "anchor")
OBJECTIVE_STATUSES = ("correct", "incorrect", "ambiguous")


@dataclass(frozen=True)
class ActivationRow:
    row_id: int
    run_id: str
    environment: str
    condition: str
    step: int
    channel: str
    message_index: int
    token_index: int
    generated_index: int
    full_position: int
    token_id: int
    token: str
    token_string: str
    activations: Mapping[str, float]


@dataclass(frozen=True)
class ControlAnchor:
    event_id: str
    control_index: int
    row: ActivationRow
    match_tier: str
    scope: str
    lexical_match: str
    coverage_distance: int
    phase_distance: float
    reused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument(
        "--activations",
        type=Path,
        default=DEFAULT_ACTIVATIONS,
        help="Compact activations.csv or full token_data.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--controls-per-event", type=int, default=5)
    parser.add_argument(
        "--min-controls-per-event",
        type=int,
        default=3,
        help="Minimum accepted when the requested number is unavailable.",
    )
    parser.add_argument(
        "--control-exclusion",
        type=int,
        default=None,
        help="Anchor-to-event-span buffer; defaults to --window.",
    )
    parser.add_argument(
        "--normalization",
        choices=("robust_zscore", "zscore", "none"),
        default="robust_zscore",
    )
    parser.add_argument(
        "--normalization-channel",
        default="analysis",
        help=(
            "Channel used to estimate within-rollout centers/scales, or 'all'; "
            "message indices are restricted to those containing aligned events."
        ),
    )
    parser.add_argument(
        "--anchor-channel",
        default="analysis",
        help="Eligible event/control channel. Events on other channels are excluded.",
    )
    parser.add_argument(
        "--match-token-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exact token, fine lexical class, or broad token category.",
    )
    parser.add_argument(
        "--unique-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Avoid reusing control anchors across events in a rollout when possible.",
    )
    parser.add_argument("--pre-start", type=int, default=-5)
    parser.add_argument("--pre-end", type=int, default=-1)
    parser.add_argument("--post-start", type=int, default=0)
    parser.add_argument("--post-end", type=int, default=3)
    parser.add_argument("--baseline-start", type=int, default=-15)
    parser.add_argument("--baseline-end", type=int, default=-5)
    parser.add_argument("--peak-start", type=int, default=-3)
    parser.add_argument("--peak-end", type=int, default=5)
    parser.add_argument(
        "--rank-metric",
        choices=PRIMARY_METRICS,
        default="delta",
        help="Paired metric used to rank arms.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20250821)
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=0,
        help="Number of ranked arms to plot; 0 plots every arm.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--allow-alignment-errors",
        action="store_true",
        help="Exclude, rather than fail on, mapped annotations that do not align.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window < 1:
        raise ValueError("--window must be positive")
    if args.controls_per_event < 1:
        raise ValueError("--controls-per-event must be positive")
    if not 1 <= args.min_controls_per_event <= args.controls_per_event:
        raise ValueError(
            "--min-controls-per-event must be positive and no larger than "
            "--controls-per-event"
        )
    if args.bootstrap_samples < 1 or args.permutations < 1:
        raise ValueError("--bootstrap-samples and --permutations must be positive")
    if args.control_exclusion is not None and args.control_exclusion < 0:
        raise ValueError("--control-exclusion must be non-negative")
    intervals = (
        ("pre", args.pre_start, args.pre_end),
        ("post", args.post_start, args.post_end),
        ("baseline", args.baseline_start, args.baseline_end),
        ("peak", args.peak_start, args.peak_end),
    )
    for name, start, end in intervals:
        if start > end:
            raise ValueError(f"{name} interval starts after it ends: {start}>{end}")
        if start < -args.window or end > args.window:
            raise ValueError(
                f"{name} interval [{start}, {end}] exceeds --window {args.window}"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_activation_csv(path: Path) -> tuple[list[ActivationRow], list[str]]:
    rows: list[ActivationRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        suffix = "_activation"
        arms = sorted(name[: -len(suffix)] for name in reader.fieldnames if name.endswith(suffix))
        if not arms:
            raise ValueError(f"No *_activation columns in {path}")
        required = {
            "row_id",
            "run_id",
            "step",
            "channel",
            "message_index",
            "token_index_in_turn",
            "generated_local_index",
            "full_position",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for record in reader:
            rows.append(
                ActivationRow(
                    row_id=int(record["row_id"]),
                    run_id=record["run_id"],
                    environment=record.get("environment", ""),
                    condition=record.get("condition", ""),
                    step=int(record["step"]),
                    channel=record.get("channel", ""),
                    message_index=int(record["message_index"]),
                    token_index=int(record["token_index_in_turn"]),
                    generated_index=int(record["generated_local_index"]),
                    full_position=int(record["full_position"]),
                    token_id=int(record.get("token_id") or -1),
                    token=record.get("token", ""),
                    token_string=record.get("token_string", ""),
                    activations={
                        arm: finite_float(record.get(f"{arm}_activation")) for arm in arms
                    },
                )
            )
    return rows, arms


def load_activation_jsonl(path: Path) -> tuple[list[ActivationRow], list[str]]:
    # Iterate physical lines.  Do not use splitlines(): decoded token strings can
    # contain Unicode line-separator characters that are not record boundaries.
    rows: list[ActivationRow] = []
    arms: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            candidates = record.get("candidates", {})
            if arms is None:
                arms = sorted(candidates)
            elif sorted(candidates) != arms:
                raise ValueError(f"Inconsistent candidate arms at {path}:{line_number}")
            rows.append(
                ActivationRow(
                    row_id=int(record["row_id"]),
                    run_id=str(record["run_id"]),
                    environment=str(record.get("environment", "")),
                    condition=str(record.get("condition", "")),
                    step=int(record["step"]),
                    channel=str(record.get("channel", "")),
                    message_index=int(record["message_index"]),
                    token_index=int(record["token_index_in_turn"]),
                    generated_index=int(record["generated_local_index"]),
                    full_position=int(record["full_position"]),
                    token_id=int(record.get("token_id", -1)),
                    token=str(record.get("token", "")),
                    token_string=str(record.get("token_string", "")),
                    activations={
                        arm: finite_float(candidates.get(arm, {}).get("activation"))
                        for arm in (arms or [])
                    },
                )
            )
    if not arms:
        raise ValueError(f"No candidate arms in {path}")
    return rows, arms


def load_activations(path: Path) -> tuple[list[ActivationRow], list[str]]:
    if path.suffix.lower() == ".csv":
        return load_activation_csv(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        return load_activation_jsonl(path)
    raise ValueError(f"Unsupported activation table: {path} (expected .csv or .jsonl)")


def broad_token_category(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "newline" if "\n" in text or "\r" in text else "whitespace"
    # A decoded token such as ".\n\n" is primarily a sentence-ending token,
    # not a newline token.  Removing embedded line breaks lets it match the
    # plain punctuation controls needed for the all-punctuation event anchors.
    stripped = stripped.replace("\n", "").replace("\r", "").strip()
    if not stripped:
        return "newline"
    if all(ch.isnumeric() or ch in ".,+-_%" for ch in stripped) and any(
        ch.isnumeric() for ch in stripped
    ):
        return "numeric"
    if any(ch.isalpha() for ch in stripped):
        return "alpha"
    if all(unicodedata.category(ch).startswith("P") for ch in stripped):
        return "punct"
    return "other"


def fine_token_class(text: str) -> str:
    broad = broad_token_category(text)
    stripped = text.strip()
    if broad != "punct":
        return broad
    if any(ch in ".?!" for ch in stripped):
        return "punct_terminal"
    if any(ch in ":;" for ch in stripped):
        return "punct_clause"
    if any(ch in ")]}>" for ch in stripped):
        return "punct_closer"
    if any(ch in "([{<" for ch in stripped):
        return "punct_opener"
    return "punct_other"


def build_indexes(
    rows: Sequence[ActivationRow],
) -> tuple[
    dict[int, ActivationRow],
    dict[tuple[str, int, str, int], dict[int, ActivationRow]],
    dict[str, list[ActivationRow]],
]:
    by_id: dict[int, ActivationRow] = {}
    by_sequence: dict[tuple[str, int, str, int], dict[int, ActivationRow]] = defaultdict(dict)
    by_run: dict[str, list[ActivationRow]] = defaultdict(list)
    for row in rows:
        if row.row_id in by_id:
            raise ValueError(f"Duplicate activation row_id {row.row_id}")
        key = (row.run_id, row.step, row.channel, row.message_index)
        if row.generated_index in by_sequence[key]:
            raise ValueError(
                f"Duplicate generated position in sequence {key}: {row.generated_index}"
            )
        by_id[row.row_id] = row
        by_sequence[key][row.generated_index] = row
        by_run[row.run_id].append(row)
    for run_rows in by_run.values():
        run_rows.sort(key=lambda row: (row.step, row.generated_index, row.row_id))
    return by_id, dict(by_sequence), dict(by_run)


def align_events(
    events: Sequence[dict[str, Any]],
    by_id: Mapping[int, ActivationRow],
    anchor_channel: str,
    allow_errors: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_event_ids: set[str] = set()
    for event in events:
        event_id = str(event.get("event_id", ""))
        if not event_id:
            message = "annotation without a nonempty event_id"
            if not allow_errors:
                errors.append(message)
            excluded.append({**event, "analysis_exclusion_reason": message})
            continue
        if event_id in seen_event_ids:
            message = f"duplicate event_id {event_id!r}"
            if not allow_errors:
                errors.append(message)
            excluded.append({**event, "analysis_exclusion_reason": message})
            continue
        seen_event_ids.add(event_id)
        if not event.get("w_arm_alignment_available", False):
            excluded.append({**event, "analysis_exclusion_reason": "w_arm_alignment_unavailable"})
            continue
        row_id = event.get("w_arm_anchor_row_id")
        row = by_id.get(int(row_id)) if row_id is not None else None
        span_start_row_id = event.get("w_arm_row_start")
        span_end_row_id = event.get("w_arm_row_end")
        span_start_row = (
            by_id.get(int(span_start_row_id)) if span_start_row_id is not None else None
        )
        span_end_row = (
            by_id.get(int(span_end_row_id)) if span_end_row_id is not None else None
        )
        problem = None
        if row is None:
            problem = f"anchor row_id {row_id!r} not present"
        elif row.run_id != event.get("run_id"):
            problem = f"run mismatch ({row.run_id!r} vs {event.get('run_id')!r})"
        elif row.step != int(event.get("step", -1)):
            problem = f"step mismatch ({row.step} vs {event.get('step')})"
        elif row.token_index != int(event.get("w_arm_anchor_token_index", -1)):
            problem = (
                "token-index mismatch "
                f"({row.token_index} vs {event.get('w_arm_anchor_token_index')})"
            )
        elif row.generated_index != int(event.get("anchor_generated_position", -1)):
            problem = (
                "generated-position mismatch "
                f"({row.generated_index} vs {event.get('anchor_generated_position')})"
            )
        elif row.full_position != int(event.get("anchor_absolute_position", -1)):
            problem = (
                "absolute-position mismatch "
                f"({row.full_position} vs {event.get('anchor_absolute_position')})"
            )
        elif row.channel != anchor_channel:
            problem = f"anchor channel {row.channel!r}, expected {anchor_channel!r}"
        elif int(event.get("w_arm_token_index_start", 0)) > int(
            event.get("w_arm_token_index_end", -1)
        ):
            problem = "event token span starts after it ends"
        elif int(event.get("w_arm_anchor_token_index", -1)) != int(
            event.get("w_arm_token_index_end", -2)
        ):
            problem = "anchor is not the final token of the annotated event span"
        elif span_start_row is None or span_end_row is None:
            problem = "one or both event-span W-arm rows are absent"
        elif (
            span_start_row.run_id,
            span_start_row.step,
            span_start_row.channel,
            span_start_row.message_index,
        ) != (row.run_id, row.step, row.channel, row.message_index) or (
            span_end_row.run_id,
            span_end_row.step,
            span_end_row.channel,
            span_end_row.message_index,
        ) != (row.run_id, row.step, row.channel, row.message_index):
            problem = "event-span rows do not belong to the anchor message"
        elif span_start_row.generated_index != int(event.get("generated_token_start", -1)):
            problem = "event-span start does not match generated_token_start"
        elif span_end_row.generated_index != int(event.get("generated_token_end", -1)):
            problem = "event-span end does not match generated_token_end"
        elif event.get("objective_status") not in OBJECTIVE_STATUSES:
            problem = f"unknown objective_status {event.get('objective_status')!r}"
        if problem:
            message = f"{event_id}: {problem}"
            if not allow_errors:
                errors.append(message)
            excluded.append({**event, "analysis_exclusion_reason": problem})
            continue
        enriched = dict(event)
        enriched["_anchor_row"] = row
        aligned.append(enriched)
    if errors:
        raise ValueError("Annotation/W-arm alignment failed:\n  " + "\n  ".join(errors))
    aligned.sort(
        key=lambda event: (
            event["run_id"],
            int(event["step"]),
            int(event["w_arm_anchor_token_index"]),
            event["event_id"],
        )
    )
    return aligned, excluded


def normalization_stats(
    by_run: Mapping[str, Sequence[ActivationRow]],
    arms: Sequence[str],
    method: str,
    channel: str,
    message_indices: frozenset[int],
) -> tuple[dict[tuple[str, str], tuple[float, float]], list[dict[str, Any]]]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    records: list[dict[str, Any]] = []
    for run_id in sorted(by_run):
        eligible = [
            row
            for row in by_run[run_id]
            if (channel == "all" or row.channel == channel)
            and row.message_index in message_indices
        ]
        if not eligible:
            # Rollouts without the event-relevant message role never enter the
            # event analysis, so they do not need a normalizer record.
            continue
        for arm in arms:
            values = np.asarray(
                [row.activations[arm] for row in eligible if math.isfinite(row.activations[arm])],
                dtype=float,
            )
            if values.size == 0:
                raise ValueError(f"No finite values for {run_id}/{arm}")
            fallback = "none"
            if method == "none":
                center, scale = 0.0, 1.0
            elif method == "zscore":
                center = float(np.mean(values))
                scale = float(np.std(values, ddof=0))
            else:
                center = float(np.median(values))
                scale = float(1.4826 * np.median(np.abs(values - center)))
                if not math.isfinite(scale) or scale <= 1e-12:
                    scale = float(np.std(values, ddof=0))
                    fallback = "standard_deviation"
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
                fallback = "unit_scale"
            stats[(run_id, arm)] = (center, scale)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "arm": arm,
                    "method": method,
                    "normalization_channel": channel,
                    "normalization_message_indices": sorted(message_indices),
                    "center": center,
                    "scale": scale,
                    "fallback": fallback,
                    "n_tokens": int(values.size),
                }
            )
    return stats, records


def available_offsets(
    sequence: Mapping[int, ActivationRow], anchor_generated_index: int, window: int
) -> frozenset[int]:
    return frozenset(
        t
        for t in range(-window, window + 1)
        if anchor_generated_index + t in sequence
    )


def event_exclusion_intervals(
    events: Sequence[dict[str, Any]], buffer_tokens: int
) -> dict[tuple[str, int, str, int], list[tuple[int, int]]]:
    intervals: dict[tuple[str, int, str, int], list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        row: ActivationRow = event["_anchor_row"]
        start = int(event["generated_token_start"]) - buffer_tokens
        end = int(event["generated_token_end"]) + buffer_tokens
        intervals[(row.run_id, row.step, row.channel, row.message_index)].append(
            (start, end)
        )
    return dict(intervals)


def is_excluded_control(
    row: ActivationRow,
    intervals: Mapping[tuple[str, int, str, int], Sequence[tuple[int, int]]],
) -> bool:
    return any(
        start <= row.generated_index <= end
        for start, end in intervals.get(
            (row.run_id, row.step, row.channel, row.message_index), ()
        )
    )


def coverage_signature(
    sequence: Mapping[int, ActivationRow], anchor_generated_index: int, window: int
) -> tuple[int, int]:
    offsets = available_offsets(sequence, anchor_generated_index, window)
    left = sum(t in offsets for t in range(-window, 0))
    right = sum(t in offsets for t in range(1, window + 1))
    return left, right


def phase(sequence: Mapping[int, ActivationRow], generated_index: int) -> float:
    indices = sequence.keys()
    low, high = min(indices), max(indices)
    return 0.0 if high == low else (generated_index - low) / (high - low)


def candidate_match_tier(
    event_row: ActivationRow, candidate: ActivationRow, match_token_class: bool
) -> tuple[int, str, str, str] | None:
    scope = "same_step" if candidate.step == event_row.step else "same_rollout"
    scope_rank = 0 if scope == "same_step" else 1
    if not match_token_class:
        return scope_rank * 3 + 2, f"{scope}_unmatched", scope, "unmatched"
    if candidate.token_id == event_row.token_id:
        return scope_rank * 3, f"{scope}_exact_token", scope, "exact_token"
    if fine_token_class(candidate.token) == fine_token_class(event_row.token):
        return scope_rank * 3 + 1, f"{scope}_fine_class", scope, "fine_class"
    if broad_token_category(candidate.token) == broad_token_category(event_row.token):
        return scope_rank * 3 + 2, f"{scope}_broad_category", scope, "broad_category"
    return None


def sample_controls(
    events: Sequence[dict[str, Any]],
    by_run: Mapping[str, Sequence[ActivationRow]],
    by_sequence: Mapping[tuple[str, int, str, int], Mapping[int, ActivationRow]],
    window: int,
    controls_per_event: int,
    min_controls_per_event: int,
    intervals: Mapping[tuple[str, int, str, int], Sequence[tuple[int, int]]],
    anchor_channel: str,
    match_token_class: bool,
    unique_controls: bool,
    seed: int,
) -> tuple[dict[str, list[ControlAnchor]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    controls_by_event: dict[str, list[ControlAnchor]] = {}
    records: list[dict[str, Any]] = []
    used_by_run: dict[str, set[int]] = defaultdict(set)

    for event in events:
        event_id = event["event_id"]
        event_row: ActivationRow = event["_anchor_row"]
        event_key = (
            event_row.run_id,
            event_row.step,
            event_row.channel,
            event_row.message_index,
        )
        event_sequence = by_sequence[event_key]
        event_offsets = available_offsets(event_sequence, event_row.generated_index, window)
        event_coverage = coverage_signature(event_sequence, event_row.generated_index, window)
        event_phase = phase(event_sequence, event_row.generated_index)

        candidates: list[tuple[tuple[Any, ...], ActivationRow, str, str, str, int, float]] = []
        for candidate in by_run[event_row.run_id]:
            if (
                candidate.channel != anchor_channel
                or candidate.message_index != event_row.message_index
                or candidate.row_id == event_row.row_id
            ):
                continue
            if is_excluded_control(candidate, intervals):
                continue
            candidate_key = (
                candidate.run_id,
                candidate.step,
                candidate.channel,
                candidate.message_index,
            )
            candidate_sequence = by_sequence[candidate_key]
            candidate_offsets = available_offsets(
                candidate_sequence, candidate.generated_index, window
            )
            if not event_offsets.issubset(candidate_offsets):
                continue
            tier = candidate_match_tier(event_row, candidate, match_token_class)
            if tier is None:
                continue
            tier_rank, tier_name, scope, lexical_match = tier
            candidate_coverage = coverage_signature(
                candidate_sequence, candidate.generated_index, window
            )
            coverage_distance = abs(event_coverage[0] - candidate_coverage[0]) + abs(
                event_coverage[1] - candidate_coverage[1]
            )
            phase_distance = abs(
                event_phase - phase(candidate_sequence, candidate.generated_index)
            )
            random_tie_breaker = float(rng.random())
            # Message-boundary geometry comes first: all event anchors are at
            # punctuation, so a mismatched distance to the end of reasoning is
            # a more dangerous confound than step identity.
            score = (coverage_distance, tier_rank, phase_distance, random_tie_breaker)
            candidates.append(
                (
                    score,
                    candidate,
                    tier_name,
                    scope,
                    lexical_match,
                    coverage_distance,
                    phase_distance,
                )
            )
        candidates.sort(key=lambda item: item[0])

        selected: list[ControlAnchor] = []
        for allow_reuse in ((False, True) if unique_controls else (True,)):
            for (
                _score,
                candidate,
                tier_name,
                scope,
                lexical_match,
                coverage_distance,
                phase_distance,
            ) in candidates:
                already_used = candidate.row_id in used_by_run[event_row.run_id]
                if not allow_reuse and already_used:
                    continue
                if any(control.row.row_id == candidate.row_id for control in selected):
                    continue
                selected.append(
                    ControlAnchor(
                        event_id=event_id,
                        control_index=len(selected) + 1,
                        row=candidate,
                        match_tier=tier_name,
                        scope=scope,
                        lexical_match=lexical_match,
                        coverage_distance=coverage_distance,
                        phase_distance=phase_distance,
                        reused=already_used,
                    )
                )
                if len(selected) == controls_per_event:
                    break
            if len(selected) == controls_per_event:
                break
        if len(selected) < min_controls_per_event:
            raise ValueError(
                f"Only {len(selected)} eligible same-rollout lexical controls for "
                f"{event_id}; minimum is {min_controls_per_event} (target "
                f"{controls_per_event}). Reduce --window or the control counts, "
                "or use --no-match-token-class."
            )
        controls_by_event[event_id] = selected
        for control in selected:
            used_by_run[event_row.run_id].add(control.row.row_id)
            record = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "event_anchor_row_id": event_row.row_id,
                "event_anchor_message_index": event_row.message_index,
                "event_anchor_generated_local_index": event_row.generated_index,
                "event_anchor_token_id": event_row.token_id,
                "event_anchor_token": event_row.token,
                "control_index": control.control_index,
                "run_id": control.row.run_id,
                "environment": control.row.environment,
                "condition": control.row.condition,
                "step": control.row.step,
                "channel": control.row.channel,
                "message_index": control.row.message_index,
                "row_id": control.row.row_id,
                "token_index_in_turn": control.row.token_index,
                "generated_local_index": control.row.generated_index,
                "full_position": control.row.full_position,
                "token_id": control.row.token_id,
                "token": control.row.token,
                "token_string": control.row.token_string,
                "broad_token_category": broad_token_category(control.row.token),
                "fine_token_class": fine_token_class(control.row.token),
                "match_tier": control.match_tier,
                "scope": control.scope,
                "lexical_match": control.lexical_match,
                "coverage_distance": control.coverage_distance,
                "phase_distance": control.phase_distance,
                "reused_across_events": control.reused,
            }
            records.append(record)
    return controls_by_event, records


def extract_window(
    anchor: ActivationRow,
    by_sequence: Mapping[tuple[str, int, str, int], Mapping[int, ActivationRow]],
    arms: Sequence[str],
    norm_stats: Mapping[tuple[str, str], tuple[float, float]],
    window: int,
) -> dict[str, Any]:
    sequence = by_sequence[
        (anchor.run_id, anchor.step, anchor.channel, anchor.message_index)
    ]
    rel = list(range(-window, window + 1))
    rows = [sequence.get(anchor.generated_index + t) for t in rel]
    raw: dict[str, np.ndarray] = {}
    normalized: dict[str, np.ndarray] = {}
    for arm in arms:
        values = np.asarray(
            [row.activations[arm] if row is not None else math.nan for row in rows],
            dtype=float,
        )
        center, scale = norm_stats[(anchor.run_id, arm)]
        raw[arm] = values
        normalized[arm] = (values - center) / scale
    return {
        "anchor": anchor,
        "relative_positions": rel,
        "available": np.asarray([row is not None for row in rows], dtype=bool),
        "row_ids": [row.row_id if row is not None else None for row in rows],
        "channels": [row.channel if row is not None else None for row in rows],
        "message_indices": [
            row.message_index if row is not None else None for row in rows
        ],
        "token_indices_in_turn": [
            row.token_index if row is not None else None for row in rows
        ],
        "generated_local_indices": [
            row.generated_index if row is not None else None for row in rows
        ],
        "full_positions": [
            row.full_position if row is not None else None for row in rows
        ],
        "token_ids": [row.token_id if row is not None else None for row in rows],
        "tokens": [row.token if row is not None else None for row in rows],
        "token_strings": [row.token_string if row is not None else None for row in rows],
        "raw": raw,
        "normalized": normalized,
    }


def complete_values(trace: np.ndarray, offsets: range, window: int) -> np.ndarray | None:
    values = np.asarray([trace[t + window] for t in offsets], dtype=float)
    return values if values.size and np.all(np.isfinite(values)) else None


def trace_metrics(trace: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    pre_values = complete_values(trace, range(args.pre_start, args.pre_end + 1), args.window)
    post_values = complete_values(trace, range(args.post_start, args.post_end + 1), args.window)
    baseline_values = complete_values(
        trace, range(args.baseline_start, args.baseline_end + 1), args.window
    )
    peak_values = complete_values(trace, range(args.peak_start, args.peak_end + 1), args.window)
    pre = float(np.mean(pre_values)) if pre_values is not None else math.nan
    post = float(np.mean(post_values)) if post_values is not None else math.nan
    baseline = float(np.mean(baseline_values)) if baseline_values is not None else math.nan
    peak = float(np.max(peak_values)) if peak_values is not None else math.nan
    trough = float(np.min(peak_values)) if peak_values is not None else math.nan
    anchor = float(trace[args.window]) if math.isfinite(float(trace[args.window])) else math.nan
    return {
        "pre": pre,
        "post": post,
        "delta": post - pre if math.isfinite(pre) and math.isfinite(post) else math.nan,
        "baseline": baseline,
        "peak": peak,
        "peak_response": peak - baseline
        if math.isfinite(peak) and math.isfinite(baseline)
        else math.nan,
        "trough": trough,
        "trough_response": trough - baseline
        if math.isfinite(trough) and math.isfinite(baseline)
        else math.nan,
        "anchor": anchor,
    }


def mean_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def sd_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan


def json_array(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_safe(dict(row)), ensure_ascii=False, allow_nan=False) + "\n"
            )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(row.get(key), float) and not math.isfinite(row[key])
                    else row.get(key, "")
                    for key in fieldnames
                }
            )


def build_windows_and_measurements(
    events: Sequence[dict[str, Any]],
    controls_by_event: Mapping[str, Sequence[ControlAnchor]],
    by_sequence: Mapping[tuple[str, int, str, int], Mapping[int, ActivationRow]],
    arms: Sequence[str],
    norm_stats: Mapping[tuple[str, str], tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    window_records: list[dict[str, Any]] = []
    event_measurements: list[dict[str, Any]] = []
    control_measurements: list[dict[str, Any]] = []
    event_data: dict[str, dict[str, Any]] = {}

    for event in events:
        event_id = event["event_id"]
        anchor: ActivationRow = event["_anchor_row"]
        event_window = extract_window(anchor, by_sequence, arms, norm_stats, args.window)
        control_windows = [
            extract_window(control.row, by_sequence, arms, norm_stats, args.window)
            for control in controls_by_event[event_id]
        ]
        event_data[event_id] = {
            "event": event,
            "event_window": event_window,
            "controls": list(zip(controls_by_event[event_id], control_windows)),
        }

        compact_event = {
            key: value
            for key, value in event.items()
            if key
            in {
                "event_id",
                "run_id",
                "environment",
                "condition",
                "step",
                "event_type",
                "objective_status",
                "confidence",
                "before",
                "event",
                "after",
                "subjective_progress",
                "objective_status_reason",
                "w_arm_token_index_start",
                "w_arm_token_index_end",
                "w_arm_anchor_token_index",
                "w_arm_anchor_row_id",
                "generated_token_start",
                "generated_token_end",
                "anchor_generated_position",
                "anchor_absolute_position",
            }
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            **compact_event,
            "anchor_policy": "final_token_overlapping_event",
            "anchor_state_semantics": "state_after_anchor_token_used_to_predict_next_token",
            "relative_coordinate": "generated_local_index",
            "relative_positions": event_window["relative_positions"],
            "event_span_relative_start": int(event["generated_token_start"])
            - anchor.generated_index,
            "event_span_relative_end": int(event["generated_token_end"])
            - anchor.generated_index,
            "event_window": {
                "row_ids": event_window["row_ids"],
                "channels": event_window["channels"],
                "message_indices": event_window["message_indices"],
                "token_indices_in_turn": event_window["token_indices_in_turn"],
                "generated_local_indices": event_window["generated_local_indices"],
                "full_positions": event_window["full_positions"],
                "token_ids": event_window["token_ids"],
                "tokens": event_window["tokens"],
                "token_strings": event_window["token_strings"],
                "raw_activations": {
                    arm: json_array(event_window["raw"][arm]) for arm in arms
                },
                "normalized_activations": {
                    arm: json_array(event_window["normalized"][arm]) for arm in arms
                },
            },
            "control_windows": [],
        }
        for control, control_window in zip(controls_by_event[event_id], control_windows):
            record["control_windows"].append(
                {
                    "control_index": control.control_index,
                    "row_id": control.row.row_id,
                    "step": control.row.step,
                    "channel": control.row.channel,
                    "message_index": control.row.message_index,
                    "token_index_in_turn": control.row.token_index,
                    "generated_local_index": control.row.generated_index,
                    "full_position": control.row.full_position,
                    "token_id": control.row.token_id,
                    "token": control.row.token,
                    "token_string": control.row.token_string,
                    "match_tier": control.match_tier,
                    "scope": control.scope,
                    "lexical_match": control.lexical_match,
                    "coverage_distance": control.coverage_distance,
                    "phase_distance": control.phase_distance,
                    "reused_across_events": control.reused,
                    "row_ids": control_window["row_ids"],
                    "channels": control_window["channels"],
                    "message_indices": control_window["message_indices"],
                    "token_indices_in_turn": control_window[
                        "token_indices_in_turn"
                    ],
                    "generated_local_indices": control_window[
                        "generated_local_indices"
                    ],
                    "full_positions": control_window["full_positions"],
                    "token_ids": control_window["token_ids"],
                    "tokens": control_window["tokens"],
                    "token_strings": control_window["token_strings"],
                    "raw_activations": {
                        arm: json_array(control_window["raw"][arm]) for arm in arms
                    },
                    "normalized_activations": {
                        arm: json_array(control_window["normalized"][arm]) for arm in arms
                    },
                }
            )
        window_records.append(record)

        for arm in arms:
            event_norm = trace_metrics(event_window["normalized"][arm], args)
            event_raw = trace_metrics(event_window["raw"][arm], args)
            control_norm_metrics = [
                trace_metrics(control_window["normalized"][arm], args)
                for control_window in control_windows
            ]
            control_raw_metrics = [
                trace_metrics(control_window["raw"][arm], args)
                for control_window in control_windows
            ]
            row: dict[str, Any] = {
                "event_id": event_id,
                "run_id": event["run_id"],
                "environment": event["environment"],
                "condition": event["condition"],
                "step": event["step"],
                "event_type": event["event_type"],
                "objective_status": event["objective_status"],
                "confidence": event["confidence"],
                "arm": arm,
                "anchor_row_id": anchor.row_id,
                "anchor_token_index": anchor.token_index,
                "anchor_message_index": anchor.message_index,
                "anchor_generated_position": anchor.generated_index,
                "anchor_full_position": anchor.full_position,
                "anchor_token_id": anchor.token_id,
                "anchor_token": anchor.token,
                "anchor_broad_category": broad_token_category(anchor.token),
                "anchor_fine_class": fine_token_class(anchor.token),
                "event_window_tokens": int(np.sum(event_window["available"])),
                "full_event_window": bool(np.all(event_window["available"])),
                "n_controls": len(control_windows),
                "same_step_controls": sum(
                    control.scope == "same_step" for control in controls_by_event[event_id]
                ),
                "exact_token_controls": sum(
                    control.lexical_match == "exact_token"
                    for control in controls_by_event[event_id]
                ),
            }
            for metric, value in event_norm.items():
                row[f"event_{metric}"] = value
                control_mean = mean_finite(item[metric] for item in control_norm_metrics)
                row[f"control_{metric}_mean"] = control_mean
                row[f"control_{metric}_sd"] = sd_finite(
                    item[metric] for item in control_norm_metrics
                )
                row[f"{metric}_contrast"] = (
                    value - control_mean
                    if math.isfinite(value) and math.isfinite(control_mean)
                    else math.nan
                )
            for metric, value in event_raw.items():
                row[f"raw_event_{metric}"] = value
                control_mean = mean_finite(item[metric] for item in control_raw_metrics)
                row[f"raw_control_{metric}_mean"] = control_mean
                row[f"raw_{metric}_contrast"] = (
                    value - control_mean
                    if math.isfinite(value) and math.isfinite(control_mean)
                    else math.nan
                )
            event_measurements.append(row)

            for control, norm_metrics, raw_metrics, control_window in zip(
                controls_by_event[event_id],
                control_norm_metrics,
                control_raw_metrics,
                control_windows,
            ):
                control_row: dict[str, Any] = {
                    "event_id": event_id,
                    "run_id": event["run_id"],
                    "environment": event["environment"],
                    "condition": event["condition"],
                    "objective_status": event["objective_status"],
                    "arm": arm,
                    "control_index": control.control_index,
                    "control_row_id": control.row.row_id,
                    "control_step": control.row.step,
                    "control_message_index": control.row.message_index,
                    "control_token_index": control.row.token_index,
                    "control_generated_position": control.row.generated_index,
                    "control_full_position": control.row.full_position,
                    "control_token_id": control.row.token_id,
                    "control_token": control.row.token,
                    "match_tier": control.match_tier,
                    "scope": control.scope,
                    "lexical_match": control.lexical_match,
                    "coverage_distance": control.coverage_distance,
                    "phase_distance": control.phase_distance,
                    "reused_across_events": control.reused,
                    "control_window_tokens": int(np.sum(control_window["available"])),
                    "full_control_window": bool(np.all(control_window["available"])),
                }
                for metric, value in norm_metrics.items():
                    control_row[metric] = value
                for metric, value in raw_metrics.items():
                    control_row[f"raw_{metric}"] = value
                control_measurements.append(control_row)
    return window_records, event_measurements, control_measurements, event_data


def nanmean_axis0(matrix: np.ndarray) -> np.ndarray:
    """Column means without warnings for columns that are entirely missing."""
    finite = np.isfinite(matrix)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, matrix, 0.0), axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def summarize_trace_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(matrix)
    counts = np.sum(finite, axis=0).astype(float)
    mean = nanmean_axis0(matrix)
    centered = np.where(finite, matrix - mean, 0.0)
    sum_squares = np.sum(centered * centered, axis=0)
    sd = np.full(matrix.shape[1], np.nan, dtype=float)
    np.divide(sum_squares, counts - 1, out=sd, where=counts > 1)
    sd = np.sqrt(sd)
    sem = np.full(matrix.shape[1], np.nan, dtype=float)
    np.divide(sd, np.sqrt(counts), out=sem, where=counts > 1)
    return mean, sem, counts


def trace_aggregates(
    event_data: Mapping[str, Mapping[str, Any]],
    arm: str,
    kind: str,
    objective_status: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    event_traces: list[tuple[str, np.ndarray]] = []
    by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in event_data.values():
        event = item["event"]
        if objective_status != "all" and event["objective_status"] != objective_status:
            continue
        if kind == "event":
            trace = item["event_window"]["normalized"][arm]
        else:
            control_traces = np.stack(
                [window["normalized"][arm] for _control, window in item["controls"]]
            )
            trace = nanmean_axis0(control_traces)
            # Controls can have a superset of an edge-truncated event's
            # offsets. Mask to that event so event and control curves use the
            # identical event/rollout support at every t.
            event_trace = item["event_window"]["normalized"][arm]
            trace = np.where(np.isfinite(event_trace), trace, np.nan)
        event_traces.append((event["run_id"], trace))
        by_run[event["run_id"]].append(trace)
    if not by_run:
        return {}
    event_matrix = np.stack([trace for _run_id, trace in event_traces])
    rollout_matrix = np.stack(
        [nanmean_axis0(np.stack(by_run[run_id])) for run_id in sorted(by_run)]
    )
    event_mean, event_sem, event_counts = summarize_trace_matrix(event_matrix)
    rollout_mean, rollout_sem, rollout_counts = summarize_trace_matrix(rollout_matrix)
    # Both estimands report both denominators so edge-related changes in sample
    # composition remain visible at every relative position.
    return {
        "event_weighted": (
            event_mean,
            event_sem,
            event_counts,
            event_counts,
            rollout_counts,
        ),
        "rollout_weighted": (
            rollout_mean,
            rollout_sem,
            rollout_counts,
            event_counts,
            rollout_counts,
        ),
    }


def build_trace_summaries(
    event_data: Mapping[str, Mapping[str, Any]], arms: Sequence[str], window: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for kind in ("event", "matched_control_event_mean"):
            for status in ("all", *OBJECTIVE_STATUSES):
                aggregates = trace_aggregates(event_data, arm, kind, status)
                if not aggregates:
                    continue
                for aggregation, (
                    mean,
                    sem,
                    n_units,
                    n_events,
                    n_rollouts,
                ) in aggregates.items():
                    for index, t in enumerate(range(-window, window + 1)):
                        rows.append(
                            {
                                "arm": arm,
                                "kind": kind,
                                "objective_status": status,
                                "aggregation": aggregation,
                                "relative_token": t,
                                "relative_coordinate": "generated_local_index",
                                "mean": float(mean[index]),
                                "sem": float(sem[index]),
                                "sem_between_events": float(sem[index])
                                if aggregation == "event_weighted"
                                else math.nan,
                                "sem_between_rollouts": float(sem[index])
                                if aggregation == "rollout_weighted"
                                else math.nan,
                                "n_units": int(n_units[index]),
                                "n_rollouts": int(n_rollouts[index]),
                                "n_events": int(n_events[index]),
                            }
                        )
    return rows


def grouped_effect_values(
    measurements: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
    status: str,
) -> tuple[list[float], dict[str, list[float]]]:
    values: list[float] = []
    by_run: dict[str, list[float]] = defaultdict(list)
    field = f"{metric}_contrast"
    for row in measurements:
        if row["arm"] != arm:
            continue
        if status != "all" and row["objective_status"] != status:
            continue
        value = float(row[field])
        if not math.isfinite(value):
            continue
        values.append(value)
        by_run[str(row["run_id"])].append(value)
    return values, dict(by_run)


def grouped_component_means(
    measurements: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
    status: str,
) -> dict[str, float]:
    event_values: list[float] = []
    control_values: list[float] = []
    event_by_run: dict[str, list[float]] = defaultdict(list)
    control_by_run: dict[str, list[float]] = defaultdict(list)
    event_field = f"event_{metric}"
    control_field = f"control_{metric}_mean"
    for row in measurements:
        if row["arm"] != arm:
            continue
        if status != "all" and row["objective_status"] != status:
            continue
        event_value = float(row[event_field])
        control_value = float(row[control_field])
        # Keep the component means on the exact paired sample used for D_i.
        if not (math.isfinite(event_value) and math.isfinite(control_value)):
            continue
        run_id = str(row["run_id"])
        event_values.append(event_value)
        control_values.append(control_value)
        event_by_run[run_id].append(event_value)
        control_by_run[run_id].append(control_value)
    if not event_values:
        return {
            "event_metric_event_weighted_mean": math.nan,
            "control_metric_event_weighted_mean": math.nan,
            "event_metric_rollout_weighted_mean": math.nan,
            "control_metric_rollout_weighted_mean": math.nan,
        }
    return {
        "event_metric_event_weighted_mean": float(np.mean(event_values)),
        "control_metric_event_weighted_mean": float(np.mean(control_values)),
        "event_metric_rollout_weighted_mean": float(
            np.mean([np.mean(event_by_run[run_id]) for run_id in sorted(event_by_run)])
        ),
        "control_metric_rollout_weighted_mean": float(
            np.mean([np.mean(control_by_run[run_id]) for run_id in sorted(control_by_run)])
        ),
    }


def grouped_inference(
    values: Sequence[float],
    by_run: Mapping[str, Sequence[float]],
    bootstrap_samples: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rollout_values = np.asarray(
        [np.mean(by_run[run_id]) for run_id in sorted(by_run)], dtype=float
    )
    n_rollouts = int(rollout_values.size)
    if n_rollouts == 0:
        return {
            "n_events": 0,
            "n_rollouts": 0,
            "event_weighted_mean": math.nan,
            "rollout_weighted_mean": math.nan,
            "rollout_sd": math.nan,
            "rollout_se": math.nan,
            "standardized_rollout_effect": math.nan,
            "bootstrap_ci_low": math.nan,
            "bootstrap_ci_high": math.nan,
            "permutation_p_two_sided": math.nan,
            "permutation_p_directional": math.nan,
            "event_consistency": math.nan,
            "rollout_consistency": math.nan,
        }
    observed = float(np.mean(rollout_values))
    rollout_sd = float(np.std(rollout_values, ddof=1)) if n_rollouts > 1 else math.nan
    rollout_se = rollout_sd / math.sqrt(n_rollouts) if n_rollouts > 1 else math.nan
    standardized = observed / rollout_sd if math.isfinite(rollout_sd) and rollout_sd > 0 else math.nan
    bootstrap_indices = rng.integers(0, n_rollouts, size=(bootstrap_samples, n_rollouts))
    bootstrap_means = np.mean(rollout_values[bootstrap_indices], axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, n_rollouts))
    null = np.mean(signs * rollout_values, axis=1)
    p_two = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1)
    direction = 1.0 if observed >= 0 else -1.0
    p_directional = (1 + int(np.sum(direction * null >= abs(observed)))) / (
        permutations + 1
    )
    finite_values = np.asarray(values, dtype=float)
    return {
        "n_events": len(values),
        "n_rollouts": n_rollouts,
        "event_weighted_mean": float(np.mean(finite_values)),
        "rollout_weighted_mean": observed,
        "rollout_sd": rollout_sd,
        "rollout_se": rollout_se,
        "standardized_rollout_effect": standardized,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "permutation_p_two_sided": p_two,
        "permutation_p_directional": p_directional,
        "event_consistency": float(np.mean(direction * finite_values > 0)),
        "rollout_consistency": float(np.mean(direction * rollout_values > 0)),
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("comparison") == "event_minus_control":
            grouped[(row["metric"], row["objective_status"])].append(row)
    for group in grouped.values():
        finite = [row for row in group if math.isfinite(row["permutation_p_two_sided"])]
        finite.sort(key=lambda row: row["permutation_p_two_sided"])
        running = 0.0
        total = len(finite)
        for index, row in enumerate(finite):
            adjusted = min(1.0, (total - index) * row["permutation_p_two_sided"])
            running = max(running, adjusted)
            row["permutation_p_holm"] = running
        for row in group:
            row.setdefault("permutation_p_holm", math.nan)


def build_grouped_statistics(
    measurements: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed + 10_000)
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        for status in ("all", *OBJECTIVE_STATUSES):
            for arm in arms:
                values, by_run = grouped_effect_values(measurements, arm, metric, status)
                inference = grouped_inference(
                    values,
                    by_run,
                    args.bootstrap_samples,
                    args.permutations,
                    rng,
                )
                rows.append(
                    {
                        "comparison": "event_minus_control",
                        "metric": metric,
                        "objective_status": status,
                        "arm": arm,
                        **grouped_component_means(
                            measurements, arm, metric, status
                        ),
                        **inference,
                    }
                )

        # Correct-minus-incorrect is restricted to rollouts containing usable
        # events of both statuses, so trajectory differences cancel.
        for arm in arms:
            _correct_values, correct_by_run = grouped_effect_values(
                measurements, arm, metric, "correct"
            )
            _incorrect_values, incorrect_by_run = grouped_effect_values(
                measurements, arm, metric, "incorrect"
            )
            shared_runs = sorted(set(correct_by_run).intersection(incorrect_by_run))
            paired = {
                run_id: [
                    float(np.mean(correct_by_run[run_id]))
                    - float(np.mean(incorrect_by_run[run_id]))
                ]
                for run_id in shared_runs
            }
            values = [items[0] for items in paired.values()]
            inference = grouped_inference(
                values,
                paired,
                args.bootstrap_samples,
                args.permutations,
                rng,
            )
            rows.append(
                {
                    "comparison": "correct_minus_incorrect_paired_rollouts",
                    "metric": metric,
                    "objective_status": "correct_minus_incorrect",
                    "arm": arm,
                    **inference,
                    "permutation_p_holm": math.nan,
                }
            )
    holm_adjust(rows)
    return rows


def build_rankings(
    grouped_rows: Sequence[Mapping[str, Any]], arms: Sequence[str], rank_metric: str
) -> list[dict[str, Any]]:
    lookup = {
        (row["arm"], row["metric"], row["objective_status"]): row
        for row in grouped_rows
        if row["comparison"] == "event_minus_control"
    }
    rankings: list[dict[str, Any]] = []
    for arm in arms:
        overall = lookup[(arm, rank_metric, "all")]
        correct = lookup[(arm, rank_metric, "correct")]
        incorrect = lookup[(arm, rank_metric, "incorrect")]
        overall_effect = float(overall["rollout_weighted_mean"])
        direction = 1.0 if overall_effect >= 0 else -1.0
        correct_effect = float(correct["rollout_weighted_mean"])
        incorrect_effect = float(incorrect["rollout_weighted_mean"])
        if math.isfinite(correct_effect) and math.isfinite(incorrect_effect):
            shared_status_effect = min(direction * correct_effect, direction * incorrect_effect)
            status_preservation = max(
                0.0,
                min(1.0, shared_status_effect / max(abs(overall_effect), 1e-12)),
            )
        else:
            shared_status_effect = math.nan
            status_preservation = 0.0
        standardized = float(overall["standardized_rollout_effect"])
        consistency = float(overall["rollout_consistency"])
        # Keep the ranking informative even when no arm preserves the effect
        # across both status subsets.  The 0.25 floor preserves ordering by
        # overall evidence while the remaining 0.75 strongly rewards the
        # correct/incorrect dissociation that motivates this experiment.
        rank_score = (
            abs(standardized)
            * consistency
            * (0.25 + 0.75 * status_preservation)
            if math.isfinite(standardized) and math.isfinite(consistency)
            else 0.0
        )
        rankings.append(
            {
                "arm": arm,
                "rank_metric": rank_metric,
                "rank_score": rank_score,
                "effect_direction": "increase" if direction > 0 else "decrease",
                "rollout_weighted_effect": overall_effect,
                "event_weighted_effect": overall["event_weighted_mean"],
                "bootstrap_ci_low": overall["bootstrap_ci_low"],
                "bootstrap_ci_high": overall["bootstrap_ci_high"],
                "standardized_rollout_effect": standardized,
                "event_consistency": overall["event_consistency"],
                "rollout_consistency": consistency,
                "permutation_p_two_sided": overall["permutation_p_two_sided"],
                "permutation_p_holm": overall["permutation_p_holm"],
                "correct_rollout_weighted_effect": correct_effect,
                "incorrect_rollout_weighted_effect": incorrect_effect,
                "shared_status_effect_oriented": shared_status_effect,
                "status_preservation_ratio": status_preservation,
                "correct_n_events": correct["n_events"],
                "incorrect_n_events": incorrect["n_events"],
                "n_events": overall["n_events"],
                "n_rollouts": overall["n_rollouts"],
            }
        )
    rankings.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            float(row["permutation_p_two_sided"])
            if math.isfinite(float(row["permutation_p_two_sided"]))
            else math.inf,
            row["arm"],
        )
    )
    for rank, row in enumerate(rankings, 1):
        row["rank"] = rank
    return rankings


def svg_polyline_path(
    xs: Sequence[float], ys: Sequence[float], xmap: Any, ymap: Any
) -> str:
    parts: list[str] = []
    drawing = False
    for x, y in zip(xs, ys):
        if not math.isfinite(float(y)):
            drawing = False
            continue
        command = "L" if drawing else "M"
        parts.append(f"{command}{xmap(float(x)):.2f},{ymap(float(y)):.2f}")
        drawing = True
    return " ".join(parts)


def svg_band_polygons(
    xs: Sequence[float], means: Sequence[float], sems: Sequence[float], xmap: Any, ymap: Any
) -> list[str]:
    polygons: list[str] = []
    segment: list[tuple[float, float, float]] = []
    for x, mean, sem in zip(xs, means, sems):
        if math.isfinite(float(mean)) and math.isfinite(float(sem)):
            segment.append((float(x), float(mean) - float(sem), float(mean) + float(sem)))
        else:
            if len(segment) >= 2:
                upper = [(xmap(x), ymap(high)) for x, _low, high in segment]
                lower = [(xmap(x), ymap(low)) for x, low, _high in reversed(segment)]
                polygons.append(" ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower))
            segment = []
    if len(segment) >= 2:
        upper = [(xmap(x), ymap(high)) for x, _low, high in segment]
        lower = [(xmap(x), ymap(low)) for x, low, _high in reversed(segment)]
        polygons.append(" ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower))
    return polygons


def write_arm_svg(
    path: Path,
    arm: str,
    trace_rows: Sequence[Mapping[str, Any]],
    ranking: Mapping[str, Any],
    normalization: str,
    window: int,
) -> None:
    series_specs = [
        ("event", "all", "all events", "#1f77b4", ""),
        ("event", "correct", "correct", "#2ca02c", ""),
        ("event", "incorrect", "incorrect", "#d62728", ""),
        ("event", "ambiguous", "ambiguous", "#9467bd", ""),
        ("matched_control_event_mean", "all", "matched controls", "#666666", "6 4"),
    ]
    lookup: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row["arm"] == arm and row["aggregation"] == "rollout_weighted":
            lookup[(str(row["kind"]), str(row["objective_status"]))].append(row)
    for rows in lookup.values():
        rows.sort(key=lambda row: int(row["relative_token"]))
    all_y: list[float] = []
    for kind, status, _label, _color, _dash in series_specs:
        for row in lookup.get((kind, status), []):
            mean = float(row["mean"])
            sem = float(row["sem_between_rollouts"])
            if math.isfinite(mean):
                all_y.extend([mean - (sem if math.isfinite(sem) else 0), mean + (sem if math.isfinite(sem) else 0)])
    if not all_y:
        return
    y_low, y_high = min(all_y), max(all_y)
    padding = max(0.15, 0.08 * (y_high - y_low if y_high > y_low else 1.0))
    y_low -= padding
    y_high += padding
    width, height = 960, 570
    left, right, top, bottom = 82, 28, 82, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    xmap = lambda value: left + (value + window) / (2 * window) * plot_w
    ymap = lambda value: top + (y_high - value) / (y_high - y_low) * plot_h
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="system-ui,sans-serif" font-size="22" font-weight="600">{html.escape(arm)} around subjective progress</text>',
        f'<text x="{left}" y="57" font-family="system-ui,sans-serif" font-size="13" fill="#555">Equal-rollout mean ± between-rollout SEM · {html.escape(normalization)}</text>',
    ]
    for fraction in np.linspace(0, 1, 5):
        value = y_low + fraction * (y_high - y_low)
        y = ymap(value)
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e7e7e7"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="11" fill="#555">{value:.2f}</text>')
    if y_low <= 0 <= y_high:
        y0 = ymap(0)
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y0:.2f}" y2="{y0:.2f}" stroke="#999" stroke-width="1"/>')
    x0 = xmap(0)
    elements.append(f'<line x1="{x0:.2f}" x2="{x0:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#111" stroke-width="1.3" stroke-dasharray="3 4"/>')
    for value in sorted(set([-window, -window // 2, 0, window // 2, window])):
        x = xmap(value)
        elements.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top + plot_h}" y2="{top + plot_h + 5}" stroke="#333"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_h + 23}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11">{value:+d}</text>')
    for kind, status, label, color, dash in series_specs:
        rows = lookup.get((kind, status), [])
        if not rows:
            continue
        xs = [float(row["relative_token"]) for row in rows]
        means = [float(row["mean"]) for row in rows]
        sems = [float(row["sem_between_rollouts"]) for row in rows]
        for polygon in svg_band_polygons(xs, means, sems, xmap, ymap):
            elements.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.10"/>')
        path_data = svg_polyline_path(xs, means, xmap, ymap)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.1"{dash_attr}/>')
    legend_x, legend_y = left + 10, top + 13
    for index, (_kind, _status, label, color, dash) in enumerate(series_specs):
        y = legend_y + 20 * index
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<line x1="{legend_x}" x2="{legend_x + 25}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.3"{dash_attr}/>')
        elements.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="system-ui,sans-serif" font-size="12" fill="#333">{html.escape(label)}</text>')
    effect = float(ranking["rollout_weighted_effect"])
    ci_low = float(ranking["bootstrap_ci_low"])
    ci_high = float(ranking["bootstrap_ci_high"])
    p_value = float(ranking["permutation_p_holm"])
    note = (
        f'{ranking["rank_metric"]} event−control = {effect:+.3f}; '
        f'cluster bootstrap 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]; '
        f'Holm p={p_value:.3g}'
    )
    elements.extend(
        [
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">generated-token position relative to final event token (t=0)</text>',
            f'<text transform="translate(20 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">within-rollout normalized activation</text>',
            f'<text x="{left + plot_w}" y="57" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#555">{html.escape(note)}</text>',
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def format_num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rankings: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    plot_files: Sequence[str],
) -> None:
    anchor_categories = summary["events"]["anchor_token_categories"]
    punctuation_count = int(anchor_categories.get("punct", 0))
    mapped_count = int(summary["events"]["mapped_analyzed"])
    if punctuation_count == mapped_count:
        lexical_caution = (
            f"All {mapped_count} annotated t=0 tokens are punctuation tokens in this "
            "dataset. Lexical matching reduces, but cannot eliminate, the "
            "sentence-boundary confound."
        )
    else:
        lexical_caution = (
            f"Anchor token categories are {anchor_categories}. Inspect lexical-match "
            "tiers before attributing a trace to subjective progress."
        )
    lines = [
        "# Subjective-progress W-arm analysis",
        "",
        "This report is exploratory candidate discovery, not held-out confirmation or a causal result.",
        "",
        "## Data and design",
        "",
        f"- Input annotations: {summary['events']['input']} ({summary['events']['mapped_analyzed']} mapped and analyzed; {summary['events']['excluded']} excluded).",
        f"- Event-bearing rollouts: {summary['events']['mapped_rollouts']}.",
        f"- Objective status among analyzed events: {summary['events']['objective_status']}.",
        f"- Window: t=-{args.window}…+{args.window} in true generated-token coordinates, bounded to one assistant message, with missing positions retained.",
        f"- Normalization: {args.normalization} within rollout using channel `{args.normalization_channel}` and event-relevant message indices {summary['activations']['normalization_message_indices']}.",
        f"- Controls: target {args.controls_per_event} per event (minimum {args.min_controls_per_event}), same rollout and message role/index, outside every event span ±{summary['controls']['exclusion_tokens']} generated tokens.",
        f"- Scalar delta: mean t={args.post_start}:{args.post_end} minus mean t={args.pre_start}:{args.pre_end}.",
        f"- Peak/trough baseline: mean t={args.baseline_start}:{args.baseline_end}; search interval t={args.peak_start}:{args.peak_end}.",
        "- Inference: event minus its mean matched control, then equal-weighted over rollout means.",
        "- Trace controls are masked to each matched event's available offsets, so plotted support is paired at every t.",
        "",
        "## Ranked candidate arms",
        "",
        f"Ranking metric: `{args.rank_metric}`. The score combines absolute rollout-standardized effect and rollout directional consistency, with a 0.25 floor plus a 0.75 multiplier for preservation of the oriented effect in both correct and incorrect events.",
        "",
        "| rank | arm | score | grouped effect | bootstrap 95% CI | Holm p | rollout consistency | correct effect | incorrect effect | preservation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rankings:
        lines.append(
            "| {rank} | {arm} | {score} | {effect} | [{low}, {high}] | {p} | {consistency} | {correct} | {incorrect} | {preservation} |".format(
                rank=row["rank"],
                arm=row["arm"],
                score=format_num(row["rank_score"]),
                effect=format_num(row["rollout_weighted_effect"]),
                low=format_num(row["bootstrap_ci_low"]),
                high=format_num(row["bootstrap_ci_high"]),
                p=format_num(row["permutation_p_holm"], 4),
                consistency=format_num(row["rollout_consistency"]),
                correct=format_num(row["correct_rollout_weighted_effect"]),
                incorrect=format_num(row["incorrect_rollout_weighted_effect"]),
                preservation=format_num(row["status_preservation_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Control audit",
            "",
            f"- Match tiers: {summary['controls']['match_tiers']}.",
            f"- Events below the target control count: {summary['controls']['events_below_target']}.",
            f"- Same-step controls: {summary['controls']['same_step']} / {summary['controls']['total']}.",
            f"- Exact-token controls: {summary['controls']['exact_token']} / {summary['controls']['total']}.",
            f"- Exact message-boundary coverage matches: {summary['controls']['exact_coverage']} / {summary['controls']['total']} (median coverage distance {summary['controls']['median_coverage_distance']}).",
            f"- Reused control anchors: {summary['controls']['reused']} / {summary['controls']['total']}.",
            "",
            "## Interpretation cautions",
            "",
            f"- {lexical_caution}",
            "- Partial windows are status-dependent; each CSV row reports the exact usable event and rollout count.",
            "- A peak statistic is selection-biased by construction, so only its matched-control contrast is interpretable.",
            "- SVD direction signs are arbitrary. A repeatable decrease can be as interesting as an increase.",
            f"- Selection and inference use the same {summary['events']['mapped_rollouts']} event-bearing rollouts. Validate any selected arm on wholly unseen rollouts before steering.",
            "- Incorrect events are a target condition, not label noise: preservation there is central to the subjective-progress hypothesis.",
            "",
            "## Files",
            "",
            "- `event_windows.jsonl`: raw and normalized event/control traces with tokens and locations.",
            "- `event_measurements.csv`: paired event-level scalar measurements.",
            "- `control_measurements.csv`: each sampled control's measurements and match metadata.",
            "- `trace_summary.csv`: event-weighted and equal-rollout event-triggered means and SEMs.",
            "- `grouped_statistics.csv`: bootstrap, sign-flip, status, and paired-status results.",
            "- `arm_rankings.csv` / `.json`: transparent ranking components.",
            "- `plot_manifest.json`: authoritative list of plots produced by this run.",
            "",
        ]
    )
    if plot_files:
        lines.extend(["Current-run plots:", ""])
        lines.extend(f"- `{name}`" for name in plot_files)
        lines.append("")
    else:
        lines.extend(["Plots were disabled for this run.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fieldnames_for(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    events_path = args.events.resolve()
    activations_path = args.activations.resolve()
    output_dir = args.output_dir.resolve()
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    if not activations_path.exists():
        raise FileNotFoundError(activations_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {events_path}")
    all_events = read_jsonl(events_path)
    print(f"Loading W-arm activations from {activations_path}")
    activation_rows, arms = load_activations(activations_path)
    by_id, by_sequence, by_run = build_indexes(activation_rows)
    events, excluded = align_events(
        all_events, by_id, args.anchor_channel, args.allow_alignment_errors
    )
    if not events:
        raise ValueError("No mapped events remain after alignment")
    print(
        f"Aligned {len(events)}/{len(all_events)} annotations across "
        f"{len({event['run_id'] for event in events})} rollouts; arms={', '.join(arms)}"
    )

    normalization_message_indices = frozenset(
        event["_anchor_row"].message_index for event in events
    )
    norm_stats, norm_records = normalization_stats(
        by_run,
        arms,
        args.normalization,
        args.normalization_channel,
        normalization_message_indices,
    )
    exclusion = args.window if args.control_exclusion is None else args.control_exclusion
    intervals = event_exclusion_intervals(events, exclusion)
    controls_by_event, control_position_records = sample_controls(
        events,
        by_run,
        by_sequence,
        args.window,
        args.controls_per_event,
        args.min_controls_per_event,
        intervals,
        args.anchor_channel,
        args.match_token_class,
        args.unique_controls,
        args.seed,
    )
    print(f"Selected {len(control_position_records)} same-rollout control anchors")
    below_target = sum(
        len(controls_by_event[event["event_id"]]) < args.controls_per_event
        for event in events
    )
    if below_target:
        print(
            f"warning: {below_target} events had fewer than the target "
            f"{args.controls_per_event} controls (all met minimum "
            f"{args.min_controls_per_event})",
            file=sys.stderr,
        )

    window_records, measurements, control_measurements, event_data = (
        build_windows_and_measurements(
            events, controls_by_event, by_sequence, arms, norm_stats, args
        )
    )
    trace_rows = build_trace_summaries(event_data, arms, args.window)
    grouped_rows = build_grouped_statistics(measurements, arms, args)
    rankings = build_rankings(grouped_rows, arms, args.rank_metric)

    event_status_counts = Counter(event["objective_status"] for event in events)
    anchor_categories = Counter(
        broad_token_category(event["_anchor_row"].token) for event in events
    )
    anchor_message_indices = Counter(
        event["_anchor_row"].message_index for event in events
    )
    full_windows = sum(
        row["full_event_window"] for row in measurements if row["arm"] == arms[0]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "events": {
            "input": len(all_events),
            "mapped_analyzed": len(events),
            "excluded": len(excluded),
            "mapped_rollouts": len({event["run_id"] for event in events}),
            "objective_status": dict(sorted(event_status_counts.items())),
            "full_windows": int(full_windows),
            "partial_windows": len(events) - int(full_windows),
            "anchor_token_categories": dict(sorted(anchor_categories.items())),
            "anchor_message_indices": {
                str(key): value for key, value in sorted(anchor_message_indices.items())
            },
        },
        "activations": {
            "rows": len(activation_rows),
            "rollouts": len(by_run),
            "arms": arms,
            "normalization_channel": args.normalization_channel,
            "normalization_message_indices": sorted(normalization_message_indices),
        },
        "controls": {
            "per_event": args.controls_per_event,
            "minimum_per_event": args.min_controls_per_event,
            "events_below_target": below_target,
            "total": len(control_position_records),
            "exclusion_tokens": exclusion,
            "same_step": sum(row["scope"] == "same_step" for row in control_position_records),
            "exact_token": sum(
                row["lexical_match"] == "exact_token" for row in control_position_records
            ),
            "exact_coverage": sum(
                row["coverage_distance"] == 0 for row in control_position_records
            ),
            "median_coverage_distance": float(
                np.median(
                    [row["coverage_distance"] for row in control_position_records]
                )
            ),
            "max_coverage_distance": max(
                row["coverage_distance"] for row in control_position_records
            ),
            "reused": sum(row["reused_across_events"] for row in control_position_records),
            "match_tiers": dict(
                sorted(Counter(row["match_tier"] for row in control_position_records).items())
            ),
        },
        "ranking": rankings,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "events_path": str(events_path),
        "activations_path": str(activations_path),
        "output_dir": str(output_dir),
        "window": args.window,
        "relative_coordinate": "generated_local_index",
        "window_boundary": "same_run_step_channel_message_index",
        "anchor_policy": "final_token_overlapping_event",
        "anchor_state_semantics": "state_after_anchor_token_used_to_predict_next_token",
        "anchor_channel": args.anchor_channel,
        "normalization": args.normalization,
        "normalization_channel": args.normalization_channel,
        "normalization_message_indices": sorted(normalization_message_indices),
        "controls_per_event": args.controls_per_event,
        "min_controls_per_event": args.min_controls_per_event,
        "control_exclusion": exclusion,
        "control_anchor_constraint": "same_run_channel_message_index_as_event",
        "match_token_class": args.match_token_class,
        "unique_controls": args.unique_controls,
        "pre_interval": [args.pre_start, args.pre_end],
        "post_interval": [args.post_start, args.post_end],
        "baseline_interval": [args.baseline_start, args.baseline_end],
        "peak_interval": [args.peak_start, args.peak_end],
        "rank_metric": args.rank_metric,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "arms": arms,
        "trace_aggregations": ["event_weighted", "rollout_weighted"],
        "plot_aggregation": "rollout_weighted",
        "inference_unit": "rollout_mean_of_paired_event_minus_mean_control_effects",
    }

    write_json(output_dir / "analysis_config.json", config)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "normalization.jsonl", norm_records)
    write_jsonl(output_dir / "excluded_events.jsonl", excluded)
    write_jsonl(output_dir / "control_positions.jsonl", control_position_records)
    write_jsonl(output_dir / "event_windows.jsonl", window_records)
    write_csv(
        output_dir / "event_measurements.csv",
        measurements,
        fieldnames_for(measurements),
    )
    write_jsonl(output_dir / "event_measurements.jsonl", measurements)
    write_csv(
        output_dir / "control_measurements.csv",
        control_measurements,
        fieldnames_for(control_measurements),
    )
    write_csv(output_dir / "trace_summary.csv", trace_rows, fieldnames_for(trace_rows))
    write_csv(
        output_dir / "grouped_statistics.csv",
        grouped_rows,
        fieldnames_for(grouped_rows),
    )
    write_jsonl(output_dir / "grouped_statistics.jsonl", grouped_rows)
    write_csv(output_dir / "arm_rankings.csv", rankings, fieldnames_for(rankings))
    write_json(output_dir / "arm_rankings.json", rankings)

    plot_files: list[str] = []
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        selected_rankings = rankings if args.plot_top_k <= 0 else rankings[: args.plot_top_k]
        for ranking in selected_rankings:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ranking["arm"]))
            relative_plot_path = f"plots/{safe_name}.svg"
            write_arm_svg(
                output_dir / relative_plot_path,
                str(ranking["arm"]),
                trace_rows,
                ranking,
                args.normalization,
                args.window,
            )
            plot_files.append(relative_plot_path)
    write_json(
        output_dir / "plot_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "plots_enabled": not args.no_plots,
            "plot_top_k": args.plot_top_k,
            "generated": plot_files,
            "note": (
                "Only files listed here were produced by the current run; the output "
                "directory is not destructively cleared between runs."
            ),
        },
    )
    write_report(output_dir / "report.md", summary, rankings, args, plot_files)

    print("Ranked arms:")
    for row in rankings:
        print(
            f"  {row['rank']}. {row['arm']}: score={row['rank_score']:.3f}, "
            f"{args.rank_metric}={row['rollout_weighted_effect']:+.3f}, "
            f"incorrect={row['incorrect_rollout_weighted_effect']:+.3f}, "
            f"Holm p={row['permutation_p_holm']:.4g}"
        )
    print(f"Wrote analysis artifacts to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
