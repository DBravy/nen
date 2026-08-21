#!/usr/bin/env python3
"""Validate and align qualitative subjective-progress annotations.

The hand-authored batch files identify a minimal verbatim event span inside an
assistant turn's ``thinking`` field.  This script checks those annotations
against the frozen transcripts and maps each span to the exact generated-token
positions saved by ``analyze_aha_rollout.py``.

The primary alignment anchor is the final generated token overlapping the
annotated event.  In the activation files, the state saved for generated token
``g`` is the state *after* consuming ``g`` (and is therefore used to predict
token ``g + 1``).
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_ROLLOUT_ROOT = ROOT / "task_gaming_v8" / "rollouts"
DEFAULT_LABELS = ROOT / "task_gaming_v8_frozen_labels.jsonl"
DEFAULT_W_ARM_DATA = ROOT / "task_gaming_candidate_data" / "token_data.jsonl"
DEFAULT_BATCHES = (
    ROOT / "subjective_progress_secret.jsonl",
    ROOT / "subjective_progress_precommit.jsonl",
    ROOT / "subjective_progress_impossible.jsonl",
)
DEFAULT_COVERAGE = (
    ROOT / "subjective_progress_secret_coverage.jsonl",
    ROOT / "subjective_progress_precommit_coverage.jsonl",
    ROOT / "subjective_progress_impossible_coverage.jsonl",
)
DEFAULT_OUT = ROOT / "task_gaming_v8_subjective_progress"

REQUIRED_EVENT_FIELDS = {
    "run_id",
    "environment",
    "condition",
    "step",
    "event_type",
    "before",
    "event",
    "after",
    "subjective_progress",
    "objective_status",
    "objective_status_reason",
    "confidence",
}
EVENT_TYPES = {
    "strategy_selection",
    "realization",
    "hypothesis_revision",
    "constraint_discovery",
    "failure_diagnosis",
    "uncertainty_resolution_action",
    "intermediate_result",
    "conclusion",
    "other",
}
OBJECTIVE_STATUSES = {"correct", "incorrect", "ambiguous"}
CONFIDENCES = {"high", "medium", "low"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            row["_source_batch"] = path.name
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def load_run_index(rollout_root: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    rollout_root = rollout_root.resolve()
    run_dirs = {
        path.parent
        for pattern in ("transcript.json", "transcript.partial.json")
        for path in rollout_root.rglob(pattern)
    }
    for run_dir in sorted(run_dirs):
        complete_path = run_dir / "transcript.json"
        partial_path = run_dir / "transcript.partial.json"
        transcript_path = complete_path if complete_path.exists() else partial_path
        trajectory_complete = complete_path.exists()
        run_id = run_dir.name
        if run_id in runs:
            raise ValueError(f"duplicate run_id: {run_id}")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assistant_items = [
            (message_index, message)
            for message_index, message in enumerate(transcript)
            if message.get("role") == "assistant"
        ]
        assistants = {int(message["step"]): message for _index, message in assistant_items}
        assistant_message_indices = {
            int(message["step"]): message_index for message_index, message in assistant_items
        }
        result_path = run_dir / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        error_path = run_dir / "error.json"
        error = (
            json.loads(error_path.read_text(encoding="utf-8"))
            if error_path.exists()
            else {}
        )
        error_text = str(error.get("error", ""))
        error_class = error_text.split("(", 1)[0] if error_text else None
        runs[run_id] = {
            "run_dir": run_dir,
            "transcript_path": transcript_path,
            "transcript": transcript,
            "assistants": assistants,
            "assistant_message_indices": assistant_message_indices,
            "environment": result.get("environment") or run_dir.parents[1].name,
            "condition": result.get("condition") or run_dir.parent.name,
            "result": result,
            "trajectory_complete": trajectory_complete,
            "trajectory_status": "complete" if trajectory_complete else "partial_error",
            "trajectory_error_class": error_class,
        }
    return runs


def load_w_arm_token_index(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Index the full-sequence candidate-direction rows by replay position."""
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    # Iterate physical lines rather than using str.splitlines(): decoded token
    # strings can legally contain Unicode line-separator characters.
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["run_id"]), int(row["step"]), int(row["generated_local_index"]))
            if key in index:
                raise ValueError(f"{path}:{line_no}: duplicate W-arm token key {key}")
            # Do not retain the large candidate/J-Lens payload for every token.
            index[key] = {
                field: row[field]
                for field in ("row_id", "full_position", "token_index_in_turn")
            }
    return index


def locate_context(text: str, excerpt: str, *, before: int | None, after: int | None) -> int:
    """Locate a context excerpt on the requested side of an event."""
    if not excerpt:
        return -1
    starts: list[int] = []
    cursor = 0
    while True:
        found = text.find(excerpt, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if before is not None:
        starts = [start for start in starts if start + len(excerpt) <= before]
        return max(starts, default=-1)
    if after is not None:
        starts = [start for start in starts if start >= after]
        return min(starts, default=-1)
    raise AssertionError("one side must be specified")


def extract_analysis_segment(raw: str) -> tuple[int, int, str]:
    prefix = "<|channel|>analysis<|message|>"
    start = raw.find(prefix)
    if start < 0:
        raise ValueError("generated text has no analysis-channel prefix")
    start += len(prefix)
    end = raw.find("<|end|>", start)
    if end < 0:
        # A malformed/recovered turn can lack the ordinary terminator.  Tool or
        # final channel tags still delimit the analysis text.
        candidates = [
            pos
            for marker in ("<|channel|>commentary", "<|channel|>final")
            if (pos := raw.find(marker, start)) >= 0
        ]
        end = min(candidates) if candidates else len(raw)
    return start, end, raw[start:end]


def map_boundary_with_diff(source: str, target: str, position: int) -> int:
    """Map a source character boundary into a nearly identical target string."""
    if not 0 <= position <= len(source):
        raise ValueError(f"source boundary out of range: {position}")
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    opcodes = matcher.get_opcodes()
    if position == len(source):
        return len(target)
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 <= position < i2:
            if tag == "equal":
                return j1 + (position - i1)
            if i2 == i1:
                return j1
            fraction = (position - i1) / (i2 - i1)
            return round(j1 + fraction * (j2 - j1))
        if position == i2:
            return j2
    raise ValueError(f"could not map character boundary {position}")


def token_alignment(
    run: dict[str, Any],
    step: int,
    thinking: str,
    char_start: int,
    char_end: int,
    event_text: str,
) -> dict[str, Any]:
    token_map_path = run["run_dir"] / "aha_analysis" / f"step_{step:02d}" / "token_map.jsonl"
    if not token_map_path.exists():
        raise ValueError(f"missing token map: {token_map_path}")
    tokens = [json.loads(line) for line in token_map_path.read_text(encoding="utf-8").splitlines() if line]
    raw = "".join(token["decoded"] for token in tokens)
    analysis_start, _analysis_end, raw_thinking = extract_analysis_segment(raw)

    raw_event_start = raw_thinking.find(event_text)
    if raw_event_start >= 0 and raw_thinking.count(event_text) == 1:
        raw_event_end = raw_event_start + len(event_text)
    else:
        # The GPT-OSS byte decoder in the saved token map can contain a Unicode
        # replacement glyph where the parsed transcript retained the original
        # character.  Sequence alignment makes this rare case deterministic.
        similarity = difflib.SequenceMatcher(a=thinking, b=raw_thinking, autojunk=False).ratio()
        if similarity < 0.98:
            raise ValueError(f"thinking/token-map text similarity too low: {similarity:.4f}")
        raw_event_start = map_boundary_with_diff(thinking, raw_thinking, char_start)
        raw_event_end = map_boundary_with_diff(thinking, raw_thinking, char_end)

    full_start = analysis_start + raw_event_start
    full_end = analysis_start + raw_event_end
    bounds: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for token in tokens:
        next_cursor = cursor + len(token["decoded"])
        bounds.append((cursor, next_cursor, token))
        cursor = next_cursor
    overlapping = [token for start, end, token in bounds if end > full_start and start < full_end]
    if not overlapping:
        raise ValueError("event span did not overlap any generated token")
    first, last = overlapping[0], overlapping[-1]
    return {
        "generated_token_start": int(first["generated_position"]),
        "generated_token_end": int(last["generated_position"]),
        "absolute_token_start": int(first["absolute_position"]),
        "absolute_token_end": int(last["absolute_position"]),
        "anchor_generated_position": int(last["generated_position"]),
        "anchor_absolute_position": int(last["absolute_position"]),
        "anchor_policy": "final_token_overlapping_event",
        "anchor_state_semantics": "state_after_anchor_token_used_to_predict_next_token",
        "token_span_decoded": "".join(token["decoded"] for token in overlapping),
        "token_map_path": str(token_map_path.relative_to(ROOT)),
    }


def validate_and_enrich_events(
    annotations: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    w_arm_index: dict[tuple[str, int, int], dict[str, Any]] | None = None,
    w_arm_path: Path | None = None,
) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    exact_keys: set[tuple[str, int, str]] = set()
    for row in annotations:
        source = f"{row['_source_batch']}:{row['_source_line']}"
        missing = REQUIRED_EVENT_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{source}: missing fields: {sorted(missing)}")
        run_id = str(row["run_id"])
        if run_id not in runs:
            raise ValueError(f"{source}: unknown run_id {run_id}")
        run = runs[run_id]
        step = int(row["step"])
        if step not in run["assistants"]:
            raise ValueError(f"{source}: run has no assistant step {step}")
        if row["environment"] != run["environment"] or row["condition"] != run["condition"]:
            raise ValueError(
                f"{source}: environment/condition mismatch; expected "
                f"{run['environment']}/{run['condition']}"
            )
        if row["event_type"] not in EVENT_TYPES:
            raise ValueError(f"{source}: invalid event_type {row['event_type']!r}")
        if row["objective_status"] not in OBJECTIVE_STATUSES:
            raise ValueError(f"{source}: invalid objective_status {row['objective_status']!r}")
        if row["confidence"] not in CONFIDENCES:
            raise ValueError(f"{source}: invalid confidence {row['confidence']!r}")

        thinking = run["assistants"][step].get("thinking") or ""
        event = row["event"]
        if not event or thinking.count(event) != 1:
            raise ValueError(
                f"{source}: event must occur exactly once in thinking; count={thinking.count(event)}"
            )
        char_start = thinking.index(event)
        char_end = char_start + len(event)
        if row["before"] and locate_context(thinking, row["before"], before=char_start, after=None) < 0:
            raise ValueError(f"{source}: `before` is not a verbatim earlier excerpt")
        if row["after"] and locate_context(thinking, row["after"], before=None, after=char_end) < 0:
            raise ValueError(f"{source}: `after` is not a verbatim later excerpt")
        exact_key = (run_id, step, event)
        if exact_key in exact_keys:
            raise ValueError(f"{source}: duplicate event annotation")
        exact_keys.add(exact_key)

        alignment = token_alignment(run, step, thinking, char_start, char_end, event)
        w_arm_alignment: dict[str, Any] = {
            "w_arm_alignment_available": False,
            "w_arm_row_start": None,
            "w_arm_row_end": None,
            "w_arm_anchor_row_id": None,
            "w_arm_token_index_start": None,
            "w_arm_token_index_end": None,
            "w_arm_anchor_token_index": None,
            "w_arm_data_path": str((w_arm_path or DEFAULT_W_ARM_DATA).resolve().relative_to(ROOT)),
        }
        if w_arm_index is not None:
            start_key = (run_id, step, alignment["generated_token_start"])
            end_key = (run_id, step, alignment["generated_token_end"])
            anchor_key = (run_id, step, alignment["anchor_generated_position"])
            keys_present = [key in w_arm_index for key in (start_key, end_key, anchor_key)]
            if any(keys_present) and not all(keys_present):
                raise ValueError(f"{source}: incomplete event span in W-arm data")
            if all(keys_present):
                w_start = w_arm_index[start_key]
                w_end = w_arm_index[end_key]
                w_anchor = w_arm_index[anchor_key]
                if int(w_start["full_position"]) != alignment["absolute_token_start"]:
                    raise ValueError(f"{source}: W-arm/start absolute-position mismatch")
                if int(w_end["full_position"]) != alignment["absolute_token_end"]:
                    raise ValueError(f"{source}: W-arm/end absolute-position mismatch")
                w_arm_alignment.update(
                    {
                        "w_arm_alignment_available": True,
                        "w_arm_row_start": int(w_start["row_id"]),
                        "w_arm_row_end": int(w_end["row_id"]),
                        "w_arm_anchor_row_id": int(w_anchor["row_id"]),
                        "w_arm_token_index_start": int(w_start["token_index_in_turn"]),
                        "w_arm_token_index_end": int(w_end["token_index_in_turn"]),
                        "w_arm_anchor_token_index": int(w_anchor["token_index_in_turn"]),
                    }
                )

        message_index = run["assistant_message_indices"][step]
        prior_message = run["transcript"][message_index - 1] if message_index > 0 else None
        prior_role = prior_message.get("role") if prior_message else None
        prior_content = (
            str(prior_message.get("content") or "")
            if prior_message and prior_role in {"tool", "harness"}
            else ""
        )
        prior_start = 0
        if len(prior_content) > 600:
            prior_start = len(prior_content) - 600
            next_newline = prior_content.find("\n", prior_start)
            if next_newline >= 0:
                prior_start = next_newline + 1
        prior_excerpt = prior_content[prior_start:] if prior_content else None

        clean = {key: value for key, value in row.items() if not key.startswith("_source_")}
        clean.update(
            {
                "thinking_char_start": char_start,
                "thinking_char_end": char_end,
                "thinking_char_end_exclusive": True,
                "transcript_message_index": message_index,
                "prior_message_index": message_index - 1 if prior_excerpt is not None else None,
                "prior_message_role": prior_role if prior_excerpt is not None else None,
                "prior_message_excerpt_start_char": prior_start if prior_excerpt is not None else None,
                "prior_message_excerpt": prior_excerpt,
                "trajectory_complete": run["trajectory_complete"],
                "trajectory_status": run["trajectory_status"],
                "trajectory_error_class": run["trajectory_error_class"],
                "transcript_path": str(run["transcript_path"].relative_to(ROOT)),
                **alignment,
                **w_arm_alignment,
            }
        )
        staged.append(clean)

    staged.sort(key=lambda row: (row["run_id"], int(row["step"]), row["thinking_char_start"]))
    counters: Counter[tuple[str, int]] = Counter()
    for row in staged:
        key = (row["run_id"], int(row["step"]))
        counters[key] += 1
        row["event_id"] = f"{row['run_id']}__step_{int(row['step']):02d}__event_{counters[key]:02d}"
        row["schema_version"] = "1.0"
        # Put identifiers and location first for easier JSONL inspection.
        order = (
            "schema_version",
            "event_id",
            "run_id",
            "environment",
            "condition",
            "step",
            "transcript_message_index",
            "prior_message_index",
            "prior_message_role",
            "prior_message_excerpt_start_char",
            "prior_message_excerpt",
            "trajectory_complete",
            "trajectory_status",
            "trajectory_error_class",
            "event_type",
            "thinking_char_start",
            "thinking_char_end",
            "thinking_char_end_exclusive",
            "generated_token_start",
            "generated_token_end",
            "absolute_token_start",
            "absolute_token_end",
            "anchor_generated_position",
            "anchor_absolute_position",
            "anchor_policy",
            "anchor_state_semantics",
            "w_arm_alignment_available",
            "w_arm_row_start",
            "w_arm_row_end",
            "w_arm_anchor_row_id",
            "w_arm_token_index_start",
            "w_arm_token_index_end",
            "w_arm_anchor_token_index",
            "before",
            "event",
            "after",
            "subjective_progress",
            "objective_status",
            "objective_status_reason",
            "confidence",
            "token_span_decoded",
            "transcript_path",
            "token_map_path",
            "w_arm_data_path",
        )
        reordered = {field: row[field] for field in order if field in row}
        row.clear()
        row.update(reordered)
    return staged


def validate_coverage(
    coverage_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_run = Counter(row["run_id"] for row in events)
    coverage: dict[str, dict[str, Any]] = {}
    for row in coverage_rows:
        source = f"{row['_source_batch']}:{row['_source_line']}"
        run_id = row.get("run_id")
        if run_id not in runs:
            raise ValueError(f"{source}: unknown coverage run_id {run_id!r}")
        if run_id in coverage:
            raise ValueError(f"{source}: duplicate coverage row for {run_id}")
        declared = int(row.get("event_count", -1))
        actual = by_run[run_id]
        if declared != actual:
            raise ValueError(f"{source}: declared event_count={declared}, actual={actual}")
        if actual == 0 and not row.get("no_event_reason"):
            raise ValueError(f"{source}: zero-event rollout needs no_event_reason")
        coverage[run_id] = row
    missing = sorted(set(runs) - set(coverage))
    extra = sorted(set(coverage) - set(runs))
    if missing or extra:
        raise ValueError(f"coverage mismatch: missing={missing}, extra={extra}")
    output: list[dict[str, Any]] = []
    for run_id in sorted(runs):
        run = runs[run_id]
        source = coverage[run_id]
        output.append(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "environment": run["environment"],
                "condition": run["condition"],
                "event_count": by_run[run_id],
                "no_event_reason": source.get("no_event_reason"),
                "trajectory_complete": run["trajectory_complete"],
                "trajectory_status": run["trajectory_status"],
                "trajectory_error_class": run["trajectory_error_class"],
                "transcript_path": str(run["transcript_path"].relative_to(ROOT)),
            }
        )
    return output


def human_review_markdown(events: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        grouped[row["run_id"]].append(row)
    lines = [
        "# Subjective-progress event review",
        "",
        "Each event is a perceived positive value update, whether or not it was objectively correct.",
        "The generated-token anchor is the final token overlapping the bold event text.",
        "",
    ]
    for rollout in coverage:
        run_id = rollout["run_id"]
        lines.extend([f"## {run_id}", ""])
        if not grouped[run_id]:
            lines.extend([f"No event selected: {rollout['no_event_reason']}", ""])
            continue
        for event in grouped[run_id]:
            location = (
                f"transcript message {event['transcript_message_index']}, step {event['step']}, thinking chars "
                f"[{event['thinking_char_start']}, {event['thinking_char_end']}), "
                f"generated tokens [{event['generated_token_start']}, "
                f"{event['generated_token_end']}]"
            )
            lines.extend(
                [
                    f"### {event['event_id']}",
                    "",
                    f"Location: {location}",
                    "",
                ]
            )
            if not event["before"] and event.get("prior_message_excerpt"):
                lines.extend(
                    [
                        "Prior tool context (exact suffix):",
                        "",
                        "```text",
                        event["prior_message_excerpt"],
                        "```",
                        "",
                    ]
                )
            lines.extend(
                [
                    f"> {event['before']} **{event['event']}** {event['after']}",
                    "",
                    f"Why subjective progress: {event['subjective_progress']}",
                    "",
                    f"Objective status: {event['objective_status']} — {event['objective_status_reason']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--w-arm-data", type=Path, default=DEFAULT_W_ARM_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batches", type=Path, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--coverage", type=Path, nargs="+", default=list(DEFAULT_COVERAGE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_run_index(args.rollout_root)
    w_arm_index = load_w_arm_token_index(args.w_arm_data)
    annotations = [row for path in args.batches for row in read_jsonl(path)]
    events = validate_and_enrich_events(annotations, runs, w_arm_index, args.w_arm_data)
    coverage_source = [row for path in args.coverage for row in read_jsonl(path)]
    coverage = validate_coverage(coverage_source, events, runs)

    frozen = read_jsonl(args.labels) if args.labels.exists() else []
    frozen_ids = {row["run_id"] for row in frozen}
    missing_transcripts = sorted(frozen_ids - set(runs))
    unexpected_transcripts = sorted(set(runs) - frozen_ids)
    summary = {
        "schema_version": "1.0",
        "available_rollouts": len(runs),
        "complete_rollouts": sum(run["trajectory_complete"] for run in runs.values()),
        "partial_rollouts": sum(not run["trajectory_complete"] for run in runs.values()),
        "annotated_events": len(events),
        "rollouts_with_events": sum(row["event_count"] > 0 for row in coverage),
        "rollouts_without_events": sum(row["event_count"] == 0 for row in coverage),
        "events_by_environment": dict(sorted(Counter(row["environment"] for row in events).items())),
        "events_by_type": dict(sorted(Counter(row["event_type"] for row in events).items())),
        "events_by_objective_status": dict(
            sorted(Counter(row["objective_status"] for row in events).items())
        ),
        "events_by_confidence": dict(sorted(Counter(row["confidence"] for row in events).items())),
        "frozen_label_rows": len(frozen),
        "frozen_labeled_runs_missing_transcript": missing_transcripts,
        "transcript_runs_missing_frozen_label": unexpected_transcripts,
        "w_arm_token_rows": len(w_arm_index),
        "events_with_w_arm_alignment": sum(row["w_arm_alignment_available"] for row in events),
        "events_without_w_arm_alignment": sum(
            not row["w_arm_alignment_available"] for row in events
        ),
        "w_arm_data_path": str(args.w_arm_data.resolve().relative_to(ROOT)),
        "primary_anchor": "final generated token overlapping the annotated event span",
        "anchor_state_semantics": "saved state is after consuming anchor token and predicts the next token",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "events.jsonl", events)
    write_jsonl(args.out / "rollout_coverage.jsonl", coverage)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "events_review.md").write_text(
        human_review_markdown(events, coverage), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
