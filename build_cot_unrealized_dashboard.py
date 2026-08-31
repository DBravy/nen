#!/usr/bin/env python3
"""Build a self-contained dashboard joining low/medium CoT and FineWeb SV evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOW_DIR = ROOT / "cot_unrealized_low"
DEFAULT_MEDIUM_DIR = ROOT / "cot_unrealized_medium"
DEFAULT_FINEWEB_DIR = ROOT / "unrealized_words_fineweb"
DEFAULT_FINEWEB_SELECTIVITY_DIR = ROOT / "unrealized_words_selectivity"
DEFAULT_PREDICTIVE_LOW_DIR = ROOT / "predictive_cot_low"
DEFAULT_PREDICTIVE_MEDIUM_DIR = ROOT / "predictive_cot_medium"
DEFAULT_PREDICTIVE_FINEWEB_DIR = ROOT / "predictive_words_scan"
DEFAULT_PREDICTIVE_FINEWEB_SELECTIVITY_DIR = ROOT / "predictive_words_scan"
DEFAULT_OUTPUT = ROOT / "cot_unrealized_report.html"

# Metadata describing each attribution basis. Both sources render through the
# exact same UI; only the residual-state/token pairing differs.
SOURCE_LABELS = {
    "unrealized": {
        "label": "Current-token",
        "tagline": "reads h[t] → token[t]",
        "note": "Current-token attribution: each direction reads the residual state at the same position as the token it describes.",
    },
    "predictive": {
        "label": "Predictive",
        "tagline": "reads h[t−1] → token[t]",
        "note": "Predictive attribution: each direction reads the residual state one position before the token it describes. Same shared SVD basis, different state/token pairing.",
    },
}
BROAD_RANK = "rank_global_mean_abs_cosine"
SELECTIVE_RANK = "rank_global_tail_selectivity"
TEXT_FIELDS = {
    "candidate",
    "selected_tail_polarity",
    "nearest_unembed_token",
    "nearest_unembed_decoded",
    "farthest_unembed_token",
    "farthest_unembed_decoded",
}
HARMONY_TOKEN = re.compile(r"<\|[^>]+?\|>")
LEFT_TOKEN_NEIGHBORS_PER_SIDE = 16
_LEFT_ARTIFACT_CACHE: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-data-dir", type=Path, default=DEFAULT_LOW_DIR)
    parser.add_argument("--medium-data-dir", type=Path, default=DEFAULT_MEDIUM_DIR)
    parser.add_argument("--fineweb-data-dir", type=Path, default=DEFAULT_FINEWEB_DIR)
    parser.add_argument(
        "--fineweb-selectivity-data-dir",
        type=Path,
        default=DEFAULT_FINEWEB_SELECTIVITY_DIR,
    )
    parser.add_argument("--predictive-low-data-dir", type=Path, default=DEFAULT_PREDICTIVE_LOW_DIR)
    parser.add_argument("--predictive-medium-data-dir", type=Path, default=DEFAULT_PREDICTIVE_MEDIUM_DIR)
    parser.add_argument("--predictive-fineweb-data-dir", type=Path, default=DEFAULT_PREDICTIVE_FINEWEB_DIR)
    parser.add_argument(
        "--predictive-fineweb-selectivity-data-dir",
        type=Path,
        default=DEFAULT_PREDICTIVE_FINEWEB_SELECTIVITY_DIR,
    )
    parser.add_argument(
        "--no-predictive",
        action="store_true",
        help="Build only the current-token source (original single-basis behavior)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top",
        type=int,
        default=250,
        help="Candidates retained per effort/discovery lens; 0 retains all (default: 250)",
    )
    parser.add_argument(
        "--contexts-per-side",
        type=int,
        default=24,
        help="Highest-ranked context events embedded per direction/polarity; 0 retains all (default: 24)",
    )
    parser.add_argument(
        "--fineweb-contexts-per-side",
        type=int,
        default=6,
        help="FineWeb contexts embedded per direction/polarity/scan output; 0 retains all (default: 6)",
    )
    return parser.parse_args()


def number(value: str) -> int | float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"expected a finite number, got {value!r}")
    return int(parsed) if parsed.is_integer() else parsed


def safe_script_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def zero_based_candidate(row: dict[str, Any]) -> str:
    return f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']):02d}"


def selected_tail_value(row: dict[str, Any], suffix: str) -> Any:
    polarity = row.get("selected_tail_polarity")
    return row.get(f"{polarity}_{suffix}") if polarity in ("positive", "negative") else None


def display_ranking_row(source: dict[str, Any]) -> dict[str, Any]:
    row = {
        ("singular_value_over_sv0" if key == "singular_value_over_sv1" else key): value
        for key, value in source.items()
        if key != "sv_rank_1based"
    }
    row["candidate"] = zero_based_candidate(source)
    for suffix in (
        "max_robust_z",
        "q99_robust_z",
        "q999_robust_z",
        "q9999_robust_z",
        "top1pct_energy_share",
        "top0_1pct_energy_share",
        "token_rate_z3",
        "token_rate_z5",
        "token_rate_z8",
        "doc_count_z3",
        "doc_count_z5",
        "doc_count_z8",
        "doc_rate_z3",
        "doc_rate_z5",
        "doc_rate_z8",
        "tail_shape_score",
        "tail_support_weight",
    ):
        row[f"selected_tail_{suffix}"] = selected_tail_value(source, suffix)
    return row


def load_rankings(path: Path, top_n: int) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate", "layer", "sv_index_0", BROAD_RANK, SELECTIVE_RANK}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} is missing required ranking columns")
        source_rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, 2):
            try:
                row = {
                    key: (value if key in TEXT_FIELDS else number(value))
                    for key, value in raw.items()
                }
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric field on {path}:{line_number}: {exc}") from exc
            expected = f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']) + 1:02d}"
            if row["candidate"] != expected:
                raise SystemExit(
                    f"Unexpected source candidate {row['candidate']!r}; expected {expected!r}"
                )
            source_rows.append(row)

    if top_n < 0:
        raise SystemExit("--top must be 0 or greater")
    source_rows.sort(key=lambda row: int(row[BROAD_RANK]))
    display_rows = [display_ranking_row(row) for row in source_rows]
    source_by_display = {
        zero_based_candidate(row): str(row["candidate"]) for row in source_rows
    }
    limit = len(source_rows) if top_n == 0 else min(top_n, len(source_rows))
    cohorts = {
        "broad": [
            zero_based_candidate(row)
            for row in sorted(source_rows, key=lambda row: int(row[BROAD_RANK]))[:limit]
        ],
        "selective": [
            zero_based_candidate(row)
            for row in sorted(source_rows, key=lambda row: int(row[SELECTIVE_RANK]))[:limit]
        ],
    }
    return display_rows, source_by_display, cohorts


def compact_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        messages.append({"role": str(item.get("role", "unknown")), "content": content})
    return messages


def load_rollouts(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rollouts: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    scanned_index = 0
    total = 0
    capped = 0
    finals = 0
    prompt_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            status = str(raw.get("status", "unknown"))
            status_counts[status] += 1
            if status != "scanned":
                continue
            source = raw.get("source") or {}
            prompt_id = str(raw.get("prompt_id") or source.get("id") or f"job_{line_number}")
            category = str(source.get("category") or "uncategorized")
            difficulty = str(source.get("difficulty") or "unspecified")
            category_counts[category] += 1
            difficulty_counts[difficulty] += 1
            hit_cap = bool(raw.get("hit_max_new_tokens"))
            final = str(raw.get("final") or "")
            capped += int(hit_cap)
            finals += int(bool(final.strip()))
            prompt_ids.append(prompt_id)
            rollouts[str(scanned_index)] = {
                "document": scanned_index,
                "job": raw.get("job_index"),
                "prompt_id": prompt_id,
                "category": category,
                "difficulty": difficulty,
                "sample": source.get("sample_index", raw.get("sample_index")),
                "messages": compact_messages(raw.get("messages")),
                "reasoning": str(raw.get("reasoning") or ""),
                "final": final,
                "status": status,
                "hit_cap": hit_cap,
                "prompt_tokens": raw.get("prompt_tokens"),
                "generated_tokens": raw.get("generated_tokens"),
                "analysis_tokens": raw.get("analysis_tokens"),
                "final_tokens": raw.get("final_tokens"),
                "sequence_tokens": raw.get("sequence_tokens"),
                "analysis_segments": raw.get("analysis_segments"),
                "final_segments": raw.get("final_segments"),
            }
            scanned_index += 1
    return rollouts, {
        "jobs": total,
        "scanned": scanned_index,
        "capped": capped,
        "with_final": finals,
        "status_counts": dict(status_counts),
        "category_counts": dict(category_counts),
        "difficulty_counts": dict(difficulty_counts),
        "prompt_ids": prompt_ids,
    }


def clean_marked_context(raw: dict[str, Any]) -> tuple[str, str]:
    marked = str(raw.get("context_marked") or raw.get("context") or "")
    pieces = HARMONY_TOKEN.split(marked)
    focused = next((piece for piece in pieces if "⟦" in piece and "⟧" in piece), marked)
    focused = focused.strip()
    plain = focused.replace("⟦", "").replace("⟧", "")
    return focused, plain


def compact_context(raw: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    marked, plain = clean_marked_context(raw)
    window = raw.get("window_meta") or {}
    prompt_tokens = int(window.get("prompt_tokens") or rollout.get("prompt_tokens") or 0)
    generated_tokens = max(
        int(window.get("generated_tokens") or rollout.get("generated_tokens") or 0), 1
    )
    position = int(raw.get("token_position") or 0)
    progress = max(0.0, min(1.0, (position - prompt_tokens) / generated_tokens))
    return {
        "polarity": raw.get("polarity"),
        "rank": raw.get("rank_within_polarity"),
        "activation": raw.get("activation"),
        "cosine": raw.get("cosine_activation"),
        "token": raw.get("token", ""),
        "marked": marked,
        "plain": plain,
        "document": raw.get("document_index"),
        "position": position,
        "progress": progress,
    }


def load_contexts(
    path: Path,
    source_candidates: dict[str, str],
    rollouts: dict[str, dict[str, Any]],
    context_limit: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if context_limit < 0:
        raise SystemExit("--contexts-per-side must be 0 or greater")
    display_by_source = {source: display for display, source in source_candidates.items()}
    contexts = {
        display: {"positive": [], "negative": []} for display in source_candidates
    }
    group_counts: Counter[tuple[str, str]] = Counter()
    summary_state: dict[tuple[str, str], dict[str, Any]] = {}
    total_source = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_source += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            source_candidate = raw.get("candidate")
            display = display_by_source.get(source_candidate)
            if display is None:
                continue
            polarity = raw.get("polarity")
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected polarity on {path}:{line_number}")
            document = str(raw.get("document_index"))
            rollout = rollouts.get(document)
            if rollout is None:
                raise SystemExit(f"Context {path}:{line_number} has no rollout {document}")
            source = raw.get("source") or {}
            if str(source.get("id")) != rollout["prompt_id"]:
                raise SystemExit(
                    f"Context/rollout prompt mismatch on {path}:{line_number}: "
                    f"{source.get('id')!r} != {rollout['prompt_id']!r}"
                )
            item = compact_context(raw, rollout)
            key = (display, polarity)
            group_counts[key] += 1
            state = summary_state.setdefault(
                key,
                {
                    "tokens": Counter(),
                    "categories": Counter(),
                    "documents": set(),
                    "progresses": [],
                    "phases": Counter(),
                },
            )
            token = str(item["token"])
            state["tokens"][token] += 1
            state["categories"][rollout["category"]] += 1
            state["documents"].add(document)
            state["progresses"].append(float(item["progress"]))
            phase = "early" if item["progress"] < 1 / 3 else "middle" if item["progress"] < 2 / 3 else "late"
            state["phases"][phase] += 1
            rank = int(item.get("rank") or 0)
            if context_limit == 0 or rank <= context_limit:
                contexts[display][polarity].append(item)

    count_values = set(group_counts.values())
    expected_groups = len(source_candidates) * 2
    if len(group_counts) != expected_groups or len(count_values) != 1:
        raise SystemExit(
            f"Inconsistent context groups in {path}: groups={len(group_counts)}/{expected_groups}, "
            f"counts={sorted(count_values)}"
        )
    source_per_side = next(iter(count_values))
    embedded_per_side = source_per_side if context_limit == 0 else min(context_limit, source_per_side)
    summaries: dict[str, dict[str, Any]] = {
        display: {"positive": {}, "negative": {}} for display in source_candidates
    }
    for display, by_polarity in contexts.items():
        for polarity, items in by_polarity.items():
            items.sort(key=lambda item: int(item.get("rank") or 0))
            if len(items) != embedded_per_side:
                raise SystemExit(
                    f"Expected {embedded_per_side} embedded contexts for {display}/{polarity}, got {len(items)}"
                )
            state = summary_state[(display, polarity)]
            total = group_counts[(display, polarity)]
            summaries[display][polarity] = {
                "events": total,
                "unique_traces": len(state["documents"]),
                "median_progress": statistics.median(state["progresses"]),
                "phases": {phase: state["phases"].get(phase, 0) for phase in ("early", "middle", "late")},
                "top_tokens": [
                    {"token": token, "count": count, "share": count / total}
                    for token, count in state["tokens"].most_common(8)
                ],
                "top_categories": [
                    {"category": category, "count": count, "share": count / total}
                    for category, count in state["categories"].most_common(6)
                ],
            }
    return contexts, summaries, {
        "total_source": total_source,
        "source_per_side": source_per_side,
        "embedded_per_side": embedded_per_side,
    }


def compact_neighbor(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("token_id"),
        "token": raw.get("token"),
        "decoded": raw.get("decoded"),
        "cosine": raw.get("cosine"),
    }


def load_unembedding(
    path: Path, source_candidates: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    display_by_source = {source: display for display, source in source_candidates.items()}
    neighbors: dict[str, Any] = {}
    total = 0
    domain_max = 0.0
    spaces: set[str] = set()
    vocab_sizes: set[int] = set()
    list_sizes: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            raw = json.loads(line)
            source = str(raw.get("candidate"))
            spaces.add(str(raw.get("space")))
            vocab_sizes.add(int(raw.get("unembedding_vocab_rows_considered", 0)))
            nearest = raw.get("nearest_tokens") or []
            farthest = raw.get("farthest_tokens") or []
            list_sizes.update((len(nearest), len(farthest)))
            domain_max = max(domain_max, abs(float(raw.get("max_abs_token_cosine", 0))))
            display = display_by_source.get(source)
            if display is None:
                continue
            if display in neighbors:
                raise SystemExit(f"Duplicate neighbor record on {path}:{line_number}")
            neighbors[display] = {
                "vocab": raw.get("unembedding_vocab_rows_considered"),
                "mean": raw.get("unembedding_cosine_mean"),
                "std": raw.get("unembedding_cosine_std"),
                "max_abs": raw.get("max_abs_token_cosine"),
                "nearest_z": raw.get("nearest_token_z"),
                "farthest_z": raw.get("farthest_token_z"),
                "nearest_margin": raw.get("nearest_cosine_margin"),
                "farthest_margin": raw.get("farthest_cosine_margin"),
                "nearest": [compact_neighbor(item) for item in nearest],
                "farthest": [compact_neighbor(item) for item in farthest],
            }
    missing = sorted(set(source_candidates) - set(neighbors))
    if missing:
        raise SystemExit(f"Missing unembedding neighbors for {len(missing)} candidates")
    if len(spaces) != 1 or len(vocab_sizes) != 1 or len(list_sizes) != 1:
        raise SystemExit(f"Inconsistent unembedding metadata in {path}")
    return neighbors, {
        "total": total,
        "space": next(iter(spaces)),
        "vocab": next(iter(vocab_sizes)),
        "per_side": next(iter(list_sizes)),
        "domain_max": domain_max,
    }


def load_basis_alignment(low_dir: Path, medium_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to compare the low/medium direction banks") from exc

    alignment: dict[str, Any] = {}
    exact = 0
    stable_99 = 0
    stable_95 = 0
    sign_flips = 0
    minimum = 1.0
    left_exact = 0
    left_stable_99 = 0
    left_stable_95 = 0
    left_sign_flips = 0
    left_minimum = 1.0
    for layer in range(23):
        low_path = low_dir / "directions" / f"L{layer:02d}.npz"
        medium_path = medium_dir / "directions" / f"L{layer:02d}.npz"
        if not low_path.is_file() or not medium_path.is_file():
            raise SystemExit(f"Missing direction file for layer {layer}")
        with np.load(low_path) as low_data, np.load(medium_path) as medium_data:
            low_v = np.asarray(low_data["V"], dtype=np.float64)
            medium_v = np.asarray(medium_data["V"], dtype=np.float64)
            low_u = np.asarray(low_data["U"], dtype=np.float64)
            medium_u = np.asarray(medium_data["U"], dtype=np.float64)
            low_v /= np.linalg.norm(low_v, axis=0, keepdims=True).clip(min=1e-12)
            medium_v /= np.linalg.norm(medium_v, axis=0, keepdims=True).clip(min=1e-12)
            low_u /= np.linalg.norm(low_u, axis=0, keepdims=True).clip(min=1e-12)
            medium_u /= np.linalg.norm(medium_u, axis=0, keepdims=True).clip(min=1e-12)
            cosine = low_v.T @ medium_v
            absolute = np.abs(cosine)
            low_best = np.argmax(absolute, axis=1)
            medium_best = np.argmax(absolute, axis=0)
            left_cosine = low_u.T @ medium_u
            left_absolute = np.abs(left_cosine)
            low_left_best = np.argmax(left_absolute, axis=1)
            medium_left_best = np.argmax(left_absolute, axis=0)
            for sv0 in range(cosine.shape[0]):
                direct = float(cosine[sv0, sv0])
                direct_abs = abs(direct)
                left_direct = float(left_cosine[sv0, sv0])
                left_direct_abs = abs(left_direct)
                candidate = f"L{layer:02d}_SV{sv0:02d}"
                exact_slot = bool(np.array_equal(low_data["V"][:, sv0], medium_data["V"][:, sv0]))
                left_exact_slot = bool(
                    np.array_equal(low_data["U"][:, sv0], medium_data["U"][:, sv0])
                )
                alignment[candidate] = {
                    "cosine": direct,
                    "abs_cosine": direct_abs,
                    "exact": exact_slot,
                    "low_best_medium_sv": int(low_best[sv0]),
                    "low_best_abs_cosine": float(absolute[sv0, low_best[sv0]]),
                    "medium_best_low_sv": int(medium_best[sv0]),
                    "medium_best_abs_cosine": float(absolute[medium_best[sv0], sv0]),
                    "left_cosine": left_direct,
                    "left_abs_cosine": left_direct_abs,
                    "left_exact": left_exact_slot,
                    "low_best_medium_left_sv": int(low_left_best[sv0]),
                    "low_best_left_abs_cosine": float(
                        left_absolute[sv0, low_left_best[sv0]]
                    ),
                    "medium_best_low_left_sv": int(medium_left_best[sv0]),
                    "medium_best_left_abs_cosine": float(
                        left_absolute[medium_left_best[sv0], sv0]
                    ),
                }
                exact += int(exact_slot)
                stable_99 += int(direct_abs >= 0.99)
                stable_95 += int(direct_abs >= 0.95)
                sign_flips += int(direct < 0)
                minimum = min(minimum, direct_abs)
                left_exact += int(left_exact_slot)
                left_stable_99 += int(left_direct_abs >= 0.99)
                left_stable_95 += int(left_direct_abs >= 0.95)
                left_sign_flips += int(left_direct < 0)
                left_minimum = min(left_minimum, left_direct_abs)
    return alignment, {
        "slots": len(alignment),
        "exact_slots": exact,
        "stable_99": stable_99,
        "stable_95": stable_95,
        "sign_flips": sign_flips,
        "minimum_abs_cosine": minimum,
        "left_exact_slots": left_exact,
        "left_stable_99": left_stable_99,
        "left_stable_95": left_stable_95,
        "left_sign_flips": left_sign_flips,
        "left_minimum_abs_cosine": left_minimum,
    }


def compact_left_token_geometry(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None

    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": item.get("token_id"),
                "token": item.get("token"),
                "decoded": item.get("decoded"),
                "cosine": item.get("cosine"),
                "dot_product": item.get("dot_product"),
            }
            for item in items[:LEFT_TOKEN_NEIGHBORS_PER_SIDE]
        ]

    return {
        "vocab": raw.get("vocab_rows_considered"),
        "mean": raw.get("cosine_mean"),
        "std": raw.get("cosine_std"),
        "max_abs": raw.get("max_abs_token_cosine"),
        "nearest_z": raw.get("nearest_token_z"),
        "farthest_z": raw.get("farthest_token_z"),
        "nearest": compact(raw.get("nearest_tokens") or []),
        "farthest": compact(raw.get("farthest_tokens") or []),
    }


def load_left_enrichment_artifact(
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    artifact_dir = artifact_dir.resolve()
    if artifact_dir in _LEFT_ARTIFACT_CACHE:
        return _LEFT_ARTIFACT_CACHE[artifact_dir]
    metadata_path = artifact_dir / "left_singular_vectors_metadata.json"
    records_path = artifact_dir / "left_singular_vectors.jsonl"
    if not metadata_path.is_file() or not records_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records: dict[str, Any] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            layer = int(raw["layer"])
            sv0 = int(raw["sv_index_0"])
            candidate = f"L{layer:02d}_SV{sv0:02d}"
            if candidate in records:
                raise SystemExit(f"Duplicate left-vector record on {records_path}:{line_number}")
            records[candidate] = {
                "source": raw.get("reconstruction_method"),
                "actual_transport_gain": raw.get("actual_transport_gain"),
                "gain_over_stored_singular_value": raw.get(
                    "gain_over_stored_singular_value"
                ),
                "transport_vs_stored_u_cosine": raw.get(
                    "transport_vs_stored_u_cosine"
                ),
                "transport_vs_sigma_stored_u_relative_error": raw.get(
                    "transport_vs_sigma_stored_u_relative_error"
                ),
                "token_geometry": compact_left_token_geometry(
                    raw.get("left_token_geometry")
                ),
                "right_left_token_overlap": raw.get("right_left_token_overlap"),
            }
    _LEFT_ARTIFACT_CACHE[artifact_dir] = (metadata, records)
    return metadata, records


def load_left_singular_geometry(
    data_dir: Path, display_candidates: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize saved U vectors without model execution or J-Lens reconstruction."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to inspect saved left singular vectors") from exc

    directions_dir = data_dir / "directions"
    u_by_layer: dict[int, Any] = {}
    u_norms_by_layer: dict[int, Any] = {}
    singular_values_by_layer: dict[int, Any] = {}
    direction_paths: dict[int, Path] = {}
    fingerprint = hashlib.sha256()
    fingerprint.update(b"jlens-vs-bank-v1\0")

    def update_fingerprint(key: str, array: Any) -> None:
        contiguous = np.ascontiguousarray(array)
        fingerprint.update(key.encode("ascii") + b"\0")
        fingerprint.update(contiguous.dtype.str.encode("ascii") + b"\0")
        fingerprint.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
        )
        fingerprint.update(b"\0")
        fingerprint.update(contiguous.tobytes())

    for path in sorted(directions_dir.glob("L[0-9][0-9].npz")):
        layer = int(path.stem[1:])
        with np.load(path) as saved:
            if not {"U", "S", "V"}.issubset(saved.files):
                raise SystemExit(f"{path} must contain U, S, and V arrays")
            raw_u = np.asarray(saved["U"], dtype=np.float64)
            singular_values = np.asarray(saved["S"], dtype=np.float64)
            fingerprint_v = np.ascontiguousarray(saved["V"], dtype=np.float32)
            fingerprint_s = np.ascontiguousarray(saved["S"].reshape(-1), dtype=np.float32)
        if raw_u.ndim != 2 or singular_values.ndim != 1:
            raise SystemExit(f"Unexpected U/S shapes in {path}")
        if raw_u.shape[1] != singular_values.shape[0]:
            raise SystemExit(f"U/S direction count mismatch in {path}")
        u_norms = np.linalg.norm(raw_u, axis=0)
        u_by_layer[layer] = raw_u / u_norms.clip(min=1e-12)
        u_norms_by_layer[layer] = u_norms
        singular_values_by_layer[layer] = singular_values
        direction_paths[layer] = path
        fingerprint.update(f"L{layer:02d}\0".encode("ascii"))
        update_fingerprint("V", fingerprint_v)
        update_fingerprint("S", fingerprint_s)
    if not u_by_layer:
        raise SystemExit(f"No saved direction banks found under {directions_dir}")

    basis_fingerprint = fingerprint.hexdigest()
    enrichment_meta: dict[str, Any] | None = None
    enrichment_records: dict[str, Any] = {}
    enrichment_dirs = [data_dir, DEFAULT_PREDICTIVE_FINEWEB_DIR, DEFAULT_FINEWEB_DIR]
    for artifact_dir in dict.fromkeys(path.resolve() for path in enrichment_dirs):
        artifact = load_left_enrichment_artifact(artifact_dir)
        if artifact is None:
            continue
        artifact_metadata, artifact_records = artifact
        if artifact_metadata.get("basis_fingerprint_sha256") != basis_fingerprint:
            continue
        enrichment_records = artifact_records
        left_arrays_dir = artifact_dir / "left_directions"
        actual_left_by_layer: dict[int, Any] = {}
        for layer, saved_u in u_by_layer.items():
            left_path = left_arrays_dir / f"L{layer:02d}.npz"
            if not left_path.is_file():
                actual_left_by_layer = {}
                break
            with np.load(left_path) as left_saved:
                actual_u = np.asarray(left_saved["U"], dtype=np.float64)
            if actual_u.shape != saved_u.shape:
                raise SystemExit(f"Enriched left-vector shape mismatch in {left_path}")
            actual_norms = np.linalg.norm(actual_u, axis=0)
            actual_left_by_layer[layer] = actual_u / actual_norms.clip(min=1e-12)
        if len(actual_left_by_layer) == len(u_by_layer):
            u_by_layer = actual_left_by_layer
        enrichment_meta = {
            "artifact_dir": str(artifact_dir),
            "reconstruction": artifact_metadata.get("reconstruction"),
            "token_geometry": artifact_metadata.get("token_geometry"),
            "left_arrays_available": len(actual_left_by_layer) == len(u_by_layer),
        }
        break

    records: dict[str, Any] = {}
    max_u_norm_error = 0.0
    max_u_orthogonality_leakage = 0.0
    for layer, normalized_u in sorted(u_by_layer.items()):
        path = direction_paths[layer]
        with np.load(path) as saved:
            raw_v = np.asarray(saved["V"], dtype=np.float64)
        if raw_v.shape != normalized_u.shape:
            raise SystemExit(f"U/V shape mismatch in {path}")
        v_norms = np.linalg.norm(raw_v, axis=0)
        normalized_v = raw_v / v_norms.clip(min=1e-12)
        u_to_v = normalized_u.T @ normalized_v
        u_gram = normalized_u.T @ normalized_u
        np.fill_diagonal(u_gram, 0.0)
        u_gram_abs = np.abs(u_gram)
        within_best = np.argmax(u_gram_abs, axis=1)
        within_best_abs = u_gram_abs[np.arange(u_gram_abs.shape[0]), within_best]
        max_u_orthogonality_leakage = max(
            max_u_orthogonality_leakage, float(np.max(within_best_abs))
        )
        max_u_norm_error = max(
            max_u_norm_error,
            float(np.max(np.abs(u_norms_by_layer[layer] - 1.0))),
        )
        best_right = np.argmax(np.abs(u_to_v), axis=1)
        singular_values = singular_values_by_layer[layer]
        saved_spectral_energy = float(np.sum(np.square(singular_values)))

        adjacent: dict[str, tuple[int, Any]] = {}
        if layer - 1 in u_by_layer:
            adjacent["previous"] = (layer - 1, normalized_u.T @ u_by_layer[layer - 1])
        if layer + 1 in u_by_layer:
            adjacent["next"] = (layer + 1, normalized_u.T @ u_by_layer[layer + 1])

        for sv0 in range(normalized_u.shape[1]):
            candidate = f"L{layer:02d}_SV{sv0:02d}"
            if candidate not in display_candidates:
                continue
            right_sv0 = int(best_right[sv0])
            record: dict[str, Any] = {
                "source": "saved_svd_u",
                "singular_value": float(singular_values[sv0]),
                "saved_spectral_energy_fraction": (
                    float(singular_values[sv0] ** 2 / saved_spectral_energy)
                    if saved_spectral_energy > 0
                    else 0.0
                ),
                "u_norm": float(u_norms_by_layer[layer][sv0]),
                "v_norm": float(v_norms[sv0]),
                "paired_u_v_cosine": float(u_to_v[sv0, sv0]),
                "best_right_sv0": right_sv0,
                "best_right_cosine": float(u_to_v[sv0, right_sv0]),
                "best_right_abs_cosine": float(abs(u_to_v[sv0, right_sv0])),
                "right_bank_projection_fraction": float(
                    np.sum(np.square(u_to_v[sv0]))
                ),
                "max_other_u_abs_cosine": float(within_best_abs[sv0]),
                "max_other_u_sv0": int(within_best[sv0]),
            }
            for side, (other_layer, cosine) in adjacent.items():
                best = int(np.argmax(np.abs(cosine[sv0])))
                record[f"{side}_layer"] = other_layer
                record[f"{side}_best_sv0"] = best
                record[f"{side}_best_cosine"] = float(cosine[sv0, best])
                record[f"{side}_best_abs_cosine"] = float(abs(cosine[sv0, best]))
                record[f"{side}_same_sv_cosine"] = (
                    float(cosine[sv0, sv0]) if sv0 < cosine.shape[1] else None
                )
            records[candidate] = record

    missing = sorted(display_candidates - records.keys())
    if missing:
        raise SystemExit(
            f"Saved left singular vectors are missing for {len(missing)} displayed candidates"
        )
    if enrichment_meta:
        for candidate, record in records.items():
            enriched = enrichment_records.get(candidate)
            if enriched:
                record.update(enriched)

    token_geometry_available = bool(
        enrichment_meta
        and (enrichment_meta.get("token_geometry") or {}).get("available")
    )
    return records, {
        "source": "directions/LXX.npz:U",
        "basis_fingerprint_sha256": basis_fingerprint,
        "candidate_count": len(records),
        "forward_passes": 0,
        "jlens_evaluations": 0,
        "max_u_norm_error": max_u_norm_error,
        "max_u_orthogonality_leakage": max_u_orthogonality_leakage,
        "token_geometry_available": token_geometry_available,
        "embedded_token_neighbors_per_side": (
            LEFT_TOKEN_NEIGHBORS_PER_SIDE if token_geometry_available else 0
        ),
        "enrichment": enrichment_meta,
    }


FINEWEB_BROAD_FIELDS = {
    "candidate",
    "layer",
    "sv_index_0",
    "n_tokens",
    "n_documents",
    "mean_activation",
    "mean_abs_activation",
    "std_activation",
    "positive_rate",
    "max_activation",
    "min_activation",
    "mean_abs_cosine",
    "top1_abs_rate",
    "top5_abs_rate",
    "doc_top5_presence_rate",
    "mean_document_peak_abs",
    "rank_global_mean_abs_cosine",
}
FINEWEB_SELECTIVE_FIELDS = FINEWEB_BROAD_FIELDS | {
    "token_sample_n",
    "median_activation",
    "robust_activation_scale",
    "stable_excess_kurtosis",
    "effective_support_fraction",
    "tail_selectivity_score",
    "selected_tail_polarity",
    "rank_global_tail_selectivity",
    "selected_tail_q999_robust_z",
    "selected_tail_top0_1pct_energy_share",
    "selected_tail_doc_count_z5",
    "selected_tail_doc_rate_z5",
    "selected_tail_top_context_largest_center_token_share",
}


def compact_fineweb_ranking(source: dict[str, Any], mode: str) -> dict[str, Any]:
    row = display_ranking_row(source)
    fields = FINEWEB_SELECTIVE_FIELDS if mode == "selective" else FINEWEB_BROAD_FIELDS
    return {key: row.get(key) for key in fields}


def load_fineweb_rankings(
    path: Path,
    display_candidates: set[str],
    rank_key: str,
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
    selected: dict[str, dict[str, Any]] = {}
    source_by_display: dict[str, str] = {}
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate", "layer", "sv_index_0", rank_key}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} is missing required FineWeb ranking columns")
        for line_number, raw in enumerate(reader, 2):
            total += 1
            try:
                row = {
                    key: (value if key in TEXT_FIELDS else number(value))
                    for key, value in raw.items()
                }
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric field on {path}:{line_number}: {exc}") from exc
            expected = f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']) + 1:02d}"
            if row["candidate"] != expected:
                raise SystemExit(
                    f"Unexpected source candidate {row['candidate']!r}; expected {expected!r}"
                )
            display = zero_based_candidate(row)
            if display not in display_candidates:
                continue
            selected[display] = compact_fineweb_ranking(row, mode)
            source_by_display[display] = str(row["candidate"])

    missing = sorted(display_candidates - selected.keys())
    if missing:
        raise SystemExit(f"FineWeb rankings are missing {len(missing)} dashboard candidates")
    return selected, source_by_display, total


def compact_fineweb_context(raw: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("source") or {}
    return {
        "polarity": raw.get("polarity"),
        "rank": raw.get("rank_within_polarity"),
        "activation": raw.get("activation"),
        "cosine": raw.get("cosine_activation"),
        "token": raw.get("token", ""),
        "marked": raw.get("context_marked") or raw.get("context") or "",
        "document": raw.get("document_index"),
        "position": raw.get("token_position"),
        "url": source.get("url"),
        "date": source.get("date"),
        "dump": source.get("dump"),
    }


def load_fineweb_contexts(
    path: Path,
    source_candidates: dict[str, str],
    context_limit: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    if context_limit < 0:
        raise SystemExit("--fineweb-contexts-per-side must be 0 or greater")
    display_by_source = {source: display for display, source in source_candidates.items()}
    contexts = {
        display: {"positive": [], "negative": []} for display in source_candidates
    }
    group_counts: Counter[tuple[str, str]] = Counter()
    total_source = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_source += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            display = display_by_source.get(raw.get("candidate"))
            if display is None:
                continue
            polarity = raw.get("polarity")
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected FineWeb polarity on {path}:{line_number}")
            key = (display, polarity)
            group_counts[key] += 1
            rank = int(raw.get("rank_within_polarity") or 0)
            if context_limit == 0 or rank <= context_limit:
                contexts[display][polarity].append(compact_fineweb_context(raw))

    expected_groups = len(source_candidates) * 2
    count_values = set(group_counts.values())
    if len(group_counts) != expected_groups or len(count_values) != 1:
        raise SystemExit(
            f"Inconsistent FineWeb context groups in {path}: "
            f"groups={len(group_counts)}/{expected_groups}, counts={sorted(count_values)}"
        )
    source_per_side = next(iter(count_values))
    embedded_per_side = source_per_side if context_limit == 0 else min(
        context_limit, source_per_side
    )
    for display, by_polarity in contexts.items():
        for polarity, items in by_polarity.items():
            items.sort(key=lambda item: int(item.get("rank") or 0))
            if len(items) != embedded_per_side:
                raise SystemExit(
                    f"Expected {embedded_per_side} FineWeb contexts for "
                    f"{display}/{polarity}, got {len(items)}"
                )
    return contexts, {
        "source_records": total_source,
        "source_per_side": source_per_side,
        "embedded_per_side": embedded_per_side,
    }


def validate_fineweb_reference_basis(
    low_dir: Path,
    fineweb_dir: Path,
    selectivity_dir: Path,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to validate the FineWeb direction bank") from exc

    slots = 0
    for layer in range(23):
        paths = {
            "CoT low": low_dir / "directions" / f"L{layer:02d}.npz",
            "FineWeb broad": fineweb_dir / "directions" / f"L{layer:02d}.npz",
            "FineWeb selective": selectivity_dir / "directions" / f"L{layer:02d}.npz",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SystemExit("Missing direction file(s): " + ", ".join(missing))
        with (
            np.load(paths["CoT low"]) as low_data,
            np.load(paths["FineWeb broad"]) as broad_data,
            np.load(paths["FineWeb selective"]) as selective_data,
        ):
            low_v = np.asarray(low_data["V"])
            broad_v = np.asarray(broad_data["V"])
            selective_v = np.asarray(selective_data["V"])
            if low_v.shape != broad_v.shape or low_v.shape != selective_v.shape:
                raise SystemExit(f"Direction-bank shape mismatch at layer {layer}")
            if not np.array_equal(low_v, broad_v) or not np.array_equal(low_v, selective_v):
                raise SystemExit(
                    "FineWeb evidence cannot be joined by layer/SV index: "
                    f"the stored V banks differ from CoT low at layer {layer}"
                )
            low_s = np.asarray(low_data["S"])
            broad_s = np.asarray(broad_data["S"])
            selective_s = np.asarray(selective_data["S"])
            if not np.array_equal(low_s, broad_s) or not np.array_equal(low_s, selective_s):
                raise SystemExit(
                    "FineWeb evidence cannot be joined by SVD rank: "
                    f"the stored singular values differ from CoT low at layer {layer}"
                )
            slots += low_v.shape[1]
    return {
        "slots": slots,
        "exact_slots": slots,
        "reference": "FineWeb broad = FineWeb selective = CoT low",
    }


def build_fineweb_payload(
    fineweb_dir: Path,
    selectivity_dir: Path,
    low_dir: Path,
    display_candidates: set[str],
    context_limit: int,
    expected_model: str | None,
    expected_layers: list[int],
    expected_k: int | None,
) -> dict[str, Any]:
    source_specs = {
        "broad": (fineweb_dir, "sv_rankings.csv", BROAD_RANK),
        "selective": (selectivity_dir, "selectivity_rankings.csv", SELECTIVE_RANK),
    }
    metadata_by_source: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    source_maps: dict[str, dict[str, str]] = {}
    for mode, (data_dir, rankings_name, rank_key) in source_specs.items():
        required = (
            data_dir / "metadata.json",
            data_dir / rankings_name,
            data_dir / "top_contexts.jsonl",
        )
        for path in required:
            if not path.is_file():
                raise SystemExit(f"Missing required FineWeb input: {path}")
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata_by_source[mode] = metadata
        rankings, source_map, total_candidates = load_fineweb_rankings(
            data_dir / rankings_name, display_candidates, rank_key, mode
        )
        contexts, context_meta = load_fineweb_contexts(
            data_dir / "top_contexts.jsonl", source_map, context_limit
        )
        source_maps[mode] = source_map
        sources[mode] = {
            "meta": {
                "primary_rank": rank_key,
                "total_candidates": total_candidates,
                **context_meta,
            },
            "rankings": rankings,
            "contexts": contexts,
        }

    broad_meta = metadata_by_source["broad"]
    selective_meta = metadata_by_source["selective"]
    comparable_fields = ("model", "dataset", "dataset_config", "layers", "k")
    mismatches = [
        field
        for field in comparable_fields
        if broad_meta.get(field) != selective_meta.get(field)
    ]
    if mismatches:
        raise SystemExit("FineWeb scan metadata differs for " + ", ".join(mismatches))
    if expected_model and broad_meta.get("model") != expected_model:
        raise SystemExit("FineWeb and CoT scans use different models")
    if broad_meta.get("layers") != expected_layers or broad_meta.get("k") != expected_k:
        raise SystemExit("FineWeb and CoT scans use different layer/SV coverage")
    if source_maps["broad"] != source_maps["selective"]:
        raise SystemExit("FineWeb scan outputs disagree on candidate numbering")
    basis_meta = validate_fineweb_reference_basis(low_dir, fineweb_dir, selectivity_dir)
    return {
        "meta": {
            "model": broad_meta.get("model"),
            "dataset": broad_meta.get("dataset"),
            "dataset_config": broad_meta.get("dataset_config"),
            "documents": broad_meta.get("documents_processed"),
            "tokens": broad_meta.get("content_tokens_processed"),
            "embedded_candidates": len(display_candidates),
            "display_sv_numbering": "zero_based",
            "basis": basis_meta,
        },
        "sources": sources,
    }


def build_effort_payload(
    effort: str,
    data_dir: Path,
    top_n: int,
    context_limit: int,
) -> dict[str, Any]:
    required = (
        data_dir / "metadata.json",
        data_dir / "rollouts.jsonl",
        data_dir / "sv_rankings.csv",
        data_dir / "top_contexts.jsonl",
        data_dir / "unembedding_neighbors.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("reasoning_effort") != effort:
        raise SystemExit(
            f"{data_dir} reports effort {metadata.get('reasoning_effort')!r}, expected {effort!r}"
        )
    rollouts, rollout_meta = load_rollouts(data_dir / "rollouts.jsonl")
    rankings, source_by_display, cohorts = load_rankings(data_dir / "sv_rankings.csv", top_n)
    selected = set(cohorts["broad"]) | set(cohorts["selective"])
    selected_sources = {candidate: source_by_display[candidate] for candidate in selected}
    contexts, context_summaries, context_meta = load_contexts(
        data_dir / "top_contexts.jsonl", selected_sources, rollouts, context_limit
    )
    unembedding, unembedding_meta = load_unembedding(
        data_dir / "unembedding_neighbors.jsonl", selected_sources
    )
    overlap = len(set(cohorts["broad"]) & set(cohorts["selective"]))
    return {
        "meta": {
            "effort": effort,
            "model": metadata.get("model"),
            "prompts_source": metadata.get("prompts_source"),
            "prompts": metadata.get("prompts"),
            "jobs": metadata.get("jobs"),
            "rollouts": metadata.get("analysis_rollouts_scanned"),
            "analysis_tokens": metadata.get("analysis_tokens_scanned"),
            "max_new_tokens": metadata.get("max_new_tokens"),
            "sampling": metadata.get("sampling"),
            "temperature": metadata.get("temperature"),
            "top_p": metadata.get("top_p"),
            "layers": metadata.get("layers", []),
            "k": metadata.get("k"),
            "total_candidates": len(rankings),
            "embedded_candidates": len(selected),
            "cohort_size": len(cohorts["broad"]),
            "cohort_overlap": overlap,
            "capped_rollouts": rollout_meta["capped"],
            "rollouts_with_final": rollout_meta["with_final"],
            "categories": len(rollout_meta["category_counts"]),
            "source_contexts_per_side": context_meta["source_per_side"],
            "embedded_contexts_per_side": context_meta["embedded_per_side"],
            "source_context_records": context_meta["total_source"],
            "unembedding_candidates": unembedding_meta["total"],
            "unembedding_vocab": unembedding_meta["vocab"],
            "unembedding_neighbors_per_side": unembedding_meta["per_side"],
            "unembedding_domain_max": unembedding_meta["domain_max"],
            "channel_filter": metadata.get("channel_filter"),
            "capture_method": metadata.get("capture_method"),
            "display_sv_numbering": "zero_based",
        },
        "cohorts": cohorts,
        "rankings": rankings,
        "contexts": contexts,
        "context_summaries": context_summaries,
        "rollouts": rollouts,
        "unembedding": unembedding,
        "rollout_meta": {
            "category_counts": rollout_meta["category_counts"],
            "difficulty_counts": rollout_meta["difficulty_counts"],
            "prompt_ids": rollout_meta["prompt_ids"],
        },
    }


def build_payload(
    low_dir: Path,
    medium_dir: Path,
    fineweb_dir: Path,
    fineweb_selectivity_dir: Path,
    top_n: int,
    context_limit: int,
    fineweb_context_limit: int,
) -> dict[str, Any]:
    low = build_effort_payload("low", low_dir, top_n, context_limit)
    medium = build_effort_payload("medium", medium_dir, top_n, context_limit)
    if low["rollout_meta"]["prompt_ids"] != medium["rollout_meta"]["prompt_ids"]:
        raise SystemExit("Low and medium runs do not contain the same ordered prompt set")
    if low["meta"]["model"] != medium["meta"]["model"]:
        raise SystemExit("Low and medium runs use different models")
    alignment, alignment_meta = load_basis_alignment(low_dir, medium_dir)
    all_candidates = set()
    for payload in (low, medium):
        all_candidates.update(payload["cohorts"]["broad"])
        all_candidates.update(payload["cohorts"]["selective"])
    # Keep full metric rows for every direction reachable in any dashboard
    # cohort. This preserves cross-effort comparisons without embedding two
    # redundant 1,472-row tables.
    for payload in (low, medium):
        payload["rankings"] = [
            row for row in payload["rankings"] if row["candidate"] in all_candidates
        ]
    low_left, low_left_meta = load_left_singular_geometry(low_dir, all_candidates)
    medium_left, medium_left_meta = load_left_singular_geometry(medium_dir, all_candidates)
    low["left_singular"] = low_left
    medium["left_singular"] = medium_left
    low["meta"]["left_singular"] = low_left_meta
    medium["meta"]["left_singular"] = medium_left_meta
    display_alignment = {
        candidate: alignment[candidate] for candidate in sorted(all_candidates)
    }
    fineweb = build_fineweb_payload(
        fineweb_dir,
        fineweb_selectivity_dir,
        low_dir,
        all_candidates,
        fineweb_context_limit,
        low["meta"]["model"],
        low["meta"]["layers"],
        low["meta"]["k"],
    )
    return {
        "default_effort": "low",
        "default_lens": "broad",
        "meta": {
            "model": low["meta"]["model"],
            "prompt_count": len(low["rollout_meta"]["prompt_ids"]),
            "display_sv_numbering": "zero_based",
            "candidate_union": len(all_candidates),
            "basis_alignment": alignment_meta,
        },
        "efforts": {"low": low, "medium": medium},
        "alignment": display_alignment,
        "fineweb": fineweb,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Cross-Corpus Singular Vector Atlas</title>
  <style>
    :root {
      --ink: #17212b;
      --ink-2: #344353;
      --paper: #eeeae2;
      --surface: #fffdf8;
      --surface-2: #f7f4ed;
      --line: #d5d0c5;
      --line-dark: #afa99e;
      --muted: #6d7680;
      --teal: #167772;
      --teal-soft: #dcecea;
      --violet: #68539a;
      --violet-soft: #e9e3f2;
      --orange: #c55f35;
      --orange-soft: #f5e4da;
      --blue: #376a96;
      --blue-soft: #e0eaf2;
      --effort: var(--teal);
      --effort-soft: var(--teal-soft);
      --shadow: 0 14px 42px rgba(31, 40, 48, .09);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --serif: Iowan Old Style, Baskerville, Georgia, serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 14px/1.5 var(--sans); }
    body[data-effort="medium"] { --effort: var(--violet); --effort-soft: var(--violet-soft); }
    button, input, select { font: inherit; }
    button, select { cursor: pointer; }
    a { color: inherit; }
    .skip-link { position: fixed; left: 12px; top: -60px; z-index: 80; background: var(--ink); color: white; padding: 8px 12px; }
    .skip-link:focus { top: 12px; }
    .eyebrow { margin: 0 0 7px; color: #e69264; font-size: 9px; font-weight: 850; letter-spacing: .18em; text-transform: uppercase; }

    .masthead { color: #f7f4ec; background: #121b23; border-bottom: 4px solid var(--orange); }
    .masthead-inner { width: min(1680px, 94vw); min-height: 164px; margin: 0 auto; display: flex; align-items: flex-end; justify-content: space-between; gap: 42px; padding: 35px 0 29px; }
    .brand { display: flex; gap: 17px; align-items: flex-start; min-width: 0; }
    .brand-mark { display: grid; place-items: center; width: 48px; height: 48px; flex: 0 0 auto; border: 1px solid rgba(255,255,255,.28); color: #f0b38e; font: 750 13px/1 var(--mono); }
    h1 { margin: 0; font: 500 clamp(30px, 4.2vw, 49px)/1.02 var(--serif); letter-spacing: -.025em; }
    h1 em { color: #9dcac4; font-weight: 400; }
    .subtitle { max-width: 760px; margin: 11px 0 0; color: #b8c2c8; font-size: 12px; }
    .dataset-facts { display: flex; justify-content: flex-end; gap: 22px; flex-wrap: wrap; padding-bottom: 3px; }
    .fact { min-width: 88px; }
    .fact b { display: block; color: white; font: 650 18px/1.15 var(--mono); }
    .fact span { color: #929ea6; font-size: 9px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }

    .effort-bar { background: #e5e0d7; border-bottom: 1px solid var(--line-dark); }
    .effort-switch { width: min(1680px, 94vw); margin: 0 auto; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; border-left: 1px solid var(--line-dark); border-right: 1px solid var(--line-dark); background: var(--line-dark); }
    .effort-tab { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 14px; align-items: center; min-width: 0; border: 0; background: #e8e4dc; color: var(--ink-2); padding: 15px 18px; text-align: left; }
    .effort-tab:hover { background: #f2efe8; }
    .effort-tab[aria-selected="true"] { position: relative; z-index: 1; background: var(--surface); box-shadow: inset 0 -4px 0 var(--effort); }
    .effort-index { display: grid; place-items: center; width: 30px; height: 30px; border: 1px solid var(--line-dark); border-radius: 50%; color: var(--muted); font: 750 9px/1 var(--mono); }
    .effort-tab[aria-selected="true"] .effort-index { border-color: var(--effort); background: var(--effort-soft); color: var(--effort); }
    .effort-tab b { display: block; color: var(--ink); font: 650 14px/1.2 var(--serif); }
    .effort-tab small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; }
    .effort-stat { color: var(--muted); font: 650 10px/1.4 var(--mono); text-align: right; white-space: nowrap; }

    .source-bar { background: #121b23; border-bottom: 1px solid var(--line-dark); }
    .source-bar-inner { width: min(1680px, 94vw); margin: 0 auto; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; padding: 10px 0; }
    .source-switch { display: inline-flex; gap: 1px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.16); }
    .source-tab { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 10px; align-items: center; border: 0; background: #1c2731; color: #b8c2c8; padding: 8px 16px; text-align: left; }
    .source-tab:hover { background: #24313d; }
    .source-tab[aria-selected="true"] { background: var(--orange); color: #fff; }
    .source-index { display: grid; place-items: center; width: 22px; height: 22px; border: 1px solid rgba(255,255,255,.3); border-radius: 50%; font: 750 8px/1 var(--mono); }
    .source-tab b { display: block; font: 650 12px/1.15 var(--serif); }
    .source-tab small { display: block; margin-top: 1px; font: 650 9px/1.3 var(--mono); opacity: .82; }
    .source-note { margin: 0; color: #929ea6; font-size: 10px; max-width: 720px; }

    .toolbar { position: sticky; top: 0; z-index: 30; border-bottom: 1px solid var(--line); background: rgba(238,234,226,.96); backdrop-filter: blur(12px); }
    .toolbar-inner { width: min(1680px, 94vw); margin: 0 auto; display: grid; grid-template-columns: 260px minmax(210px, 1.2fr) 145px minmax(165px, .65fr) 120px auto; gap: 10px; align-items: end; padding: 12px 0; }
    .control { display: block; min-width: 0; color: var(--muted); font-size: 8px; font-weight: 850; letter-spacing: .11em; text-transform: uppercase; }
    .control input, .control select { width: 100%; height: 38px; margin-top: 4px; border: 1px solid var(--line-dark); border-radius: 2px; outline: 0; background: var(--surface); color: var(--ink); padding: 0 10px; text-transform: none; letter-spacing: normal; font-size: 11px; }
    .control input:focus, .control select:focus { border-color: var(--effort); box-shadow: 0 0 0 3px color-mix(in srgb, var(--effort) 14%, transparent); }
    .lens-control > span { display: block; margin-bottom: 4px; }
    .lens-switch { display: grid; grid-template-columns: repeat(2, 1fr); height: 38px; border: 1px solid var(--line-dark); background: var(--line); gap: 1px; }
    .lens-button { border: 0; background: var(--surface); color: var(--muted); padding: 0 10px; font-size: 10px; font-weight: 750; }
    .lens-button[aria-pressed="true"] { background: var(--effort-soft); color: var(--effort); }
    .search-wrap { position: relative; }

    .token-finder { border-bottom: 1px solid var(--line); background: var(--surface-2); }
    .token-finder-inner { width: min(1680px, 94vw); margin: 0 auto; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; padding: 10px 0; }
    .token-field { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 8px; font-weight: 850; letter-spacing: .11em; text-transform: uppercase; white-space: nowrap; }
    .token-wrap { position: relative; }
    .token-field input { width: 240px; height: 34px; border: 1px solid var(--line-dark); border-radius: 2px; outline: 0; background: var(--surface); color: var(--ink); padding: 0 10px; text-transform: none; letter-spacing: normal; font: 12px/1 var(--mono); }
    .token-field input:focus { border-color: var(--effort); box-shadow: 0 0 0 3px color-mix(in srgb, var(--effort) 14%, transparent); }
    .token-summary { margin: 0; color: var(--muted); font-size: 11px; min-width: 0; }
    .token-summary.active { color: var(--ink-2); }
    .token-summary code { font: 700 11px/1 var(--mono); background: var(--effort-soft); color: var(--effort); padding: 2px 6px; border-radius: 3px; }
    .token-clear { border: 1px solid var(--line-dark); background: var(--surface); color: var(--ink-2); height: 30px; padding: 0 12px; border-radius: 2px; font-size: 10px; font-weight: 750; }
    .token-clear:hover { background: var(--surface-2); }
    .rank-token { display: flex; align-items: center; gap: 6px; margin-top: 6px; color: var(--muted); font: 8px/1.25 var(--mono); }
    .rank-token code { font: 700 9px/1 var(--mono); color: var(--ink-2); }
    .token-hit { display: inline-flex; align-items: center; gap: 4px; font: 700 8px/1 var(--mono); padding: 2px 5px; border-radius: 3px; text-transform: uppercase; white-space: nowrap; }
    .token-hit code { font: inherit; background: none; padding: 0; color: inherit; }
    .token-hit.positive { background: var(--teal-soft); color: var(--teal); }
    .token-hit.negative { background: var(--orange-soft); color: var(--orange); }
    .search-wrap input { padding-right: 42px; }
    .shortcut { position: absolute; right: 8px; bottom: 8px; border: 1px solid var(--line); border-radius: 3px; background: var(--surface-2); color: var(--muted); padding: 1px 6px; font: 9px/1.45 var(--mono); }
    .reset { height: 38px; border: 1px solid var(--line-dark); background: transparent; color: var(--ink-2); padding: 0 13px; font-size: 10px; font-weight: 800; }
    .reset:hover { background: var(--surface); }

    .shell { width: min(1680px, 94vw); margin: 0 auto; padding: 24px 0 68px; }
    .dashboard-grid { display: grid; grid-template-columns: minmax(315px, 380px) minmax(0, 1fr); gap: 22px; align-items: start; }
    .panel { border: 1px solid var(--line); background: var(--surface); box-shadow: var(--shadow); }
    .ranking-panel { position: sticky; top: 82px; height: calc(100vh - 106px); min-height: 520px; display: flex; flex-direction: column; }
    .panel-head { border-bottom: 1px solid var(--line); padding: 18px 18px 14px; }
    .panel-head-row { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .panel h2, .detail-title { margin: 0; font: 500 23px/1.1 var(--serif); }
    .count { color: var(--muted); font: 10px/1.2 var(--mono); }
    .microcopy { margin: 7px 0 0; color: var(--muted); font-size: 10px; }
    .ranking-list { overflow: auto; overscroll-behavior: contain; }
    .rank-row { width: 100%; display: grid; grid-template-columns: 43px minmax(0, 1fr); gap: 10px; border: 0; border-bottom: 1px solid #e7e2d8; background: transparent; color: var(--ink); padding: 12px 14px 12px 11px; text-align: left; }
    .rank-row:hover { background: var(--surface-2); }
    .rank-row.active { background: var(--effort-soft); box-shadow: inset 4px 0 0 var(--effort); }
    .rank-number { padding-top: 2px; color: var(--muted); font: 10px/1.2 var(--mono); text-align: right; }
    .rank-row.active .rank-number { color: var(--effort); font-weight: 850; }
    .rank-topline { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
    .candidate { font: 750 12px/1.3 var(--mono); }
    .layer-note { color: var(--muted); font-size: 9px; white-space: nowrap; }
    .rank-measure { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 6px; color: var(--muted); font-size: 9px; }
    .rank-measure b { color: var(--ink-2); font: 650 9px/1 var(--mono); }
    .mini-track { display: block; height: 3px; margin-top: 7px; overflow: hidden; background: #e3ded4; }
    .mini-track i { display: block; height: 100%; background: var(--effort); }
    .rank-compare { display: flex; justify-content: space-between; gap: 8px; margin-top: 5px; color: var(--muted); font: 8px/1.25 var(--mono); }
    .tail-pill, .status-pill { display: inline-block; margin-left: 5px; border: 1px solid var(--line); border-radius: 99px; background: var(--surface); color: var(--orange); padding: 2px 5px; font: 750 7px/1 var(--mono); text-transform: uppercase; vertical-align: 1px; }
    .status-pill { margin: 0; color: var(--effort); }
    .status-pill.warn { color: var(--orange); }
    .empty { padding: 30px 20px; color: var(--muted); text-align: center; }

    .detail-panel { min-width: 0; }
    .detail-hero { padding: clamp(22px, 3vw, 34px); border-bottom: 1px solid var(--line); background: linear-gradient(125deg, #fffdf8, #f6f2e9); }
    .detail-nav { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
    .detail-rank { color: var(--orange); font-size: 9px; font-weight: 850; letter-spacing: .13em; text-transform: uppercase; }
    .nav-buttons { display: flex; gap: 6px; }
    .icon-button, .copy-button { height: 33px; border: 1px solid var(--line-dark); background: var(--surface); color: var(--ink-2); padding: 0 10px; font-size: 10px; }
    .icon-button { width: 35px; padding: 0; font-size: 15px; }
    .icon-button:disabled { cursor: default; opacity: .35; }
    .icon-button:hover:not(:disabled), .copy-button:hover { border-color: var(--effort); color: var(--effort); }
    .title-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
    .detail-title { font-size: clamp(29px, 4vw, 43px); }
    .id-chip { border-radius: 99px; background: var(--effort-soft); color: var(--effort); padding: 4px 9px; font: 750 9px/1.2 var(--mono); }
    .detail-summary { max-width: 940px; margin: 10px 0 0; color: var(--muted); font-size: 12px; }
    .detail-summary b { color: var(--ink-2); }
    .metric-grid { display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr)); gap: 1px; margin-top: 22px; border: 1px solid var(--line); background: var(--line); }
    .metric-card { min-width: 0; background: rgba(255,253,248,.88); padding: 11px; }
    .metric-card span { display: block; min-height: 27px; color: var(--muted); font-size: 8px; font-weight: 850; letter-spacing: .065em; text-transform: uppercase; }
    .metric-card b { display: block; overflow: hidden; margin-top: 4px; color: var(--ink); font: 650 clamp(13px, 1.4vw, 17px)/1.15 var(--mono); text-overflow: ellipsis; }
    .activation-profile { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 25px; align-items: center; margin-top: 18px; }
    .profile-labels { display: flex; justify-content: space-between; margin-bottom: 5px; color: var(--muted); font: 8px/1.2 var(--mono); }
    .axis { position: relative; height: 9px; border-radius: 99px; background: linear-gradient(90deg, #edd8cf, #e6e2d8 50%, #d7e9e4); }
    .zero-marker, .mean-marker { position: absolute; top: -5px; width: 1px; height: 19px; background: var(--ink-2); }
    .mean-marker { width: 3px; background: var(--effort); }
    .profile-stats { display: flex; justify-content: space-between; gap: 12px; border-left: 1px solid var(--line); padding-left: 20px; }
    .profile-stats span { color: var(--muted); font-size: 9px; }
    .profile-stats b { display: block; margin-top: 2px; color: var(--effort); font: 650 15px/1.2 var(--mono); }

    .comparison { margin-top: 19px; border: 1px solid var(--line); background: var(--surface); }
    .comparison-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; border-bottom: 1px solid var(--line); padding: 12px 14px; }
    .comparison-head h3 { margin: 0; font: 550 18px/1.15 var(--serif); }
    .comparison-head p { margin: 4px 0 0; color: var(--muted); font-size: 9px; }
    .alignment-badge { flex: 0 0 auto; border-radius: 99px; background: var(--teal-soft); color: var(--teal); padding: 5px 9px; font: 750 9px/1 var(--mono); }
    .alignment-badge.caution { background: #f4ead4; color: #9b681b; }
    .alignment-badge.warn { background: var(--orange-soft); color: var(--orange); }
    .effort-compare-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; background: var(--line); }
    .effort-compare-card { background: #faf8f2; padding: 13px 14px; }
    .effort-compare-card.current { background: var(--effort-soft); }
    .effort-compare-title { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 9px; font-weight: 800; text-transform: uppercase; }
    .compare-values { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 10px; }
    .compare-value span { display: block; min-height: 22px; color: var(--muted); font-size: 8px; text-transform: uppercase; }
    .compare-value b { display: block; color: var(--ink-2); font: 650 11px/1.2 var(--mono); }
    .alignment-note { margin: 0; border-top: 1px solid var(--line); padding: 9px 14px; color: var(--muted); font-size: 9px; }
    .alignment-note strong { color: var(--ink-2); }

    .tail-profile { margin-top: 18px; border: 1px solid var(--line); background: var(--surface); }
    .tail-profile-head { border-bottom: 1px solid var(--line); padding: 12px 14px; }
    .tail-profile-head h3 { margin: 0; font: 550 18px/1.15 var(--serif); }
    .tail-profile-head p { margin: 4px 0 0; color: var(--muted); font-size: 9px; }
    .tail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: var(--line); }
    .tail-side { min-width: 0; background: #faf8f2; padding: 13px 14px; }
    .tail-side.selected { background: var(--orange-soft); box-shadow: inset 0 3px 0 var(--orange); }
    .tail-side-head { display: flex; justify-content: space-between; gap: 10px; }
    .tail-side-head h4 { margin: 0; font: 700 10px/1.2 var(--mono); }
    .tail-side-head span { color: var(--muted); font: 8px/1.2 var(--mono); }
    .tail-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px 13px; margin-top: 11px; }
    .tail-stat span { display: block; min-height: 20px; color: var(--muted); font-size: 7px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
    .tail-stat b { display: block; color: var(--ink-2); font: 650 11px/1.2 var(--mono); }

    .metric-details { margin-top: 16px; }
    .metric-details summary { width: max-content; color: var(--effort); cursor: pointer; font-size: 10px; font-weight: 800; }
    .metric-groups { display: grid; gap: 14px; margin-top: 12px; }
    .metric-group h4 { margin: 0 0 5px; color: var(--muted); font-size: 8px; letter-spacing: .1em; text-transform: uppercase; }
    .all-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0 22px; border: 1px solid var(--line); background: var(--surface); padding: 11px 13px; }
    .all-metric { display: flex; justify-content: space-between; gap: 9px; border-bottom: 1px dotted var(--line); padding: 5px 0; }
    .all-metric span { min-width: 0; color: var(--muted); font-size: 9px; }
    .all-metric b { flex: 0 0 auto; font: 600 9px/1.4 var(--mono); }

    .trace-section { padding: clamp(22px, 3vw, 34px); border-bottom: 1px solid var(--line); background: #fbfaf6; }
    .section-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 22px; }
    .section-heading h3 { margin: 0; font: 500 26px/1.12 var(--serif); }
    .section-heading p { max-width: 780px; margin: 7px 0 0; color: var(--muted); font-size: 10px; }
    .sign-note { display: inline-block; margin-top: 8px; border-left: 3px solid var(--orange); padding-left: 9px; color: var(--muted); font-size: 9px; }
    .trace-controls { display: grid; grid-template-columns: minmax(190px, 1fr) 110px auto; gap: 8px; align-items: end; margin: 19px 0 13px; }
    .check { display: flex; align-items: center; gap: 6px; height: 38px; color: var(--ink-2); font-size: 10px; white-space: nowrap; }
    .check input { width: 14px; height: 14px; accent-color: var(--effort); }
    .trace-workbench { display: grid; grid-template-columns: minmax(310px, 390px) minmax(0, 1fr); gap: 16px; align-items: start; }
    .event-browser { min-width: 0; }
    .polarity-switch { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; border: 1px solid var(--line); background: var(--line); }
    .polarity-button { min-height: 43px; border: 0; background: var(--surface); color: var(--muted); padding: 7px 10px; text-align: left; }
    .polarity-button b { display: block; color: var(--ink-2); font: 650 10px/1.2 var(--mono); }
    .polarity-button small { font-size: 8px; }
    .polarity-button[aria-pressed="true"] { background: var(--effort-soft); }
    .polarity-button[aria-pressed="true"] b { color: var(--effort); }
    .footprint { margin-top: 8px; border: 1px solid var(--line); background: var(--surface); padding: 10px; }
    .footprint-row { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 8px; }
    .phase-bar { display: flex; height: 5px; margin-top: 7px; overflow: hidden; background: #e4dfd5; }
    .phase-bar i:nth-child(1) { background: #78aaa1; }
    .phase-bar i:nth-child(2) { background: #d1a762; }
    .phase-bar i:nth-child(3) { background: #b76d59; }
    .chip-row { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 8px; }
    .data-chip { max-width: 180px; overflow: hidden; border: 1px solid var(--line); border-radius: 99px; background: var(--surface-2); color: var(--muted); padding: 3px 6px; font: 8px/1.2 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .data-chip b { color: var(--ink-2); }
    .event-list { display: grid; gap: 7px; max-height: 760px; margin-top: 8px; overflow: auto; overscroll-behavior: contain; }
    .event-card { border: 1px solid var(--line); background: var(--surface); }
    .event-card.active { border-color: var(--effort); box-shadow: inset 3px 0 0 var(--effort); }
    .event-meta { display: flex; justify-content: space-between; gap: 9px; border-bottom: 1px solid #e9e4da; background: var(--surface-2); padding: 7px 9px; color: var(--muted); font: 8px/1.25 var(--mono); }
    .event-meta b { color: var(--ink-2); }
    .event-copy { margin: 0; padding: 10px 10px 7px; color: #2a3540; font: 11px/1.55 var(--serif); white-space: pre-wrap; overflow-wrap: anywhere; }
    mark { border-radius: 2px; background: #f2d39f; color: #151b20; padding: 1px 2px; box-shadow: 0 0 0 1px rgba(156,95,31,.13); }
    .event-footer { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 0 9px 9px; }
    .event-token { display: inline-block; max-width: 145px; overflow: hidden; border: 1px solid var(--line); border-radius: 3px; background: white; color: var(--ink-2); padding: 2px 5px; font: 8px/1.25 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .open-trace { border: 0; background: transparent; color: var(--effort); padding: 2px 0; font-size: 9px; font-weight: 800; }
    .event-progress { display: block; height: 3px; background: #e7e2d8; }
    .event-progress i { display: block; height: 100%; background: var(--effort); }

    .trace-viewer { min-width: 0; border: 1px solid var(--line); background: var(--surface); }
    .trace-viewer-head { display: flex; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); background: #f5f2eb; padding: 13px 15px; }
    .trace-viewer-head h4 { margin: 0; font: 600 18px/1.2 var(--serif); }
    .trace-viewer-head p { margin: 4px 0 0; color: var(--muted); font-size: 9px; }
    .trace-badges { display: flex; align-items: flex-start; justify-content: flex-end; gap: 5px; flex-wrap: wrap; }
    .trace-body { max-height: 1040px; overflow: auto; overscroll-behavior: contain; }
    .trace-block { border-bottom: 1px solid var(--line); padding: 14px 16px; }
    .trace-block:last-child { border-bottom: 0; }
    .trace-block-label { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; color: var(--muted); font-size: 8px; font-weight: 850; letter-spacing: .1em; text-transform: uppercase; }
    .trace-text { margin: 0; color: #28343f; font: 11px/1.62 var(--serif); white-space: pre-wrap; overflow-wrap: anywhere; }
    .trace-focus { display: inline; border-radius: 3px; background: #edf4f2; box-shadow: 0 0 0 3px #edf4f2; }
    .trace-empty { color: var(--muted); font-style: italic; }

    .fineweb-section { padding: clamp(22px, 3vw, 34px); border-bottom: 1px solid var(--line); background: #f2f5f2; }
    .fineweb-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 22px; }
    .fineweb-heading h3 { margin: 0; font: 500 26px/1.12 var(--serif); }
    .fineweb-heading p { max-width: 800px; margin: 7px 0 0; color: var(--muted); font-size: 10px; }
    .basis-badge { flex: 0 0 auto; border-radius: 99px; background: var(--teal-soft); color: var(--teal); padding: 6px 10px; font: 750 9px/1.2 var(--mono); }
    .basis-badge.flip { background: #f4ead4; color: #916015; }
    .basis-badge.warn { background: var(--orange-soft); color: var(--orange); }
    .fineweb-scan-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; margin-top: 18px; border: 1px solid var(--line); background: var(--line); }
    .fineweb-scan-card { min-width: 0; border: 0; background: var(--surface); color: var(--ink); padding: 13px 14px; text-align: left; }
    .fineweb-scan-card:hover { background: #faf8f2; }
    .fineweb-scan-card[aria-pressed="true"] { background: var(--blue-soft); box-shadow: inset 0 3px 0 var(--blue); }
    .fineweb-scan-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .fineweb-scan-title b { font: 650 12px/1.2 var(--serif); }
    .fineweb-scan-title span { color: var(--blue); font: 750 9px/1.2 var(--mono); }
    .fineweb-scan-values { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 9px; }
    .fineweb-scan-values span { display: block; min-height: 20px; color: var(--muted); font-size: 7px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
    .fineweb-scan-values b { display: block; overflow: hidden; color: var(--ink-2); font: 650 10px/1.2 var(--mono); text-overflow: ellipsis; }
    .fineweb-context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; margin-top: 15px; }
    .fineweb-column { min-width: 0; }
    .fineweb-column-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; border-top: 3px solid; padding: 9px 2px 7px; }
    .fineweb-column.positive .fineweb-column-head { border-color: var(--teal); }
    .fineweb-column.negative .fineweb-column-head { border-color: var(--orange); }
    .fineweb-column-head h4 { margin: 0; font: 650 11px/1.2 var(--mono); }
    .fineweb-column-head span { color: var(--muted); font-size: 8px; text-align: right; }
    .fineweb-list { display: grid; gap: 7px; }
    .fineweb-context-card { min-width: 0; border: 1px solid var(--line); background: var(--surface); }
    .fineweb-context-meta { display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid #e9e4da; background: var(--surface-2); padding: 7px 9px; color: var(--muted); font: 8px/1.25 var(--mono); }
    .fineweb-context-meta b { color: var(--ink-2); }
    .fineweb-context-copy { margin: 0; padding: 10px 10px 8px; color: #2a3540; font: 11px/1.55 var(--serif); white-space: pre-wrap; overflow-wrap: anywhere; }
    .fineweb-context-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 9px 9px; }
    .fineweb-source { min-width: 0; overflow: hidden; color: var(--muted); font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
    .fineweb-source a { color: var(--blue); font-weight: 750; text-decoration: none; }
    .fineweb-source a:hover { text-decoration: underline; }

    .left-section { padding: clamp(22px, 3vw, 34px); border-bottom: 1px solid var(--line); background: #eef3f2; }
    .left-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 22px; }
    .left-heading h3 { margin: 0; font: 500 26px/1.12 var(--serif); }
    .left-heading p { max-width: 850px; margin: 7px 0 0; color: var(--muted); font-size: 10px; }
    .left-source-badge { flex: 0 0 auto; border-radius: 99px; background: var(--teal-soft); color: var(--teal); padding: 6px 10px; font: 750 9px/1.2 var(--mono); }
    .left-source-badge.saved { background: #f4ead4; color: #916015; }
    .sv-map { display: grid; grid-template-columns: minmax(0, 1fr) 165px minmax(0, 1fr); gap: 10px; align-items: stretch; margin-top: 18px; }
    .sv-node, .sv-operator { display: grid; align-content: center; min-height: 86px; border: 1px solid var(--line); background: var(--surface); padding: 12px 14px; }
    .sv-node span, .sv-operator span { color: var(--muted); font-size: 8px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .sv-node b { margin-top: 5px; color: var(--ink); font: 650 16px/1.2 var(--mono); }
    .sv-node small { margin-top: 4px; color: var(--muted); font-size: 8px; }
    .sv-node.left { border-color: color-mix(in srgb, var(--effort) 45%, var(--line)); box-shadow: inset 3px 0 0 var(--effort); }
    .sv-operator { justify-items: center; background: #17242b; color: white; text-align: center; }
    .sv-operator b { margin: 4px 0; color: white; font: 650 13px/1.2 var(--mono); }
    .sv-operator span, .sv-operator small { color: #aebbc1; }
    .left-metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 1px; margin-top: 12px; border: 1px solid var(--line); background: var(--line); }
    .left-metric { min-width: 0; background: var(--surface); padding: 11px 12px; }
    .left-metric span { display: block; min-height: 21px; color: var(--muted); font-size: 7px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
    .left-metric b { display: block; overflow: hidden; color: var(--ink-2); font: 650 11px/1.2 var(--mono); text-overflow: ellipsis; }
    .left-relations { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .left-relation { border: 1px solid var(--line); background: rgba(255,253,248,.75); padding: 11px 12px; }
    .left-relation h4 { margin: 0; color: var(--ink-2); font: 700 9px/1.2 var(--mono); text-transform: uppercase; }
    .left-relation b { display: block; margin-top: 6px; color: var(--effort); font: 650 11px/1.3 var(--mono); }
    .left-relation p { margin: 4px 0 0; color: var(--muted); font-size: 8px; }
    .left-caveat { margin: 11px 0 0; border-left: 3px solid var(--blue); padding-left: 9px; color: var(--muted); font-size: 9px; }

    .left-token-section { margin-top: 14px; border: 1px solid var(--line); background: rgba(255,253,248,.72); }
    .left-token-section > summary { cursor: pointer; list-style: none; padding: 12px 14px; }
    .left-token-section > summary::-webkit-details-marker { display: none; }
    .left-token-section .token-content { padding: 0 14px 14px; }

    .token-section { border-bottom: 1px solid var(--line); background: #f7f6f1; }
    .token-section > summary { cursor: pointer; list-style: none; padding: 18px clamp(22px, 3vw, 34px); }
    .token-section > summary::-webkit-details-marker { display: none; }
    .token-summary { display: flex; justify-content: space-between; align-items: center; gap: 18px; }
    .token-summary h3 { margin: 0; font: 500 23px/1.15 var(--serif); }
    .token-summary p { margin: 4px 0 0; color: var(--muted); font-size: 9px; }
    .token-summary b { color: var(--effort); font: 650 10px/1.2 var(--mono); }
    .token-content { padding: 0 clamp(22px, 3vw, 34px) clamp(22px, 3vw, 34px); }
    .token-toolbar { display: flex; justify-content: flex-end; margin-bottom: 10px; }
    .token-limit { width: 120px; }
    .token-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
    .token-column { min-width: 0; }
    .token-head { display: flex; justify-content: space-between; gap: 10px; border-top: 3px solid; padding: 8px 2px; }
    .token-column.aligned .token-head { border-color: var(--teal); }
    .token-column.opposed .token-head { border-color: var(--orange); }
    .token-head h4 { margin: 0; font: 650 11px/1.2 var(--mono); }
    .token-head span { color: var(--muted); font-size: 8px; }
    .token-list { display: grid; gap: 5px; }
    .token-row { display: grid; grid-template-columns: 25px minmax(0, 1fr) auto; gap: 8px; align-items: center; border: 1px solid var(--line); background: var(--surface); padding: 7px 9px; }
    .token-rank { color: var(--muted); font: 650 8px/1 var(--mono); text-align: right; }
    .token-name { min-width: 0; }
    .token-name b { display: block; overflow: hidden; font: 650 10px/1.2 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .token-name small { display: block; overflow: hidden; margin-top: 2px; color: var(--muted); font: 7px/1.2 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .token-value { color: var(--ink-2); font: 650 9px/1.2 var(--mono); }

    .footer { width: min(1680px, 94vw); margin: -38px auto 32px; color: var(--muted); font-size: 9px; }
    .footer code { font-family: var(--mono); }
    :focus-visible { outline: 3px solid color-mix(in srgb, var(--effort) 28%, transparent); outline-offset: 2px; }

    @media (max-width: 1220px) {
      .toolbar-inner { grid-template-columns: 230px minmax(190px, 1fr) repeat(3, 130px) auto; }
      .metric-grid { grid-template-columns: repeat(3, 1fr); }
      .left-metrics { grid-template-columns: repeat(3, 1fr); }
      .all-metrics { grid-template-columns: repeat(2, 1fr); }
      .trace-workbench { grid-template-columns: 340px minmax(0, 1fr); }
    }
    @media (max-width: 940px) {
      .masthead-inner { align-items: flex-start; flex-direction: column; min-height: 0; }
      .dataset-facts { justify-content: flex-start; }
      .toolbar-inner { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .lens-control, .search-control { grid-column: 1 / -1; }
      .dashboard-grid { grid-template-columns: 1fr; }
      .ranking-panel { position: static; height: auto; min-height: 0; }
      .ranking-list { max-height: 430px; }
      .trace-workbench { grid-template-columns: 1fr; }
      .event-list { max-height: 520px; }
    }
    @media (max-width: 650px) {
      .masthead-inner, .effort-switch, .toolbar-inner, .shell, .footer { width: min(92vw, 1680px); }
      .brand-mark { display: none; }
      .effort-switch { grid-template-columns: 1fr; }
      .effort-stat { display: none; }
      .toolbar-inner { grid-template-columns: 1fr; }
      .lens-control, .search-control { grid-column: auto; }
      .metric-grid, .tail-grid, .effort-compare-grid, .token-grid, .fineweb-scan-grid, .fineweb-context-grid, .left-relations, .sv-map { grid-template-columns: 1fr; }
      .left-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .all-metrics { grid-template-columns: 1fr; }
      .activation-profile { grid-template-columns: 1fr; }
      .profile-stats { border-left: 0; border-top: 1px solid var(--line); padding: 12px 0 0; }
      .trace-controls { grid-template-columns: 1fr; }
      .compare-values, .tail-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .detail-hero, .trace-section, .fineweb-section, .left-section { padding: 20px 15px; }
      .comparison-head, .trace-viewer-head, .token-summary, .fineweb-heading, .left-heading { align-items: flex-start; flex-direction: column; }
    }
    @media print {
      .toolbar, .ranking-panel, .nav-buttons, .trace-controls, .event-browser, .footer { display: none !important; }
      .masthead { color: var(--ink); background: white; }
      .masthead * { color: var(--ink) !important; }
      .dashboard-grid { display: block; }
      .panel { box-shadow: none; }
      .trace-body { max-height: none; overflow: visible; }
    }
  </style>
</head>
<body data-effort="low">
  <a class="skip-link" href="#detail">Skip to selected direction</a>
  <header class="masthead">
    <div class="masthead-inner">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">SV×</div>
        <div>
          <p class="eyebrow">Cross-corpus interpretability workbench · GPT-OSS-20B</p>
          <h1>Reasoning <em>SV Atlas</em></h1>
          <p class="subtitle">Explore singular directions over generated analysis-channel tokens, compare low and medium reasoning effort, then check the same direction against broad and selective FineWeb context scans on the same page.</p>
        </div>
      </div>
      <div class="dataset-facts" id="datasetFacts" aria-label="Current scan summary"></div>
    </div>
  </header>

  <nav class="source-bar" id="sourceBar" aria-label="Attribution basis">
    <div class="source-bar-inner">
      <div class="source-switch" id="sourceSwitch" role="tablist" aria-label="Attribution basis"></div>
      <p class="source-note" id="sourceNote"></p>
    </div>
  </nav>

  <nav class="effort-bar" aria-label="Reasoning effort">
    <div class="effort-switch" role="tablist" aria-label="Reasoning effort scan">
      <button class="effort-tab" type="button" role="tab" data-effort="low" aria-selected="true">
        <span class="effort-index">01</span><span><b>Low reasoning effort</b><small>Shorter analysis traces · 512-token generation cap</small></span><span class="effort-stat" id="lowEffortStat"></span>
      </button>
      <button class="effort-tab" type="button" role="tab" data-effort="medium" aria-selected="false">
        <span class="effort-index">02</span><span><b>Medium reasoning effort</b><small>Longer analysis traces · 768-token generation cap</small></span><span class="effort-stat" id="mediumEffortStat"></span>
      </button>
    </div>
  </nav>

  <section class="toolbar" aria-label="Direction controls">
    <div class="toolbar-inner">
      <div class="control lens-control"><span>Discovery lens</span><div class="lens-switch" role="group" aria-label="Discovery lens"><button class="lens-button" type="button" data-lens="broad" aria-pressed="true">Broad activity</button><button class="lens-button" type="button" data-lens="selective" aria-pressed="false">Selective tails</button></div></div>
      <label class="control search-control">Find a direction<span class="search-wrap"><input id="candidateSearch" type="search" placeholder="Candidate ID, layer, or SV index…" autocomplete="off"><kbd class="shortcut">/</kbd></span></label>
      <label class="control">Layer<select id="layerFilter"></select></label>
      <label class="control">Rank by<select id="sortMetric"></select></label>
      <label class="control">Show<select id="rowLimit"></select></label>
      <button class="reset" id="resetFilters" type="button">Reset</button>
    </div>
  </section>

  <section class="token-finder" aria-label="Find directions by context token">
    <div class="token-finder-inner">
      <label class="token-field">Token in top contexts
        <span class="token-wrap"><input id="tokenSearch" type="search" placeholder="e.g. maybe · wait · def · →" autocomplete="off" spellcheck="false"></span>
      </label>
      <p class="token-summary" id="tokenSummary"></p>
      <button class="token-clear" id="tokenClear" type="button" hidden>Clear token</button>
    </div>
  </section>

  <main class="shell">
    <div class="dashboard-grid">
      <aside class="panel ranking-panel" aria-label="Ranked singular directions">
        <div class="panel-head"><div class="panel-head-row"><h2 id="rankingTitle">Ranked directions</h2><span class="count" id="rankingCount"></span></div><p class="microcopy" id="rankingCaption"></p></div>
        <div class="ranking-list" id="rankingList"></div>
      </aside>
      <article class="panel detail-panel" id="detail" tabindex="-1"></article>
    </div>
  </main>
  <footer class="footer" id="footerNote"></footer>

  <script>
    const ALL = __COT_DASHBOARD_PAYLOAD__;
    let activeSource = ALL.default_source;
    let DATA = ALL.sources[activeSource];
    const BROAD_METRICS = {
      mean_abs_cosine: { label: "Mean |cosine|", short: "mean |cos|", kind: "decimal", help: "Mean absolute projection normalized by residual-stream norm." },
      doc_top5_presence_rate: { label: "Trace top-5 presence", short: "trace top-5", kind: "percent", help: "Share of reasoning traces where this direction enters the layer's top five." },
      mean_document_peak_abs: { label: "Mean trace peak |act|", short: "trace peak", kind: "number", help: "Mean per-trace maximum absolute projection." },
      top1_abs_rate: { label: "Top-1 reasoning-token share", short: "top-1 share", kind: "percent", help: "Share of analysis tokens where this is the strongest direction in the top-k bank." },
      top5_abs_rate: { label: "Top-5 reasoning-token share", short: "top-5 share", kind: "percent", help: "Share of analysis tokens where this direction enters the top five." },
      std_activation: { label: "Activation variability", short: "activation std", kind: "number", help: "Standard deviation over generated analysis tokens." },
      dynamicity_std_over_abs_mean: { label: "Dynamicity", short: "dynamicity", kind: "decimal", help: "Activation standard deviation divided by mean absolute activation." },
      max_abs_unembed_token_cosine: { label: "Max |token cosine|", short: "max |token cos|", kind: "decimal", help: "Strongest absolute output-token unembedding cosine." }
    };
    const SELECTIVE_METRICS = {
      tail_selectivity_score: { label: "Tail selectivity score", short: "tail score", kind: "decimal", help: "Selected-tail q99.9 robust-z × √(top-0.1% energy share) × support weight." },
      selected_tail_q999_robust_z: { label: "Selected-tail q99.9 robust z", short: "q99.9 robust z", kind: "decimal", help: "Score-driving tail's 99.9th percentile after median/MAD normalization." },
      selected_tail_top0_1pct_energy_share: { label: "Selected-tail top-0.1% energy", short: "top-0.1% energy", kind: "percent", help: "Squared robust-z energy carried by the strongest 0.1% of sampled reasoning tokens." },
      selected_tail_doc_rate_z5: { label: "Traces above z=5 · low first", short: "traces >5z", kind: "percent", direction: "asc", help: "Share of reasoning traces whose score-driving tail peak exceeds robust z=5." },
      effective_support_fraction: { label: "Effective support · low first", short: "effective support", kind: "percent", direction: "asc", help: "M2²/(N×M4); lower values mean heavier or more concentrated tails." },
      stable_excess_kurtosis: { label: "Excess kurtosis", short: "excess kurtosis", kind: "number", help: "Fourth-moment tail weight minus three; high values can also reflect artifacts." },
      selected_tail_top_context_largest_center_token_share: { label: "Largest center-token share · low first", short: "largest token", kind: "percent", direction: "asc", help: "Largest token share among retained score-driving events. Lower is more lexically diverse." },
      selected_tail_top_context_effective_center_tokens: { label: "Effective center tokens", short: "effective tokens", kind: "number", help: "Entropy-derived effective count of center tokens in retained score-driving events." },
      mean_abs_cosine: { label: "Mean |cosine|", short: "mean |cos|", kind: "decimal", help: "Broad-activity score for comparison." },
      max_abs_unembed_token_cosine: { label: "Max |token cosine|", short: "max |token cos|", kind: "decimal", help: "Strongest absolute output-token cosine; high can indicate lexical anchoring." }
    };
    const LENS_CONFIG = {
      broad: { label: "Broad activity", title: "Broad CoT activity", primaryRank: "rank_global_mean_abs_cosine", defaultMetric: "mean_abs_cosine", metrics: BROAD_METRICS, caption: "Primary rank uses mean absolute cosine over generated analysis tokens." },
      selective: { label: "Selective tails", title: "Selective CoT tails", primaryRank: "rank_global_tail_selectivity", defaultMetric: "tail_selectivity_score", metrics: SELECTIVE_METRICS, caption: "Primary rank uses a robust tail-concentration score, not semantic cleanliness." }
    };
    const LABELS = {
      candidate: "Candidate", layer: "Layer", sv_index_0: "SV index (zero-based)", singular_value: "Singular value", singular_value_over_sv0: "Singular value / SV0",
      n_tokens: "Analysis tokens", n_documents: "Reasoning traces", mean_activation: "Mean activation", mean_abs_activation: "Mean |activation|", rms_activation: "RMS activation", std_activation: "Activation std",
      positive_rate: "Positive rate", max_activation: "Maximum activation", min_activation: "Minimum activation", mean_abs_cosine: "Mean |cosine|", rms_cosine: "RMS cosine",
      top1_abs_rate: "Top-1 token rate", top5_abs_rate: "Top-5 token rate", doc_top5_presence_rate: "Trace top-5 presence", mean_document_peak_abs: "Mean trace peak |activation|",
      mean_layer_residual_norm: "Mean residual norm", sigma_weighted_mean_abs: "σ-weighted mean |activation|", sigma_weighted_std: "σ-weighted activation std", dynamicity_std_over_abs_mean: "Dynamicity",
      token_sample_n: "Robust-stat token sample", median_activation: "Median activation", mad_activation: "Median absolute deviation", robust_activation_scale: "Robust activation scale",
      stable_skewness: "Stable skewness", stable_kurtosis: "Stable kurtosis", stable_excess_kurtosis: "Stable excess kurtosis", effective_support_fraction: "Effective support fraction",
      tail_selectivity_score: "Tail selectivity score", selected_tail_polarity: "Score-driving tail", rank_global_mean_abs_cosine: "Global broad-activity rank", rank_global_tail_selectivity: "Global tail-selectivity rank",
      rank_layer_mean_abs_cosine: "Layer broad-activity rank", rank_layer_tail_selectivity: "Layer tail-selectivity rank", rank_global_excess_kurtosis: "Global excess-kurtosis rank",
      max_abs_unembed_token_cosine: "Maximum |token cosine|", rank_global_max_abs_unembed_token_cosine: "Global token-likeness rank",
      selected_tail_q999_robust_z: "Selected-tail q99.9 robust z", selected_tail_top0_1pct_energy_share: "Selected-tail top-0.1% energy share",
      selected_tail_doc_count_z5: "Selected-tail traces above z=5", selected_tail_doc_rate_z5: "Selected-tail trace rate above z=5",
      selected_tail_top_context_effective_center_tokens: "Selected-tail effective center tokens", selected_tail_top_context_largest_center_token_share: "Selected-tail largest center-token share"
    };

    const buildRankingIndexes = () => Object.fromEntries(Object.entries(DATA.efforts).map(([effort, payload]) => [effort, new Map(payload.rankings.map(row => [row.candidate, row]))]));
    const buildFinewebIndexes = () => Object.fromEntries(Object.entries(DATA.fineweb.sources).map(([source, payload]) => [source, payload.rankings]));
    const buildViews = () => Object.fromEntries(Object.entries(DATA.efforts).map(([effort, payload]) => [effort, Object.fromEntries(Object.entries(payload.cohorts).map(([lens, candidates]) => [lens, {
      query: "", layer: "all", metric: LENS_CONFIG[lens].defaultMetric, limit: Math.min(50, candidates.length), selected: candidates[0] || null
    }]))]));
    let rankingIndexes = buildRankingIndexes();
    let finewebIndexes = buildFinewebIndexes();
    const state = {
      effort: DATA.default_effort || "low",
      lens: DATA.default_lens || "broad",
      views: buildViews(),
      contextQuery: "",
      contextLimit: 6,
      dedupe: true,
      polarity: "positive",
      traceFocus: null,
      traceKey: null,
      tokenLimit: 8,
      leftTokenLimit: 8,
      finewebSource: "broad",
      tokenQuery: ""
    };
    const $ = selector => document.querySelector(selector);
    const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
    const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    const integer = new Intl.NumberFormat();
    const effortData = () => DATA.efforts[state.effort];
    const viewState = () => state.views[state.effort][state.lens];
    const lensConfig = () => LENS_CONFIG[state.lens];
    const rowIndex = (effort = state.effort) => rankingIndexes[effort];
    const currentRow = () => rowIndex().get(viewState().selected);
    const otherEffort = effort => effort === "low" ? "medium" : "low";
    const cohortRows = () => effortData().cohorts[state.lens].map(candidate => rowIndex().get(candidate));

    function decimal(value, digits = 3) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      const magnitude = Math.abs(number);
      if (magnitude !== 0 && (magnitude < .001 || magnitude >= 10000)) return number.toExponential(2);
      return number.toLocaleString(undefined, { maximumFractionDigits: digits });
    }
    function signed(value, digits = 3) {
      const number = Number(value);
      return Number.isFinite(number) ? `${number >= 0 ? "+" : ""}${decimal(number, digits)}` : "—";
    }
    function percent(value, digits = 1) {
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
    }
    function formatMetric(value, kind) {
      if (kind === "percent") return percent(value);
      if (kind === "integer") return integer.format(value);
      return decimal(value, kind === "decimal" ? 4 : 3);
    }
    function visibleToken(token) {
      return JSON.stringify(String(token ?? "")).replaceAll("\\n", "↵").replaceAll("\\t", "⇥");
    }

    function setup() {
      const initial = parseHash();
      if (initial.effort && DATA.efforts[initial.effort]) state.effort = initial.effort;
      if (initial.lens && LENS_CONFIG[initial.lens]) state.lens = initial.lens;
      if (initial.candidate && rowIndex().has(initial.candidate) && effortData().cohorts[state.lens].includes(initial.candidate)) {
        viewState().selected = initial.candidate;
        expandLimitToCandidate(initial.candidate);
      }
      buildSourceTabs();
      configureControls();
      document.querySelectorAll(".effort-tab").forEach(button => button.addEventListener("click", () => switchEffort(button.dataset.effort)));
      document.querySelectorAll(".lens-button").forEach(button => button.addEventListener("click", () => switchLens(button.dataset.lens)));
      $("#candidateSearch").addEventListener("input", event => { viewState().query = event.target.value.trim().toLowerCase(); render(); });
      $("#tokenSearch").addEventListener("input", event => { state.tokenQuery = event.target.value; render(); });
      $("#tokenClear").addEventListener("click", () => { state.tokenQuery = ""; $("#tokenSearch").value = ""; render(); $("#tokenSearch").focus(); });
      $("#layerFilter").addEventListener("change", event => { viewState().layer = event.target.value; render(); });
      $("#sortMetric").addEventListener("change", event => { viewState().metric = event.target.value; render(); });
      $("#rowLimit").addEventListener("change", event => { viewState().limit = Number(event.target.value); render(); });
      $("#resetFilters").addEventListener("click", resetFilters);
      addEventListener("hashchange", () => {
        const target = parseHash();
        if (!target.effort || !DATA.efforts[target.effort] || !LENS_CONFIG[target.lens] || !target.candidate) return;
        state.effort = target.effort;
        state.lens = target.lens;
        configureControls();
        if (effortData().cohorts[state.lens].includes(target.candidate)) revealCandidate(target.candidate, false);
      });
      addEventListener("keydown", event => {
        const tag = document.activeElement?.tagName;
        if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(tag)) { event.preventDefault(); $("#candidateSearch").focus(); }
        if (event.key === "Escape" && document.activeElement === $("#candidateSearch")) { viewState().query = ""; $("#candidateSearch").value = ""; render(); $("#candidateSearch").blur(); }
      });
      render();
      syncHash();
    }

    function parseHash() {
      let raw = location.hash.slice(1);
      try { raw = decodeURIComponent(raw); } catch (_) {}
      if (!raw) return { effort: DATA.default_effort || "low", lens: DATA.default_lens || "broad", candidate: null };
      const parts = raw.split("/");
      if (parts.length >= 3 && DATA.efforts[parts[0]] && LENS_CONFIG[parts[1]]) return { effort: parts[0], lens: parts[1], candidate: parts.slice(2).join("/") };
      return { effort: "low", lens: "broad", candidate: raw };
    }

    function configureControls() {
      const payload = effortData();
      const meta = payload.meta;
      const view = viewState();
      document.body.dataset.effort = state.effort;
      document.body.dataset.source = activeSource;
      document.querySelectorAll(".source-tab").forEach(button => button.setAttribute("aria-selected", String(button.dataset.source === activeSource)));
      const sourceInfo = (ALL.source_labels || {})[activeSource];
      if ($("#sourceNote")) $("#sourceNote").textContent = sourceInfo ? sourceInfo.note : "";
      document.querySelectorAll(".effort-tab").forEach(button => button.setAttribute("aria-selected", String(button.dataset.effort === state.effort)));
      document.querySelectorAll(".lens-button").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.lens === state.lens)));
      $("#lowEffortStat").textContent = `${compact.format(DATA.efforts.low.meta.analysis_tokens)} analysis tokens`;
      $("#mediumEffortStat").textContent = `${compact.format(DATA.efforts.medium.meta.analysis_tokens)} analysis tokens`;
      $("#datasetFacts").innerHTML = [
        [compact.format(meta.analysis_tokens), "analysis tokens"], [integer.format(meta.rollouts), "reasoning traces"], [`${meta.capped_rollouts}/${meta.rollouts}`, "hit gen cap"], [integer.format(meta.categories), "task categories"]
      ].map(([value, label]) => `<div class="fact"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join("");
      $("#rankingTitle").textContent = `${state.effort === "low" ? "Low" : "Medium"} · ${lensConfig().title}`;
      const rows = cohortRows();
      const layers = [...new Set(rows.map(row => Number(row.layer)))].sort((a, b) => a - b);
      $("#layerFilter").innerHTML = `<option value="all">All embedded layers</option>` + layers.map(layer => `<option value="${layer}">Layer ${String(layer).padStart(2,"0")}</option>`).join("");
      $("#sortMetric").innerHTML = Object.entries(lensConfig().metrics).map(([key, item]) => `<option value="${key}">${esc(item.label)}</option>`).join("");
      if (!lensConfig().metrics[view.metric]) view.metric = lensConfig().defaultMetric;
      const choices = [25, 50, 100, 250].filter(value => value < rows.length); choices.push(rows.length);
      $("#rowLimit").innerHTML = [...new Set(choices)].map(value => `<option value="${value}">${value === rows.length ? `All ${integer.format(value)}` : `Top ${value}`}</option>`).join("");
      if (view.limit > rows.length) view.limit = rows.length;
      $("#candidateSearch").value = view.query;
      if ($("#tokenSearch")) $("#tokenSearch").value = state.tokenQuery;
      $("#layerFilter").value = view.layer;
      $("#sortMetric").value = view.metric;
      $("#rowLimit").value = String(view.limit);
      const fw = DATA.fineweb;
      $("#footerNote").innerHTML = `SV identifiers are zero-based (<code>SV00</code> is the first vector); rank numbers remain one-based. The dashboard embeds paired saved left-vector geometry for every displayed right vector, plus validated J·V and U-token neighbors where a matching basis fingerprint is available. It also includes the top ${integer.format(meta.cohort_size)} directions per CoT effort/lens, top ${integer.format(meta.embedded_contexts_per_side)} of ${integer.format(meta.source_contexts_per_side)} CoT events per polarity, all ${integer.format(meta.rollouts)} compact rollouts, right-vector token neighbors, and ${integer.format(fw.sources.broad.meta.embedded_per_side)} broad plus ${integer.format(fw.sources.selective.meta.embedded_per_side)} selective FineWeb contexts per polarity. Prompt/control/final tokens were excluded from CoT activation statistics. FineWeb and low CoT share an exact direction bank; medium was independently materialized, so right- and left-basis same-index alignment are reported explicitly.`;
    }

    function switchEffort(effort) {
      if (!DATA.efforts[effort] || effort === state.effort) return;
      state.effort = effort;
      resetTraceState();
      configureControls(); render(); syncHash();
    }
    function switchLens(lens) {
      if (!LENS_CONFIG[lens] || lens === state.lens) return;
      state.lens = lens;
      resetTraceState();
      configureControls(); render(); syncHash();
    }
    function switchSource(source) {
      if (!ALL.sources[source] || source === activeSource) return;
      activeSource = source;
      DATA = ALL.sources[source];
      rankingIndexes = buildRankingIndexes();
      finewebIndexes = buildFinewebIndexes();
      state.effort = DATA.default_effort || "low";
      state.lens = DATA.default_lens || "broad";
      state.views = buildViews();
      state.finewebSource = "broad";
      state.polarity = "positive";
      state.tokenQuery = "";
      resetTraceState();
      configureControls(); render(); syncHash();
    }
    function buildSourceTabs() {
      const bar = $("#sourceBar"), order = ALL.source_order || [];
      if (order.length < 2) { if (bar) bar.style.display = "none"; return; }
      $("#sourceSwitch").innerHTML = order.map((source, index) => {
        const info = (ALL.source_labels || {})[source] || { label: source, tagline: "" };
        return `<button class="source-tab" type="button" role="tab" data-source="${esc(source)}" aria-selected="${String(source === activeSource)}"><span class="source-index">${String(index + 1).padStart(2, "0")}</span><span><b>${esc(info.label)}</b><small>${esc(info.tagline)}</small></span></button>`;
      }).join("");
      $("#sourceSwitch").querySelectorAll(".source-tab").forEach(button => button.addEventListener("click", () => switchSource(button.dataset.source)));
    }
    function resetTraceState() { state.contextQuery = ""; state.contextLimit = 6; state.traceFocus = null; state.traceKey = null; }
    function resetFilters() {
      const view = viewState();
      view.query = ""; view.layer = "all"; view.metric = lensConfig().defaultMetric; view.limit = Math.min(50, cohortRows().length); state.tokenQuery = "";
      $("#candidateSearch").value = ""; $("#layerFilter").value = "all"; $("#sortMetric").value = view.metric; $("#rowLimit").value = String(view.limit); if ($("#tokenSearch")) $("#tokenSearch").value = ""; render();
    }
    function compareRows(a, b) {
      const metric = lensConfig().metrics[viewState().metric];
      const direction = metric.direction === "asc" ? 1 : -1;
      const av = Number(a[viewState().metric]), bv = Number(b[viewState().metric]);
      const diff = Number.isFinite(av) && Number.isFinite(bv) ? (av - bv) * direction : 0;
      return diff || Number(a[lensConfig().primaryRank]) - Number(b[lensConfig().primaryRank]);
    }
    function embeddedPerSide() { return Number(effortData().meta.embedded_contexts_per_side) || 0; }
    function candidateTokenHits(candidate, queryLower) {
      const grouped = effortData().contexts[candidate];
      if (!grouped) return null;
      let best = null; const seen = new Map(); const sides = new Set();
      for (const polarity of ["positive", "negative"]) {
        for (const item of grouped[polarity] || []) {
          if (!String(item.token ?? "").toLowerCase().includes(queryLower)) continue;
          sides.add(polarity);
          if (!seen.has(item.token)) seen.set(item.token, { token: item.token, polarity, rank: item.rank });
          if (!best || Number(item.rank) < Number(best.rank)) best = { token: item.token, polarity, rank: item.rank };
        }
      }
      return best ? { best, sides: [...sides], tokens: [...seen.values()], count: seen.size } : null;
    }
    function tokenHitBadge(candidate) {
      const q = state.tokenQuery.trim().toLowerCase();
      if (!q) return "";
      const hit = candidateTokenHits(candidate, q);
      if (!hit) return "";
      const label = hit.best.polarity === "positive" ? "+ high" : "− low";
      const extra = hit.count > 1 ? ` · ${integer.format(hit.count)} tokens` : "";
      return `<span class="rank-token"><span class="token-hit ${hit.best.polarity}">${label} · rank ${integer.format(hit.best.rank)}/${embeddedPerSide()}</span><code>${esc(visibleToken(hit.best.token))}</code>${extra}</span>`;
    }
    function updateTokenSummary(rows) {
      const summary = $("#tokenSummary"), clear = $("#tokenClear");
      if (!summary) return;
      const q = state.tokenQuery.trim();
      if (!q) {
        summary.classList.remove("active");
        summary.textContent = `Type a token to keep only directions whose top-${embeddedPerSide()} activating contexts include it (matches either tail, current effort and lens).`;
        if (clear) clear.hidden = true;
        return;
      }
      const ql = q.toLowerCase();
      let pos = 0, neg = 0;
      rows.forEach(row => { const hit = candidateTokenHits(row.candidate, ql); if (!hit) return; if (hit.sides.includes("positive")) pos++; if (hit.sides.includes("negative")) neg++; });
      summary.classList.add("active");
      summary.innerHTML = `<b>${integer.format(rows.length)}</b> of ${integer.format(cohortRows().length)} ${esc(state.lens)} directions include <code>${esc(visibleToken(q))}</code> in top-${embeddedPerSide()} · <span class="token-hit positive">+ ${integer.format(pos)} high</span> <span class="token-hit negative">− ${integer.format(neg)} low</span>`;
      if (clear) clear.hidden = false;
    }
    function filteredRows(applyLimit = true) {
      const view = viewState(), query = view.query, tokenQuery = state.tokenQuery.trim().toLowerCase();
      const rows = cohortRows().filter(row => {
        if (view.layer !== "all" && Number(row.layer) !== Number(view.layer)) return false;
        if (query && !`${row.candidate} layer ${row.layer} l${String(row.layer).padStart(2,"0")} sv ${row.sv_index_0} sv${String(row.sv_index_0).padStart(2,"0")}`.toLowerCase().includes(query)) return false;
        if (tokenQuery && !candidateTokenHits(row.candidate, tokenQuery)) return false;
        return true;
      }).sort(compareRows);
      return applyLimit ? rows.slice(0, view.limit) : rows;
    }
    function render() {
      const view = viewState(), previous = view.selected, all = filteredRows(false), rows = all.slice(0, view.limit);
      if (rows.length && !rows.some(row => row.candidate === view.selected)) view.selected = rows[0].candidate;
      if (!rows.length) view.selected = null;
      if (view.selected !== previous) { state.traceFocus = null; state.traceKey = null; syncHash(); }
      renderRankings(rows, all.length); renderDetail(rows); updateTokenSummary(all);
    }
    function renderRankings(rows, matchCount) {
      const view = viewState(), metric = lensConfig().metrics[view.metric];
      const values = rows.map(row => Number(row[view.metric])).filter(Number.isFinite);
      const min = values.length ? Math.min(...values) : 0, max = values.length ? Math.max(...values) : 0;
      $("#rankingCount").textContent = `${integer.format(rows.length)} / ${integer.format(matchCount)}`;
      const tokenNote = state.tokenQuery.trim() ? `Filtered to directions with “${state.tokenQuery.trim()}” in top-${embeddedPerSide()} contexts. ` : "";
      $("#rankingCaption").textContent = `${tokenNote}${metric.label} · ${metric.direction === "asc" ? "ascending" : "descending"}. ${lensConfig().caption} SV indices are zero-based.`;
      if (!rows.length) { $("#rankingList").innerHTML = `<div class="empty">No directions match these filters.</div>`; return; }
      const other = otherEffort(state.effort), otherIndex = rowIndex(other);
      $("#rankingList").innerHTML = rows.map(row => {
        const value = Number(row[view.metric]), span = Math.max(max - min, 1e-12);
        const normalized = metric.direction === "asc" ? (max - value) / span : (value - min) / span;
        const width = values.length === 1 ? 100 : Math.max(2, Math.min(100, normalized * 100));
        const otherRow = otherIndex.get(row.candidate);
        const tail = state.lens === "selective" ? `<span class="tail-pill">${row.selected_tail_polarity === "positive" ? "+ high" : "− low"}</span>` : "";
        return `<button class="rank-row ${row.candidate === view.selected ? "active" : ""}" type="button" data-candidate="${esc(row.candidate)}" ${row.candidate === view.selected ? 'aria-current="true"' : ""}>
          <span class="rank-number">#${integer.format(row[lensConfig().primaryRank])}</span><span><span class="rank-topline"><span class="candidate">${esc(row.candidate)}${tail}</span><span class="layer-note">L${String(row.layer).padStart(2,"0")} · SV${String(row.sv_index_0).padStart(2,"0")}</span></span>
          <span class="rank-measure"><span>${esc(metric.short)}</span><b>${esc(formatMetric(value, metric.kind))}</b></span><span class="mini-track"><i style="width:${width.toFixed(2)}%"></i></span>
          <span class="rank-compare"><span>${other} effort #${integer.format(otherRow?.[lensConfig().primaryRank])}</span><span>${state.lens === "selective" ? `${percent(row.selected_tail_top_context_largest_center_token_share, 0)} top token` : `${percent(row.doc_top5_presence_rate, 0)} traces`}</span></span>${tokenHitBadge(row.candidate)}</span></button>`;
      }).join("");
      $("#rankingList").querySelectorAll(".rank-row").forEach(button => button.addEventListener("click", () => { selectCandidate(button.dataset.candidate); if (innerWidth <= 940) $("#detail").scrollIntoView({behavior:"smooth",block:"start"}); }));
    }
    function selectCandidate(candidate, updateHash = true) {
      if (!effortData().cohorts[state.lens].includes(candidate)) return;
      viewState().selected = candidate; state.traceFocus = null; state.traceKey = null; if (updateHash) syncHash(); render();
    }
    function expandLimitToCandidate(candidate) {
      const sorted = cohortRows().slice().sort(compareRows), needed = sorted.findIndex(row => row.candidate === candidate) + 1;
      const choices = [...new Set([25,50,100,250,sorted.length])].filter(value => value <= sorted.length).sort((a,b) => a-b);
      viewState().limit = choices.find(value => value >= needed) || sorted.length;
    }
    function revealCandidate(candidate, updateHash = true) {
      const view = viewState(); view.query = ""; view.layer = "all"; $("#candidateSearch").value = ""; $("#layerFilter").value = "all";
      expandLimitToCandidate(candidate); $("#rowLimit").value = String(view.limit); view.selected = candidate; state.traceFocus = null; state.traceKey = null; if (updateHash) syncHash(); render();
    }
    function syncHash() {
      const selected = viewState().selected, next = selected ? `#${state.effort}/${state.lens}/${encodeURIComponent(selected)}` : "";
      if (location.hash === next) return;
      try { history.replaceState(null, "", next || `${location.pathname}${location.search}`); } catch (_) { location.hash = next; }
    }

    function metricCard(label, value, title) { return `<div class="metric-card" title="${esc(title || "")}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`; }
    function renderDetail(visibleRows) {
      const root = $("#detail"), row = currentRow();
      if (!row) { root.innerHTML = `<div class="empty">Choose a broader filter to inspect a direction.</div>`; return; }
      const key = `${state.effort}/${state.lens}/${row.candidate}`;
      if (state.traceKey !== key) { state.traceKey = key; state.traceFocus = null; state.polarity = state.lens === "selective" ? row.selected_tail_polarity : "positive"; }
      const selectedIndex = visibleRows.findIndex(item => item.candidate === row.candidate);
      const lo = Math.min(Number(row.min_activation), 0), hi = Math.max(Number(row.max_activation), 0), spread = Math.max(hi - lo, 1e-9);
      const zero = (0 - lo) / spread * 100, mean = Math.max(0, Math.min(100, (Number(row.mean_activation) - lo) / spread * 100));
      const selective = state.lens === "selective";
      const summary = selective
        ? `This zero-based layer ${row.layer} / SV ${row.sv_index_0} direction ranks <b>#${integer.format(row.rank_global_tail_selectivity)} by tail selectivity</b> in the ${state.effort}-effort CoT scan and <b>#${integer.format(row.rank_global_mean_abs_cosine)} by broad activity</b>. The tail score rewards robust extremes and concentration with a small-trace support weight; inspect trace prevalence and center-token concentration before assigning a concept.`
        : `This zero-based layer ${row.layer} / SV ${row.sv_index_0} direction ranks <b>#${integer.format(row.rank_global_mean_abs_cosine)} by mean absolute cosine</b> over ${integer.format(row.n_tokens)} generated analysis tokens from ${integer.format(row.n_documents)} ${state.effort}-effort reasoning traces. Prompt, Harmony control, and final-channel tokens were excluded.`;
      root.innerHTML = `<section class="detail-hero"><div class="detail-nav"><span class="detail-rank">${esc(state.effort)} effort · ${esc(lensConfig().label)} rank #${integer.format(row[lensConfig().primaryRank])}</span><div class="nav-buttons"><button class="icon-button" id="previousCandidate" type="button" ${selectedIndex <= 0 ? "disabled" : ""} aria-label="Previous visible direction">←</button><button class="icon-button" id="nextCandidate" type="button" ${selectedIndex < 0 || selectedIndex >= visibleRows.length-1 ? "disabled" : ""} aria-label="Next visible direction">→</button><button class="copy-button" id="copyCandidate" type="button">Copy ID</button></div></div>
        <div class="title-row"><h2 class="detail-title">${esc(row.candidate)}</h2><span class="id-chip">${state.effort.toUpperCase()} CoT · L${String(row.layer).padStart(2,"0")} / SV${String(row.sv_index_0).padStart(2,"0")}</span></div><p class="detail-summary">${summary}</p>
        <div class="metric-grid">${selective ? selectiveHero(row) : broadHero(row)}</div>
        <div class="activation-profile"><div><div class="profile-labels"><span>${signed(lo)}</span><span>analysis-token activation range</span><span>${signed(hi)}</span></div><div class="axis"><i class="zero-marker" style="left:${zero.toFixed(2)}%"></i><i class="mean-marker" style="left:${mean.toFixed(2)}%"></i></div></div><div class="profile-stats"><span>Mean<b>${signed(row.mean_activation)}</b></span><span>Positive tokens<b>${percent(row.positive_rate,1)}</b></span></div></div>
        ${comparisonSection(row)}${selective ? tailProfile(row) : ""}<details class="metric-details"><summary>Inspect all ${integer.format(Object.keys(row).length-1)} scan metrics</summary>${allMetricGroups(row)}</details></section>
        ${leftSingularSection(row)}${traceSection(row)}${finewebSection(row)}${tokenSection(row)}`;
      $("#previousCandidate").addEventListener("click", () => selectedIndex > 0 && selectCandidate(visibleRows[selectedIndex-1].candidate));
      $("#nextCandidate").addEventListener("click", () => selectedIndex >= 0 && selectedIndex < visibleRows.length-1 && selectCandidate(visibleRows[selectedIndex+1].candidate));
      $("#copyCandidate").addEventListener("click", copyCandidate);
      bindTraceControls(row);
      bindFinewebControls(row);
      bindLeftTokenControls(row);
      bindTokenControls(row);
    }
    function broadHero(row) {
      return [metricCard("Mean |cosine|",decimal(row.mean_abs_cosine,4),BROAD_METRICS.mean_abs_cosine.help),metricCard("Trace top-5 presence",percent(row.doc_top5_presence_rate,1),BROAD_METRICS.doc_top5_presence_rate.help),metricCard("Mean trace peak |act|",decimal(row.mean_document_peak_abs),BROAD_METRICS.mean_document_peak_abs.help),metricCard("Top-1 token share",percent(row.top1_abs_rate,1),BROAD_METRICS.top1_abs_rate.help),metricCard("Top-5 token share",percent(row.top5_abs_rate,1),BROAD_METRICS.top5_abs_rate.help),metricCard("Activation std",decimal(row.std_activation),BROAD_METRICS.std_activation.help)].join("");
    }
    function selectiveHero(row) {
      return [metricCard("Tail selectivity score",decimal(row.tail_selectivity_score,4),SELECTIVE_METRICS.tail_selectivity_score.help),metricCard("Score-driving tail",row.selected_tail_polarity === "positive" ? "High (+)" : "Low (−)","Arbitrary SV orientation with the larger score."),metricCard("Selected q99.9 robust z",decimal(row.selected_tail_q999_robust_z,3),SELECTIVE_METRICS.selected_tail_q999_robust_z.help),metricCard("Top-0.1% tail energy",percent(row.selected_tail_top0_1pct_energy_share,2),SELECTIVE_METRICS.selected_tail_top0_1pct_energy_share.help),metricCard("Traces above z=5",`${integer.format(row.selected_tail_doc_count_z5)} · ${percent(row.selected_tail_doc_rate_z5,1)}`,SELECTIVE_METRICS.selected_tail_doc_rate_z5.help),metricCard("Largest center token",percent(row.selected_tail_top_context_largest_center_token_share,1),SELECTIVE_METRICS.selected_tail_top_context_largest_center_token_share.help)].join("");
    }
    function comparisonSection(row) {
      const alignment = DATA.alignment[row.candidate], abs = Number(alignment?.abs_cosine), direct = Number(alignment?.cosine);
      const status = abs >= .99 ? {label:"high alignment",cls:""} : abs >= .95 ? {label:"moderate alignment",cls:"caution"} : {label:"basis drift",cls:"warn"};
      const effortCards = ["low","medium"].map(effort => {
        const item = rowIndex(effort).get(row.candidate), meta = DATA.efforts[effort].meta;
        return `<article class="effort-compare-card ${effort === state.effort ? "current" : ""}"><div class="effort-compare-title"><span>${effort} effort</span>${effort === state.effort ? '<span class="status-pill">current</span>' : ""}</div><div class="compare-values"><div class="compare-value"><span>Broad rank / score</span><b>#${integer.format(item?.rank_global_mean_abs_cosine)} · ${decimal(item?.mean_abs_cosine,4)}</b></div><div class="compare-value"><span>Tail rank / score</span><b>#${integer.format(item?.rank_global_tail_selectivity)} · ${decimal(item?.tail_selectivity_score,3)}</b></div><div class="compare-value"><span>Analysis exposure</span><b>${compact.format(meta.analysis_tokens)} tokens</b></div></div></article>`;
      }).join("");
      const fwBroad = finewebIndexes.broad[row.candidate], fwSelective = finewebIndexes.selective[row.candidate], fwMeta = DATA.fineweb.meta;
      const finewebCard = `<article class="effort-compare-card"><div class="effort-compare-title"><span>FineWeb reference</span><span class="status-pill">canonical bank</span></div><div class="compare-values"><div class="compare-value"><span>Broad rank / score</span><b>#${integer.format(fwBroad?.rank_global_mean_abs_cosine)} · ${decimal(fwBroad?.mean_abs_cosine,4)}</b></div><div class="compare-value"><span>Tail rank / score</span><b>#${integer.format(fwSelective?.rank_global_tail_selectivity)} · ${decimal(fwSelective?.tail_selectivity_score,3)}</b></div><div class="compare-value"><span>Corpus exposure</span><b>${compact.format(fwMeta.tokens)} tokens</b></div></div></article>`;
      let note = `FineWeb broad, FineWeb selective, and low CoT use a bit-identical direction bank. Medium ↔ canonical same-index cosine is <strong>${signed(direct,4)}</strong> (absolute ${decimal(abs,4)}). `;
      if (direct < 0) note += `<strong>The medium orientation flips</strong>, so its high/low tails reverse relative to FineWeb and low CoT. `;
      if (abs < .95) note += `This component drifts enough that same-index metrics and contexts may describe rotated directions. `;
      const bestIndex = state.effort === "low" ? alignment?.low_best_medium_sv : alignment?.medium_best_low_sv;
      const bestAbs = state.effort === "low" ? alignment?.low_best_abs_cosine : alignment?.medium_best_abs_cosine;
      if (Number(bestIndex) !== Number(row.sv_index_0) && Number(bestAbs) > abs + .05) note += `Its strongest cross-effort match is SV${String(bestIndex).padStart(2,"0")} at |cos| ${decimal(bestAbs,4)}.`;
      else note += `Same-index comparison is the strongest cross-effort match for this component.`;
      return `<section class="comparison"><div class="comparison-head"><div><h3>FineWeb ↔ CoT comparison</h3><p>Same model and indexed SV slot; corpus and reasoning observations remain separately normalized.</p></div><span class="alignment-badge ${status.cls}">medium ↔ canonical · ${status.label} · |cos| ${decimal(abs,3)}</span></div><div class="effort-compare-grid">${effortCards}${finewebCard}</div><p class="alignment-note">${note}</p></section>`;
    }
    function tailProfile(row) {
      return `<section class="tail-profile"><div class="tail-profile-head"><h3>Two-sided robust tail profile</h3><p>High (+) and low (−) are arbitrary effort-local SV orientations. Trace counts use per-trace extrema over analysis tokens.</p></div><div class="tail-grid">${tailSide(row,"positive")}${tailSide(row,"negative")}</div></section>`;
    }
    function tailSide(row, polarity) {
      const p = `${polarity}_`, selected = row.selected_tail_polarity === polarity;
      return `<article class="tail-side ${selected ? "selected" : ""}"><div class="tail-side-head"><h4>${polarity === "positive" ? "+ High tail" : "− Low tail"}</h4><span>${selected ? "score driver" : `score ${decimal(row[`${p}tail_selectivity_score`],3)}`}</span></div><div class="tail-stats">${tailStat("q99 robust z",decimal(row[`${p}q99_robust_z`],3))}${tailStat("q99.9 robust z",decimal(row[`${p}q999_robust_z`],3))}${tailStat("max robust z",decimal(row[`${p}max_robust_z`],3))}${tailStat("top-.1% z² energy",percent(row[`${p}top0_1pct_energy_share`],2))}${tailStat("traces above z=5",`${integer.format(row[`${p}doc_count_z5`])} · ${percent(row[`${p}doc_rate_z5`],1)}`)}${tailStat("largest center token",percent(row[`${p}top_context_largest_center_token_share`],1))}</div></article>`;
    }
    function tailStat(label,value) { return `<div class="tail-stat"><span>${esc(label)}</span><b>${esc(value)}</b></div>`; }
    function allMetricGroups(row) {
      const groups = new Map();
      Object.entries(row).filter(([key]) => key !== "candidate").forEach(([key,value]) => {
        let group = "Core activity";
        if (key.startsWith("rank_")) group = "Ranks"; else if (key.includes("unembed")) group = "Token geometry"; else if (key.includes("top_context")) group = "Retained-event diversity"; else if (key.startsWith("positive_")) group = "+ high tail"; else if (key.startsWith("negative_")) group = "− low tail"; else if (key.startsWith("abs_") || key.startsWith("selected_tail_") || ["token_sample_n","median_activation","mad_activation","robust_activation_scale","stable_skewness","stable_kurtosis","stable_excess_kurtosis","effective_support_fraction","tail_selectivity_score"].includes(key)) group = "Robust distribution";
        if (!groups.has(group)) groups.set(group,[]);
        groups.get(group).push(`<div class="all-metric"><span>${esc(LABELS[key] || key.replaceAll("_"," "))}</span><b>${esc(formatAny(key,value))}</b></div>`);
      });
      return `<div class="metric-groups">${[...groups.entries()].map(([label,items]) => `<section class="metric-group"><h4>${esc(label)}</h4><div class="all-metrics">${items.join("")}</div></section>`).join("")}</div>`;
    }
    function formatAny(key,value) {
      if (value == null) return "—"; if (typeof value === "string") return visibleToken(value);
      if (key.startsWith("rank_") || key.includes("_count_z") || ["layer","sv_index_0","n_tokens","n_documents","token_sample_n"].includes(key)) return integer.format(value);
      if (/(?:rate|share|fraction|weight)$/.test(key) || key.includes("energy_share")) return percent(value,2);
      return decimal(value,5);
    }

    function leftMetric(label,value,title="") {
      return `<div class="left-metric" title="${esc(title)}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
    }
    function svCandidate(layer,sv0) {
      return Number.isFinite(Number(layer)) && Number.isFinite(Number(sv0)) ? `L${String(layer).padStart(2,"0")}_SV${String(sv0).padStart(2,"0")}` : "—";
    }
    function leftSingularSection(row) {
      const data=effortData().left_singular?.[row.candidate];
      if(!data) return "";
      const alignment=DATA.alignment[row.candidate] || {};
      const validated=data.source === "normalized_jlens_times_saved_v";
      const bestCrossSv=state.effort === "low" ? alignment.low_best_medium_left_sv : alignment.medium_best_low_left_sv;
      const bestCrossAbs=state.effort === "low" ? alignment.low_best_left_abs_cosine : alignment.medium_best_left_abs_cosine;
      const other=otherEffort(state.effort);
      const previous=Number.isFinite(Number(data.previous_layer)) ? `${svCandidate(data.previous_layer,data.previous_best_sv0)} · |cos| ${decimal(data.previous_best_abs_cosine,4)}` : "first saved layer";
      const next=Number.isFinite(Number(data.next_layer)) ? `${svCandidate(data.next_layer,data.next_best_sv0)} · |cos| ${decimal(data.next_best_abs_cosine,4)}` : "last saved layer";
      const sameLayerBest=svCandidate(row.layer,data.best_right_sv0);
      const sameLayerNote=Number(data.best_right_sv0) === Number(row.sv_index_0) ? "The paired V is also the closest saved right vector." : `The paired V cosine is ${signed(data.paired_u_v_cosine,4)}; a different right component is closer in raw residual coordinates.`;
      const crossNote=Number(bestCrossSv) === Number(row.sv_index_0) ? "The same indexed U is the strongest cross-effort match." : `The strongest ${other}-effort U match is ${svCandidate(row.layer,bestCrossSv)} at |cos| ${decimal(bestCrossAbs,4)}.`;
      const tokenGeometry=data.token_geometry;
      const validationMetrics=validated ? `${leftMetric("Actual ||J·V||",decimal(data.actual_transport_gain,5),"Transport gain reconstructed directly from the J-Lens.")}${leftMetric("actual gain / σ",decimal(data.gain_over_stored_singular_value,7),"Agreement between direct J·V gain and the stored singular value.")}${leftMetric("cos(J·V, saved U)",decimal(data.transport_vs_stored_u_cosine,7),"Direction agreement between reconstructed transport and saved SVD U.")}${leftMetric("σU relative error",decimal(data.transport_vs_sigma_stored_u_relative_error,7),"Relative error between direct J·V and stored σU.")}` : "";
      const tokenPanel=tokenGeometry ? `<details class="left-token-section" open><summary><div class="token-summary"><div><p class="eyebrow">Output-side vocabulary geometry · U</p><h3>Left-vector token directions</h3><p>Full-vocabulary lm_head-row cosine against normalized J·V. These are geometric output-token neighbors, not observed activations.</p></div><b>max |cos| ${decimal(tokenGeometry.max_abs,4)} · ${integer.format(tokenGeometry.vocab)} tokens</b></div></summary><div class="token-content"><div class="token-toolbar"><label class="control token-limit">Tokens per side<select id="leftTokenLimit"></select></label></div><div class="token-grid"><section class="token-column aligned"><div class="token-head"><h4>+ Aligned with U</h4><span id="leftAlignedTokenCount"></span></div><div class="token-list" id="leftAlignedTokenList"></div></section><section class="token-column opposed"><div class="token-head"><h4>− Opposed to U</h4><span id="leftOpposedTokenCount"></span></div><div class="token-list" id="leftOpposedTokenList"></div></section></div></div></details>` : `<p class="left-caveat"><strong>U token neighbors are not embedded for this basis.</strong> This independently materialized bank still has its complete saved U diagnostics above. Full-vocabulary token projection can be added from the J-Lens and lm_head weights with zero transformer forward passes.</p>`;
      return `<section class="left-section"><div class="left-heading"><div><p class="eyebrow">Paired SVD output direction</p><h3>Corresponding left singular vector</h3><p>The selected right vector V is the input direction inspected by the activation scans. This panel uses normalized J·V when a fingerprint-matched reconstruction is available; otherwise it uses the scanner's paired saved U.</p></div><span class="left-source-badge ${validated ? "" : "saved"}">${validated ? "J·V validated" : "saved SVD U"}</span></div>
        <div class="sv-map"><div class="sv-node"><span>Right singular vector · input side</span><b>V · ${esc(row.candidate)}</b><small>Projected against residual states in the CoT and FineWeb scans</small></div><div class="sv-operator"><span>J-Lens transport</span><b>J · V = σU</b><small>σ ${decimal(data.singular_value,4)}</small></div><div class="sv-node left"><span>Left singular vector · output side</span><b>U · ${esc(row.candidate)}</b><small>${validated ? "Reconstructed from normalized J·V and checked against saved U" : "Loaded directly from the paired scanner SVD file"}</small></div></div>
        <div class="left-metrics">${leftMetric("Singular value σ",decimal(data.singular_value,5),"Gain assigned to this singular pair by the saved SVD.")}${leftMetric("Top-64 σ² share",percent(data.saved_spectral_energy_fraction,2),"Share of squared singular value mass within the saved top-64 bank, not the full spectrum.")}${leftMetric("Saved ||U||",decimal(data.u_norm,7),"Norm of the SVD file's U column before dashboard normalization.")}${leftMetric("Max other-U |cos|",decimal(data.max_other_u_abs_cosine,6),validated ? "Largest absolute cosine with another directly reconstructed J·V direction in this layer; nonzero values expose randomized-SVD transport mismatch." : "Largest absolute cosine with another saved U in this layer; near zero is expected.")}${leftMetric("paired cos(U,V)",signed(data.paired_u_v_cosine,5),"Raw residual-coordinate cosine. This is descriptive, not the SVD pairing criterion.")}${leftMetric("U energy in V₆₄ span",percent(data.right_bank_projection_fraction,2),"Squared projection into the layer's saved top-64 right-vector span.")}${validationMetrics}</div>
        <div class="left-relations"><article class="left-relation"><h4>Closest right vector in this layer</h4><b>${sameLayerBest} · |cos| ${decimal(data.best_right_abs_cosine,4)}</b><p>${sameLayerNote}</p></article><article class="left-relation"><h4>Left-vector continuity across layers</h4><b>← ${previous}</b><b>→ ${next}</b><p>Best absolute-cosine U match in each adjacent saved layer; signs may flip.</p></article><article class="left-relation"><h4>Low ↔ medium left-basis alignment</h4><b>same slot cos ${signed(alignment.left_cosine,4)}</b><p>${crossNote}</p></article></div>
        <p class="left-caveat">U and V occupy the same residual coordinate system, but they have different roles: V says which input perturbation the lens is sensitive to; U says the output direction that perturbation becomes. Sign is joint and arbitrary—flipping both U and V leaves the singular pair unchanged.</p>${tokenPanel}</section>`;
    }

    function bindLeftTokenControls(row) {
      const data=effortData().left_singular?.[row.candidate]?.token_geometry;
      if(!data || !$("#leftTokenLimit")) return;
      const max=Math.max(data.nearest.length,data.farthest.length), limits=[...new Set([8,16,max].filter(value=>value<=max))].sort((a,b)=>a-b);
      if(!limits.includes(state.leftTokenLimit)) state.leftTokenLimit=limits[0]||max;
      $("#leftTokenLimit").innerHTML=limits.map(value=>`<option value="${value}">${value===max?`All ${value}`:value}</option>`).join("");
      $("#leftTokenLimit").value=String(state.leftTokenLimit);
      $("#leftTokenLimit").addEventListener("change",event=>{state.leftTokenLimit=Number(event.target.value);renderLeftTokenRows(data);});
      renderLeftTokenRows(data);
    }
    function renderLeftTokenRows(data) {
      renderGeometryTokenColumn("leftAligned",data.nearest,data,state.leftTokenLimit);
      renderGeometryTokenColumn("leftOpposed",data.farthest,data,state.leftTokenLimit);
    }
    function renderGeometryTokenColumn(side,items,data,limit) {
      const list=$(`#${side}TokenList`); if(!list)return;
      const shown=items.slice(0,limit); $(`#${side}TokenCount`).textContent=`${shown.length} of ${items.length}`;
      list.innerHTML=shown.map((item,index)=>{const label=item.decoded||item.token||`token #${item.id}`,z=(Number(item.cosine)-Number(data.mean))/Math.max(Number(data.std),1e-9);return `<article class="token-row"><span class="token-rank">#${index+1}</span><span class="token-name"><b>${esc(visibleToken(label))}</b><small>${item.token?`raw ${esc(visibleToken(item.token))} · `:""}id ${integer.format(item.id)}</small></span><span class="token-value">${signed(item.cosine,4)}<br>${signed(z,2)}σ</span></article>`;}).join("");
    }

    function traceSection(row) {
      return `<section class="trace-section"><div class="section-heading"><div><p class="eyebrow">Reasoning trace workbench · ${esc(state.effort)} effort</p><h3>Activation events in chain of thought</h3><p>Browse retained high/low projection events, see where they occur in the generated response, and open the joined prompt → analysis → final trace. Full traces are rendered only for the selected event.</p><span class="sign-note">Timeline position is (full-sequence token position − prompt tokens) / generated tokens. The highlighted center token is an observed activation, not a causal attribution.</span></div></div>
        <div class="trace-controls"><label class="control">Search events<input id="contextSearch" type="search" value="${esc(state.contextQuery)}" placeholder="Snippet, task, category, prompt…"></label><label class="control">Events shown<select id="contextLimit"></select></label><label class="check"><input id="dedupeTraces" type="checkbox" ${state.dedupe ? "checked" : ""}> One event per trace</label></div>
        <div class="trace-workbench"><aside class="event-browser"><div class="polarity-switch"><button class="polarity-button" type="button" data-polarity="positive"><b>+ High tail</b><small id="positiveEventCount"></small></button><button class="polarity-button" type="button" data-polarity="negative"><b>− Low tail</b><small id="negativeEventCount"></small></button></div><div class="footprint" id="footprint"></div><div class="event-list" id="eventList"></div></aside><article class="trace-viewer" id="traceViewer"></article></div></section>`;
    }
    function bindTraceControls(row) {
      const max = effortData().meta.embedded_contexts_per_side;
      const limits = [...new Set([6,12,max].filter(value => value > 0 && value <= max))].sort((a,b)=>a-b);
      if (!limits.includes(state.contextLimit)) state.contextLimit = limits[0] || max;
      $("#contextLimit").innerHTML = limits.map(value => `<option value="${value}">${value === max ? `All ${value}` : value}</option>`).join(""); $("#contextLimit").value = String(state.contextLimit);
      $("#contextSearch").addEventListener("input", event => { state.contextQuery = event.target.value.trim().toLowerCase(); renderEvents(row); });
      $("#contextLimit").addEventListener("change", event => { state.contextLimit = Number(event.target.value); renderEvents(row); });
      $("#dedupeTraces").addEventListener("change", event => { state.dedupe = event.target.checked; renderEvents(row); });
      document.querySelectorAll(".polarity-button").forEach(button => button.addEventListener("click", () => { state.polarity = button.dataset.polarity; state.traceFocus = null; renderEvents(row); }));
      renderEvents(row);
    }
    function filteredEvents(items) {
      const seen = new Set(), query = state.contextQuery;
      return items.filter(item => {
        const rollout = effortData().rollouts[String(item.document)];
        const prompt = (rollout?.messages || []).map(message => message.content).join(" ");
        if (query && !`${item.marked} ${item.token} ${rollout?.prompt_id} ${rollout?.category} ${rollout?.difficulty} ${prompt}`.toLowerCase().includes(query)) return false;
        if (state.dedupe && seen.has(item.document)) return false;
        if (state.dedupe) seen.add(item.document); return true;
      });
    }
    function eventKey(item) { return item ? `${item.polarity}/${item.rank}/${item.document}` : ""; }
    function renderEvents(row) {
      const grouped = effortData().contexts[row.candidate] || {positive:[],negative:[]};
      const all = grouped[state.polarity] || [], filtered = filteredEvents(all), shown = filtered.slice(0,state.contextLimit);
      document.querySelectorAll(".polarity-button").forEach(button => button.setAttribute("aria-pressed",String(button.dataset.polarity === state.polarity)));
      $("#positiveEventCount").textContent = `${filteredEvents(grouped.positive).length} visible`;
      $("#negativeEventCount").textContent = `${filteredEvents(grouped.negative).length} visible`;
      renderFootprint(row);
      if (!state.traceFocus || state.traceFocus.polarity !== state.polarity || !all.some(item => eventKey(item) === eventKey(state.traceFocus))) state.traceFocus = shown[0] || all[0] || null;
      const list = $("#eventList"); list.replaceChildren();
      if (!shown.length) { const empty=document.createElement("div"); empty.className="empty"; empty.textContent="No retained events match this search."; list.append(empty); renderTraceViewer(row); return; }
      shown.forEach(item => list.append(eventCard(item,row)));
      renderTraceViewer(row);
    }
    function renderFootprint(row) {
      const summary = effortData().context_summaries[row.candidate]?.[state.polarity]; if (!summary) return;
      const phases = summary.phases, total = Math.max(summary.events,1);
      const tokenChips = summary.top_tokens.slice(0,5).map(item => `<span class="data-chip" title="${esc(item.count)} of ${esc(summary.events)} retained source events"><b>${esc(visibleToken(item.token))}</b> · ${percent(item.share,0)}</span>`).join("");
      const categoryChips = summary.top_categories.slice(0,3).map(item => `<span class="data-chip"><b>${esc(item.category)}</b> · ${integer.format(item.count)}</span>`).join("");
      $("#footprint").innerHTML = `<div class="footprint-row"><span>${integer.format(summary.unique_traces)} unique traces / ${integer.format(summary.events)} source events</span><span>median response position ${percent(summary.median_progress,0)}</span></div><div class="phase-bar" title="Early ${phases.early}, middle ${phases.middle}, late ${phases.late}"><i style="width:${phases.early/total*100}%"></i><i style="width:${phases.middle/total*100}%"></i><i style="width:${phases.late/total*100}%"></i></div><div class="chip-row">${tokenChips}</div><div class="chip-row">${categoryChips}</div>`;
    }
    function eventCard(item,row) {
      const rollout = effortData().rollouts[String(item.document)], card=document.createElement("article");
      card.className=`event-card ${eventKey(item) === eventKey(state.traceFocus) ? "active" : ""}`;
      const scale=Math.max(Number(row.robust_activation_scale),1e-12), centered=(Number(item.activation)-Number(row.median_activation))/scale, tailZ=item.polarity === "positive" ? centered : -centered;
      const meta=document.createElement("div"); meta.className="event-meta"; meta.innerHTML=`<span>${item.polarity === "positive" ? "+" : "−"} event #${integer.format(item.rank)} · <b>${esc(rollout?.prompt_id)}</b></span><span>act <b>${esc(signed(item.activation))}</b> · z <b>${esc(decimal(tailZ,2))}</b></span>`;
      const copy=document.createElement("p"); copy.className="event-copy"; appendMarkedText(copy,item.marked || "");
      const footer=document.createElement("div"); footer.className="event-footer"; const left=document.createElement("span"); left.innerHTML=`<span class="event-token" title="Raw token: ${esc(item.token)}">${esc(visibleToken(item.token))}</span> <span class="count">${esc(rollout?.category)} · ${percent(item.progress,0)} response</span>`;
      const open=document.createElement("button"); open.type="button"; open.className="open-trace"; open.textContent="Read full trace →"; open.addEventListener("click",()=>openTrace(item,row)); footer.append(left,open);
      const progress=document.createElement("span"); progress.className="event-progress"; progress.innerHTML=`<i style="width:${Math.max(1,item.progress*100).toFixed(2)}%"></i>`;
      card.addEventListener("click",event=>{ if (!event.target.closest("button")) openTrace(item,row); }); card.append(meta,copy,footer,progress); return card;
    }
    function openTrace(item,row) { state.traceFocus=item; renderEvents(row); $("#traceViewer").scrollIntoView({behavior:"smooth",block:"nearest"}); }
    function renderTraceViewer(row) {
      const root=$("#traceViewer"), item=state.traceFocus;
      if (!item) { root.innerHTML=`<div class="empty">Choose an activation event to open its reasoning trace.</div>`; return; }
      const rollout=effortData().rollouts[String(item.document)];
      if (!rollout) { root.innerHTML=`<div class="empty">The joined rollout is unavailable.</div>`; return; }
      root.innerHTML=`<div class="trace-viewer-head"><div><h4>${esc(rollout.prompt_id)}</h4><p>${esc(rollout.category)} · ${esc(rollout.difficulty)} · event #${integer.format(item.rank)} at ${percent(item.progress,0)} of generated response</p></div><div class="trace-badges"><span class="status-pill">${esc(state.effort)} effort</span>${rollout.hit_cap ? '<span class="status-pill warn">hit generation cap</span>' : ""}${rollout.final?.trim() ? '<span class="status-pill">final captured</span>' : '<span class="status-pill warn">no final channel</span>'}</div></div><div class="trace-body"><section class="trace-block"><div class="trace-block-label"><span>Prompt</span><span>excluded from activation scan · ${integer.format(rollout.prompt_tokens)} tokens</span></div><div id="tracePrompt"></div></section><section class="trace-block"><div class="trace-block-label"><span>Generated analysis</span><span>scanned · ${integer.format(rollout.analysis_tokens)} tokens</span></div><p class="trace-text" id="traceReasoning"></p></section><section class="trace-block"><div class="trace-block-label"><span>Final response</span><span>excluded · ${integer.format(rollout.final_tokens)} tokens</span></div><p class="trace-text" id="traceFinal"></p></section></div>`;
      const promptRoot=$("#tracePrompt"); (rollout.messages || []).forEach((message,index)=>{ const p=document.createElement("p"); p.className="trace-text"; p.textContent=`${String(message.role).toUpperCase()}\n${message.content}`; if(index) p.style.marginTop="12px"; promptRoot.append(p); });
      appendFocusedReasoning($("#traceReasoning"),rollout.reasoning || "",item);
      $("#traceFinal").textContent=rollout.final?.trim() || (rollout.hit_cap ? "No final-channel response was captured before the generation cap." : "No final-channel response was captured.");
      if (!rollout.final?.trim()) $("#traceFinal").classList.add("trace-empty");
    }
    function appendFocusedReasoning(target,text,item) {
      if (!text) { target.textContent="No analysis text was captured."; target.classList.add("trace-empty"); return; }
      const plain=String(item.plain || ""), index=plain ? text.indexOf(plain) : -1;
      if (index < 0) { const focus=document.createElement("span"); focus.className="trace-focus"; appendMarkedText(focus,item.marked || ""); target.append(focus,document.createTextNode(`\n\n${text}`)); return; }
      target.append(document.createTextNode(text.slice(0,index))); const focus=document.createElement("span"); focus.className="trace-focus"; appendMarkedText(focus,item.marked || plain); target.append(focus,document.createTextNode(text.slice(index+plain.length)));
    }
    function appendMarkedText(target,text) {
      let cursor=0; while(cursor<text.length){ const start=text.indexOf("⟦",cursor); if(start<0){target.append(document.createTextNode(text.slice(cursor)));break;} const end=text.indexOf("⟧",start+1); if(end<0){target.append(document.createTextNode(text.slice(cursor)));break;} target.append(document.createTextNode(text.slice(cursor,start))); const mark=document.createElement("mark"); mark.textContent=text.slice(start+1,end); target.append(mark); cursor=end+1; } if(!text.length) target.textContent="(empty context)";
    }

    function finewebBasisStatus() {
      if (state.effort === "low") return { cls:"", label:"exact shared basis · polarity aligned", detail:"FineWeb and low-effort CoT use bit-identical vectors, so native high/low orientation is directly comparable." };
      const alignment=DATA.alignment[currentRow()?.candidate], direct=Number(alignment?.cosine), abs=Number(alignment?.abs_cosine);
      if (abs < .95) return { cls:"warn", label:`basis drift · |cos| ${decimal(abs,3)}`, detail:"The same-index medium component has materially drifted. These FineWeb contexts are a canonical-slot reference, not observed evidence for the current medium vector; polarity correspondence is unreliable." };
      if (direct < 0) return { cls:"flip", label:`orientation flips · cos ${signed(direct,3)}`, detail:"The same-index medium direction is aligned in magnitude but sign-reversed: FineWeb high corresponds approximately to medium CoT low, and vice versa." };
      return { cls:abs >= .99 ? "" : "flip", label:`same-index aligned · cos ${signed(direct,3)}`, detail:"The FineWeb canonical vector and current medium vector are aligned at the same SVD slot; the displayed contexts remain FineWeb observations, not CoT activations." };
    }
    function finewebSection(row) {
      const broad=finewebIndexes.broad[row.candidate], selective=finewebIndexes.selective[row.candidate];
      if (!broad || !selective) return "";
      const status=finewebBasisStatus();
      const broadCard=`<button class="fineweb-scan-card" type="button" data-fineweb-source="broad"><span class="fineweb-scan-title"><b>Broad-run extremes</b><span>FineWeb #${integer.format(broad.rank_global_mean_abs_cosine)}</span></span><span class="fineweb-scan-values"><span>Mean |cosine|<b>${decimal(broad.mean_abs_cosine,4)}</b></span><span>Top-5 document presence<b>${percent(broad.doc_top5_presence_rate,1)}</b></span><span>Mean document peak<b>${decimal(broad.mean_document_peak_abs,3)}</b></span></span></button>`;
      const driver=selective.selected_tail_polarity === "positive" ? "+ high" : "− low";
      const selectiveCard=`<button class="fineweb-scan-card" type="button" data-fineweb-source="selective"><span class="fineweb-scan-title"><b>Selectivity-run extremes</b><span>FineWeb #${integer.format(selective.rank_global_tail_selectivity)}</span></span><span class="fineweb-scan-values"><span>Tail score<b>${decimal(selective.tail_selectivity_score,3)}</b></span><span>Selected q99.9 robust z<b>${decimal(selective.selected_tail_q999_robust_z,3)}</b></span><span>Score-driving tail<b>${driver}</b></span></span></button>`;
      return `<section class="fineweb-section" id="finewebEvidence"><div class="fineweb-heading"><div><p class="eyebrow">Cross-corpus transfer check · FineWeb</p><h3>How this SV responds in general web text</h3><p>These cards are observed projection extremes from ${integer.format(DATA.fineweb.meta.tokens)} tokens across ${integer.format(DATA.fineweb.meta.documents)} FineWeb windows. Switch between the original broad-run reservoir and the larger selectivity-run reservoir; tied quantized activations can produce different retained examples.</p><span class="sign-note">${status.detail} High (+) and low (−) are arbitrary vector orientations, not sentiment.</span></div><span class="basis-badge ${status.cls}">${status.label}</span></div><div class="fineweb-scan-grid" role="group" aria-label="FineWeb scan output">${broadCard}${selectiveCard}</div><div class="fineweb-context-grid"><section class="fineweb-column positive"><div class="fineweb-column-head"><h4>FineWeb + high</h4><span id="finewebPositiveMapping"></span></div><div class="fineweb-list" id="finewebPositiveList"></div></section><section class="fineweb-column negative"><div class="fineweb-column-head"><h4>FineWeb − low</h4><span id="finewebNegativeMapping"></span></div><div class="fineweb-list" id="finewebNegativeList"></div></section></div></section>`;
    }
    function bindFinewebControls(row) {
      if (!$("#finewebEvidence")) return;
      document.querySelectorAll("[data-fineweb-source]").forEach(button => button.addEventListener("click",()=>{ state.finewebSource=button.dataset.finewebSource; renderFineweb(row); }));
      renderFineweb(row);
    }
    function finewebPolarityMapping(polarity) {
      if (state.effort === "low") return `matches low CoT ${polarity === "positive" ? "+ high" : "− low"}`;
      const alignment=DATA.alignment[currentRow()?.candidate], direct=Number(alignment?.cosine), abs=Number(alignment?.abs_cosine);
      if (abs < .95) return "same-index reference only";
      const mapped=direct < 0 ? (polarity === "positive" ? "− low" : "+ high") : (polarity === "positive" ? "+ high" : "− low");
      return `≈ medium CoT ${mapped}`;
    }
    function renderFineweb(row) {
      document.querySelectorAll("[data-fineweb-source]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.finewebSource === state.finewebSource)));
      const source=DATA.fineweb.sources[state.finewebSource], grouped=source?.contexts[row.candidate] || {positive:[],negative:[]};
      $("#finewebPositiveMapping").textContent=`${finewebPolarityMapping("positive")} · ${grouped.positive.length} shown`;
      $("#finewebNegativeMapping").textContent=`${finewebPolarityMapping("negative")} · ${grouped.negative.length} shown`;
      renderFinewebColumn("finewebPositiveList",grouped.positive,"positive",row);
      renderFinewebColumn("finewebNegativeList",grouped.negative,"negative",row);
    }
    function renderFinewebColumn(id,items,polarity,row) {
      const root=$(`#${id}`); root.replaceChildren();
      if (!items.length) { const empty=document.createElement("div"); empty.className="empty"; empty.textContent="No embedded FineWeb contexts for this side."; root.append(empty); return; }
      items.forEach(item=>root.append(finewebContextCard(item,polarity,row)));
    }
    function safeHttpUrl(value) {
      try { const url=new URL(String(value)); return ["http:","https:"].includes(url.protocol) ? url : null; } catch (_) { return null; }
    }
    function finewebContextCard(item,polarity,row) {
      const card=document.createElement("article"); card.className="fineweb-context-card";
      const metricRow=finewebIndexes[state.finewebSource][row.candidate], scale=Math.max(Number(metricRow?.robust_activation_scale),1e-12), centered=(Number(item.activation)-Number(metricRow?.median_activation))/scale;
      const tailZ=polarity === "positive" ? centered : -centered, zText=state.finewebSource === "selective" && Number.isFinite(tailZ) ? ` · z <b>${esc(decimal(tailZ,2))}</b>` : "";
      const meta=document.createElement("div"); meta.className="fineweb-context-meta"; meta.innerHTML=`<span>${polarity === "positive" ? "+" : "−"} context #${integer.format(item.rank)}</span><span>act <b>${esc(signed(item.activation))}</b> · cos <b>${esc(signed(item.cosine,4))}</b>${zText}</span>`;
      const copy=document.createElement("p"); copy.className="fineweb-context-copy"; appendMarkedText(copy,item.marked || "");
      const footer=document.createElement("div"); footer.className="fineweb-context-footer";
      const token=document.createElement("span"); token.className="event-token"; token.title=`Raw token: ${item.token ?? ""}`; token.textContent=visibleToken(item.token);
      const source=document.createElement("span"); source.className="fineweb-source"; const url=safeHttpUrl(item.url);
      if (url) { const link=document.createElement("a"); link.href=url.href; link.target="_blank"; link.rel="noreferrer noopener"; link.textContent=url.hostname.replace(/^www\./,""); source.append(link); }
      else source.append(document.createTextNode(item.dump || "FineWeb source"));
      if (item.date) source.append(document.createTextNode(` · ${String(item.date).slice(0,10)}`));
      footer.append(token,source); card.append(meta,copy,footer); return card;
    }

    function tokenSection(row) {
      const data=effortData().unembedding[row.candidate]; if(!data) return "";
      return `<details class="token-section"><summary><div class="token-summary"><div><p class="eyebrow">Input-side vocabulary geometry · V</p><h3>Right-vector token-space neighbors</h3><p>Geometric neighbors of the right singular vector used by the ${state.effort}-effort activation scan. They are not observed activations or semantic proof.</p></div><b>max |cos| ${decimal(data.max_abs,4)} · expand ↓</b></div></summary><div class="token-content"><div class="token-toolbar"><label class="control token-limit">Tokens per side<select id="tokenLimit"></select></label></div><div class="token-grid"><section class="token-column aligned"><div class="token-head"><h4>+ Aligned with V</h4><span id="alignedTokenCount"></span></div><div class="token-list" id="alignedTokenList"></div></section><section class="token-column opposed"><div class="token-head"><h4>− Opposed to V</h4><span id="opposedTokenCount"></span></div><div class="token-list" id="opposedTokenList"></div></section></div></div></details>`;
    }
    function bindTokenControls(row) {
      const data=effortData().unembedding[row.candidate]; if(!data || !$("#tokenLimit")) return;
      const max=Math.max(data.nearest.length,data.farthest.length), limits=[...new Set([8,16,max].filter(value=>value<=max))].sort((a,b)=>a-b);
      if(!limits.includes(state.tokenLimit)) state.tokenLimit=limits[0]||max;
      $("#tokenLimit").innerHTML=limits.map(value=>`<option value="${value}">${value===max?`All ${value}`:value}</option>`).join(""); $("#tokenLimit").value=String(state.tokenLimit); $("#tokenLimit").addEventListener("change",event=>{state.tokenLimit=Number(event.target.value);renderTokenRows(data);}); renderTokenRows(data);
    }
    function renderTokenRows(data) { renderTokenColumn("aligned",data.nearest,data); renderTokenColumn("opposed",data.farthest,data); }
    function renderTokenColumn(side,items,data) {
      const list=$(`#${side}TokenList`); if(!list)return; const shown=items.slice(0,state.tokenLimit); $(`#${side}TokenCount`).textContent=`${shown.length} of ${items.length}`; list.innerHTML=shown.map((item,index)=>{ const label=item.decoded||item.token||`token #${item.id}`, z=(Number(item.cosine)-Number(data.mean))/Math.max(Number(data.std),1e-9); return `<article class="token-row"><span class="token-rank">#${index+1}</span><span class="token-name"><b>${esc(visibleToken(label))}</b><small>${item.token?`raw ${esc(visibleToken(item.token))} · `:""}id ${integer.format(item.id)}</small></span><span class="token-value">${signed(item.cosine,4)}<br>${signed(z,2)}σ</span></article>`;}).join("");
    }
    function copyCandidate() {
      const button=$("#copyCandidate"), value=viewState().selected, done=()=>{button.textContent="Copied";setTimeout(()=>button.textContent="Copy ID",1200);};
      if(navigator.clipboard?.writeText)navigator.clipboard.writeText(value).then(done).catch(()=>fallbackCopy(value,done));else fallbackCopy(value,done);
    }
    function fallbackCopy(text,done){const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.append(area);area.select();document.execCommand("copy");area.remove();done();}

    setup();
  </script>
</body>
</html>'''


def summarize_source(name: str, payload: dict[str, Any]) -> str:
    embedded_contexts = sum(
        len(items)
        for effort in payload["efforts"].values()
        for candidate in effort["contexts"].values()
        for items in candidate.values()
    )
    embedded_neighbors = sum(
        len(items)
        for effort in payload["efforts"].values()
        for candidate in effort["unembedding"].values()
        for side, items in candidate.items()
        if side in ("nearest", "farthest")
    )
    embedded_fineweb_contexts = sum(
        len(items)
        for source in payload["fineweb"]["sources"].values()
        for candidate in source["contexts"].values()
        for items in candidate.values()
    )
    embedded_left_vectors = sum(
        len(effort["left_singular"]) for effort in payload["efforts"].values()
    )
    embedded_left_neighbors = sum(
        len(items)
        for effort in payload["efforts"].values()
        for record in effort["left_singular"].values()
        for side, items in (record.get("token_geometry") or {}).items()
        if side in ("nearest", "farthest")
    )
    low_meta = payload["efforts"]["low"]["meta"]
    medium_meta = payload["efforts"]["medium"]["meta"]
    return (
        f"[{name}] {low_meta['embedded_candidates']:,} low-effort and "
        f"{medium_meta['embedded_candidates']:,} medium-effort candidates, "
        f"{embedded_contexts:,} CoT context events, "
        f"{embedded_fineweb_contexts:,} FineWeb contexts, "
        f"{embedded_neighbors:,} token neighbors, "
        f"{embedded_left_vectors:,} paired left vectors with "
        f"{embedded_left_neighbors:,} left-token neighbors, "
        f"and {low_meta['rollouts'] + medium_meta['rollouts']:,} full rollouts"
    )


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()

    def build_source(low: Path, medium: Path, fineweb: Path, selectivity: Path) -> dict[str, Any]:
        return build_payload(
            low.expanduser().resolve(),
            medium.expanduser().resolve(),
            fineweb.expanduser().resolve(),
            selectivity.expanduser().resolve(),
            args.top,
            args.contexts_per_side,
            args.fineweb_contexts_per_side,
        )

    sources: dict[str, Any] = {}
    order: list[str] = []

    # Current-token (original) source — unchanged default behavior.
    sources["unrealized"] = build_source(
        args.low_data_dir,
        args.medium_data_dir,
        args.fineweb_data_dir,
        args.fineweb_selectivity_data_dir,
    )
    order.append("unrealized")
    print(summarize_source("current-token", sources["unrealized"]))

    # Predictive source — added when its scan directories are present.
    if not args.no_predictive:
        predictive_dirs = [
            args.predictive_low_data_dir,
            args.predictive_medium_data_dir,
            args.predictive_fineweb_data_dir,
            args.predictive_fineweb_selectivity_data_dir,
        ]
        if all(path.expanduser().resolve().exists() for path in predictive_dirs):
            sources["predictive"] = build_source(*predictive_dirs)
            order.append("predictive")
            print(summarize_source("predictive", sources["predictive"]))
        else:
            print("Predictive scan directories not found; building current-token source only.")

    combined = {
        "default_source": "unrealized",
        "source_order": order,
        "source_labels": {name: SOURCE_LABELS[name] for name in order},
        "sources": sources,
    }
    html = HTML_TEMPLATE.replace("__COT_DASHBOARD_PAYLOAD__", safe_script_json(combined))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"Wrote {output}")
    print(f"Attribution sources embedded: {', '.join(order)}")
    print(f"Output size: {output.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
