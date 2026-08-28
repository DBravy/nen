#!/usr/bin/env python3
"""Build a self-contained atlas for predictively aligned J-Lens directions.

The canonical scan reads h[t-1] and attributes the projection to target token
t.  A mirror directory is validated instead of duplicated in the payload, and
the original current-token scan is included as a same-basis reference.
"""

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
DEFAULT_INPUT = ROOT / "predictive_words_scan"
DEFAULT_MIRROR = ROOT / "predictive_words_fineweb"
DEFAULT_ORIGINAL_BASIS = ROOT / "unrealized_words_fineweb"
DEFAULT_ORIGINAL_RANKINGS = ROOT / "unrealized_words_selectivity" / "selectivity_rankings.csv"
DEFAULT_COT_LOW = ROOT / "predictive_cot_low"
DEFAULT_COT_MEDIUM = ROOT / "predictive_cot_medium"
DEFAULT_OUTPUT = ROOT / "predictive_words_report.html"
BROAD_RANK = "rank_global_mean_abs_cosine"
TAIL_RANK = "rank_global_tail_selectivity"
TEXT_FIELDS = {
    "candidate",
    "selected_tail_polarity",
    "nearest_unembed_token",
    "nearest_unembed_decoded",
    "farthest_unembed_token",
    "farthest_unembed_decoded",
}
SPECIAL_TOKEN = re.compile(r"^<\|[^>]+\|>$")
HARMONY_TOKEN = re.compile(r"<\|[^>]+\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical predictive scan directory")
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR, help="Mirror predictive scan directory to validate")
    parser.add_argument(
        "--original-basis",
        type=Path,
        default=DEFAULT_ORIGINAL_BASIS,
        help="Original current-token scan whose V/S bank must equal the canonical bank",
    )
    parser.add_argument(
        "--original-rankings",
        type=Path,
        default=DEFAULT_ORIGINAL_RANKINGS,
        help="Original current-token selectivity_rankings.csv used for metric comparison",
    )
    parser.add_argument("--cot-low-dir", type=Path, default=DEFAULT_COT_LOW)
    parser.add_argument("--cot-medium-dir", type=Path, default=DEFAULT_COT_MEDIUM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=250, help="Candidates retained per lens; 0 retains all")
    parser.add_argument(
        "--contexts-per-side",
        type=int,
        default=24,
        help="Highest-ranked contexts embedded per candidate/polarity; 0 retains all",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.10,
        help="Maximum selected-tail special-read share eligible for the screened-tail lens",
    )
    parser.add_argument(
        "--cot-contexts-per-side",
        type=int,
        default=12,
        help="Highest-ranked CoT transition events embedded per direction/polarity; 0 retains all",
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


def display_candidate(layer: int, sv0: int) -> str:
    return f"L{layer:02d}_SV{sv0:02d}"


def source_candidate(layer: int, sv0: int) -> str:
    return f"L{layer:02d}_SV{sv0 + 1:02d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_scan_files(data_dir: Path) -> tuple[Path, ...]:
    return tuple(
        data_dir / name
        for name in (
            "metadata.json",
            "sv_rankings.csv",
            "selectivity_rankings.csv",
            "top_contexts.jsonl",
            "unembedding_neighbors.jsonl",
        )
    )


def validate_inputs(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input(s): " + ", ".join(missing))


def validate_mirror(canonical: Path, mirror: Path) -> dict[str, Any]:
    validate_inputs(*required_scan_files(canonical), *required_scan_files(mirror))
    substantive = (
        "sv_rankings.csv",
        "selectivity_rankings.csv",
        "top_contexts.jsonl",
        "unembedding_neighbors.jsonl",
    )
    artifact_hashes: dict[str, str] = {}
    for name in substantive:
        left = file_sha256(canonical / name)
        right = file_sha256(mirror / name)
        if left != right:
            raise SystemExit(f"Predictive mirror differs for substantive artifact {name}")
        artifact_hashes[name] = left

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to validate direction banks") from exc

    canonical_meta = json.loads((canonical / "metadata.json").read_text(encoding="utf-8"))
    mirror_meta = json.loads((mirror / "metadata.json").read_text(encoding="utf-8"))
    layers = [int(layer) for layer in canonical_meta.get("layers", [])]
    if layers != [int(layer) for layer in mirror_meta.get("layers", [])]:
        raise SystemExit("Predictive mirror metadata has different layers")
    slots = 0
    for layer in layers:
        left_path = canonical / "directions" / f"L{layer:02d}.npz"
        right_path = mirror / "directions" / f"L{layer:02d}.npz"
        validate_inputs(left_path, right_path)
        with np.load(left_path) as left, np.load(right_path) as right:
            for key in ("V", "S"):
                if key not in left or key not in right or not np.array_equal(left[key], right[key]):
                    raise SystemExit(f"Predictive mirror {key} differs at layer {layer}")
            slots += int(left["S"].shape[0])

    ignored = {"svd_method"}
    left_meta = {key: value for key, value in canonical_meta.items() if key not in ignored}
    right_meta = {key: value for key, value in mirror_meta.items() if key not in ignored}
    if left_meta != right_meta:
        keys = sorted(key for key in set(left_meta) | set(right_meta) if left_meta.get(key) != right_meta.get(key))
        raise SystemExit("Predictive mirror metadata differs beyond svd_method: " + ", ".join(keys))
    return {
        "canonical_svd_method": canonical_meta.get("svd_method"),
        "mirror_svd_method": mirror_meta.get("svd_method"),
        "metadata_conflict": canonical_meta.get("svd_method") != mirror_meta.get("svd_method"),
        "substantive_artifacts_identical": True,
        "direction_arrays_identical": True,
        "direction_slots": slots,
        "artifact_hashes": artifact_hashes,
    }


def validate_original_basis(canonical: Path, original_basis: Path, layers: list[int]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to validate the original direction bank") from exc
    slots = 0
    for layer in layers:
        predictive_path = canonical / "directions" / f"L{layer:02d}.npz"
        original_path = original_basis / "directions" / f"L{layer:02d}.npz"
        validate_inputs(predictive_path, original_path)
        with np.load(predictive_path) as predictive, np.load(original_path) as original:
            for key in ("V", "S"):
                if key not in predictive or key not in original or not np.array_equal(predictive[key], original[key]):
                    raise SystemExit(f"Predictive and original {key} banks differ at layer {layer}")
            slots += int(predictive["S"].shape[0])
    return {"arrays_identical": True, "direction_slots": slots, "reference": str(original_basis)}


def read_rankings(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate", "layer", "sv_index_0", BROAD_RANK, TAIL_RANK, "selected_tail_polarity"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} is missing required ranking fields")
        rows: list[dict[str, Any]] = []
        source_by_display: dict[str, str] = {}
        for line_number, raw in enumerate(reader, 2):
            try:
                row = {key: (value if key in TEXT_FIELDS else number(value)) for key, value in raw.items()}
            except ValueError as exc:
                raise SystemExit(f"Invalid ranking value on {path}:{line_number}: {exc}") from exc
            layer, sv0 = int(row["layer"]), int(row["sv_index_0"])
            expected = source_candidate(layer, sv0)
            if row["candidate"] != expected:
                raise SystemExit(f"Unexpected source candidate {row['candidate']!r}; expected {expected!r}")
            display = display_candidate(layer, sv0)
            row["candidate"] = display
            row.pop("sv_rank_1based", None)
            if "singular_value_over_sv1" in row:
                row["singular_value_over_sv0"] = row.pop("singular_value_over_sv1")
            polarity = row.get("selected_tail_polarity")
            for suffix in (
                "max_robust_z",
                "q99_robust_z",
                "q999_robust_z",
                "q9999_robust_z",
                "top0_1pct_energy_share",
                "doc_count_z5",
                "doc_rate_z5",
            ):
                row[f"selected_tail_{suffix}"] = row.get(f"{polarity}_{suffix}")
            rows.append(row)
            source_by_display[display] = expected
    return rows, source_by_display


def is_special_read(raw: dict[str, Any]) -> bool:
    previous = str(raw.get("prev_token") or "")
    return int(raw.get("read_position") or 0) == 0 or bool(SPECIAL_TOKEN.match(previous))


def transition_label(previous: str, target: str) -> str:
    return f"{previous} → {target}"


def scan_context_summaries(path: Path, source_by_display: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    display_by_source = {source: display for display, source in source_by_display.items()}
    state: dict[tuple[str, str], dict[str, Any]] = {}
    total_source = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_source += 1
            raw = json.loads(line)
            display = display_by_source.get(str(raw.get("candidate")))
            if display is None:
                continue
            polarity = str(raw.get("polarity"))
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected polarity on {path}:{line_number}")
            key = (display, polarity)
            item = state.setdefault(
                key,
                {
                    "events": 0,
                    "special": 0,
                    "documents": set(),
                    "previous": Counter(),
                    "targets": Counter(),
                    "transitions": Counter(),
                },
            )
            previous = str(raw.get("prev_token") or "")
            target = str(raw.get("token") or "")
            item["events"] += 1
            item["special"] += int(is_special_read(raw))
            item["documents"].add(int(raw.get("document_index") or 0))
            item["previous"][previous] += 1
            item["targets"][target] += 1
            item["transitions"][(previous, target)] += 1

    counts = {int(item["events"]) for item in state.values()}
    expected_groups = len(source_by_display) * 2
    if len(state) != expected_groups or len(counts) != 1:
        raise SystemExit(
            f"Context summary groups are inconsistent: {len(state)}/{expected_groups}, counts={sorted(counts)}"
        )
    summaries: dict[str, Any] = {candidate: {} for candidate in source_by_display}
    for (candidate, polarity), item in state.items():
        events = int(item["events"])
        summaries[candidate][polarity] = {
            "events": events,
            "special_read_count": int(item["special"]),
            "special_read_share": item["special"] / events,
            "unique_documents": len(item["documents"]),
            "unique_previous_tokens": len(item["previous"]),
            "unique_target_tokens": len(item["targets"]),
            "unique_transitions": len(item["transitions"]),
            "top_previous": [
                {"token": token, "count": count, "share": count / events}
                for token, count in item["previous"].most_common(6)
            ],
            "top_targets": [
                {"token": token, "count": count, "share": count / events}
                for token, count in item["targets"].most_common(6)
            ],
            "top_transitions": [
                {
                    "previous": pair[0],
                    "target": pair[1],
                    "label": transition_label(*pair),
                    "count": count,
                    "share": count / events,
                }
                for pair, count in item["transitions"].most_common(6)
            ],
        }
    return summaries, {"source_records": total_source, "contexts_per_polarity": next(iter(counts))}


def compact_context(raw: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("source") or {}
    activation = float(raw.get("activation") or 0)
    scale = max(float(row.get("robust_activation_scale") or 0), 1e-12)
    robust_z = (activation - float(row.get("median_activation") or 0)) / scale
    return {
        "polarity": raw.get("polarity"),
        "rank": raw.get("rank_within_polarity"),
        "activation": activation,
        "cosine": raw.get("cosine_activation"),
        "robust_z": robust_z,
        "target": raw.get("token", ""),
        "previous": raw.get("prev_token", ""),
        "previous_id": raw.get("prev_token_id"),
        "target_position": raw.get("token_position"),
        "read_position": raw.get("read_position"),
        "special_read": is_special_read(raw),
        "context": raw.get("context_marked") or raw.get("context", ""),
        "document": raw.get("document_index"),
        "url": source.get("url"),
        "date": source.get("date"),
        "dump": source.get("dump"),
    }


def load_selected_contexts(
    path: Path,
    selected: set[str],
    source_by_display: dict[str, str],
    rows_by_candidate: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    display_by_source = {source_by_display[candidate]: candidate for candidate in selected}
    contexts = {candidate: {"positive": [], "negative": []} for candidate in selected}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            candidate = display_by_source.get(str(raw.get("candidate")))
            if candidate is None:
                continue
            rank = int(raw.get("rank_within_polarity") or 0)
            if limit and rank > limit:
                continue
            polarity = str(raw.get("polarity"))
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected polarity on {path}:{line_number}")
            contexts[candidate][polarity].append(compact_context(raw, rows_by_candidate[candidate]))
    expected = limit
    for candidate, grouped in contexts.items():
        for polarity, items in grouped.items():
            items.sort(key=lambda item: int(item.get("rank") or 0))
            if expected and len(items) != expected:
                raise SystemExit(f"Expected {expected} contexts for {candidate}/{polarity}, got {len(items)}")
    return {candidate: contexts[candidate] for candidate in sorted(contexts)}


def compact_neighbor(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("token_id"),
        "token": raw.get("token"),
        "decoded": raw.get("decoded"),
        "cosine": raw.get("cosine"),
    }


def load_unembedding(path: Path, selected: set[str], source_by_display: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    display_by_source = {source_by_display[candidate]: candidate for candidate in selected}
    output: dict[str, Any] = {}
    total = 0
    domain_max = 0.0
    spaces: set[str] = set()
    vocab: set[int] = set()
    per_side: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            raw = json.loads(line)
            spaces.add(str(raw.get("space")))
            vocab.add(int(raw.get("unembedding_vocab_rows_considered") or 0))
            nearest = raw.get("nearest_tokens") or []
            farthest = raw.get("farthest_tokens") or []
            per_side.update((len(nearest), len(farthest)))
            domain_max = max(domain_max, abs(float(raw.get("max_abs_token_cosine") or 0)))
            candidate = display_by_source.get(str(raw.get("candidate")))
            if candidate is None:
                continue
            if candidate in output:
                raise SystemExit(f"Duplicate unembedding record on {path}:{line_number}")
            output[candidate] = {
                "mean": raw.get("unembedding_cosine_mean"),
                "std": raw.get("unembedding_cosine_std"),
                "max_abs": raw.get("max_abs_token_cosine"),
                "nearest": [compact_neighbor(item) for item in nearest],
                "farthest": [compact_neighbor(item) for item in farthest],
            }
    missing = selected - set(output)
    if missing:
        raise SystemExit(f"Missing unembedding records for {len(missing)} selected candidates")
    if len(spaces) != 1 or len(vocab) != 1 or len(per_side) != 1:
        raise SystemExit("Inconsistent unembedding metadata")
    return {candidate: output[candidate] for candidate in sorted(output)}, {
        "source_candidates": total,
        "space": next(iter(spaces)),
        "vocab": next(iter(vocab)),
        "per_side": next(iter(per_side)),
        "domain_max": domain_max,
    }


def load_original_metrics(path: Path, selected: set[str]) -> dict[str, Any]:
    rows, _ = read_rankings(path)
    wanted = set(selected)
    fields = (
        "rank_global_mean_abs_cosine",
        "mean_abs_cosine",
        "rank_global_tail_selectivity",
        "tail_selectivity_score",
        "selected_tail_polarity",
        "selected_tail_q999_robust_z",
        "selected_tail_top0_1pct_energy_share",
        "selected_tail_doc_count_z5",
        "selected_tail_doc_rate_z5",
        "selected_tail_top_context_largest_center_token_share",
    )
    output = {
        row["candidate"]: {field: row.get(field) for field in fields}
        for row in rows
        if row["candidate"] in wanted
    }
    missing = wanted - set(output)
    if missing:
        raise SystemExit(f"Original rankings are missing {len(missing)} selected candidates")
    return {candidate: output[candidate] for candidate in sorted(output)}


def compact_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    output: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        output.append({"role": str(item.get("role", "unknown")), "content": content})
    return output


def load_cot_rollouts(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rollouts: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    prompt_ids: list[str] = []
    messages_by_prompt: dict[str, list[dict[str, str]]] = {}
    jobs = scanned = capped = finals = analysis_tokens = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            jobs += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid CoT rollout JSON on {path}:{line_number}: {exc}") from exc
            status = str(raw.get("status", "unknown"))
            status_counts[status] += 1
            if status != "scanned":
                continue
            source = raw.get("source") or {}
            prompt_id = str(raw.get("prompt_id") or source.get("id") or f"job_{line_number}")
            if prompt_id in messages_by_prompt:
                raise SystemExit(f"Duplicate CoT prompt id {prompt_id!r} in {path}")
            category = str(source.get("category") or "uncategorized")
            difficulty = str(source.get("difficulty") or "unspecified")
            messages = compact_messages(raw.get("messages"))
            hit_cap = bool(raw.get("hit_max_new_tokens"))
            final = str(raw.get("final") or "")
            prompt_ids.append(prompt_id)
            messages_by_prompt[prompt_id] = messages
            category_counts[category] += 1
            difficulty_counts[difficulty] += 1
            capped += int(hit_cap)
            finals += int(bool(final.strip()))
            analysis_tokens += int(raw.get("analysis_tokens") or 0)
            rollouts[str(scanned)] = {
                "document": scanned,
                "job": raw.get("job_index"),
                "prompt_id": prompt_id,
                "category": category,
                "difficulty": difficulty,
                "sample": source.get("sample_index", raw.get("sample_index")),
                "messages": messages,
                "reasoning": str(raw.get("reasoning") or ""),
                "final": final,
                "hit_cap": hit_cap,
                "prompt_tokens": raw.get("prompt_tokens"),
                "generated_tokens": raw.get("generated_tokens"),
                "analysis_tokens": raw.get("analysis_tokens"),
                "final_tokens": raw.get("final_tokens"),
            }
            scanned += 1
    return rollouts, {
        "jobs": jobs,
        "scanned": scanned,
        "capped": capped,
        "with_final": finals,
        "analysis_tokens": analysis_tokens,
        "status_counts": dict(status_counts),
        "category_counts": dict(category_counts),
        "difficulty_counts": dict(difficulty_counts),
        "prompt_ids": prompt_ids,
        "messages_by_prompt": messages_by_prompt,
    }


def load_cot_catalog(effort: str, data_dir: Path, top_n: int) -> dict[str, Any]:
    required = (
        data_dir / "metadata.json",
        data_dir / "rollouts.jsonl",
        data_dir / "sv_rankings.csv",
        data_dir / "unembedding_neighbors.jsonl",
    )
    validate_inputs(*required)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("reasoning_effort") != effort:
        raise SystemExit(
            f"{data_dir} reports effort {metadata.get('reasoning_effort')!r}, expected {effort!r}"
        )
    rows, source_by_display = read_rankings(data_dir / "sv_rankings.csv")
    if not rows:
        raise SystemExit(f"CoT rankings are empty: {data_dir}")
    rows_by_candidate = {str(row["candidate"]): row for row in rows}
    limit = len(rows) if top_n == 0 else min(top_n, len(rows))
    broad = [
        str(row["candidate"])
        for row in sorted(rows, key=lambda row: int(row[BROAD_RANK]))[:limit]
    ]
    raw = [
        str(row["candidate"])
        for row in sorted(rows, key=lambda row: int(row[TAIL_RANK]))[:limit]
    ]
    union = sorted(set(broad) | set(raw), key=lambda candidate: int(rows_by_candidate[candidate][BROAD_RANK]))
    rollouts, rollout_meta = load_cot_rollouts(data_dir / "rollouts.jsonl")
    return {
        "effort": effort,
        "data_dir": data_dir,
        "metadata": metadata,
        "rows": rows,
        "rows_by_candidate": rows_by_candidate,
        "source_by_display": source_by_display,
        "cohorts": {"broad": broad, "raw": raw, "union": union},
        "rollouts": rollouts,
        "rollout_meta": rollout_meta,
    }


def validate_cot_shared_artifacts(
    canonical: Path,
    canonical_metadata: dict[str, Any],
    catalogs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to validate predictive CoT direction banks") from exc

    layers = [int(layer) for layer in canonical_metadata.get("layers", [])]
    canonical_neighbors = canonical / "unembedding_neighbors.jsonl"
    validate_inputs(canonical_neighbors)
    neighbor_hash = file_sha256(canonical_neighbors)
    slots = 0
    for effort, catalog in catalogs.items():
        metadata = catalog["metadata"]
        for field in ("model", "layers", "k"):
            if metadata.get(field) != canonical_metadata.get(field):
                raise SystemExit(f"Predictive CoT {effort} and FineWeb differ on metadata field {field}")
        cot_neighbors = catalog["data_dir"] / "unembedding_neighbors.jsonl"
        if file_sha256(cot_neighbors) != neighbor_hash:
            raise SystemExit(f"Predictive CoT {effort} token-neighbor artifact differs from FineWeb")

    for layer in layers:
        canonical_path = canonical / "directions" / f"L{layer:02d}.npz"
        validate_inputs(canonical_path)
        with np.load(canonical_path) as reference:
            for effort, catalog in catalogs.items():
                cot_path = catalog["data_dir"] / "directions" / f"L{layer:02d}.npz"
                validate_inputs(cot_path)
                with np.load(cot_path) as cot:
                    for key in ("V", "S"):
                        if key not in reference or key not in cot or not np.array_equal(reference[key], cot[key]):
                            raise SystemExit(
                                f"Predictive CoT {effort} {key} differs from FineWeb at layer {layer}"
                            )
            slots += int(reference["S"].shape[0])

    low_rollouts = catalogs["low"]["rollout_meta"]
    medium_rollouts = catalogs["medium"]["rollout_meta"]
    if low_rollouts["prompt_ids"] != medium_rollouts["prompt_ids"]:
        raise SystemExit("Predictive CoT low and medium do not contain the same ordered prompts")
    if low_rollouts["messages_by_prompt"] != medium_rollouts["messages_by_prompt"]:
        raise SystemExit("Predictive CoT low and medium prompt messages differ")
    return {
        "arrays_identical": True,
        "direction_slots": slots,
        "neighbors_identical": True,
        "neighbor_sha256": neighbor_hash,
        "reference": "FineWeb predictive = CoT low predictive = CoT medium predictive",
    }


def clean_cot_marked_context(raw: dict[str, Any]) -> tuple[str, str]:
    marked = str(raw.get("context_marked") or raw.get("context") or "")
    pieces = HARMONY_TOKEN.split(marked)
    focused = next((piece for piece in pieces if "⟦" in piece and "⟧" in piece), marked).strip()
    return focused, focused.replace("⟦", "").replace("⟧", "")


def compact_cot_context(
    raw: dict[str, Any],
    rollout: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    marked, plain = clean_cot_marked_context(raw)
    window = raw.get("window_meta") or {}
    prompt_tokens = int(window.get("prompt_tokens") or rollout.get("prompt_tokens") or 0)
    generated_tokens = max(int(window.get("generated_tokens") or rollout.get("generated_tokens") or 0), 1)
    target_position = int(raw.get("token_position") or 0)
    activation = float(raw.get("activation") or 0)
    scale = max(float(row.get("robust_activation_scale") or 0), 1e-12)
    centered = (activation - float(row.get("median_activation") or 0)) / scale
    polarity = str(raw.get("polarity"))
    return {
        "polarity": polarity,
        "rank": raw.get("rank_within_polarity"),
        "activation": activation,
        "cosine": raw.get("cosine_activation"),
        "tail_z": centered if polarity == "positive" else -centered,
        "target": raw.get("token", ""),
        "previous": raw.get("prev_token", ""),
        "previous_id": raw.get("prev_token_id"),
        "target_position": target_position,
        "read_position": raw.get("read_position"),
        "special_read": is_special_read(raw),
        "marked": marked,
        "plain": plain,
        "document": raw.get("document_index"),
        "progress": max(0.0, min(1.0, (target_position - prompt_tokens) / generated_tokens)),
        "prompt_id": rollout.get("prompt_id"),
        "category": rollout.get("category"),
    }


def load_cot_contexts(
    catalog: dict[str, Any],
    selected: set[str],
    context_limit: int,
    required: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if context_limit < 0:
        raise SystemExit("--cot-contexts-per-side must be 0 or greater")
    path = catalog["data_dir"] / "top_contexts.jsonl"
    if not path.is_file():
        if required:
            raise SystemExit(f"Missing required predictive CoT contexts: {path}")
        return {}, {}, {
            "available": False,
            "missing_path": str(path),
            "source_records": 0,
            "source_per_side": None,
            "embedded_per_side": 0,
        }

    source_by_display = catalog["source_by_display"]
    display_by_source = {source: display for display, source in source_by_display.items()}
    contexts = {candidate: {"positive": [], "negative": []} for candidate in selected}
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
                raise SystemExit(f"Invalid CoT context JSON on {path}:{line_number}: {exc}") from exc
            source_candidate = str(raw.get("candidate"))
            display = display_by_source.get(source_candidate)
            if display is None:
                raise SystemExit(f"Unknown CoT context candidate on {path}:{line_number}: {source_candidate!r}")
            polarity = str(raw.get("polarity"))
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected CoT context polarity on {path}:{line_number}")
            read_position = int(raw.get("read_position") or 0)
            target_position = int(raw.get("token_position") or 0)
            if read_position + 1 != target_position:
                raise SystemExit(f"Non-adjacent predictive CoT context on {path}:{line_number}")
            document = str(raw.get("document_index"))
            rollout = catalog["rollouts"].get(document)
            if rollout is None:
                raise SystemExit(f"CoT context {path}:{line_number} has no rollout {document}")
            source = raw.get("source") or {}
            if str(source.get("id")) != rollout["prompt_id"]:
                raise SystemExit(
                    f"CoT context/rollout prompt mismatch on {path}:{line_number}: "
                    f"{source.get('id')!r} != {rollout['prompt_id']!r}"
                )
            key = (display, polarity)
            group_counts[key] += 1
            if display not in selected:
                continue
            item = compact_cot_context(raw, rollout, catalog["rows_by_candidate"][display])
            state = summary_state.setdefault(
                key,
                {
                    "documents": set(),
                    "previous": Counter(),
                    "targets": Counter(),
                    "transitions": Counter(),
                    "categories": Counter(),
                    "progresses": [],
                    "phases": Counter(),
                    "special": 0,
                },
            )
            state["documents"].add(document)
            state["previous"][str(item["previous"])] += 1
            state["targets"][str(item["target"])] += 1
            state["transitions"][(str(item["previous"]), str(item["target"]))] += 1
            state["categories"][str(rollout["category"])] += 1
            state["progresses"].append(float(item["progress"]))
            phase = "early" if item["progress"] < 1 / 3 else "middle" if item["progress"] < 2 / 3 else "late"
            state["phases"][phase] += 1
            state["special"] += int(item["special_read"])
            rank = int(item.get("rank") or 0)
            if context_limit == 0 or rank <= context_limit:
                contexts[display][polarity].append(item)

    expected_groups = len(source_by_display) * 2
    count_values = set(group_counts.values())
    if len(group_counts) != expected_groups or len(count_values) != 1:
        raise SystemExit(
            f"Incomplete CoT context groups in {path}: groups={len(group_counts)}/{expected_groups}, "
            f"counts={sorted(count_values)}"
        )
    source_per_side = next(iter(count_values))
    declared = int(catalog["metadata"].get("top_contexts_per_polarity") or source_per_side)
    if source_per_side != declared:
        raise SystemExit(f"CoT contexts have {source_per_side} events/side, metadata declares {declared}")
    embedded_per_side = source_per_side if context_limit == 0 else min(context_limit, source_per_side)
    summaries: dict[str, Any] = {candidate: {} for candidate in selected}
    for candidate in selected:
        for polarity in ("positive", "negative"):
            items = contexts[candidate][polarity]
            items.sort(key=lambda item: int(item.get("rank") or 0))
            if len(items) != embedded_per_side:
                raise SystemExit(
                    f"Expected {embedded_per_side} CoT contexts for {candidate}/{polarity}, got {len(items)}"
                )
            state = summary_state[(candidate, polarity)]
            events = group_counts[(candidate, polarity)]
            summaries[candidate][polarity] = {
                "events": events,
                "special_read_count": int(state["special"]),
                "special_read_share": state["special"] / events,
                "unique_traces": len(state["documents"]),
                "unique_previous_tokens": len(state["previous"]),
                "unique_target_tokens": len(state["targets"]),
                "unique_transitions": len(state["transitions"]),
                "median_progress": statistics.median(state["progresses"]),
                "phases": {phase: state["phases"].get(phase, 0) for phase in ("early", "middle", "late")},
                "top_previous": [
                    {"token": token, "count": count, "share": count / events}
                    for token, count in state["previous"].most_common(6)
                ],
                "top_targets": [
                    {"token": token, "count": count, "share": count / events}
                    for token, count in state["targets"].most_common(6)
                ],
                "top_transitions": [
                    {
                        "previous": pair[0],
                        "target": pair[1],
                        "label": transition_label(*pair),
                        "count": count,
                        "share": count / events,
                    }
                    for pair, count in state["transitions"].most_common(6)
                ],
                "top_categories": [
                    {"category": category, "count": count, "share": count / events}
                    for category, count in state["categories"].most_common(5)
                ],
            }
    return contexts, summaries, {
        "available": True,
        "missing_path": None,
        "source_records": total_source,
        "source_per_side": source_per_side,
        "embedded_per_side": embedded_per_side,
    }


COT_RANKING_FIELDS = (
    "candidate", "layer", "sv_index_0", "singular_value", "singular_value_over_sv0",
    "n_tokens", "n_documents", "mean_activation", "mean_abs_activation", "rms_activation",
    "std_activation", "positive_rate", "max_activation", "min_activation", "mean_abs_cosine",
    "rms_cosine", "top1_abs_rate", "top5_abs_rate", "doc_top5_presence_rate",
    "mean_document_peak_abs", "dynamicity_std_over_abs_mean", "token_sample_n",
    "median_activation", "mad_activation", "robust_activation_scale", "stable_excess_kurtosis",
    "effective_support_fraction", "positive_tail_selectivity_score", "negative_tail_selectivity_score",
    "tail_selectivity_score", "selected_tail_polarity", "rank_global_mean_abs_cosine",
    "rank_global_tail_selectivity", "positive_q999_robust_z", "negative_q999_robust_z",
    "positive_max_robust_z", "negative_max_robust_z", "positive_top0_1pct_energy_share",
    "negative_top0_1pct_energy_share", "positive_doc_count_z5", "negative_doc_count_z5",
    "positive_doc_rate_z5", "negative_doc_rate_z5", "selected_tail_q999_robust_z",
    "selected_tail_top0_1pct_energy_share", "selected_tail_doc_count_z5",
    "selected_tail_doc_rate_z5", "selected_tail_top_context_largest_center_token_share",
    "selected_tail_top_context_effective_center_tokens", "selected_tail_special_read_share",
    "selected_tail_unique_previous_tokens", "selected_tail_unique_target_tokens",
    "selected_tail_unique_transitions",
)


def compact_cot_ranking(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in COT_RANKING_FIELDS}


def build_payload(
    canonical: Path,
    mirror: Path,
    original_basis: Path,
    original_rankings: Path,
    cot_low_dir: Path,
    cot_medium_dir: Path,
    top_n: int,
    context_limit: int,
    cot_context_limit: int,
    threshold: float,
) -> dict[str, Any]:
    if top_n < 0 or context_limit < 0 or cot_context_limit < 0:
        raise SystemExit("--top and context limits must be 0 or greater")
    if not 0 <= threshold <= 1:
        raise SystemExit("--boundary-threshold must be between 0 and 1")
    validate_inputs(*required_scan_files(canonical), original_rankings)
    mirror_meta = validate_mirror(canonical, mirror)
    metadata = json.loads((canonical / "metadata.json").read_text(encoding="utf-8"))
    layers = [int(layer) for layer in metadata.get("layers", [])]
    basis_meta = validate_original_basis(canonical, original_basis, layers)
    rows, source_by_display = read_rankings(canonical / "sv_rankings.csv")
    if not rows:
        raise SystemExit("Canonical rankings are empty")
    rows_by_candidate = {str(row["candidate"]): row for row in rows}
    summaries, context_meta = scan_context_summaries(canonical / "top_contexts.jsonl", source_by_display)

    cot_catalogs = {
        "low": load_cot_catalog("low", cot_low_dir, top_n),
        "medium": load_cot_catalog("medium", cot_medium_dir, top_n),
    }
    cot_basis = validate_cot_shared_artifacts(canonical, metadata, cot_catalogs)

    for row in rows:
        candidate = str(row["candidate"])
        polarity = str(row.get("selected_tail_polarity"))
        summary = summaries[candidate][polarity]
        row["selected_tail_special_read_share"] = summary["special_read_share"]
        row["selected_tail_unique_previous_tokens"] = summary["unique_previous_tokens"]
        row["selected_tail_unique_target_tokens"] = summary["unique_target_tokens"]
        row["selected_tail_unique_transitions"] = summary["unique_transitions"]

    broad_all = sorted(rows, key=lambda row: int(row[BROAD_RANK]))
    raw_all = sorted(rows, key=lambda row: int(row[TAIL_RANK]))
    eligible = [
        row for row in raw_all if float(row["selected_tail_special_read_share"]) <= threshold
    ]
    for rank, row in enumerate(eligible, 1):
        row["rank_boundary_screened_tail"] = rank

    limit = len(rows) if top_n == 0 else min(top_n, len(rows))
    cohorts: dict[str, list[str]] = {
        "broad": [str(row["candidate"]) for row in broad_all[:limit]],
        "screened": [str(row["candidate"]) for row in eligible[: min(limit, len(eligible))]],
        "raw": [str(row["candidate"]) for row in raw_all[:limit]],
        "cot_low": list(cot_catalogs["low"]["cohorts"]["union"]),
        "cot_medium": list(cot_catalogs["medium"]["cohorts"]["union"]),
    }
    selected = set().union(*map(set, cohorts.values()))
    if not selected:
        raise SystemExit("No candidates survived cohort selection")
    if context_limit == 0:
        context_limit = int(context_meta["contexts_per_polarity"])
    context_limit = min(context_limit, int(context_meta["contexts_per_polarity"]))
    contexts = load_selected_contexts(
        canonical / "top_contexts.jsonl",
        selected,
        source_by_display,
        rows_by_candidate,
        context_limit,
    )
    unembedding, unembedding_meta = load_unembedding(
        canonical / "unembedding_neighbors.jsonl", selected, source_by_display
    )
    original = load_original_metrics(original_rankings, selected)
    display_rows = [rows_by_candidate[candidate] for candidate in sorted(selected)]

    cot_payloads: dict[str, Any] = {}
    for effort, catalog in cot_catalogs.items():
        cot_contexts, cot_summaries, cot_context_meta = load_cot_contexts(
            catalog,
            selected,
            cot_context_limit,
            required=effort == "medium",
        )
        for candidate in selected:
            row = catalog["rows_by_candidate"][candidate]
            row["selected_tail_special_read_share"] = None
            row["selected_tail_unique_previous_tokens"] = None
            row["selected_tail_unique_target_tokens"] = None
            row["selected_tail_unique_transitions"] = None
            if cot_context_meta["available"]:
                polarity = str(row.get("selected_tail_polarity"))
                summary = cot_summaries[candidate][polarity]
                row["selected_tail_special_read_share"] = summary["special_read_share"]
                row["selected_tail_unique_previous_tokens"] = summary["unique_previous_tokens"]
                row["selected_tail_unique_target_tokens"] = summary["unique_target_tokens"]
                row["selected_tail_unique_transitions"] = summary["unique_transitions"]
        rollout_meta = catalog["rollout_meta"]
        effort_meta = catalog["metadata"]
        raw_boundary_count = None
        if cot_context_meta["available"]:
            raw_boundary_count = sum(
                float(catalog["rows_by_candidate"][candidate]["selected_tail_special_read_share"]) > threshold
                for candidate in catalog["cohorts"]["raw"]
            )
        cot_payloads[effort] = {
            "meta": {
                "effort": effort,
                "model": effort_meta.get("model"),
                "prompts": effort_meta.get("prompts"),
                "rollouts": rollout_meta["scanned"],
                "analysis_target_tokens": effort_meta.get("analysis_target_tokens_scanned")
                or rollout_meta["analysis_tokens"],
                "max_new_tokens": effort_meta.get("max_new_tokens"),
                "capped_rollouts": rollout_meta["capped"],
                "rollouts_with_final": rollout_meta["with_final"],
                "categories": len(rollout_meta["category_counts"]),
                "total_candidates": len(catalog["rows"]),
                "broad_cohort_size": len(catalog["cohorts"]["broad"]),
                "raw_cohort_size": len(catalog["cohorts"]["raw"]),
                "union_cohort_size": len(catalog["cohorts"]["union"]),
                "broad_raw_overlap": len(
                    set(catalog["cohorts"]["broad"]) & set(catalog["cohorts"]["raw"])
                ),
                "contexts_available": cot_context_meta["available"],
                "contexts_missing_path": cot_context_meta["missing_path"],
                "source_context_records": cot_context_meta["source_records"],
                "source_contexts_per_side": cot_context_meta["source_per_side"],
                "embedded_contexts_per_side": cot_context_meta["embedded_per_side"],
                "raw_top_boundary_dominated": raw_boundary_count,
                "channel_filter": effort_meta.get("channel_filter"),
                "activation_alignment": effort_meta.get("activation_alignment"),
                "display_sv_numbering": "zero_based",
            },
            "rankings": {
                candidate: compact_cot_ranking(catalog["rows_by_candidate"][candidate])
                for candidate in sorted(selected)
            },
            "contexts": cot_contexts,
            "context_summaries": cot_summaries,
            "rollouts": catalog["rollouts"],
        }

    raw_boundary = sum(
        float(rows_by_candidate[candidate]["selected_tail_special_read_share"]) > threshold
        for candidate in cohorts["raw"]
    )
    overlap = {
        "broad_screened": len(set(cohorts["broad"]) & set(cohorts["screened"])),
        "broad_raw": len(set(cohorts["broad"]) & set(cohorts["raw"])),
        "screened_raw": len(set(cohorts["screened"]) & set(cohorts["raw"])),
    }
    return {
        "default_lens": "broad",
        "meta": {
            "model": metadata.get("model"),
            "dataset": metadata.get("dataset"),
            "dataset_config": metadata.get("dataset_config"),
            "documents": metadata.get("documents_processed"),
            "target_tokens": metadata.get("target_tokens_processed"),
            "layers": layers,
            "k": metadata.get("k"),
            "total_candidates": len(rows),
            "candidate_union": len(selected),
            "top_per_lens": limit,
            "contexts_per_polarity_source": context_meta["contexts_per_polarity"],
            "contexts_per_polarity_embedded": context_limit,
            "boundary_threshold": threshold,
            "screened_eligible_candidates": len(eligible),
            "raw_top_boundary_dominated": raw_boundary,
            "raw_cohort_size": len(cohorts["raw"]),
            "activation_alignment": metadata.get("activation_alignment"),
            "target_token_rule": metadata.get("target_token_rule"),
            "mirror": mirror_meta,
            "original_basis": basis_meta,
            "unembedding": unembedding_meta,
            "cohort_overlap": overlap,
            "cohort_sizes": {key: len(value) for key, value in cohorts.items()},
            "display_sv_numbering": "zero_based",
        },
        "lenses": {
            "broad": {"candidates": cohorts["broad"]},
            "screened": {"candidates": cohorts["screened"]},
            "raw": {"candidates": cohorts["raw"]},
            "cot_low": {"candidates": cohorts["cot_low"]},
            "cot_medium": {"candidates": cohorts["cot_medium"]},
        },
        "rankings": display_rows,
        "contexts": contexts,
        "context_summaries": {candidate: summaries[candidate] for candidate in sorted(selected)},
        "unembedding": unembedding,
        "original": original,
        "cot": {
            "default_effort": "medium",
            "basis": cot_basis,
            "efforts": cot_payloads,
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Predictive Alignment SV Atlas</title>
  <style>
    :root {
      --ink:#14202a; --ink2:#324453; --muted:#6d7880; --paper:#eceae4; --surface:#fffdf8;
      --surface2:#f5f2ea; --line:#d8d3c8; --line2:#aaa69d; --navy:#142b3c; --cyan:#247c85;
      --cyan-soft:#dceeee; --gold:#bd7b28; --gold-soft:#f4e7cf; --red:#aa4f45; --red-soft:#f4ded9;
      --violet:#625697; --violet-soft:#e7e2f3; --accent:var(--cyan); --accent-soft:var(--cyan-soft);
      --shadow:0 14px 42px rgba(31,40,48,.09); --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
      --sans:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      --serif:Iowan Old Style,Baskerville,Georgia,serif;
    }
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 var(--sans)}
    body[data-lens="screened"]{--accent:var(--violet);--accent-soft:var(--violet-soft)} body[data-lens="raw"]{--accent:var(--red);--accent-soft:var(--red-soft)} body[data-lens="cot_low"]{--accent:var(--gold);--accent-soft:var(--gold-soft)} body[data-lens="cot_medium"]{--accent:var(--violet);--accent-soft:var(--violet-soft)}
    button,input,select{font:inherit}button,select{cursor:pointer}a{color:inherit}.skip{position:fixed;top:-60px;left:12px;z-index:90;background:var(--ink);color:#fff;padding:8px 12px}.skip:focus{top:12px}
    .masthead{background:var(--navy);color:#f7f4ec;border-bottom:4px solid var(--gold)}.mast-inner{width:min(1660px,94vw);min-height:172px;margin:auto;display:flex;align-items:flex-end;justify-content:space-between;gap:42px;padding:35px 0 30px}
    .brand{display:flex;align-items:flex-start;gap:17px;min-width:0}.brand-mark{display:grid;place-items:center;width:50px;height:50px;flex:0 0 auto;border:1px solid rgba(255,255,255,.28);color:#82cbd0;font:800 11px/1.15 var(--mono);text-align:center}
    .eyebrow{margin:0 0 7px;color:#e3a96b;font-size:9px;font-weight:850;letter-spacing:.17em;text-transform:uppercase}h1{margin:0;font:500 clamp(30px,4.2vw,50px)/1.02 var(--serif);letter-spacing:-.025em}h1 em{color:#90cbd0;font-weight:400}.subtitle{max-width:780px;margin:11px 0 0;color:#b8c4cb;font-size:12px}
    .facts{display:flex;justify-content:flex-end;gap:22px;flex-wrap:wrap}.fact{min-width:88px}.fact b{display:block;color:#fff;font:650 18px/1.15 var(--mono)}.fact span{color:#93a1aa;font-size:8px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}
    .provenance{border-bottom:1px solid var(--line2);background:#e2dfd7}.provenance-inner{width:min(1660px,94vw);margin:auto;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:9px 0;color:var(--muted);font-size:9px}.provenance b{color:var(--ink2)}.prov-badges{display:flex;gap:6px;flex-wrap:wrap}.badge{display:inline-block;border:1px solid var(--line2);border-radius:99px;background:var(--surface);padding:4px 7px;font:750 8px/1.1 var(--mono)}.badge.warn{border-color:#d5a89c;background:var(--red-soft);color:var(--red)}.badge.ok{border-color:#9abdbd;background:var(--cyan-soft);color:var(--cyan)}
    .lensbar{border-bottom:1px solid var(--line2);background:#ddd9d1}.lens-switch{width:min(1660px,94vw);margin:auto;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;border-inline:1px solid var(--line2);background:var(--line2)}.lens-tab{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:11px;border:0;background:#e8e4dc;color:var(--muted);padding:13px 14px;text-align:left}.lens-tab[data-kind="cot"]{background:#e4e0d8}.lens-tab.cot-start{box-shadow:inset 4px 0 0 #b6afa2}.lens-tab:hover{background:#f1eee7}.lens-tab[aria-selected="true"]{background:var(--surface);box-shadow:inset 0 -4px 0 var(--accent)}.lens-tab[aria-selected="true"].cot-start{box-shadow:inset 4px 0 0 #b6afa2,inset 0 -4px 0 var(--accent)}.lens-index{display:grid;place-items:center;width:29px;height:29px;border:1px solid var(--line2);border-radius:50%;font:750 9px/1 var(--mono)}.lens-tab[aria-selected="true"] .lens-index{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}.lens-tab b{display:block;color:var(--ink);font:650 12px/1.2 var(--serif)}.lens-tab small{display:block;margin-top:3px;font-size:8px;line-height:1.3}
    .toolbar{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--line);background:rgba(236,234,228,.96);backdrop-filter:blur(12px)}.toolbar-inner{width:min(1660px,94vw);margin:auto;display:grid;grid-template-columns:minmax(230px,1.2fr) 145px minmax(180px,.7fr) 120px auto;align-items:end;gap:9px;padding:11px 0}.control{display:block;min-width:0;color:var(--muted);font-size:8px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}.control input,.control select{width:100%;height:38px;margin-top:4px;border:1px solid var(--line2);border-radius:2px;outline:0;background:var(--surface);color:var(--ink);padding:0 10px;font-size:11px;letter-spacing:normal;text-transform:none}.control input:focus,.control select:focus{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 15%,transparent)}.search-wrap{position:relative}.search-wrap input{padding-right:42px}.shortcut{position:absolute;right:8px;bottom:8px;border:1px solid var(--line);border-radius:3px;background:var(--surface2);padding:1px 6px;font:9px/1.45 var(--mono)}.reset{height:38px;border:1px solid var(--line2);background:transparent;color:var(--ink2);padding:0 13px;font-size:10px;font-weight:800}.reset:hover{background:var(--surface)}
    .shell{width:min(1660px,94vw);margin:auto;padding:23px 0 68px}.grid{display:grid;grid-template-columns:minmax(315px,380px) minmax(0,1fr);gap:22px;align-items:start}.panel{border:1px solid var(--line);background:var(--surface);box-shadow:var(--shadow)}.ranking-panel{position:sticky;top:74px;height:calc(100vh - 98px);min-height:520px;display:flex;flex-direction:column}.panel-head{border-bottom:1px solid var(--line);padding:17px 18px 14px}.panel-head-row{display:flex;align-items:baseline;justify-content:space-between;gap:12px}.panel h2,.detail-title{margin:0;font:500 23px/1.1 var(--serif)}.count{color:var(--muted);font:10px/1.2 var(--mono)}.micro{margin:7px 0 0;color:var(--muted);font-size:9px}.ranking-list{overflow:auto;overscroll-behavior:contain}.rank-row{width:100%;display:grid;grid-template-columns:48px minmax(0,1fr);gap:10px;border:0;border-bottom:1px solid #e7e2d8;background:transparent;color:var(--ink);padding:11px 14px 11px 9px;text-align:left}.rank-row:hover{background:var(--surface2)}.rank-row.active{background:var(--accent-soft);box-shadow:inset 4px 0 0 var(--accent)}.rank-number{padding-top:2px;color:var(--muted);font:10px/1.2 var(--mono);text-align:right}.rank-row.active .rank-number{color:var(--accent);font-weight:850}.rank-top{display:flex;align-items:baseline;justify-content:space-between;gap:8px}.candidate{font:750 12px/1.3 var(--mono)}.layer-note{color:var(--muted);font-size:8px}.rank-measure,.rank-foot{display:flex;justify-content:space-between;gap:8px;margin-top:6px;color:var(--muted);font-size:8px}.rank-measure b{color:var(--ink2);font:650 9px/1 var(--mono)}.mini{display:block;height:3px;margin-top:6px;background:#e1ddd4}.mini i{display:block;height:100%;background:var(--accent)}.boundary-pill{display:inline-block;margin-left:5px;border:1px solid #d7a99e;border-radius:99px;background:var(--red-soft);color:var(--red);padding:2px 5px;font:800 7px/1 var(--mono);vertical-align:1px}.empty{padding:30px 20px;color:var(--muted);text-align:center}
    .detail-panel{min-width:0}.hero{padding:clamp(22px,3vw,34px);border-bottom:1px solid var(--line);background:linear-gradient(125deg,#fffdf8,#f4f1e9)}.detail-nav{display:flex;align-items:center;justify-content:space-between;gap:16px}.detail-rank{color:var(--accent);font-size:9px;font-weight:850;letter-spacing:.12em;text-transform:uppercase}.nav-buttons{display:flex;gap:6px}.icon,.copy{height:33px;border:1px solid var(--line2);background:var(--surface);color:var(--ink2);padding:0 10px;font-size:10px}.icon{width:35px;padding:0;font-size:15px}.icon:disabled{cursor:default;opacity:.35}.icon:hover:not(:disabled),.copy:hover{border-color:var(--accent);color:var(--accent)}.title-row{display:flex;align-items:center;gap:11px;flex-wrap:wrap;margin-top:14px}.detail-title{font-size:clamp(29px,4vw,43px)}.id-chip{border-radius:99px;background:var(--accent-soft);color:var(--accent);padding:4px 9px;font:750 9px/1.2 var(--mono)}.detail-summary{max-width:950px;margin:10px 0 0;color:var(--muted);font-size:12px}.detail-summary b{color:var(--ink2)}
    .semantic-note{display:grid;grid-template-columns:auto 1fr;gap:13px;align-items:center;margin-top:18px;border:1px solid #aac6c7;background:#edf6f5;padding:12px 14px}.time-diagram{display:flex;align-items:center;gap:7px;white-space:nowrap;font:750 9px/1 var(--mono)}.time-node{border:1px solid #8fb5b7;background:#fff;padding:7px 8px}.time-arrow{color:var(--cyan);font-size:16px}.semantic-note p{margin:0;color:var(--muted);font-size:9px}.semantic-note strong{color:var(--ink2)}
    .warning{margin-top:14px;border-left:4px solid var(--red);background:var(--red-soft);padding:10px 12px;color:#77463f;font-size:10px}.warning.screen{border-color:var(--violet);background:var(--violet-soft);color:#50466f}.metric-grid{display:grid;grid-template-columns:repeat(6,minmax(100px,1fr));gap:1px;margin-top:20px;border:1px solid var(--line);background:var(--line)}.metric-card{min-width:0;background:rgba(255,253,248,.9);padding:11px}.metric-card span{display:block;min-height:27px;color:var(--muted);font-size:8px;font-weight:850;letter-spacing:.055em;text-transform:uppercase}.metric-card b{display:block;overflow:hidden;margin-top:4px;color:var(--ink);font:650 clamp(13px,1.35vw,17px)/1.15 var(--mono);text-overflow:ellipsis}
    .comparison{margin-top:18px;border:1px solid var(--line);background:var(--surface)}.comparison-head{display:flex;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);padding:12px 14px}.comparison-head h3,.tail-head h3{margin:0;font:550 18px/1.15 var(--serif)}.comparison-head p,.tail-head p{margin:4px 0 0;color:var(--muted);font-size:9px}.pair-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.pair-card{background:#faf8f2;padding:13px 14px}.pair-card.predictive{background:#e8f2f1}.pair-title{display:flex;justify-content:space-between;gap:9px;color:var(--muted);font-size:8px;font-weight:850;text-transform:uppercase}.pair-values{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;margin-top:10px}.pair-value span{display:block;min-height:21px;color:var(--muted);font-size:7px;text-transform:uppercase}.pair-value b{display:block;color:var(--ink2);font:650 11px/1.2 var(--mono)}.comparison-note{margin:0;border-top:1px solid var(--line);padding:9px 14px;color:var(--muted);font-size:9px}.comparison-note strong{color:var(--ink2)}
    .tail-profile{margin-top:17px;border:1px solid var(--line)}.tail-head{border-bottom:1px solid var(--line);padding:12px 14px}.tail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.tail-side{background:#faf8f2;padding:12px 14px}.tail-side.selected{background:var(--accent-soft);box-shadow:inset 0 3px 0 var(--accent)}.tail-side-title{display:flex;justify-content:space-between;gap:10px}.tail-side-title h4{margin:0;font:700 10px/1.2 var(--mono)}.tail-side-title span{color:var(--muted);font:8px/1.2 var(--mono)}.tail-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 13px;margin-top:11px}.tail-stat span{display:block;min-height:20px;color:var(--muted);font-size:7px;font-weight:800;text-transform:uppercase}.tail-stat b{display:block;color:var(--ink2);font:650 11px/1.2 var(--mono)}
    .metric-details{margin-top:15px}.metric-details summary{width:max-content;color:var(--accent);cursor:pointer;font-size:10px;font-weight:800}.all-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 22px;margin-top:10px;border:1px solid var(--line);background:var(--surface);padding:11px 13px}.all-metric{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dotted var(--line);padding:5px 0}.all-metric span{min-width:0;color:var(--muted);font-size:8px}.all-metric b{flex:0 0 auto;font:600 8px/1.4 var(--mono)}
    .contexts{padding:clamp(22px,3vw,34px);border-bottom:1px solid var(--line);background:#fbfaf6}.section-heading h3{margin:0;font:500 26px/1.12 var(--serif)}.section-heading p{max-width:840px;margin:7px 0 0;color:var(--muted);font-size:10px}.sign-note{display:inline-block;margin-top:8px;border-left:3px solid var(--gold);padding-left:9px;color:var(--muted);font-size:9px}.context-controls{display:grid;grid-template-columns:minmax(200px,1fr) 110px auto;gap:8px;align-items:end;margin:19px 0 14px}.check{display:flex;align-items:center;gap:6px;height:38px;color:var(--ink2);font-size:10px;white-space:nowrap}.check input{width:14px;height:14px;accent-color:var(--accent)}.context-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}.context-column{min-width:0}.column-head{display:flex;justify-content:space-between;gap:10px;border-top:3px solid var(--cyan);padding:8px 2px}.context-column.negative .column-head{border-color:var(--gold)}.column-head h4{margin:0;font:650 11px/1.2 var(--mono)}.column-head span{color:var(--muted);font-size:8px}
    .transition-summary{border:1px solid var(--line);background:var(--surface);padding:10px;margin-bottom:8px}.summary-line{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:8px}.summary-line b{color:var(--ink2)}.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.data-chip{max-width:185px;overflow:hidden;border:1px solid var(--line);border-radius:99px;background:var(--surface2);padding:3px 6px;color:var(--muted);font:8px/1.2 var(--mono);text-overflow:ellipsis;white-space:nowrap}.data-chip b{color:var(--ink2)}.context-list{display:grid;gap:8px}.context-card{overflow:hidden;border:1px solid var(--line);background:var(--surface)}.context-card.boundary{border-color:#d3a398}.context-meta{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #e9e4da;background:var(--surface2);padding:7px 9px;color:var(--muted);font:8px/1.25 var(--mono)}.context-meta b{color:var(--ink2)}.transition{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:8px;padding:9px 10px 2px}.transition-box{min-width:0;border:1px solid var(--line);background:#fff;padding:7px 8px}.transition-box span{display:block;color:var(--muted);font-size:7px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}.transition-box b{display:block;overflow:hidden;margin-top:3px;font:650 10px/1.2 var(--mono);text-overflow:ellipsis;white-space:nowrap}.transition-arrow{color:var(--accent);font-size:17px}.context-copy{margin:0;padding:10px;color:#293741;font:11px/1.58 var(--serif);white-space:pre-wrap;overflow-wrap:anywhere}mark{border-radius:2px;background:#f0cf91;color:#17202a;padding:1px 2px;box-shadow:0 0 0 1px rgba(156,95,31,.13)}.source-row{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:0 10px 9px;color:var(--muted);font-size:8px}.source-link{color:var(--cyan);font-weight:800;text-decoration:none}.source-link:hover{text-decoration:underline}.boundary-badge{border-radius:99px;background:var(--red-soft);color:var(--red);padding:3px 6px;font:800 7px/1 var(--mono);text-transform:uppercase}
    .cot-compare{margin-top:17px;border:1px solid var(--line)}.cot-compare-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line)}.cot-compare-card{border:0;background:#faf8f2;padding:13px 14px;text-align:left}.cot-compare-card.current{background:var(--accent-soft)}.cot-compare-card .pair-title{margin-bottom:9px}.cot-compare-card .pair-values{margin:0}
    .cot-section{padding:clamp(22px,3vw,34px);border-bottom:1px solid var(--line);background:#f1eee7}.cot-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:22px}.cot-heading h3{margin:0;font:500 26px/1.12 var(--serif)}.cot-heading p{max-width:880px;margin:7px 0 0;color:var(--muted);font-size:10px}.cot-effort-switch{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));min-width:340px;border:1px solid var(--line2);background:var(--line2);gap:1px}.cot-effort-button{border:0;background:#e7e3da;color:var(--muted);padding:10px 12px;text-align:left}.cot-effort-button b,.cot-effort-button small{display:block}.cot-effort-button b{color:var(--ink2);font:650 11px/1.2 var(--serif)}.cot-effort-button small{margin-top:3px;font:8px/1.3 var(--mono)}.cot-effort-button[aria-selected="true"]{background:var(--surface);box-shadow:inset 0 -3px 0 var(--accent)}.cot-metrics{display:grid;grid-template-columns:repeat(6,minmax(90px,1fr));gap:1px;margin-top:16px;border:1px solid var(--line);background:var(--line)}.cot-caveat,.cot-missing{margin:14px 0;border-left:4px solid var(--red);background:var(--red-soft);color:#77463f;padding:11px 13px;font-size:9px}.cot-missing{border-color:var(--gold);background:var(--gold-soft);color:#6e512d}.cot-controls{display:grid;grid-template-columns:minmax(190px,1fr) 105px auto auto;gap:8px;align-items:end;margin:15px 0 11px}.cot-workbench{display:grid;grid-template-columns:minmax(320px,430px) minmax(0,1fr);gap:14px;align-items:start}.cot-browser,.cot-trace-viewer{min-width:0;border:1px solid var(--line);background:var(--surface)}.cot-polarity-switch{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line)}.cot-polarity-button{border:0;background:#eeeae2;padding:9px 11px;text-align:left}.cot-polarity-button b,.cot-polarity-button small{display:block}.cot-polarity-button b{font:700 9px/1.2 var(--mono)}.cot-polarity-button small{margin-top:2px;color:var(--muted);font-size:8px}.cot-polarity-button[aria-pressed="true"]{background:var(--accent-soft);box-shadow:inset 0 -3px 0 var(--accent)}.cot-footprint{border-bottom:1px solid var(--line);padding:9px 10px}.cot-event-list{display:grid;gap:7px;padding:9px;max-height:620px;overflow:auto}.cot-event{overflow:hidden;border:1px solid var(--line);background:#fff}.cot-event.active{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}.cot-event .transition{padding-top:7px}.cot-event .context-copy{font-size:10px;max-height:150px;overflow:auto}.cot-event-footer{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:0 10px 9px;color:var(--muted);font-size:8px}.cot-open{border:0;background:transparent;color:var(--accent);padding:2px 0;font-size:8px;font-weight:850}.cot-progress{display:block;height:3px;background:#e1ddd4}.cot-progress i{display:block;height:100%;background:var(--accent)}.cot-trace-head{display:flex;justify-content:space-between;gap:15px;border-bottom:1px solid var(--line);padding:12px 14px}.cot-trace-head h4{margin:0;font:550 18px/1.15 var(--serif)}.cot-trace-head p{margin:4px 0 0;color:var(--muted);font-size:8px}.trace-badges{display:flex;gap:5px;align-items:flex-start;flex-wrap:wrap}.cot-trace-body{display:grid;gap:1px;background:var(--line)}.cot-trace-block{min-width:0;background:#fbfaf6;padding:13px 15px}.cot-trace-label{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;color:var(--muted);font-size:7px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}.cot-trace-text{margin:0;color:#283640;font:11px/1.62 var(--serif);white-space:pre-wrap;overflow-wrap:anywhere}.cot-trace-text+.cot-trace-text{margin-top:11px}.trace-focus{display:block;border-left:4px solid var(--accent);background:var(--accent-soft);margin:10px -5px;padding:8px 9px}.trace-empty{color:var(--muted);font-style:italic}.cot-rollout-controls{display:grid;grid-template-columns:minmax(220px,1fr) auto;gap:9px;align-items:end;margin:13px 0}.cot-rollout-controls .control{max-width:560px}.unlinked-badge{border-radius:99px;background:var(--gold-soft);color:#6e512d;padding:4px 7px;font:800 7px/1 var(--mono);text-transform:uppercase}
    .token-section{border-bottom:1px solid var(--line);background:#f6f5f0}.token-section>summary{cursor:pointer;list-style:none;padding:18px clamp(22px,3vw,34px)}.token-section>summary::-webkit-details-marker{display:none}.token-summary{display:flex;justify-content:space-between;align-items:center;gap:18px}.token-summary h3{margin:0;font:500 23px/1.15 var(--serif)}.token-summary p{margin:4px 0 0;color:var(--muted);font-size:9px}.token-summary b{color:var(--accent);font:650 10px/1.2 var(--mono)}.token-content{padding:0 clamp(22px,3vw,34px) clamp(22px,3vw,34px)}.token-toolbar{display:flex;justify-content:flex-end;margin-bottom:9px}.token-limit{width:120px}.token-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.token-head{display:flex;justify-content:space-between;gap:10px;border-top:3px solid var(--cyan);padding:8px 2px}.token-column.opposed .token-head{border-color:var(--gold)}.token-head h4{margin:0;font:650 11px/1.2 var(--mono)}.token-head span{color:var(--muted);font-size:8px}.token-list{display:grid;gap:5px}.token-row{display:grid;grid-template-columns:25px minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid var(--line);background:var(--surface);padding:7px 9px}.token-rank{color:var(--muted);font:650 8px/1 var(--mono);text-align:right}.token-name{min-width:0}.token-name b,.token-name small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.token-name b{font:650 10px/1.2 var(--mono)}.token-name small{margin-top:2px;color:var(--muted);font:7px/1.2 var(--mono)}.token-value{color:var(--ink2);font:650 9px/1.2 var(--mono)}
    .footer{width:min(1660px,94vw);margin:-38px auto 30px;color:var(--muted);font-size:9px}.footer code{font-family:var(--mono)}:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 28%,transparent);outline-offset:2px}
    @media(max-width:1180px){.metric-grid,.cot-metrics{grid-template-columns:repeat(3,1fr)}.all-metrics{grid-template-columns:repeat(2,1fr)}.cot-workbench{grid-template-columns:minmax(300px,390px) minmax(0,1fr)}}
    @media(max-width:940px){.mast-inner{align-items:flex-start;flex-direction:column;min-height:0}.facts{justify-content:flex-start}.provenance-inner{align-items:flex-start;flex-direction:column}.lens-switch{overflow-x:auto;grid-template-columns:repeat(5,minmax(210px,1fr))}.toolbar-inner{grid-template-columns:repeat(2,minmax(0,1fr))}.toolbar-inner .search-control{grid-column:1/-1}.grid{grid-template-columns:1fr}.ranking-panel{position:static;height:auto;min-height:0}.ranking-list{max-height:430px}.cot-heading{flex-direction:column}.cot-effort-switch{width:100%;min-width:0}.cot-workbench{grid-template-columns:1fr}.cot-event-list{max-height:430px}}
    @media(max-width:650px){.mast-inner,.provenance-inner,.lens-switch,.toolbar-inner,.shell,.footer{width:min(92vw,1660px)}.brand-mark{display:none}.lens-switch{display:flex;overflow-x:auto}.lens-tab{min-width:210px}.toolbar-inner{grid-template-columns:1fr}.toolbar-inner .search-control{grid-column:auto}.metric-grid,.pair-grid,.tail-grid,.context-grid,.token-grid,.cot-compare-grid,.cot-metrics{grid-template-columns:1fr}.all-metrics{grid-template-columns:1fr}.pair-values,.tail-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.context-controls,.cot-controls{grid-template-columns:1fr}.hero,.contexts,.cot-section{padding:20px 15px}.comparison-head,.token-summary{align-items:flex-start;flex-direction:column}.semantic-note{grid-template-columns:1fr}.cot-effort-switch{grid-template-columns:1fr}.cot-trace-head{flex-direction:column}.cot-rollout-controls{grid-template-columns:1fr}}
    @media print{.toolbar,.ranking-panel,.nav-buttons,.context-controls,.footer{display:none!important}.masthead{background:#fff;color:var(--ink)}.masthead *{color:var(--ink)!important}.grid{display:block}.panel{box-shadow:none}}
  </style>
</head>
<body data-lens="broad">
  <a class="skip" href="#detail">Skip to selected direction</a>
  <header class="masthead"><div class="mast-inner"><div class="brand"><div class="brand-mark">t−1<br>→ t</div><div><p class="eyebrow">Temporal alignment workbench · GPT-OSS-20B</p><h1>Predictive Alignment <em>SV Atlas</em></h1><p class="subtitle">Inspect the same exact J-Lens directions before FineWeb targets and generated reasoning tokens, distinguish content-linked signals from boundary artifacts, and compare low/medium chain-of-thought evidence without conflating corpus-local ranks.</p></div></div><div class="facts" id="facts"></div></div></header>
  <section class="provenance"><div class="provenance-inner"><span id="provenanceText"></span><div class="prov-badges" id="provenanceBadges"></div></div></section>
  <nav class="lensbar" aria-label="Predictive discovery lens"><div class="lens-switch" role="tablist">
    <button class="lens-tab" type="button" role="tab" data-lens="broad" aria-selected="true"><span class="lens-index">01</span><span><b>Broad predictive activity</b><small>Frequently recruited pre-target states · mean absolute cosine</small></span></button>
    <button class="lens-tab" type="button" role="tab" data-lens="screened" aria-selected="false"><span class="lens-index">02</span><span><b>Boundary-screened tails</b><small>Tail score, excluding directions whose retained driver exceeds the special-read threshold</small></span></button>
    <button class="lens-tab" type="button" role="tab" data-lens="raw" aria-selected="false"><span class="lens-index">03</span><span><b>Raw tails · diagnostic</b><small>Unscreened source ordering · prominently exposes BOS artifacts</small></span></button>
    <button class="lens-tab cot-start" type="button" role="tab" data-kind="cot" data-lens="cot_low" aria-selected="false"><span class="lens-index">04</span><span><b>CoT · low effort</b><small>Top broad + raw-tail union · transition-linked reasoning traces</small></span></button>
    <button class="lens-tab" type="button" role="tab" data-kind="cot" data-lens="cot_medium" aria-selected="false"><span class="lens-index">05</span><span><b>CoT · medium effort</b><small>Top broad + raw-tail union · transition-linked reasoning traces</small></span></button>
  </div></nav>
  <section class="toolbar" aria-label="Ranking controls"><div class="toolbar-inner">
    <label class="control search-control">Find a direction or transition<span class="search-wrap"><input id="candidateSearch" type="search" placeholder="Candidate, layer, previous or target token…" autocomplete="off"><kbd class="shortcut">/</kbd></span></label>
    <label class="control">Layer<select id="layerFilter"></select></label><label class="control">Rank by<select id="sortMetric"></select></label><label class="control">Show<select id="rowLimit"></select></label><button class="reset" id="resetFilters" type="button">Reset</button>
  </div></section>
  <main class="shell"><div class="grid"><aside class="panel ranking-panel"><div class="panel-head"><div class="panel-head-row"><h2 id="rankingTitle">Ranked directions</h2><span class="count" id="rankingCount"></span></div><p class="micro" id="rankingCaption"></p></div><div class="ranking-list" id="rankingList"></div></aside><article class="panel detail-panel" id="detail" tabindex="-1"></article></div></main>
  <footer class="footer" id="footer"></footer>
  <script>
    const DATA = __PREDICTIVE_PAYLOAD__;
    const BROAD_METRICS = {
      mean_abs_cosine:{label:"Predictive mean |cosine|",short:"mean |cos|",kind:"decimal",help:"Mean |h[t−1]·V| divided by the norm of h[t−1]."},
      doc_top5_presence_rate:{label:"Window top-5 presence",short:"window top-5",kind:"percent",help:"Share of FineWeb windows where the direction enters the layer's top five at least once."},
      mean_document_peak_abs:{label:"Mean window peak |act|",short:"window peak",kind:"number",help:"Mean per-window maximum absolute pre-target projection."},
      top1_abs_rate:{label:"Target top-1 share",short:"top-1 share",kind:"percent",help:"Share of target tokens whose preceding state recruits this direction most strongly in the top-k bank."},
      top5_abs_rate:{label:"Target top-5 share",short:"top-5 share",kind:"percent",help:"Share of target tokens whose preceding state recruits this direction in the layer's top five."},
      std_activation:{label:"Predictive activation variability",short:"activation std",kind:"number",help:"Standard deviation of h[t−1]·V over target tokens."},
      max_abs_unembed_token_cosine:{label:"Max |token cosine|",short:"max |token cos|",kind:"decimal",help:"Strongest geometric cosine between V and an output-token unembedding row."}
    };
    const TAIL_METRICS = {
      tail_selectivity_score:{label:"Source tail score",short:"tail score",kind:"decimal",help:"Unmodified source score computed with special predecessor states included."},
      selected_tail_special_read_share:{label:"Selected-tail special-read share · low first",short:"special-read",kind:"percent",direction:"asc",help:"Share of all 64 retained score-driving events read from start-of-text or another special token."},
      selected_tail_q999_robust_z:{label:"Selected-tail q99.9 robust z",short:"q99.9 z",kind:"decimal",help:"Score-driving tail's sampled q99.9 robust z; source moments include special reads."},
      selected_tail_top0_1pct_energy_share:{label:"Selected-tail top-0.1% energy",short:"top-.1% energy",kind:"percent",help:"Share of one-sided robust-z² energy in the strongest 0.1% of sampled targets."},
      selected_tail_doc_rate_z5:{label:"Windows above z=5",short:"windows >5z",kind:"percent",help:"Share of windows whose score-driving tail peak exceeds robust z=5."},
      selected_tail_unique_target_tokens:{label:"Unique retained targets",short:"unique targets",kind:"integer",help:"Unique target tokens among all 64 retained score-driving events."},
      selected_tail_unique_transitions:{label:"Unique retained transitions",short:"transitions",kind:"integer",help:"Unique previous-token → target-token pairs among all retained score-driving events."},
      selected_tail_top_context_largest_center_token_share:{label:"Largest target-token share · low first",short:"largest target",kind:"percent",direction:"asc",help:"Largest target-token share among retained score-driving events."},
      mean_abs_cosine:{label:"Predictive mean |cosine|",short:"mean |cos|",kind:"decimal",help:"Broad predictive-activity score for comparison."}
    };
    const COT_METRICS = {
      mean_abs_cosine:{label:"CoT mean |cosine|",short:"mean |cos|",kind:"decimal",help:"Mean absolute pre-target cosine over generated analysis targets for this effort."},
      doc_top5_presence_rate:{label:"Trace top-5 presence",short:"trace top-5",kind:"percent",help:"Share of reasoning traces where this direction enters the layer top five."},
      mean_document_peak_abs:{label:"Mean trace peak |act|",short:"trace peak",kind:"number",help:"Mean per-trace maximum absolute pre-target projection."},
      tail_selectivity_score:{label:"Raw CoT tail score",short:"raw tail",kind:"decimal",help:"Unscreened effort-local tail score; source moments and heaps include special predecessor states."},
      rank_global_tail_selectivity:{label:"Raw CoT tail rank · low first",short:"tail rank",kind:"integer",direction:"asc",help:"One-based source tail rank with no boundary correction."},
      selected_tail_special_read_share:{label:"Selected-tail special-read share · low first",short:"special-read",kind:"percent",direction:"asc",help:"Available only when the effort's complete retained-event file can be audited."},
      selected_tail_top_context_largest_center_token_share:{label:"Largest target-token share · low first",short:"largest target",kind:"percent",direction:"asc",help:"Lexical concentration among retained score-driving targets."},
      effective_support_fraction:{label:"Effective support · low first",short:"support",kind:"percent",direction:"asc",help:"Lower values indicate activation energy supported by fewer targets."}
    };
    const LENSES = {
      broad:{kind:"fineweb",label:"FineWeb broad activity",title:"FineWeb · broad predictive activity",primaryRank:"rank_global_mean_abs_cosine",defaultMetric:"mean_abs_cosine",metrics:BROAD_METRICS,caption:"Source rank uses mean absolute pre-target cosine over all target tokens."},
      screened:{kind:"fineweb",label:"FineWeb boundary-screened tails",title:"FineWeb · boundary-screened predictive tails",primaryRank:"rank_boundary_screened_tail",defaultMetric:"tail_selectivity_score",metrics:TAIL_METRICS,caption:"Rank among directions whose retained score-driving tail has special-read share at or below the threshold; scores are not recomputed."},
      raw:{kind:"fineweb",label:"FineWeb raw tails · diagnostic",title:"FineWeb · raw predictive tails",primaryRank:"rank_global_tail_selectivity",defaultMetric:"tail_selectivity_score",metrics:TAIL_METRICS,caption:"Unmodified source ordering, including start-of-text predecessor states."},
      cot_low:{kind:"cot",effort:"low",label:"Low-effort CoT discoveries",title:"CoT low · broad + raw-tail union",primaryRank:"rank_global_mean_abs_cosine",defaultMetric:"mean_abs_cosine",metrics:COT_METRICS,caption:"Candidate pool is the union of low-effort top broad and top raw-tail source ranks; default order is broad. No screened CoT rank is inferred."},
      cot_medium:{kind:"cot",effort:"medium",label:"Medium-effort CoT discoveries",title:"CoT medium · broad + raw-tail union",primaryRank:"rank_global_mean_abs_cosine",defaultMetric:"mean_abs_cosine",metrics:COT_METRICS,caption:"Candidate pool is the union of medium-effort top broad and top raw-tail source ranks; default order is broad. Raw tails include channel-entry states."}
    };
    const rowIndex = new Map(DATA.rankings.map(row=>[row.candidate,row]));
    const cotIndexes = Object.fromEntries(Object.entries(DATA.cot.efforts).map(([effort,payload])=>[effort,new Map(Object.entries(payload.rankings))]));
    const state = {
      lens:DATA.default_lens||"broad",
      views:Object.fromEntries(Object.entries(DATA.lenses).map(([lens,payload])=>[lens,{query:"",layer:"all",metric:LENSES[lens].defaultMetric,limit:Math.min(50,payload.candidates.length),selected:payload.candidates[0]||null}])),
      contexts:{
        broad:{query:"",limit:6,includeSpecial:true},
        screened:{query:"",limit:6,includeSpecial:false},
        raw:{query:"",limit:6,includeSpecial:true},
        cot_low:{query:"",limit:6,includeSpecial:false},
        cot_medium:{query:"",limit:6,includeSpecial:false}
      },
      cotEffort:DATA.cot.default_effort||"medium",
      cotViews:{
        low:{query:"",limit:6,includeSpecial:false,dedupe:true,polarity:"positive",focus:null,focusCandidate:null},
        medium:{query:"",limit:6,includeSpecial:false,dedupe:true,polarity:"positive",focus:null,focusCandidate:null}
      },
      cotPromptId:null,
      tokenLimit:8
    };
    const $=selector=>document.querySelector(selector);
    const esc=value=>String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
    const compact=new Intl.NumberFormat(undefined,{notation:"compact",maximumFractionDigits:1});
    const integer=new Intl.NumberFormat();
    const lensData=()=>DATA.lenses[state.lens], lensConfig=()=>LENSES[state.lens], viewState=()=>state.views[state.lens], contextState=()=>state.contexts[state.lens];
    const activeIndex=()=>lensConfig().kind==="cot"?cotIndexes[lensConfig().effort]:rowIndex;
    const activeRow=()=>activeIndex().get(viewState().selected);
    const cotData=(effort=state.cotEffort)=>DATA.cot.efforts[effort];
    const cotView=(effort=state.cotEffort)=>state.cotViews[effort];
    const cohortRows=()=>lensData().candidates.map(candidate=>activeIndex().get(candidate)).filter(Boolean);
    function decimal(value,digits=3){const n=Number(value);if(!Number.isFinite(n))return"—";const m=Math.abs(n);if(m!==0&&(m<.001||m>=10000))return n.toExponential(2);return n.toLocaleString(undefined,{maximumFractionDigits:digits});}
    function signed(value,digits=3){const n=Number(value);return Number.isFinite(n)?`${n>=0?"+":""}${decimal(n,digits)}`:"—";}
    function percent(value,digits=1){const n=Number(value);return Number.isFinite(n)?`${(n*100).toFixed(digits)}%`:"—";}
    function formatMetric(value,kind){if(kind==="percent")return percent(value);if(kind==="integer")return integer.format(value);return decimal(value,kind==="decimal"?4:3);}
    function visibleToken(value){return JSON.stringify(String(value??"")).replaceAll("\\n","↵").replaceAll("\\t","⇥");}
    function primaryRank(row){return row?.[lensConfig().primaryRank];}
    function polarityLabel(value){return value==="positive"?"High (+)":"Low (−)";}
    function searchBlob(row){
      const fineweb=DATA.context_summaries[row.candidate]||{}, cot=lensConfig().kind==="cot"?cotData(lensConfig().effort).context_summaries[row.candidate]||{}:{};
      const tokens=[fineweb,cot].flatMap(summary=>["positive","negative"].flatMap(p=>[...(summary[p]?.top_previous||[]).map(x=>x.token),...(summary[p]?.top_targets||[]).map(x=>x.token),...(summary[p]?.top_transitions||[]).map(x=>x.label)]));
      return `${row.candidate} layer ${row.layer} l${String(row.layer).padStart(2,"0")} sv ${row.sv_index_0} sv${String(row.sv_index_0).padStart(2,"0")} ${tokens.join(" ")}`.toLowerCase();
    }
    function setup(){
      const initial=parseHash();
      if(initial.lens&&DATA.lenses[initial.lens])state.lens=initial.lens;
      if(LENSES[state.lens].effort)state.cotEffort=LENSES[state.lens].effort;
      if(initial.candidate&&lensData().candidates.includes(initial.candidate)){viewState().selected=initial.candidate;expandLimit(initial.candidate);}
      document.querySelectorAll(".lens-tab").forEach(button=>button.addEventListener("click",()=>switchLens(button.dataset.lens)));
      $("#candidateSearch").addEventListener("input",event=>{viewState().query=event.target.value.trim().toLowerCase();render();});
      $("#layerFilter").addEventListener("change",event=>{viewState().layer=event.target.value;render();});
      $("#sortMetric").addEventListener("change",event=>{viewState().metric=event.target.value;render();});
      $("#rowLimit").addEventListener("change",event=>{viewState().limit=Number(event.target.value);render();});
      $("#resetFilters").addEventListener("click",resetFilters);
      addEventListener("hashchange",()=>{const target=parseHash();if(!target.lens||!DATA.lenses[target.lens]||!target.candidate)return;state.lens=target.lens;if(LENSES[state.lens].effort)state.cotEffort=LENSES[state.lens].effort;if(lensData().candidates.includes(target.candidate)){revealCandidate(target.candidate,false);return;}configure();render();syncHash();});
      addEventListener("keydown",event=>{const tag=document.activeElement?.tagName;if(event.key==="/"&&!['INPUT','SELECT','TEXTAREA'].includes(tag)){event.preventDefault();$("#candidateSearch").focus();}if(event.key==="Escape"&&document.activeElement===$("#candidateSearch")){viewState().query="";$("#candidateSearch").value="";render();}});
      configure();render();syncHash();
    }
    function parseHash(){let raw=location.hash.slice(1);try{raw=decodeURIComponent(raw);}catch(_){}if(!raw)return{lens:DATA.default_lens||"broad",candidate:null};const slash=raw.indexOf("/");if(slash>0&&DATA.lenses[raw.slice(0,slash)])return{lens:raw.slice(0,slash),candidate:raw.slice(slash+1)};return{lens:"broad",candidate:raw};}
    function syncHash(){const candidate=viewState().selected,next=candidate?`#${state.lens}/${encodeURIComponent(candidate)}`:"";if(location.hash===next)return;try{history.replaceState(null,"",next||`${location.pathname}${location.search}`);}catch(_){location.hash=next;}}
    function configure(){
      const meta=DATA.meta,view=viewState(),rows=cohortRows();document.body.dataset.lens=state.lens;
      document.querySelectorAll(".lens-tab").forEach(button=>button.setAttribute("aria-selected",String(button.dataset.lens===state.lens)));
      const effort=lensConfig().effort,cotMeta=effort?cotData(effort).meta:null;
      $("#facts").innerHTML=(cotMeta?[[compact.format(cotMeta.analysis_target_tokens),"analysis targets"],[integer.format(cotMeta.rollouts),`${effort} traces`],[integer.format(cotMeta.union_cohort_size),"broad + tail union"],[cotMeta.contexts_available?integer.format(cotMeta.source_contexts_per_side):"missing","events / tail"]]:[[compact.format(meta.target_tokens),"target tokens"],[integer.format(meta.documents),"FineWeb windows"],[integer.format(meta.total_candidates),"directions"],[integer.format(meta.contexts_per_polarity_source),"events / tail"]]).map(([v,l])=>`<div class="fact"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join("");
      const mirror=meta.mirror;
      $("#provenanceText").innerHTML=`Input <b>predictive_words_scan</b> and mirror <b>predictive_words_fineweb</b> store one shared evidence set. Their requested-method metadata conflicts and cannot establish which decomposition method produced the cached bank.`;
      $("#provenanceBadges").innerHTML=`<span class="badge ok">FineWeb + CoT V/S exact</span><span class="badge ok">one shared neighbor bank</span><span class="badge">scan label · ${esc(mirror.canonical_svd_method)}</span><span class="badge warn">fineweb label · ${esc(mirror.mirror_svd_method)} · conflict</span>`;
      $("#rankingTitle").textContent=lensConfig().title;
      const layers=[...new Set(rows.map(row=>Number(row.layer)))].sort((a,b)=>a-b);
      $("#layerFilter").innerHTML=`<option value="all">All embedded layers</option>`+layers.map(layer=>`<option value="${layer}">Layer ${String(layer).padStart(2,"0")}</option>`).join("");
      $("#sortMetric").innerHTML=Object.entries(lensConfig().metrics).map(([key,item])=>`<option value="${key}">${esc(item.label)}</option>`).join("");
      if(!lensConfig().metrics[view.metric])view.metric=lensConfig().defaultMetric;
      const choices=[25,50,100,250].filter(value=>value<rows.length);choices.push(rows.length);
      $("#rowLimit").innerHTML=[...new Set(choices)].map(value=>`<option value="${value}">${value===rows.length?`All ${integer.format(value)}`:`Top ${value}`}</option>`).join("");
      if(view.limit>rows.length)view.limit=rows.length;
      $("#candidateSearch").value=view.query;$("#layerFilter").value=view.layer;$("#sortMetric").value=view.metric;$("#rowLimit").value=String(view.limit);
      $("#footer").innerHTML=`SV identifiers are zero-based; rank numbers are one-based. FineWeb screened-tail rank is derived among candidates with retained selected-tail special-read share ≤ <b>${percent(meta.boundary_threshold,0)}</b>; its score, moments, heaps, and source raw rank are unchanged. No screened CoT rank is invented. Low CoT has rankings and rollouts but no retained-event file, so its traces cannot be linked to an SV. FineWeb predictive, low CoT, medium CoT, and the original reference share bit-identical V/S arrays; the single token-neighbor section is shared geometry.`;
    }
    function switchLens(lens){if(!DATA.lenses[lens]||lens===state.lens)return;state.lens=lens;if(LENSES[lens].effort)state.cotEffort=LENSES[lens].effort;configure();render();syncHash();}
    function resetFilters(){const view=viewState();view.query="";view.layer="all";view.metric=lensConfig().defaultMetric;view.limit=Math.min(50,cohortRows().length);configure();render();}
    function metricNumber(value){return value===null||value===undefined?NaN:Number(value);}
    function compareRows(a,b){const metric=lensConfig().metrics[viewState().metric],direction=metric.direction==="asc"?1:-1,av=metricNumber(a[viewState().metric]),bv=metricNumber(b[viewState().metric]);if(Number.isFinite(av)!==Number.isFinite(bv))return Number.isFinite(av)?-1:1;const diff=Number.isFinite(av)&&Number.isFinite(bv)?(av-bv)*direction:0;return diff||Number(primaryRank(a))-Number(primaryRank(b));}
    function filteredRows(applyLimit=true){const view=viewState(),query=view.query;const rows=cohortRows().filter(row=>{if(view.layer!=="all"&&Number(row.layer)!==Number(view.layer))return false;return!query||searchBlob(row).includes(query);}).sort(compareRows);return applyLimit?rows.slice(0,view.limit):rows;}
    function render(){const view=viewState(),previous=view.selected,all=filteredRows(false),rows=all.slice(0,view.limit);if(rows.length&&!rows.some(row=>row.candidate===view.selected))view.selected=rows[0].candidate;if(!rows.length)view.selected=null;if(view.selected!==previous)syncHash();renderRankings(rows,all.length);renderDetail(rows);}
    function renderRankings(rows,matchCount){
      const view=viewState(),metric=lensConfig().metrics[view.metric],values=rows.map(row=>metricNumber(row[view.metric])).filter(Number.isFinite),min=values.length?Math.min(...values):0,max=values.length?Math.max(...values):0;
      $("#rankingCount").textContent=`${integer.format(rows.length)} / ${integer.format(matchCount)}`;$("#rankingCaption").textContent=`${metric.label} · ${metric.direction==="asc"?"ascending":"descending"}. ${lensConfig().caption} SV indices are zero-based.`;
      if(!rows.length){$("#rankingList").innerHTML=`<div class="empty">No directions match these filters.</div>`;return;}
      $("#rankingList").innerHTML=rows.map(row=>{const value=metricNumber(row[view.metric]),span=Math.max(max-min,1e-12),norm=metric.direction==="asc"?(max-value)/span:(value-min)/span,width=!Number.isFinite(value)?2:values.length===1?100:Math.max(2,Math.min(100,norm*100)),shareValue=row.selected_tail_special_read_share,share=Number(shareValue),audited=shareValue!==null&&shareValue!==undefined&&Number.isFinite(share),boundary=audited&&share>DATA.meta.boundary_threshold?`<span class="boundary-pill">special ${percent(share,0)}</span>`:"",rank=primaryRank(row),cot=lensConfig().kind==="cot",sourceRank=cot?`raw tail #${integer.format(row.rank_global_tail_selectivity)} · ${polarityLabel(row.selected_tail_polarity)}`:state.lens==="screened"?`source raw #${integer.format(row.rank_global_tail_selectivity)}`:`${polarityLabel(row.selected_tail_polarity)} tail`,foot=cot?`FineWeb broad #${integer.format(rowIndex.get(row.candidate)?.rank_global_mean_abs_cosine)}`:`original tail #${integer.format(DATA.original[row.candidate]?.rank_global_tail_selectivity)}`;
        return `<button class="rank-row ${row.candidate===view.selected?"active":""}" type="button" data-candidate="${esc(row.candidate)}" ${row.candidate===view.selected?'aria-current="true"':""}><span class="rank-number">#${integer.format(rank)}</span><span><span class="rank-top"><span class="candidate">${esc(row.candidate)}${boundary}</span><span class="layer-note">L${String(row.layer).padStart(2,"0")} · SV${String(row.sv_index_0).padStart(2,"0")}</span></span><span class="rank-measure"><span>${esc(metric.short)}</span><b>${esc(formatMetric(value,metric.kind))}</b></span><span class="mini"><i style="width:${width.toFixed(2)}%"></i></span><span class="rank-foot"><span>${esc(sourceRank)}</span><span>${esc(foot)}</span></span></span></button>`;}).join("");
      $("#rankingList").querySelectorAll(".rank-row").forEach(button=>button.addEventListener("click",()=>{selectCandidate(button.dataset.candidate);if(innerWidth<=940)$("#detail").scrollIntoView({behavior:"smooth",block:"start"});}));
    }
    function selectCandidate(candidate,updateHash=true){if(!lensData().candidates.includes(candidate))return;viewState().selected=candidate;if(updateHash)syncHash();render();}
    function expandLimit(candidate){const sorted=cohortRows().slice().sort(compareRows),needed=sorted.findIndex(row=>row.candidate===candidate)+1,choices=[...new Set([25,50,100,250,sorted.length])].filter(value=>value<=sorted.length).sort((a,b)=>a-b);viewState().limit=choices.find(value=>value>=needed)||sorted.length;}
    function revealCandidate(candidate,updateHash=true){const view=viewState();view.query="";view.layer="all";expandLimit(candidate);view.selected=candidate;if(updateHash)syncHash();configure();render();}
    function metricCard(label,value,title){return `<div class="metric-card" title="${esc(title||"")}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;}
    function renderDetail(visibleRows){
      const root=$("#detail"),row=activeRow(),canonical=rowIndex.get(viewState().selected);if(!row||!canonical){root.innerHTML=`<div class="empty">Choose a broader filter to inspect a direction.</div>`;return;}
      const selectedIndex=visibleRows.findIndex(item=>item.candidate===row.candidate),cot=lensConfig().kind==="cot",selective=!cot&&state.lens!=="broad",shareValue=row.selected_tail_special_read_share,share=Number(shareValue),audited=shareValue!==null&&shareValue!==undefined&&Number.isFinite(share);
      const summary=cot?`This zero-based layer ${row.layer} / SV ${row.sv_index_0} direction is in the <b>${lensConfig().effort}-effort broad + raw-tail candidate union</b>. It has broad rank <b>#${integer.format(row.rank_global_mean_abs_cosine)}</b> over ${integer.format(row.n_tokens)} generated analysis targets and raw tail rank <b>#${integer.format(row.rank_global_tail_selectivity)}</b>. These are effort-local observations of the exact FineWeb direction.`:selective?`This zero-based layer ${row.layer} / SV ${row.sv_index_0} direction has source tail rank <b>#${integer.format(row.rank_global_tail_selectivity)}</b> and tail score <b>${decimal(row.tail_selectivity_score,4)}</b>. ${state.lens==="screened"?`It ranks <b>#${integer.format(row.rank_boundary_screened_tail)}</b> among directions passing the retained-event boundary screen.`:"The raw view deliberately preserves the unscreened source ordering."}`:`This zero-based layer ${row.layer} / SV ${row.sv_index_0} direction ranks <b>#${integer.format(row.rank_global_mean_abs_cosine)}</b> by mean absolute cosine over ${integer.format(row.n_tokens)} FineWeb targets. Each activation is read one causal position before the highlighted target.`;
      const cotMeta=cot?cotData(lensConfig().effort).meta:null;
      const warning=cot&&!cotMeta.contexts_available?`<div class="warning"><strong>Rankings only for low CoT:</strong> <code>top_contexts.jsonl</code> is absent, so raw tail ranks cannot be audited for special predecessors and no rollout location can be attributed to this SV. The rollout browser below is explicitly non-direction-linked.</div>`:cot?`<div class="warning"><strong>Raw CoT tail audit:</strong> ${integer.format(cotMeta.raw_top_boundary_dominated)} of ${integer.format(cotMeta.raw_cohort_size)} medium raw-tail leaders exceed the ${percent(DATA.meta.boundary_threshold,0)} retained special-predecessor threshold. ${audited?`This direction's selected-tail share is ${percent(share,0)}.`:""} Hiding marker events changes only the cards shown; moments, heaps, scores, and ranks remain uncorrected.</div>`:state.lens==="raw"?`<div class="warning"><strong>Raw-tail audit:</strong> ${integer.format(DATA.meta.raw_top_boundary_dominated)} of ${integer.format(DATA.meta.raw_cohort_size)} embedded raw leaders exceed the ${percent(DATA.meta.boundary_threshold,0)} special-read threshold. Their source moments, quantiles, document peaks, and heaps include start-of-text states; semantic selectivity claims are not valid without rescanning.</div>`:state.lens==="screened"?`<div class="warning screen"><strong>Diagnostic screen, not corrected statistics:</strong> this view excludes directions whose ${integer.format(DATA.meta.contexts_per_polarity_source)} retained score-driving contexts exceed the special-read threshold. Only the post-hoc screened order is new; source moments, scores, heaps, retained contexts, and the raw rank are not recomputed. A full rescan is required for valid content-only selectivity.</div>`:share>DATA.meta.boundary_threshold?`<div class="warning"><strong>Tail diagnostic:</strong> ${percent(share,0)} of this direction's retained score-driving events read from a special predecessor. Its broad-activity rank remains a whole-corpus summary, but its extreme contexts may be boundary artifacts.</div>`:"";
      root.innerHTML=`<section class="hero"><div class="detail-nav"><span class="detail-rank">${esc(lensConfig().label)} · broad rank #${integer.format(primaryRank(row))}</span><div class="nav-buttons"><button class="icon" id="previousCandidate" type="button" ${selectedIndex<=0?"disabled":""}>←</button><button class="icon" id="nextCandidate" type="button" ${selectedIndex<0||selectedIndex>=visibleRows.length-1?"disabled":""}>→</button><button class="copy" id="copyCandidate" type="button">Copy ID</button></div></div><div class="title-row"><h2 class="detail-title">${esc(row.candidate)}</h2><span class="id-chip">${cot?`${lensConfig().effort.toUpperCase()} COT`:`FINEWEB`} · L${String(row.layer).padStart(2,"0")} / SV${String(row.sv_index_0).padStart(2,"0")}</span></div><p class="detail-summary">${summary}</p><div class="semantic-note"><div class="time-diagram"><span class="time-node">READ h[t−1]</span><span class="time-arrow">→</span><span class="time-node">TARGET token[t]</span></div><p><strong>Predictive alignment changes the state/token pairing, not the shared SVD basis.</strong> FineWeb and both CoT efforts project bit-identical V/S slots. Temporal precedence is useful evidence, but it is not a causal intervention or a token probability.</p></div>${warning}<div class="metric-grid">${cot?cotHero(row,lensConfig().effort):selective?tailHero(row):broadHero(row)}</div>${comparisonSection(canonical)}${cotComparisonSection(canonical)}${selective?tailProfile(row):""}<details class="metric-details"><summary>Inspect all ${integer.format(Object.keys(row).length-1)} ${cot?`${lensConfig().effort}-effort CoT`:"FineWeb"} scan metrics</summary>${allMetrics(row)}</details></section>${contextSection(canonical)}${cotSection(canonical)}${tokenSection(canonical)}`;
      $("#previousCandidate").addEventListener("click",()=>selectedIndex>0&&selectCandidate(visibleRows[selectedIndex-1].candidate));$("#nextCandidate").addEventListener("click",()=>selectedIndex>=0&&selectedIndex<visibleRows.length-1&&selectCandidate(visibleRows[selectedIndex+1].candidate));$("#copyCandidate").addEventListener("click",copyCandidate);bindContextControls(canonical);bindCotControls(canonical);bindTokenControls(canonical);
    }
    function broadHero(row){return [metricCard("Predictive mean |cosine|",decimal(row.mean_abs_cosine,4),BROAD_METRICS.mean_abs_cosine.help),metricCard("Window top-5 presence",percent(row.doc_top5_presence_rate,1),BROAD_METRICS.doc_top5_presence_rate.help),metricCard("Mean window peak |act|",decimal(row.mean_document_peak_abs),BROAD_METRICS.mean_document_peak_abs.help),metricCard("Target top-1 share",percent(row.top1_abs_rate,1),BROAD_METRICS.top1_abs_rate.help),metricCard("Target top-5 share",percent(row.top5_abs_rate,1),BROAD_METRICS.top5_abs_rate.help),metricCard("Selected-tail special reads",percent(row.selected_tail_special_read_share,1),TAIL_METRICS.selected_tail_special_read_share.help)].join("");}
    function tailHero(row){return [metricCard("Source tail score",decimal(row.tail_selectivity_score,4),TAIL_METRICS.tail_selectivity_score.help),metricCard("Score-driving tail",polarityLabel(row.selected_tail_polarity),"Arbitrary SVD orientation whose source tail score is larger."),metricCard("Selected q99.9 robust z",decimal(row.selected_tail_q999_robust_z,3),TAIL_METRICS.selected_tail_q999_robust_z.help),metricCard("Top-0.1% tail energy",percent(row.selected_tail_top0_1pct_energy_share,2),TAIL_METRICS.selected_tail_top0_1pct_energy_share.help),metricCard("Windows above z=5",`${integer.format(row.selected_tail_doc_count_z5)} · ${percent(row.selected_tail_doc_rate_z5,1)}`,TAIL_METRICS.selected_tail_doc_rate_z5.help),metricCard("Selected-tail special reads",percent(row.selected_tail_special_read_share,1),TAIL_METRICS.selected_tail_special_read_share.help)].join("");}
    function cotHero(row,effort){const audit=row.selected_tail_special_read_share==null?"audit unavailable":percent(row.selected_tail_special_read_share,1);return [metricCard(`${effort} mean |cosine|`,decimal(row.mean_abs_cosine,4),COT_METRICS.mean_abs_cosine.help),metricCard("Trace top-5 presence",percent(row.doc_top5_presence_rate,1),COT_METRICS.doc_top5_presence_rate.help),metricCard("Mean trace peak |act|",decimal(row.mean_document_peak_abs),COT_METRICS.mean_document_peak_abs.help),metricCard("Raw tail rank / score",`#${integer.format(row.rank_global_tail_selectivity)} · ${decimal(row.tail_selectivity_score,3)}`,COT_METRICS.tail_selectivity_score.help),metricCard("Score-driving tail",polarityLabel(row.selected_tail_polarity),"Arbitrary shared-vector orientation."),metricCard("Special predecessor share",audit,COT_METRICS.selected_tail_special_read_share.help)].join("");}
    function comparisonSection(row){const original=DATA.original[row.candidate];return `<section class="comparison"><div class="comparison-head"><div><h3>One basis, two temporal alignments</h3><p>The direction arrays and singular values are exactly identical; only the projected state and attributed token differ.</p></div><span class="badge ok">V/S exact match</span></div><div class="pair-grid"><article class="pair-card predictive"><div class="pair-title"><span>Predictive alignment</span><span>h[t−1] → target t</span></div><div class="pair-values">${pairValue("Broad rank / mean |cos|",`#${integer.format(row.rank_global_mean_abs_cosine)} · ${decimal(row.mean_abs_cosine,4)}`)}${pairValue("Raw tail rank / score",`#${integer.format(row.rank_global_tail_selectivity)} · ${decimal(row.tail_selectivity_score,3)}`)}${pairValue("Selected tail",polarityLabel(row.selected_tail_polarity))}${pairValue("Special-read share",percent(row.selected_tail_special_read_share,1))}</div></article><article class="pair-card"><div class="pair-title"><span>Original current-token alignment</span><span>h[t] at token t</span></div><div class="pair-values">${pairValue("Broad rank / mean |cos|",`#${integer.format(original?.rank_global_mean_abs_cosine)} · ${decimal(original?.mean_abs_cosine,4)}`)}${pairValue("Tail rank / score",`#${integer.format(original?.rank_global_tail_selectivity)} · ${decimal(original?.tail_selectivity_score,3)}`)}${pairValue("Selected tail",polarityLabel(original?.selected_tail_polarity))}${pairValue("Windows above z=5",`${integer.format(original?.selected_tail_doc_count_z5)} · ${percent(original?.selected_tail_doc_rate_z5,1)}`)}</div></article></div><p class="comparison-note"><strong>Original:</strong> h[t] already contains token t, so activation is coincident with that token and predicts later tokens. <strong>Predictive:</strong> h[t−1] precedes the highlighted target, but special predecessor states—especially start-of-text—must be handled explicitly.</p></section>`;}
    function cotComparisonSection(row){const low=cotIndexes.low.get(row.candidate),medium=cotIndexes.medium.get(row.candidate);const card=(label,item,meta,extra,unit)=>`<article class="cot-compare-card"><div class="pair-title"><span>${esc(label)}</span><span>${esc(extra)}</span></div><div class="pair-values">${pairValue("Broad rank / mean |cos|",`#${integer.format(item?.rank_global_mean_abs_cosine)} · ${decimal(item?.mean_abs_cosine,4)}`)}${pairValue("Raw tail rank / score",`#${integer.format(item?.rank_global_tail_selectivity)} · ${decimal(item?.tail_selectivity_score,3)}`)}${pairValue("Exposure",`${compact.format(meta.analysis_target_tokens)} targets · ${integer.format(meta.rollouts)} ${unit}`)}${pairValue("Selected-tail special",item?.selected_tail_special_read_share==null?"audit unavailable":percent(item.selected_tail_special_read_share,1))}</div></article>`;return `<section class="cot-compare"><div class="comparison-head"><div><h3>One predictive direction, three corpora</h3><p>FineWeb and both reasoning efforts use bit-identical V/S arrays and one token-neighbor bank. Each rank, robust scale, and tail score is normalized within its own corpus/effort.</p></div><span class="badge ok">1,472 exact slots</span></div><div class="cot-compare-grid">${card("FineWeb predictive",row,{analysis_target_tokens:DATA.meta.target_tokens,rollouts:DATA.meta.documents},"web windows","windows")}${card("Low-effort CoT",low,cotData("low").meta,"independent rollout sample","traces")}${card("Medium-effort CoT",medium,cotData("medium").meta,"independent rollout sample","traces")}</div><p class="comparison-note">Compare rank and prevalence as descriptive transfer evidence, not as a controlled low-versus-medium effect size. The generated responses differ, and raw tail scores include special predecessor states.</p></section>`;}
    function pairValue(label,value){return `<div class="pair-value"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;}
    function tailProfile(row){return `<section class="tail-profile"><div class="tail-head"><h3>Two-sided predictive tail profile</h3><p>Robust statistics come from the unmodified source scan. Special-read shares and transition diversity summarize all ${integer.format(DATA.meta.contexts_per_polarity_source)} retained events per side.</p></div><div class="tail-grid">${tailSide(row,"positive")}${tailSide(row,"negative")}</div></section>`;}
    function tailSide(row,polarity){const p=`${polarity}_`,summary=DATA.context_summaries[row.candidate]?.[polarity],selected=row.selected_tail_polarity===polarity;return `<article class="tail-side ${selected?"selected":""}"><div class="tail-side-title"><h4>${polarity==="positive"?"+ High tail":"− Low tail"}</h4><span>${selected?"score driver":`score ${decimal(row[`${p}tail_selectivity_score`],3)}`}</span></div><div class="tail-stats">${tailStat("q99.9 robust z",decimal(row[`${p}q999_robust_z`],3))}${tailStat("max robust z",decimal(row[`${p}max_robust_z`],3))}${tailStat("top-.1% z² energy",percent(row[`${p}top0_1pct_energy_share`],2))}${tailStat("windows > z5",`${integer.format(row[`${p}doc_count_z5`])} · ${percent(row[`${p}doc_rate_z5`],1)}`)}${tailStat("special reads",percent(summary?.special_read_share,1))}${tailStat("unique transitions",integer.format(summary?.unique_transitions))}</div></article>`;}
    function tailStat(label,value){return `<div class="tail-stat"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;}
    function allMetrics(row){return `<div class="all-metrics">${Object.entries(row).filter(([key])=>key!=="candidate").map(([key,value])=>`<div class="all-metric"><span>${esc(key.replaceAll("_"," "))}</span><b>${esc(formatAny(key,value))}</b></div>`).join("")}</div>`;}
    function formatAny(key,value){if(value==null)return"—";if(typeof value==="string")return value;if(key.startsWith("rank_")||key.includes("_count_z")||key.startsWith("selected_tail_unique_")||["layer","sv_index_0","n_tokens","n_documents","token_sample_n"].includes(key))return integer.format(value);if(/(?:rate|share|fraction|weight)$/.test(key)||key.includes("energy_share"))return percent(value,2);return decimal(value,5);}
    function contextSection(row){const cfg=contextState();return `<section class="contexts"><div class="section-heading"><p class="eyebrow">Observed pre-target evidence · FineWeb</p><h3>Prediction-state transitions</h3><p>Every card projects the residual state at <b>READ h[t−1]</b> and attributes it to the adjacent <b>TARGET token[t]</b>. The context highlight marks the target; the transition header names the actual predecessor token whose state was read.</p><span class="sign-note">High (+) and low (−) are arbitrary SVD orientations. Contexts demonstrate association, not that this one direction caused the target token.</span></div><div class="context-controls"><label class="control">Search these contexts<input id="contextSearch" type="search" value="${esc(cfg.query)}" placeholder="Context, previous token, target, domain…"></label><label class="control">Per side<select id="contextLimit"></select></label><label class="check"><input id="includeSpecial" type="checkbox" ${cfg.includeSpecial?"checked":""}> Include special-read events</label></div><div class="context-grid"><section class="context-column positive"><div class="column-head"><h4>+ High predictive tail</h4><span id="positiveCount"></span></div><div id="positiveSummary"></div><div class="context-list" id="positiveList"></div></section><section class="context-column negative"><div class="column-head"><h4>− Low predictive tail</h4><span id="negativeCount"></span></div><div id="negativeSummary"></div><div class="context-list" id="negativeList"></div></section></div></section>`;}
    function bindContextControls(row){const max=DATA.meta.contexts_per_polarity_embedded,cfg=contextState(),limits=[...new Set([6,12,max].filter(value=>value>0&&value<=max))].sort((a,b)=>a-b);if(!limits.includes(cfg.limit))cfg.limit=limits[0]||max;$("#contextLimit").innerHTML=limits.map(value=>`<option value="${value}">${value===max?`All ${value}`:value}</option>`).join("");$("#contextLimit").value=String(cfg.limit);$("#contextSearch").addEventListener("input",event=>{cfg.query=event.target.value.trim().toLowerCase();renderContexts(row);});$("#contextLimit").addEventListener("change",event=>{cfg.limit=Number(event.target.value);renderContexts(row);});$("#includeSpecial").addEventListener("change",event=>{cfg.includeSpecial=event.target.checked;renderContexts(row);});renderContexts(row);}
    function filteredContexts(items){const cfg=contextState();return items.filter(item=>{if(!cfg.includeSpecial&&item.special_read)return false;if(!cfg.query)return true;return `${item.context} ${item.previous} ${item.target} ${hostname(item.url||"")} ${item.date||""}`.toLowerCase().includes(cfg.query);});}
    function renderContexts(row){const grouped=DATA.contexts[row.candidate]||{positive:[],negative:[]};for(const polarity of ["positive","negative"]){const filtered=filteredContexts(grouped[polarity]),shown=filtered.slice(0,contextState().limit),list=$(`#${polarity}List`);$(`#${polarity}Count`).textContent=`${shown.length} of ${filtered.length} visible`;renderTransitionSummary(row,polarity);list.replaceChildren();if(!shown.length){const empty=document.createElement("div");empty.className="empty";empty.textContent=contextState().includeSpecial?"No retained events match this search.":"No non-special retained events are embedded; enable special-read events to inspect the boundary signal.";list.append(empty);continue;}shown.forEach(item=>list.append(contextCard(item,polarity)));}}
    function renderTransitionSummary(row,polarity){const summary=DATA.context_summaries[row.candidate]?.[polarity],root=$(`#${polarity}Summary`);if(!summary){root.innerHTML="";return;}const chips=(label,items,formatter)=>`<div class="chips">${items.slice(0,3).map(item=>`<span class="data-chip"><b>${esc(label)}</b> ${esc(formatter(item))} · ${percent(item.share,0)}</span>`).join("")}</div>`;root.innerHTML=`<div class="transition-summary"><div class="summary-line"><span><b>${integer.format(summary.unique_documents)}</b> windows · <b>${integer.format(summary.unique_transitions)}</b> transitions</span><span>special reads <b>${percent(summary.special_read_share,0)}</b></span></div>${chips("PREV",summary.top_previous,item=>visibleToken(item.token))}${chips("TARGET",summary.top_targets,item=>visibleToken(item.token))}${chips("PAIR",summary.top_transitions,item=>`${visibleToken(item.previous)} → ${visibleToken(item.target)}`)}</div>`;}
    function contextCard(item,polarity){const card=document.createElement("article");card.className=`context-card ${item.special_read?"boundary":""}`;const meta=document.createElement("div");meta.className="context-meta";meta.innerHTML=`<span>${polarity==="positive"?"+":"−"} context #${integer.format(item.rank)}</span><span>act <b>${esc(signed(item.activation))}</b> · cos <b>${esc(signed(item.cosine,4))}</b> · z <b>${esc(signed(item.robust_z,2))}</b></span>`;const transition=document.createElement("div");transition.className="transition";transition.innerHTML=`<div class="transition-box"><span>READ h[t−1] · pos ${integer.format(item.read_position)}</span><b title="token id ${esc(item.previous_id)}">${esc(visibleToken(item.previous))}</b></div><span class="transition-arrow">→</span><div class="transition-box"><span>TARGET token[t] · pos ${integer.format(item.target_position)}</span><b>${esc(visibleToken(item.target))}</b></div>`;const copy=document.createElement("p");copy.className="context-copy";appendMarkedText(copy,item.context||"");const source=document.createElement("div");source.className="source-row";const details=[hostname(item.url||""),item.date?String(item.date).slice(0,10):null,item.document!=null?`window ${item.document}`:null].filter(Boolean).join(" · ");const left=document.createElement("span");left.textContent=details;source.append(left);if(item.special_read){const badge=document.createElement("span");badge.className="boundary-badge";badge.textContent="special predecessor";source.append(badge);}else if(/^https?:\/\//i.test(item.url||"")){const link=document.createElement("a");link.className="source-link";link.href=item.url;link.target="_blank";link.rel="noopener noreferrer";link.textContent="source ↗";source.append(link);}card.append(meta,transition,copy,source);return card;}
    function appendMarkedText(target,text){let cursor=0;while(cursor<text.length){const start=text.indexOf("⟦",cursor);if(start<0){target.append(document.createTextNode(text.slice(cursor)));break;}const end=text.indexOf("⟧",start+1);if(end<0){target.append(document.createTextNode(text.slice(cursor)));break;}target.append(document.createTextNode(text.slice(cursor,start)));const mark=document.createElement("mark");mark.textContent=text.slice(start+1,end);target.append(mark);cursor=end+1;}if(!text.length)target.textContent="(empty context)";}
    function hostname(url){try{return new URL(url).hostname.replace(/^www\./,"");}catch(_){return"unknown source";}}
    function cotSection(row){const low=cotData("low").meta,medium=cotData("medium").meta;return `<section class="cot-section" id="cotEvidence"><div class="cot-heading"><div><p class="eyebrow">Cross-corpus transfer · generated analysis</p><h3>Predictive CoT evidence</h3><p>Switch effort without changing the selected zero-based SV or FineWeb discovery lens. Medium events join each pre-target projection to its full prompt → analysis → final rollout. Low rankings and rollout text remain available, but its missing retained-event file prevents direction-linked localization.</p><span class="sign-note">Both efforts use this exact FineWeb V/S slot. Their responses are independent generated samples, and corpus-local ranks or robust z-scores are not controlled effort effects.</span></div><div class="cot-effort-switch" role="tablist" aria-label="CoT reasoning effort"><button class="cot-effort-button" type="button" role="tab" data-cot-effort="low"><b>Low effort</b><small>${compact.format(low.analysis_target_tokens)} targets · ${integer.format(low.rollouts)} traces · events missing</small></button><button class="cot-effort-button" type="button" role="tab" data-cot-effort="medium"><b>Medium effort</b><small>${compact.format(medium.analysis_target_tokens)} targets · ${integer.format(medium.rollouts)} traces · ${integer.format(medium.source_contexts_per_side)}/tail</small></button></div></div><div id="cotPanel"></div></section>`;}
    function bindCotControls(row){document.querySelectorAll(".cot-effort-button").forEach(button=>button.addEventListener("click",()=>{state.cotEffort=button.dataset.cotEffort;renderCotPanel(row);}));renderCotPanel(row);}
    function cotMetricCards(row,effort){const special=row?.selected_tail_special_read_share==null?"audit unavailable":percent(row.selected_tail_special_read_share,1);return [metricCard("Broad rank / mean |cos|",`#${integer.format(row?.rank_global_mean_abs_cosine)} · ${decimal(row?.mean_abs_cosine,4)}`),metricCard("Trace top-5 presence",percent(row?.doc_top5_presence_rate,1)),metricCard("Mean trace peak |act|",decimal(row?.mean_document_peak_abs)),metricCard("Raw tail rank / score",`#${integer.format(row?.rank_global_tail_selectivity)} · ${decimal(row?.tail_selectivity_score,3)}`),metricCard("Score-driving tail",polarityLabel(row?.selected_tail_polarity)),metricCard("Selected-tail special",special)].join("");}
    function renderCotPanel(row){document.querySelectorAll(".cot-effort-button").forEach(button=>button.setAttribute("aria-selected",String(button.dataset.cotEffort===state.cotEffort)));const payload=cotData(),meta=payload.meta,item=cotIndexes[state.cotEffort].get(row.candidate),root=$("#cotPanel");if(!root)return;const current=cotView();if(current.focusCandidate!==row.candidate){current.focusCandidate=row.candidate;current.focus=null;current.polarity=item?.selected_tail_polarity||"positive";}if(!meta.contexts_available){root.innerHTML=`<div class="cot-metrics">${cotMetricCards(item,state.cotEffort)}</div><div class="cot-missing"><strong>Direction-linked low-effort events are unavailable.</strong> The scan directory has rankings, 55 rollout records, and the exact shared direction bank, but no <code>top_contexts.jsonl</code>. No activation location, predecessor transition, special-read share, or highlighted trace can be attributed to ${esc(row.candidate)}. The browser below is a corpus reference only.</div><div class="cot-rollout-controls"><label class="control">Browse low-effort rollout (not SV-linked)<select id="cotRolloutSelect"></select></label><span class="unlinked-badge">not direction-linked</span></div><article class="cot-trace-viewer" id="cotTraceViewer"></article>`;bindCotRolloutBrowser();return;}const share=item?.selected_tail_special_read_share,caveat=`<div class="cot-caveat"><strong>Retained-event diagnostic:</strong> ${integer.format(meta.raw_top_boundary_dominated)} of ${integer.format(meta.raw_cohort_size)} medium raw-tail leaders exceed the ${percent(DATA.meta.boundary_threshold,0)} special-predecessor threshold; this direction's selected tail is ${percent(share,0)}. Special events are hidden by default. That filter changes only this browser—not source moments, quantiles, heaps, scores, or ranks. A valid content-only ranking requires a rescan.</div>`;root.innerHTML=`<div class="cot-metrics">${cotMetricCards(item,state.cotEffort)}</div>${caveat}<div class="cot-controls"><label class="control">Search CoT events<input id="cotContextSearch" type="search" value="${esc(current.query)}" placeholder="Transition, task, category, snippet…"></label><label class="control">Events shown<select id="cotContextLimit"></select></label><label class="check"><input id="cotIncludeSpecial" type="checkbox" ${current.includeSpecial?"checked":""}> Include special predecessors</label><label class="check"><input id="cotDedupe" type="checkbox" ${current.dedupe?"checked":""}> One event per trace</label></div><div class="cot-workbench"><aside class="cot-browser"><div class="cot-polarity-switch"><button class="cot-polarity-button" type="button" data-cot-polarity="positive"><b>+ High tail</b><small id="cotPositiveCount"></small></button><button class="cot-polarity-button" type="button" data-cot-polarity="negative"><b>− Low tail</b><small id="cotNegativeCount"></small></button></div><div class="cot-footprint" id="cotFootprint"></div><div class="cot-event-list" id="cotEventList"></div></aside><article class="cot-trace-viewer" id="cotTraceViewer"></article></div>`;const max=meta.embedded_contexts_per_side,limits=[...new Set([6,max].filter(value=>value>0&&value<=max))].sort((a,b)=>a-b);if(!limits.includes(current.limit))current.limit=limits[0]||max;$("#cotContextLimit").innerHTML=limits.map(value=>`<option value="${value}">${value===max?`All ${value}`:value}</option>`).join("");$("#cotContextLimit").value=String(current.limit);$("#cotContextSearch").addEventListener("input",event=>{current.query=event.target.value.trim().toLowerCase();renderCotEvents(row);});$("#cotContextLimit").addEventListener("change",event=>{current.limit=Number(event.target.value);renderCotEvents(row);});$("#cotIncludeSpecial").addEventListener("change",event=>{current.includeSpecial=event.target.checked;renderCotEvents(row);});$("#cotDedupe").addEventListener("change",event=>{current.dedupe=event.target.checked;renderCotEvents(row);});document.querySelectorAll(".cot-polarity-button").forEach(button=>button.addEventListener("click",()=>{current.polarity=button.dataset.cotPolarity;current.focus=null;renderCotEvents(row);}));renderCotEvents(row);}
    function cotRollouts(effort=state.cotEffort){return Object.values(cotData(effort).rollouts);}
    function rolloutByPrompt(promptId,effort=state.cotEffort){return cotRollouts(effort).find(item=>item.prompt_id===promptId)||null;}
    function bindCotRolloutBrowser(){const rollouts=cotRollouts();if(!state.cotPromptId||!rolloutByPrompt(state.cotPromptId))state.cotPromptId=rollouts[0]?.prompt_id||null;const select=$("#cotRolloutSelect");select.innerHTML=rollouts.map(item=>`<option value="${esc(item.prompt_id)}">${esc(item.prompt_id)} · ${esc(item.category)}</option>`).join("");select.value=state.cotPromptId||"";select.addEventListener("change",event=>{state.cotPromptId=event.target.value;renderCotTrace(null,true);});renderCotTrace(null,true);}
    function cotFilteredEvents(items){const current=cotView(),seen=new Set();return items.filter(item=>{if(!current.includeSpecial&&item.special_read)return false;const rollout=cotData().rollouts[String(item.document)],prompt=(rollout?.messages||[]).map(message=>message.content).join(" ");if(current.query&&!`${item.marked} ${item.previous} ${item.target} ${rollout?.prompt_id} ${rollout?.category} ${rollout?.difficulty} ${prompt}`.toLowerCase().includes(current.query))return false;if(current.dedupe&&seen.has(item.document))return false;if(current.dedupe)seen.add(item.document);return true;});}
    function cotEventKey(item){return item?`${item.polarity}/${item.rank}/${item.document}`:"";}
    function renderCotEvents(row){const payload=cotData(),current=cotView(),grouped=payload.contexts[row.candidate]||{positive:[],negative:[]},all=grouped[current.polarity]||[],filtered=cotFilteredEvents(all),shown=filtered.slice(0,current.limit);document.querySelectorAll(".cot-polarity-button").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.cotPolarity===current.polarity)));$("#cotPositiveCount").textContent=`${cotFilteredEvents(grouped.positive).length} visible`;$("#cotNegativeCount").textContent=`${cotFilteredEvents(grouped.negative).length} visible`;renderCotFootprint(row);if(!current.focus||!shown.some(item=>cotEventKey(item)===cotEventKey(current.focus)))current.focus=shown[0]||null;const root=$("#cotEventList");root.replaceChildren();if(!shown.length){const empty=document.createElement("div");empty.className="empty";empty.textContent=current.includeSpecial?"No retained CoT events match this search.":"No non-special embedded events match. Enable special predecessors only to audit the marker signal.";root.append(empty);renderCotTrace(null,false);return;}shown.forEach(item=>root.append(cotEventCard(item,row)));renderCotTrace(current.focus,false);}
    function renderCotFootprint(row){const summary=cotData().context_summaries[row.candidate]?.[cotView().polarity],root=$("#cotFootprint");if(!summary||!root)return;const chips=(label,items,format)=>`<div class="chips">${items.slice(0,3).map(item=>`<span class="data-chip"><b>${label}</b> ${esc(format(item))} · ${percent(item.share,0)}</span>`).join("")}</div>`;root.innerHTML=`<div class="summary-line"><span><b>${integer.format(summary.unique_traces)}</b> traces · <b>${integer.format(summary.unique_transitions)}</b> transitions</span><span>special <b>${percent(summary.special_read_share,0)}</b> · median ${percent(summary.median_progress,0)}</span></div>${chips("PREV",summary.top_previous,item=>visibleToken(item.token))}${chips("TARGET",summary.top_targets,item=>visibleToken(item.token))}${chips("PAIR",summary.top_transitions,item=>`${visibleToken(item.previous)} → ${visibleToken(item.target)}`)}${chips("TASK",summary.top_categories,item=>item.category)}`;}
    function cotEventCard(item,row){const rollout=cotData().rollouts[String(item.document)],card=document.createElement("article");card.className=`cot-event ${cotEventKey(item)===cotEventKey(cotView().focus)?"active":""} ${item.special_read?"boundary":""}`;const meta=document.createElement("div");meta.className="context-meta";meta.innerHTML=`<span>${item.polarity==="positive"?"+":"−"} event #${integer.format(item.rank)} · <b>${esc(rollout?.prompt_id)}</b></span><span>act <b>${esc(signed(item.activation))}</b> · cos <b>${esc(signed(item.cosine,4))}</b> · tail z <b>${esc(decimal(item.tail_z,2))}</b></span>`;const transition=document.createElement("div");transition.className="transition";transition.innerHTML=`<div class="transition-box"><span>READ h[t−1] · pos ${integer.format(item.read_position)}</span><b title="token id ${esc(item.previous_id)}">${esc(visibleToken(item.previous))}</b></div><span class="transition-arrow">→</span><div class="transition-box"><span>TARGET token[t] · pos ${integer.format(item.target_position)}</span><b>${esc(visibleToken(item.target))}</b></div>`;const copy=document.createElement("p");copy.className="context-copy";appendMarkedText(copy,item.marked||"");const footer=document.createElement("div");footer.className="cot-event-footer";const left=document.createElement("span");left.textContent=`${rollout?.category||"task"} · ${percent(item.progress,0)} response`;if(item.special_read){const badge=document.createElement("span");badge.className="boundary-badge";badge.textContent="special predecessor";left.append(document.createTextNode(" "),badge);}const open=document.createElement("button");open.type="button";open.className="cot-open";open.textContent="Read full trace →";open.addEventListener("click",()=>openCotTrace(item,row));footer.append(left,open);const progress=document.createElement("span");progress.className="cot-progress";progress.innerHTML=`<i style="width:${Math.max(1,item.progress*100).toFixed(2)}%"></i>`;card.addEventListener("click",event=>{if(!event.target.closest("button"))openCotTrace(item,row);});card.append(meta,transition,copy,footer,progress);return card;}
    function openCotTrace(item,row){cotView().focus=item;state.cotPromptId=item.prompt_id;renderCotEvents(row);$("#cotTraceViewer")?.scrollIntoView({behavior:"smooth",block:"nearest"});}
    function renderCotTrace(item,unlinked){const root=$("#cotTraceViewer");if(!root)return;const rollout=item?cotData().rollouts[String(item.document)]:rolloutByPrompt(state.cotPromptId);if(!rollout){root.innerHTML=`<div class="empty">${unlinked?"Choose a rollout to read the corpus reference.":"Choose a visible transition event to open its linked trace."}</div>`;return;}state.cotPromptId=rollout.prompt_id;root.innerHTML=`<div class="cot-trace-head"><div><h4>${esc(rollout.prompt_id)}</h4><p>${esc(rollout.category)} · ${esc(rollout.difficulty)}${item?` · event #${integer.format(item.rank)} at ${percent(item.progress,0)} of response`:" · rollout corpus reference"}</p></div><div class="trace-badges"><span class="badge ok">${esc(state.cotEffort)} effort</span>${unlinked?'<span class="unlinked-badge">not direction-linked</span>':""}${item?.special_read?'<span class="badge warn">special predecessor</span>':""}${rollout.hit_cap?'<span class="badge warn">hit generation cap</span>':""}</div></div><div class="cot-trace-body"><section class="cot-trace-block"><div class="cot-trace-label"><span>Prompt</span><span>excluded from activation scan · ${integer.format(rollout.prompt_tokens)} tokens</span></div><div id="cotTracePrompt"></div></section><section class="cot-trace-block"><div class="cot-trace-label"><span>Generated analysis</span><span>${unlinked?"present, but not localized to this SV":"target highlighted; projection read one token earlier"} · ${integer.format(rollout.analysis_tokens)} tokens</span></div><p class="cot-trace-text" id="cotTraceReasoning"></p></section><section class="cot-trace-block"><div class="cot-trace-label"><span>Final response</span><span>excluded from activation scan · ${integer.format(rollout.final_tokens)} tokens</span></div><p class="cot-trace-text" id="cotTraceFinal"></p></section></div>`;const promptRoot=$("#cotTracePrompt");(rollout.messages||[]).forEach(message=>{const p=document.createElement("p");p.className="cot-trace-text";p.textContent=`${String(message.role).toUpperCase()}\n${message.content}`;promptRoot.append(p);});const reasoning=$("#cotTraceReasoning");if(item)appendCotFocusedReasoning(reasoning,rollout.reasoning||"",item);else reasoning.textContent=rollout.reasoning||"No analysis text was captured.";const final=$("#cotTraceFinal");final.textContent=rollout.final?.trim()||(rollout.hit_cap?"No final-channel response was captured before the generation cap.":"No final-channel response was captured.");if(!rollout.final?.trim())final.classList.add("trace-empty");}
    function appendCotFocusedReasoning(target,text,item){if(!text){target.textContent="No analysis text was captured.";target.classList.add("trace-empty");return;}const plain=String(item.plain||""),index=plain?text.indexOf(plain):-1;if(index<0){const focus=document.createElement("span");focus.className="trace-focus";appendMarkedText(focus,item.marked||"");target.append(focus,document.createTextNode(`\n\n${text}`));return;}target.append(document.createTextNode(text.slice(0,index)));const focus=document.createElement("span");focus.className="trace-focus";appendMarkedText(focus,item.marked||plain);target.append(focus,document.createTextNode(text.slice(index+plain.length)));}
    function tokenSection(row){const data=DATA.unembedding[row.candidate];if(!data)return"";return `<details class="token-section"><summary><div class="token-summary"><div><p class="eyebrow">Shared direction geometry</p><h3>Token-space neighbors</h3><p>FineWeb predictive, low CoT, medium CoT, and the original current-token reference use this exact V direction; their neighbor artifact is byte-identical. Neighbors are geometric references, not observed next-token probabilities.</p></div><b>max |cos| ${decimal(data.max_abs,4)} · expand ↓</b></div></summary><div class="token-content"><div class="token-toolbar"><label class="control token-limit">Tokens per side<select id="tokenLimit"></select></label></div><div class="token-grid"><section class="token-column aligned"><div class="token-head"><h4>+ Aligned tokens</h4><span id="alignedTokenCount"></span></div><div class="token-list" id="alignedTokenList"></div></section><section class="token-column opposed"><div class="token-head"><h4>− Opposed tokens</h4><span id="opposedTokenCount"></span></div><div class="token-list" id="opposedTokenList"></div></section></div></div></details>`;}
    function bindTokenControls(row){const data=DATA.unembedding[row.candidate];if(!data||!$("#tokenLimit"))return;const max=Math.max(data.nearest.length,data.farthest.length),limits=[...new Set([8,16,max].filter(value=>value<=max))].sort((a,b)=>a-b);if(!limits.includes(state.tokenLimit))state.tokenLimit=limits[0]||max;$("#tokenLimit").innerHTML=limits.map(value=>`<option value="${value}">${value===max?`All ${value}`:value}</option>`).join("");$("#tokenLimit").value=String(state.tokenLimit);$("#tokenLimit").addEventListener("change",event=>{state.tokenLimit=Number(event.target.value);renderTokenRows(data);});renderTokenRows(data);}
    function renderTokenRows(data){renderTokenColumn("aligned",data.nearest,data);renderTokenColumn("opposed",data.farthest,data);}
    function renderTokenColumn(side,items,data){const list=$(`#${side}TokenList`);if(!list)return;const shown=items.slice(0,state.tokenLimit);$(`#${side}TokenCount`).textContent=`${shown.length} of ${items.length}`;list.innerHTML=shown.map((item,index)=>{const label=item.decoded||item.token||`token #${item.id}`,z=(Number(item.cosine)-Number(data.mean))/Math.max(Number(data.std),1e-9);return `<article class="token-row"><span class="token-rank">#${index+1}</span><span class="token-name"><b>${esc(visibleToken(label))}</b><small>${item.token?`raw ${esc(visibleToken(item.token))} · `:""}id ${integer.format(item.id)}</small></span><span class="token-value">${signed(item.cosine,4)}<br>${signed(z,2)}σ</span></article>`;}).join("");}
    function copyCandidate(){const button=$("#copyCandidate"),value=viewState().selected,done=()=>{button.textContent="Copied";setTimeout(()=>button.textContent="Copy ID",1200);};if(navigator.clipboard?.writeText)navigator.clipboard.writeText(value).then(done).catch(()=>fallbackCopy(value,done));else fallbackCopy(value,done);}
    function fallbackCopy(text,done){const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.append(area);area.select();document.execCommand("copy");area.remove();done();}
    setup();
  </script>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    canonical = args.input.expanduser().resolve()
    mirror = args.mirror.expanduser().resolve()
    original_basis = args.original_basis.expanduser().resolve()
    original_rankings = args.original_rankings.expanduser().resolve()
    cot_low_dir = args.cot_low_dir.expanduser().resolve()
    cot_medium_dir = args.cot_medium_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = build_payload(
        canonical,
        mirror,
        original_basis,
        original_rankings,
        cot_low_dir,
        cot_medium_dir,
        args.top,
        args.contexts_per_side,
        args.cot_contexts_per_side,
        args.boundary_threshold,
    )
    html = HTML_TEMPLATE.replace("__PREDICTIVE_PAYLOAD__", safe_script_json(payload))
    if "__JAVASCRIPT_CONTINUES__" in html:
        raise SystemExit("Dashboard JavaScript template is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    contexts = sum(
        len(items)
        for grouped in payload["contexts"].values()
        for items in grouped.values()
    )
    cot_contexts = sum(
        len(items)
        for effort in payload["cot"]["efforts"].values()
        for grouped in effort["contexts"].values()
        for items in grouped.values()
    )
    neighbors = sum(
        len(candidate[side])
        for candidate in payload["unembedding"].values()
        for side in ("nearest", "farthest")
    )
    print(f"Wrote {output}")
    print(
        f"Embedded {payload['meta']['candidate_union']:,} unique candidates, "
        f"{contexts:,} FineWeb contexts, {cot_contexts:,} CoT contexts, "
        f"and {neighbors:,} shared token neighbors"
    )
    print(
        "Mirror substantive artifacts and direction arrays are identical; "
        f"svd_method metadata={payload['meta']['mirror']['canonical_svd_method']!r} vs "
        f"{payload['meta']['mirror']['mirror_svd_method']!r}"
    )
    print(f"Output size: {output.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
