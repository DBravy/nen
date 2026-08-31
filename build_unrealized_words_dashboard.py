#!/usr/bin/env python3
"""Build a self-contained browser dashboard for two FineWeb SV scans.

The source data files are intentionally large. This script keeps the highest
ranked candidates from both the broad-activity and selective-tail rankings,
prunes contexts and token neighbors to the fields used by the UI, and embeds the
result in one HTML file that can be opened directly from disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "unrealized_words_fineweb"
DEFAULT_SELECTIVITY_DATA_DIR = ROOT / "unrealized_words_selectivity"
DEFAULT_LEFT_DATA_DIR = ROOT / "predictive_words_scan"
LEFT_TOKEN_NEIGHBORS_PER_SIDE = 16
BROAD_RANK = "rank_global_mean_abs_cosine"
SELECTIVITY_RANK = "rank_global_tail_selectivity"
RANKING_TEXT_FIELDS = {
    "candidate",
    "selected_tail_polarity",
    "nearest_unembed_token",
    "nearest_unembed_decoded",
    "farthest_unembed_token",
    "farthest_unembed_decoded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing rankings, contexts, unembedding neighbors, and metadata",
    )
    parser.add_argument(
        "--selectivity-data-dir",
        type=Path,
        default=DEFAULT_SELECTIVITY_DATA_DIR,
        help="Directory containing the selective-tail scan outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: DATA_DIR/report.html)",
    )
    parser.add_argument(
        "--left-data-dir",
        type=Path,
        default=DEFAULT_LEFT_DATA_DIR,
        help=(
            "Directory containing fingerprinted left_singular_vectors.jsonl and metadata "
            "(default: predictive_words_scan, whose V/S bank matches this FineWeb scan)"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=250,
        help="Number of globally top-ranked SVs to embed; 0 embeds all (default: 250)",
    )
    parser.add_argument(
        "--selectivity-top",
        type=int,
        default=250,
        help="Number of top selective-tail SVs to embed; 0 embeds all (default: 250)",
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


def load_rankings(path: Path, top_n: int, rank_key: str) -> tuple[list[dict[str, Any]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "candidate" not in reader.fieldnames or rank_key not in reader.fieldnames:
            raise SystemExit(f"{path} is missing candidate or {rank_key}")
        rows: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, 2):
            try:
                row = {key: (value if key in RANKING_TEXT_FIELDS else number(value)) for key, value in raw.items()}
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric field on {path}:{row_number}: {exc}") from exc
            rows.append(row)

    rows.sort(key=lambda row: int(row[rank_key]))
    total = len(rows)
    if top_n < 0:
        raise SystemExit("top counts must be 0 or greater")
    return (rows if top_n == 0 else rows[:top_n]), total


def compact_context(raw: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("source") or {}
    return {
        "polarity": raw.get("polarity"),
        "rank": raw.get("rank_within_polarity"),
        "activation": raw.get("activation"),
        "cosine": raw.get("cosine_activation"),
        "token": raw.get("token", ""),
        "context": raw.get("context_marked") or raw.get("context", ""),
        "document": raw.get("document_index"),
        "position": raw.get("token_position"),
        "url": source.get("url"),
        "date": source.get("date"),
        "dump": source.get("dump"),
    }


def load_contexts(path: Path, candidates: list[str]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], int]:
    contexts = {candidate: {"positive": [], "negative": []} for candidate in candidates}
    candidate_set = set(candidates)
    source_total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source_total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            candidate = raw.get("candidate")
            if candidate not in candidate_set:
                continue
            polarity = raw.get("polarity")
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected polarity {polarity!r} on {path}:{line_number}")
            contexts[candidate][polarity].append(compact_context(raw))

    for by_polarity in contexts.values():
        for values in by_polarity.values():
            values.sort(key=lambda item: int(item.get("rank") or 0))
    return contexts, source_total


def compact_token_neighbor(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("token_id"),
        "token": raw.get("token"),
        "decoded": raw.get("decoded"),
        "cosine": raw.get("cosine"),
    }


def load_unembedding_neighbors(
    path: Path, candidates: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    neighbors: dict[str, dict[str, Any]] = {}
    candidate_set = set(candidates)
    source_total = 0
    domain_max = 0.0
    spaces: set[str] = set()
    vocab_sizes: set[int] = set()
    list_sizes: set[int] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source_total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc

            candidate = raw.get("candidate")
            if not isinstance(candidate, str):
                raise SystemExit(f"Missing candidate on {path}:{line_number}")
            spaces.add(str(raw.get("space")))
            vocab_sizes.add(int(raw.get("unembedding_vocab_rows_considered", 0)))
            nearest = raw.get("nearest_tokens") or []
            farthest = raw.get("farthest_tokens") or []
            list_sizes.update((len(nearest), len(farthest)))
            domain_max = max(domain_max, abs(float(raw.get("max_abs_token_cosine", 0))))

            if candidate not in candidate_set:
                continue
            if candidate in neighbors:
                raise SystemExit(f"Duplicate unembedding candidate {candidate!r} on {path}:{line_number}")
            neighbors[candidate] = {
                "vocab": raw.get("unembedding_vocab_rows_considered"),
                "mean": raw.get("unembedding_cosine_mean"),
                "std": raw.get("unembedding_cosine_std"),
                "max_abs": raw.get("max_abs_token_cosine"),
                "nearest_z": raw.get("nearest_token_z"),
                "farthest_z": raw.get("farthest_token_z"),
                "nearest_margin": raw.get("nearest_cosine_margin"),
                "farthest_margin": raw.get("farthest_cosine_margin"),
                "nearest": [compact_token_neighbor(item) for item in nearest],
                "farthest": [compact_token_neighbor(item) for item in farthest],
            }

    missing = sorted(candidate_set - neighbors.keys())
    if missing:
        preview = ", ".join(missing[:8])
        raise SystemExit(f"Missing unembedding neighbors for {len(missing)} candidates: {preview}")
    if len(spaces) != 1 or len(vocab_sizes) != 1 or len(list_sizes) != 1:
        raise SystemExit(
            f"Inconsistent unembedding metadata: spaces={sorted(spaces)}, "
            f"vocab_sizes={sorted(vocab_sizes)}, list_sizes={sorted(list_sizes)}"
        )
    ordered_neighbors = {candidate: neighbors[candidate] for candidate in candidates}
    return ordered_neighbors, {
        "total": source_total,
        "space": next(iter(spaces)),
        "vocab": next(iter(vocab_sizes)),
        "per_side": next(iter(list_sizes)),
        "domain_max": domain_max,
    }


def safe_script_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def direction_bank_fingerprint(data_dir: Path) -> tuple[str, dict[str, float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required to validate the left-vector basis") from exc

    digest = hashlib.sha256()
    digest.update(b"jlens-vs-bank-v1\0")
    spectral_fractions: dict[str, float] = {}

    def update(key: str, array: Any) -> None:
        contiguous = np.ascontiguousarray(array)
        digest.update(key.encode("ascii") + b"\0")
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(contiguous.tobytes())

    paths = sorted((data_dir / "directions").glob("L[0-9][0-9].npz"))
    if not paths:
        raise SystemExit(f"No LXX.npz direction banks found under {data_dir / 'directions'}")
    for path in paths:
        layer = int(path.stem[1:])
        with np.load(path) as saved:
            if not {"V", "S"}.issubset(saved.files):
                raise SystemExit(f"{path} must contain V and S arrays")
            V = np.ascontiguousarray(saved["V"], dtype=np.float32)
            S = np.ascontiguousarray(saved["S"].reshape(-1), dtype=np.float32)
        digest.update(f"L{layer:02d}\0".encode("ascii"))
        update("V", V)
        update("S", S)
        energy = float(np.sum(np.square(S, dtype=np.float64)))
        for sv0, singular_value in enumerate(S):
            spectral_fractions[f"L{layer:02d}_SV{sv0:02d}"] = (
                float(float(singular_value) ** 2 / energy) if energy > 0 else 0.0
            )
    return digest.hexdigest(), spectral_fractions


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


def load_left_singular_payload(
    artifact_dir: Path,
    broad_data_dir: Path,
    selectivity_data_dir: Path,
    display_candidates: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = artifact_dir / "left_singular_vectors_metadata.json"
    records_path = artifact_dir / "left_singular_vectors.jsonl"
    for path in (metadata_path, records_path):
        if not path.is_file():
            raise SystemExit(f"Missing required left-vector artifact: {path}")

    broad_fingerprint, spectral_fractions = direction_bank_fingerprint(broad_data_dir)
    selective_fingerprint, _ = direction_bank_fingerprint(selectivity_data_dir)
    if broad_fingerprint != selective_fingerprint:
        raise SystemExit(
            "Broad and selective FineWeb direction banks differ; left vectors cannot be shared"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_fingerprint = metadata.get("basis_fingerprint_sha256")
    if artifact_fingerprint != broad_fingerprint:
        raise SystemExit(
            "Left-vector artifact basis fingerprint does not match the FineWeb V/S bank"
        )

    records: dict[str, Any] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            layer = int(raw["layer"])
            sv0 = int(raw["sv_index_0"])
            candidate = f"L{layer:02d}_SV{sv0:02d}"
            if candidate not in display_candidates:
                continue
            if candidate in records:
                raise SystemExit(f"Duplicate left-vector record on {records_path}:{line_number}")
            records[candidate] = {
                "source": raw.get("reconstruction_method"),
                "singular_value": raw.get("stored_singular_value"),
                "saved_spectral_energy_fraction": spectral_fractions[candidate],
                "source_v_norm": raw.get("source_v_norm_before_normalization"),
                "saved_u_norm": raw.get("stored_u_norm_before_normalization"),
                "actual_transport_gain": raw.get("actual_transport_gain"),
                "gain_over_stored_singular_value": raw.get(
                    "gain_over_stored_singular_value"
                ),
                "paired_u_v_cosine": raw.get("left_right_coordinate_cosine"),
                "transport_vs_stored_u_cosine": raw.get(
                    "transport_vs_stored_u_cosine"
                ),
                "transport_vs_sigma_stored_u_relative_error": raw.get(
                    "transport_vs_sigma_stored_u_relative_error"
                ),
                "max_other_u_abs_cosine": raw.get(
                    "largest_abs_left_cosine_with_other_sv_same_layer"
                ),
                "max_other_u_sv0": raw.get(
                    "most_overlapping_left_sv_index_0_same_layer"
                ),
                "max_other_u_cosine": raw.get(
                    "most_overlapping_left_sv_cosine_same_layer"
                ),
                "previous_layer_match": raw.get("previous_layer_left_match"),
                "next_layer_match": raw.get("next_layer_left_match"),
                "token_geometry": compact_left_token_geometry(
                    raw.get("left_token_geometry")
                ),
                "right_left_token_overlap": raw.get("right_left_token_overlap"),
            }

    missing = sorted(display_candidates - records.keys())
    if missing:
        raise SystemExit(f"Left-vector artifact is missing {len(missing)} dashboard candidates")
    token_meta = metadata.get("token_geometry") or {}
    return records, {
        "basis_fingerprint_sha256": broad_fingerprint,
        "candidate_count": len(records),
        "source": str(records_path),
        "reconstruction": metadata.get("reconstruction"),
        "token_geometry": token_meta,
        "embedded_token_neighbors_per_side": LEFT_TOKEN_NEIGHBORS_PER_SIDE,
        "forward_passes": 0,
    }


def zero_based_candidate(row: dict[str, Any]) -> str:
    return f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']):02d}"


def selected_tail_value(row: dict[str, Any], suffix: str) -> Any:
    polarity = row.get("selected_tail_polarity")
    return row.get(f"{polarity}_{suffix}") if polarity in ("positive", "negative") else None


def display_row(source_row: dict[str, Any], mode: str) -> dict[str, Any]:
    row = {
        ("singular_value_over_sv0" if key == "singular_value_over_sv1" else key): value
        for key, value in source_row.items()
        if key != "sv_rank_1based"
    }
    row["candidate"] = zero_based_candidate(source_row)
    if mode == "selective":
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
            row[f"selected_tail_{suffix}"] = selected_tail_value(source_row, suffix)
    return row


def build_mode_payload(
    data_dir: Path,
    rankings_filename: str,
    rank_key: str,
    top_n: int,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rankings_path = data_dir / rankings_filename
    contexts_path = data_dir / "top_contexts.jsonl"
    metadata_path = data_dir / "metadata.json"
    for path in (rankings_path, contexts_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rankings, total_candidates = load_rankings(rankings_path, top_n, rank_key)
    candidates = [row["candidate"] for row in rankings]
    contexts, total_contexts = load_contexts(contexts_path, candidates)

    expected = int(metadata.get("top_contexts_per_polarity", 0))
    incomplete: list[str] = []
    for candidate, by_polarity in contexts.items():
        if expected and any(len(by_polarity[polarity]) != expected for polarity in ("positive", "negative")):
            incomplete.append(candidate)
    if incomplete:
        preview = ", ".join(incomplete[:8])
        raise SystemExit(f"Incomplete context sets for {len(incomplete)} candidates: {preview}")

    display_rankings: list[dict[str, Any]] = []
    display_contexts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rankings:
        source_candidate = str(row["candidate"])
        expected_source = f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']) + 1:02d}"
        if source_candidate != expected_source:
            raise SystemExit(
                f"Unexpected source candidate {source_candidate!r}; expected {expected_source!r} "
                "from layer and sv_index_0"
            )
        display_candidate = zero_based_candidate(row)
        display_rankings.append(display_row(row, mode))
        display_contexts[display_candidate] = contexts[source_candidate]

    return ({
        "meta": {
            "mode": mode,
            "model": metadata.get("model"),
            "dataset": metadata.get("dataset"),
            "dataset_config": metadata.get("dataset_config"),
            "documents": metadata.get("documents_processed"),
            "tokens": metadata.get("content_tokens_processed"),
            "layers": metadata.get("layers", []),
            "k": metadata.get("k"),
            "primary_rank": rank_key,
            "rankings_file": rankings_filename,
            "contexts_per_polarity": expected,
            "total_candidates": total_candidates,
            "embedded_candidates": len(rankings),
            "total_contexts": total_contexts,
            "display_sv_numbering": "zero_based",
            "token_sample_per_layer": metadata.get("token_sample_actual_per_layer", {}),
            "min_tail_docs_for_full_score_weight": metadata.get("min_tail_docs_for_full_score_weight"),
        },
        "rankings": display_rankings,
        "contexts": display_contexts,
    }, rankings)


def build_payload(
    data_dir: Path,
    selectivity_data_dir: Path,
    left_data_dir: Path,
    top_n: int,
    selectivity_top_n: int,
) -> dict[str, Any]:
    broad, broad_source_rows = build_mode_payload(
        data_dir, "sv_rankings.csv", BROAD_RANK, top_n, "broad"
    )
    selective, selective_source_rows = build_mode_payload(
        selectivity_data_dir,
        "selectivity_rankings.csv",
        SELECTIVITY_RANK,
        selectivity_top_n,
        "selective",
    )

    comparable_fields = (
        "model",
        "dataset",
        "dataset_config",
        "documents",
        "tokens",
        "layers",
        "k",
        "total_candidates",
    )
    mismatches = [
        field
        for field in comparable_fields
        if broad["meta"].get(field) != selective["meta"].get(field)
    ]
    if mismatches:
        raise SystemExit(
            "The scans are not directly comparable; metadata differs for " + ", ".join(mismatches)
        )

    source_rows: dict[str, dict[str, Any]] = {}
    for row in broad_source_rows + selective_source_rows:
        source_rows.setdefault(str(row["candidate"]), row)
    source_candidates = list(source_rows)
    unembedding_path = data_dir / "unembedding_neighbors.jsonl"
    if not unembedding_path.is_file():
        raise SystemExit(f"Missing required input: {unembedding_path}")
    source_unembedding, unembedding_meta = load_unembedding_neighbors(
        unembedding_path, source_candidates
    )
    display_unembedding = {
        zero_based_candidate(source_rows[source_candidate]): source_unembedding[source_candidate]
        for source_candidate in source_candidates
    }
    left_singular, left_meta = load_left_singular_payload(
        left_data_dir,
        data_dir,
        selectivity_data_dir,
        set(display_unembedding),
    )

    overlap = len(
        {row["candidate"] for row in broad["rankings"]}
        & {row["candidate"] for row in selective["rankings"]}
    )
    return {
        "default_mode": "broad",
        "meta": {
            "model": broad["meta"]["model"],
            "dataset": broad["meta"]["dataset"],
            "dataset_config": broad["meta"]["dataset_config"],
            "display_sv_numbering": "zero_based",
            "unembedding_candidates": unembedding_meta["total"],
            "unembedding_space": unembedding_meta["space"],
            "unembedding_vocab_rows": unembedding_meta["vocab"],
            "unembedding_neighbors_per_side": unembedding_meta["per_side"],
            "unembedding_domain_max": unembedding_meta["domain_max"],
            "embedded_candidate_union": len(display_unembedding),
            "embedded_candidate_overlap": overlap,
            "left_singular": left_meta,
        },
        "modes": {"broad": broad, "selective": selective},
        "unembedding": display_unembedding,
        "left_singular": left_singular,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>FineWeb Singular Vector Atlas</title>
  <style>
    :root {
      --ink: #18201d;
      --ink-2: #31413b;
      --paper: #f2efe7;
      --surface: #fffdf8;
      --surface-2: #f8f5ed;
      --line: #d9d4c7;
      --line-dark: #b9b3a5;
      --muted: #6d756f;
      --green: #176a53;
      --green-soft: #dcece4;
      --orange: #ca6337;
      --orange-soft: #f5e4da;
      --blue: #416b92;
      --blue-soft: #e3ebf3;
      --mode-accent: var(--green);
      --mode-soft: var(--green-soft);
      --shadow: 0 14px 40px rgba(42, 49, 44, .08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --serif: Iowan Old Style, Baskerville, Georgia, serif;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; color: var(--ink); background: var(--paper); font: 14px/1.5 var(--sans); }
    body[data-mode="selective"] { --mode-accent: var(--blue); --mode-soft: var(--blue-soft); }
    button, input, select { font: inherit; }
    button, select { cursor: pointer; }
    a { color: inherit; }
    .skip-link { position: fixed; left: 12px; top: -60px; z-index: 50; background: var(--ink); color: white; padding: 8px 12px; }
    .skip-link:focus { top: 12px; }

    .masthead { color: #f8f5ed; background: var(--ink); border-bottom: 4px solid var(--orange); }
    .masthead-inner { width: min(1640px, 94vw); margin: 0 auto; min-height: 156px; display: flex; align-items: flex-end; justify-content: space-between; gap: 42px; padding: 34px 0 28px; }
    .brand { display: flex; align-items: flex-start; gap: 17px; min-width: 0; }
    .brand-mark { display: grid; place-items: center; flex: 0 0 auto; width: 46px; height: 46px; margin-top: 2px; border: 1px solid rgba(255,255,255,.28); color: #f7bd95; font: 700 13px/1 var(--mono); letter-spacing: .08em; }
    .eyebrow { margin: 0 0 7px; color: #e59a6d; font-size: 10px; font-weight: 800; letter-spacing: .19em; text-transform: uppercase; }
    h1 { margin: 0; font: 500 clamp(28px, 4vw, 47px)/1.02 var(--serif); letter-spacing: -.025em; }
    h1 em { color: #9fcbb9; font-weight: 400; }
    .subtitle { max-width: 690px; margin: 11px 0 0; color: #bdc7c1; font-size: 13px; }
    .dataset-facts { display: flex; justify-content: flex-end; gap: 22px; flex-wrap: wrap; padding-bottom: 3px; }
    .fact { min-width: 90px; }
    .fact b { display: block; color: #fff; font: 600 18px/1.15 var(--mono); }
    .fact span { color: #9ba8a1; font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }

    .mode-bar { border-bottom: 1px solid var(--line); background: #e9e5dc; }
    .mode-switcher { width: min(1640px, 94vw); margin: 0 auto; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: var(--line-dark); border-left: 1px solid var(--line-dark); border-right: 1px solid var(--line-dark); }
    .mode-tab { min-width: 0; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 15px; align-items: center; border: 0; background: #ece8df; color: var(--ink-2); padding: 15px 18px; text-align: left; }
    .mode-tab:hover { background: #f4f1e9; }
    .mode-tab[aria-selected="true"] { position: relative; z-index: 1; background: var(--surface); box-shadow: inset 0 -4px 0 var(--mode-accent); }
    .mode-tab-index { display: grid; place-items: center; width: 29px; height: 29px; border: 1px solid var(--line-dark); border-radius: 50%; color: var(--muted); font: 700 10px/1 var(--mono); }
    .mode-tab[aria-selected="true"] .mode-tab-index { border-color: var(--mode-accent); background: var(--mode-soft); color: var(--mode-accent); }
    .mode-tab b { display: block; color: var(--ink); font: 650 14px/1.2 var(--serif); }
    .mode-tab small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; line-height: 1.35; }

    .toolbar { position: sticky; top: 0; z-index: 20; background: rgba(242, 239, 231, .96); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }
    .toolbar-inner { width: min(1640px, 94vw); margin: 0 auto; display: grid; grid-template-columns: minmax(220px, 1.4fr) repeat(3, minmax(145px, .55fr)) auto; gap: 11px; align-items: end; padding: 13px 0; }
    .control { display: block; min-width: 0; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
    .control input, .control select { width: 100%; height: 38px; margin-top: 4px; border: 1px solid var(--line-dark); border-radius: 2px; outline: none; background: var(--surface); color: var(--ink); padding: 0 11px; text-transform: none; letter-spacing: normal; font-size: 12px; }
    .control input:focus, .control select:focus { border-color: var(--green); box-shadow: 0 0 0 3px rgba(23,106,83,.12); }
    .search-wrap { position: relative; }
    .search-wrap input { padding-right: 46px; }
    .shortcut { position: absolute; right: 9px; bottom: 8px; border: 1px solid var(--line); border-radius: 3px; background: var(--surface-2); color: var(--muted); padding: 1px 6px; font: 10px/1.45 var(--mono); }
    .reset { height: 38px; border: 1px solid var(--line-dark); border-radius: 2px; background: transparent; color: var(--ink-2); padding: 0 14px; font-size: 11px; font-weight: 750; }
    .reset:hover { background: var(--surface); }

    .shell { width: min(1640px, 94vw); margin: 0 auto; padding: 24px 0 68px; }
    .dashboard-grid { display: grid; grid-template-columns: minmax(320px, 390px) minmax(0, 1fr); gap: 22px; align-items: start; }
    .panel { background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow); }
    .panel-head { padding: 18px 19px 14px; border-bottom: 1px solid var(--line); }
    .panel-head-row { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .panel h2, .detail-title { margin: 0; font: 500 23px/1.1 var(--serif); letter-spacing: -.01em; }
    .count { color: var(--muted); font: 11px/1.2 var(--mono); }
    .microcopy { margin: 7px 0 0; color: var(--muted); font-size: 11px; }
    .ranking-panel { position: sticky; top: 90px; height: calc(100vh - 114px); min-height: 520px; display: flex; flex-direction: column; }
    .ranking-list { overflow: auto; overscroll-behavior: contain; }
    .rank-row { width: 100%; display: grid; grid-template-columns: 45px minmax(0, 1fr); gap: 11px; border: 0; border-bottom: 1px solid #e8e4da; background: transparent; color: var(--ink); padding: 13px 15px 13px 12px; text-align: left; }
    .rank-row:hover { background: var(--surface-2); }
    .rank-row.active { background: #edf3ee; box-shadow: inset 4px 0 0 var(--green); }
    body[data-mode="selective"] .rank-row.active { background: var(--blue-soft); box-shadow: inset 4px 0 0 var(--blue); }
    .rank-number { padding-top: 2px; color: var(--muted); font: 11px/1.2 var(--mono); text-align: right; }
    .rank-row.active .rank-number { color: var(--green); font-weight: 800; }
    body[data-mode="selective"] .rank-row.active .rank-number { color: var(--blue); }
    .rank-topline { display: flex; align-items: baseline; justify-content: space-between; gap: 9px; }
    .candidate { font: 700 13px/1.3 var(--mono); letter-spacing: .01em; }
    .layer-note { color: var(--muted); font-size: 10px; white-space: nowrap; }
    .rank-measure { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 6px; color: var(--muted); font-size: 10px; }
    .rank-measure b { color: var(--ink-2); font: 600 10px/1 var(--mono); }
    .mini-track { display: block; height: 3px; margin-top: 7px; overflow: hidden; background: #e5e0d5; }
    .mini-track i { display: block; height: 100%; background: var(--green); }
    body[data-mode="selective"] .mini-track i { background: var(--blue); }
    .tail-pill { display: inline-block; margin-left: 6px; border: 1px solid var(--line); border-radius: 99px; background: var(--surface); color: var(--blue); padding: 2px 5px; font: 750 8px/1 var(--mono); text-transform: uppercase; vertical-align: 1px; }
    .rank-diagnostic { display: block; margin-top: 5px; color: var(--muted); font: 8px/1.3 var(--mono); }
    .empty { padding: 32px 20px; color: var(--muted); text-align: center; }

    .detail-panel { min-width: 0; }
    .detail-hero { padding: clamp(20px, 3vw, 34px); border-bottom: 1px solid var(--line); background: linear-gradient(125deg, #fffdf8 0%, #f7f3e9 100%); }
    .detail-nav { display: flex; justify-content: space-between; align-items: center; gap: 18px; }
    .detail-rank { color: var(--orange); font-size: 10px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    .nav-buttons { display: flex; gap: 6px; }
    .icon-button, .copy-button { height: 33px; border: 1px solid var(--line-dark); background: var(--surface); color: var(--ink-2); padding: 0 10px; font-size: 11px; }
    .icon-button { width: 35px; padding: 0; font-size: 16px; }
    .icon-button:hover, .copy-button:hover { border-color: var(--green); color: var(--green); }
    .icon-button:disabled { cursor: default; opacity: .35; }
    .title-row { display: flex; align-items: center; gap: 13px; flex-wrap: wrap; margin-top: 15px; }
    .detail-title { font-size: clamp(29px, 4vw, 43px); }
    .id-chip { border-radius: 99px; background: var(--green-soft); color: var(--green); padding: 4px 9px; font: 700 10px/1.2 var(--mono); }
    body[data-mode="selective"] .id-chip { background: var(--blue-soft); color: var(--blue); }
    .detail-summary { max-width: 820px; margin: 10px 0 0; color: var(--muted); }
    .detail-summary b { color: var(--ink-2); }
    .metric-grid { display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr)); gap: 1px; margin-top: 24px; border: 1px solid var(--line); background: var(--line); }
    .metric-card { min-width: 0; background: rgba(255,253,248,.86); padding: 12px; }
    .metric-card span { display: block; min-height: 28px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .metric-card b { display: block; margin-top: 4px; overflow: hidden; color: var(--ink); font: 650 clamp(14px, 1.5vw, 18px)/1.1 var(--mono); text-overflow: ellipsis; }

    .tail-profile { margin-top: 20px; border: 1px solid var(--line); background: var(--surface); }
    .tail-profile-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; border-bottom: 1px solid var(--line); padding: 13px 15px; }
    .tail-profile-head h3 { margin: 0; font: 550 18px/1.15 var(--serif); }
    .tail-profile-head p { max-width: 680px; margin: 4px 0 0; color: var(--muted); font-size: 10px; }
    .tail-profile-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: var(--line); }
    .tail-side { min-width: 0; background: #faf8f2; padding: 14px 15px 13px; }
    .tail-side.selected { position: relative; background: #f4f7fa; box-shadow: inset 0 3px 0 var(--blue); }
    .tail-side-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .tail-side-head h4 { margin: 0; font: 700 11px/1.2 var(--mono); }
    .tail-side-head span { color: var(--muted); font: 9px/1.2 var(--mono); }
    .tail-side.selected .tail-side-head span { color: var(--blue); font-weight: 750; }
    .tail-stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px 14px; margin-top: 13px; }
    .tail-stat { min-width: 0; }
    .tail-stat span { display: block; min-height: 22px; color: var(--muted); font-size: 8px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
    .tail-stat b { display: block; overflow: hidden; color: var(--ink-2); font: 650 12px/1.2 var(--mono); text-overflow: ellipsis; }
    .tail-artifact-note { margin: 13px 0 0; border-top: 1px dotted var(--line); padding-top: 10px; color: var(--muted); font-size: 9px; }
    .tail-artifact-note strong { color: var(--ink-2); }
    .artifact-warning { color: var(--orange); font-weight: 750; }

    .profile { display: grid; grid-template-columns: minmax(0, 1fr) 170px; gap: 28px; align-items: center; margin-top: 20px; }
    .profile-labels { display: flex; justify-content: space-between; margin-bottom: 5px; color: var(--muted); font: 9px/1.2 var(--mono); }
    .axis { position: relative; height: 9px; border-radius: 99px; background: linear-gradient(90deg, #edd7d0, #e7e2d8 50%, #d8ebe1); }
    .zero-marker, .mean-marker { position: absolute; top: -5px; width: 1px; height: 19px; background: var(--ink-2); }
    .mean-marker { width: 3px; background: var(--orange); }
    .profile-stat { display: flex; justify-content: space-between; gap: 12px; border-left: 1px solid var(--line); padding-left: 22px; }
    .profile-stat span { color: var(--muted); font-size: 10px; }
    .profile-stat b { display: block; margin-top: 2px; color: var(--green); font: 650 17px/1.2 var(--mono); }

    .metric-details { margin-top: 18px; }
    .metric-details summary { width: max-content; color: var(--green); cursor: pointer; font-size: 11px; font-weight: 750; }
    .all-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0 24px; margin-top: 13px; padding: 13px 15px; border: 1px solid var(--line); background: var(--surface); }
    .metric-groups { display: grid; gap: 15px; margin-top: 13px; }
    .metric-group h4 { margin: 0 0 6px; color: var(--muted); font-size: 9px; letter-spacing: .1em; text-transform: uppercase; }
    .metric-group .all-metrics { margin-top: 0; }
    .all-metric { display: flex; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px dotted var(--line); }
    .all-metric span { min-width: 0; color: var(--muted); font-size: 10px; }
    .all-metric b { flex: 0 0 auto; font: 600 10px/1.4 var(--mono); }

    .left-section { padding: clamp(20px, 3vw, 34px); border-bottom: 1px solid var(--line); background: #eef4f1; }
    .left-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 22px; }
    .left-heading h3 { margin: 0; font: 500 25px/1.15 var(--serif); }
    .left-heading p { max-width: 820px; margin: 7px 0 0; color: var(--muted); font-size: 11px; }
    .left-badge { flex: 0 0 auto; border-radius: 99px; background: var(--green-soft); color: var(--green); padding: 6px 10px; font: 750 9px/1.2 var(--mono); }
    .sv-map { display: grid; grid-template-columns: minmax(0, 1fr) 170px minmax(0, 1fr); gap: 10px; margin-top: 19px; }
    .sv-node, .sv-operator { display: grid; align-content: center; min-height: 88px; border: 1px solid var(--line); background: var(--surface); padding: 12px 14px; }
    .sv-node span, .sv-operator span { color: var(--muted); font-size: 8px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .sv-node b { margin-top: 5px; color: var(--ink); font: 650 16px/1.2 var(--mono); }
    .sv-node small { margin-top: 4px; color: var(--muted); font-size: 8px; }
    .sv-node.left { border-color: color-mix(in srgb, var(--green) 45%, var(--line)); box-shadow: inset 3px 0 0 var(--green); }
    .sv-operator { justify-items: center; background: #17251f; color: white; text-align: center; }
    .sv-operator b { margin: 4px 0; color: white; font: 650 13px/1.2 var(--mono); }
    .sv-operator span, .sv-operator small { color: #b3c0ba; }
    .left-metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin-top: 12px; border: 1px solid var(--line); background: var(--line); }
    .left-metric { min-width: 0; background: var(--surface); padding: 11px 12px; }
    .left-metric span { display: block; min-height: 22px; color: var(--muted); font-size: 8px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
    .left-metric b { display: block; overflow: hidden; color: var(--ink-2); font: 650 12px/1.2 var(--mono); text-overflow: ellipsis; }
    .left-relations { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .left-relation { border: 1px solid var(--line); background: rgba(255,253,248,.78); padding: 11px 12px; }
    .left-relation h4 { margin: 0; color: var(--ink-2); font: 700 9px/1.2 var(--mono); text-transform: uppercase; }
    .left-relation b { display: block; margin-top: 6px; color: var(--green); font: 650 11px/1.3 var(--mono); }
    .left-relation p { margin: 4px 0 0; color: var(--muted); font-size: 8px; }
    .left-note { margin: 11px 0 0; border-left: 3px solid var(--blue); padding-left: 9px; color: var(--muted); font-size: 9px; }
    .left-token-details { margin-top: 15px; border: 1px solid var(--line); background: rgba(255,253,248,.76); }
    .left-token-details > summary { cursor: pointer; list-style: none; padding: 13px 14px; }
    .left-token-details > summary::-webkit-details-marker { display: none; }
    .left-token-summary { display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    .left-token-summary h4 { margin: 0; font: 500 20px/1.15 var(--serif); }
    .left-token-summary p { margin: 4px 0 0; color: var(--muted); font-size: 9px; }
    .left-token-summary b { color: var(--green); font: 650 10px/1.2 var(--mono); }
    .left-token-content { padding: 0 14px 14px; }
    .left-token-toolbar { display: flex; justify-content: flex-end; margin-bottom: 10px; }

    .token-space-section { padding: clamp(20px, 3vw, 34px); border-bottom: 1px solid var(--line); background: #fbfaf5; }
    .token-space-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 22px; }
    .token-space-heading h3 { margin: 0; font: 500 25px/1.15 var(--serif); }
    .token-space-heading p { max-width: 760px; margin: 7px 0 0; color: var(--muted); font-size: 11px; }
    .token-limit { flex: 0 0 118px; }
    .token-limit select { background: #fff; }
    .unembed-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; margin-top: 20px; border: 1px solid var(--line); background: var(--line); }
    .unembed-stat { min-width: 0; background: var(--surface); padding: 11px 12px; }
    .unembed-stat span { display: block; min-height: 25px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
    .unembed-stat b { display: block; color: var(--ink); font: 650 16px/1.2 var(--mono); }
    .unembed-spectrum { margin-top: 18px; border: 1px solid var(--line); background: var(--surface); padding: 14px 15px 11px; }
    .spectrum-topline { display: flex; justify-content: space-between; gap: 6px 20px; flex-wrap: wrap; color: var(--muted); font-size: 9px; }
    .spectrum-topline b { color: var(--ink-2); font-weight: 700; }
    .spectrum-track { position: relative; height: 54px; margin: 11px 0 7px; overflow: hidden; border: 1px solid #e1ddd3; background: linear-gradient(90deg, var(--orange-soft), #f1eee6 50%, var(--green-soft)); }
    .spectrum-band { position: absolute; top: 0; bottom: 0; background: rgba(91, 102, 96, .13); border-left: 1px solid rgba(49,65,59,.22); border-right: 1px solid rgba(49,65,59,.22); }
    .spectrum-zero, .spectrum-mean { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(24,32,29,.62); }
    .spectrum-mean { width: 2px; background: var(--blue); }
    .spectrum-dot { position: absolute; z-index: 2; width: 7px; height: 7px; transform: translate(-50%, -50%); border: 1px solid rgba(255,255,255,.9); border-radius: 50%; box-shadow: 0 1px 2px rgba(24,32,29,.2); }
    .spectrum-dot.aligned { background: var(--green); }
    .spectrum-dot.opposed { background: var(--orange); }
    .spectrum-labels { display: grid; grid-template-columns: 1fr auto 1fr; color: var(--muted); font: 9px/1.2 var(--mono); }
    .spectrum-labels span:nth-child(2) { text-align: center; }
    .spectrum-labels span:last-child { text-align: right; }
    .spectrum-legend { display: flex; gap: 6px 16px; flex-wrap: wrap; margin-top: 9px; color: var(--muted); font-size: 9px; }
    .legend-dot { display: inline-block; width: 7px; height: 7px; margin-right: 4px; border-radius: 50%; }
    .legend-dot.aligned { background: var(--green); }
    .legend-dot.opposed { background: var(--orange); }
    .legend-band { display: inline-block; width: 12px; height: 7px; margin-right: 4px; background: rgba(91,102,96,.18); }
    .neighbor-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 17px; margin-top: 17px; align-items: start; }
    .neighbor-column { min-width: 0; }
    .neighbor-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; min-height: 39px; border-top: 3px solid; padding: 8px 2px 0; }
    .neighbor-column.aligned .neighbor-head { border-color: var(--green); }
    .neighbor-column.opposed .neighbor-head { border-color: var(--orange); }
    .neighbor-head h4 { margin: 0; font: 650 13px/1.2 var(--mono); }
    .neighbor-head span { color: var(--muted); font-size: 9px; }
    .neighbor-list { display: grid; gap: 5px; }
    .neighbor-row { min-width: 0; border: 1px solid var(--line); background: var(--surface); padding: 8px 10px 7px; }
    .neighbor-main { display: grid; grid-template-columns: 25px minmax(0, 1fr) auto; gap: 9px; align-items: center; }
    .neighbor-rank { color: var(--muted); font: 650 9px/1 var(--mono); text-align: right; }
    .neighbor-name { min-width: 0; }
    .neighbor-token { display: block; overflow: hidden; color: var(--ink); font: 650 12px/1.25 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .neighbor-raw { display: block; overflow: hidden; margin-top: 2px; color: var(--muted); font: 8px/1.2 var(--mono); text-overflow: ellipsis; white-space: nowrap; }
    .neighbor-values { display: flex; gap: 10px; color: var(--muted); font: 8px/1.2 var(--mono); text-align: right; }
    .neighbor-values b { display: block; margin-top: 2px; color: var(--ink-2); font-size: 10px; }
    .neighbor-column.aligned .neighbor-values .cosine { color: var(--green); }
    .neighbor-column.opposed .neighbor-values .cosine { color: var(--orange); }
    .neighbor-bar { display: block; height: 2px; margin: 7px 0 0 34px; background: #ebe7dd; }
    .neighbor-bar i { display: block; height: 100%; }
    .neighbor-column.aligned .neighbor-bar i { background: var(--green); }
    .neighbor-column.opposed .neighbor-bar i { background: var(--orange); }

    .context-section { padding: clamp(20px, 3vw, 34px); }
    .section-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; }
    .section-heading h3 { margin: 0; font: 500 25px/1.15 var(--serif); }
    .section-heading p { max-width: 720px; margin: 7px 0 0; color: var(--muted); font-size: 11px; }
    .sign-note { display: inline-block; margin-top: 9px; border-left: 3px solid var(--orange); padding-left: 10px; color: var(--muted); font-size: 10px; }
    .context-controls { display: grid; grid-template-columns: minmax(190px, 1fr) 105px auto; gap: 9px; align-items: end; margin: 22px 0 16px; }
    .context-controls .control input, .context-controls .control select { background: #fff; }
    .check { display: flex; align-items: center; gap: 7px; height: 38px; color: var(--ink-2); font-size: 11px; white-space: nowrap; }
    .check input { width: 15px; height: 15px; accent-color: var(--green); }
    .context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 17px; align-items: start; }
    .context-column { min-width: 0; }
    .column-head { display: flex; align-items: center; justify-content: space-between; min-height: 42px; margin-bottom: 7px; border-top: 3px solid; padding: 9px 2px 0; }
    .context-column.positive .column-head { border-color: var(--green); }
    .context-column.negative .column-head { border-color: var(--orange); }
    body[data-mode="selective"] .context-column.score-driver { outline: 1px solid rgba(65,107,146,.35); outline-offset: 4px; }
    body[data-mode="selective"] .context-column.score-driver .column-head { background: var(--blue-soft); padding-left: 8px; padding-right: 8px; }
    .column-head h4 { margin: 0; font: 600 13px/1.2 var(--mono); }
    .column-head span { color: var(--muted); font-size: 10px; }
    .column-head .tail-pill { color: var(--blue); font-size: 8px; }
    .context-list { display: grid; gap: 8px; }
    .context-card { overflow: hidden; border: 1px solid var(--line); background: var(--surface); }
    .context-meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; border-bottom: 1px solid #ebe7de; background: var(--surface-2); padding: 8px 11px; }
    .context-rank { color: var(--muted); font: 650 9px/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; }
    .context-values { display: flex; gap: 10px; color: var(--muted); font: 9px/1.2 var(--mono); }
    .context-values span { white-space: nowrap; }
    .context-values b { color: var(--ink-2); }
    .positive .activation { color: var(--green); }
    .negative .activation { color: var(--orange); }
    .context-copy { margin: 0; padding: 13px 13px 10px; color: #28312d; font: 12px/1.63 var(--serif); white-space: pre-wrap; overflow-wrap: anywhere; }
    mark { border-radius: 2px; background: #f5d6a8; color: #161b19; padding: 1px 2px; box-shadow: 0 0 0 1px rgba(170,104,38,.13); }
    .source-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 12px 11px; color: var(--muted); font-size: 9px; }
    .source-info { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .token-chip { display: inline-block; max-width: 160px; overflow: hidden; border: 1px solid var(--line); border-radius: 3px; background: #fff; padding: 2px 5px; color: var(--ink-2); font: 9px/1.25 var(--mono); text-overflow: ellipsis; vertical-align: middle; }
    .source-link { flex: 0 0 auto; color: var(--blue); font-weight: 700; text-decoration: none; }
    .source-link:hover { text-decoration: underline; }
    .footer { width: min(1640px, 94vw); margin: -38px auto 32px; color: var(--muted); font-size: 10px; }
    .footer code { font-family: var(--mono); }

    :focus-visible { outline: 3px solid rgba(23,106,83,.28); outline-offset: 2px; }
    @media (max-width: 1120px) {
      .metric-grid { grid-template-columns: repeat(3, 1fr); }
      .left-metrics { grid-template-columns: repeat(3, 1fr); }
      .tail-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .all-metrics { grid-template-columns: repeat(2, 1fr); }
      .unembed-stats { grid-template-columns: repeat(2, 1fr); }
      .toolbar-inner { grid-template-columns: minmax(210px, 1.3fr) repeat(3, minmax(125px, .55fr)) auto; }
    }
    @media (max-width: 880px) {
      .masthead-inner { align-items: flex-start; flex-direction: column; min-height: 0; }
      .mode-switcher { grid-template-columns: 1fr; }
      .dataset-facts { justify-content: flex-start; }
      .toolbar { position: static; }
      .toolbar-inner { grid-template-columns: 1fr 1fr; }
      .search-control { grid-column: 1 / -1; }
      .reset { align-self: end; }
      .dashboard-grid { grid-template-columns: 1fr; }
      .ranking-panel { position: static; height: auto; min-height: 0; }
      .ranking-list { max-height: 430px; }
      .profile { grid-template-columns: 1fr; gap: 16px; }
      .profile-stat { border-left: 0; border-top: 1px solid var(--line); padding: 13px 0 0; }
    }
    @media (max-width: 660px) {
      .masthead-inner, .shell, .toolbar-inner, .footer { width: min(92vw, 1640px); }
      .brand-mark { display: none; }
      .dataset-facts { gap: 14px 22px; }
      .toolbar-inner { grid-template-columns: 1fr; }
      .search-control { grid-column: auto; }
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
      .left-metrics { grid-template-columns: repeat(2, 1fr); }
      .sv-map, .left-relations { grid-template-columns: 1fr; }
      .left-token-summary, .left-heading { align-items: flex-start; flex-direction: column; }
      .all-metrics { grid-template-columns: 1fr; }
      .token-space-heading { align-items: flex-start; flex-direction: column; }
      .token-limit { width: 100%; flex-basis: auto; }
      .neighbor-grid { grid-template-columns: 1fr; }
      .context-grid { grid-template-columns: 1fr; }
      .tail-profile-grid { grid-template-columns: 1fr; }
      .context-controls { grid-template-columns: 1fr 1fr; }
      .context-controls .context-search { grid-column: 1 / -1; }
      .detail-hero, .left-section, .token-space-section, .context-section { padding: 20px 16px; }
      .section-heading { align-items: flex-start; flex-direction: column; }
    }
    @media print {
      .toolbar, .ranking-panel, .nav-buttons, .context-controls, .token-limit, .footer { display: none !important; }
      .masthead { color: var(--ink); background: white; border-color: var(--ink); }
      .masthead * { color: var(--ink) !important; }
      .masthead-inner { min-height: 0; padding: 18px 0; }
      .dashboard-grid { display: block; }
      .panel { box-shadow: none; }
      .context-card, .neighbor-row, .unembed-spectrum { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#detail">Skip to selected direction</a>
  <header class="masthead">
    <div class="masthead-inner">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">SV</div>
        <div>
          <p class="eyebrow">Interpretability workbench · FineWeb</p>
          <h1>Singular Vector <em>Atlas</em></h1>
          <p class="subtitle">Explore one shared bank of singular directions through two ranking lenses, compare each right vector with its transported left vector, then inspect both token geometries and the FineWeb contexts at each activation extreme.</p>
        </div>
      </div>
      <div class="dataset-facts" id="datasetFacts" aria-label="Dataset summary"></div>
    </div>
  </header>

  <nav class="mode-bar" aria-label="Analysis mode">
    <div class="mode-switcher" role="tablist" aria-label="Direction ranking lens">
      <button class="mode-tab" id="broadMode" type="button" role="tab" data-mode="broad" aria-selected="true">
        <span class="mode-tab-index">01</span><span><b>Broad activity</b><small>Directions used strongly on average · ranked by mean absolute cosine</small></span>
      </button>
      <button class="mode-tab" id="selectiveMode" type="button" role="tab" data-mode="selective" aria-selected="false">
        <span class="mode-tab-index">02</span><span><b>Selective tails</b><small>Directions with unusually sharp robust tails · not automatically rare or semantic</small></span>
      </button>
    </div>
  </nav>

  <section class="toolbar" aria-label="Ranking controls">
    <div class="toolbar-inner">
      <label class="control search-control">Find a direction
        <span class="search-wrap"><input id="candidateSearch" type="search" placeholder="Candidate ID, layer, or SV index…" autocomplete="off"><kbd class="shortcut">/</kbd></span>
      </label>
      <label class="control">Layer<select id="layerFilter"></select></label>
      <label class="control">Rank by<select id="sortMetric"></select></label>
      <label class="control">Show<select id="rowLimit"></select></label>
      <button class="reset" id="resetFilters" type="button">Reset</button>
    </div>
  </section>

  <main class="shell">
    <div class="dashboard-grid">
      <aside class="panel ranking-panel" aria-label="Ranked singular vectors">
        <div class="panel-head">
          <div class="panel-head-row"><h2 id="rankingTitle">Ranked directions</h2><span class="count" id="rankingCount"></span></div>
          <p class="microcopy" id="rankingCaption"></p>
        </div>
        <div class="ranking-list" id="rankingList"></div>
      </aside>
      <article class="panel detail-panel" id="detail" tabindex="-1"></article>
    </div>
  </main>
  <footer class="footer" id="footerNote"></footer>

  <script>
    const DATA = __DASHBOARD_PAYLOAD__;

    const BROAD_METRICS = {
      mean_abs_cosine: { label: "Mean |cosine|", short: "mean |cos|", kind: "decimal", help: "Mean absolute activation divided by residual-stream norm; the default cross-layer score." },
      top1_abs_rate: { label: "Top-1 token share", short: "top-1 share", kind: "percent", help: "Fraction of tokens where this direction has the largest absolute activation in the layer's top-k bank." },
      top5_abs_rate: { label: "Top-5 token share", short: "top-5 share", kind: "percent", help: "Fraction of tokens where this direction is among the five largest absolute activations." },
      sigma_weighted_mean_abs: { label: "σ × mean |activation|", short: "σ × mean |act|", kind: "number", help: "Singular value multiplied by mean absolute activation." },
      std_activation: { label: "Activation variability", short: "activation std", kind: "number", help: "Standard deviation across token activations." },
      singular_value: { label: "Singular value", short: "singular value", kind: "number", help: "Singular value for this direction within its layer." },
      dynamicity_std_over_abs_mean: { label: "Dynamicity", short: "dynamicity", kind: "decimal", help: "Activation standard deviation divided by mean absolute activation." },
      max_abs_unembed_token_cosine: { label: "Max |token cosine|", short: "max |token cos|", kind: "decimal", help: "Strongest absolute cosine between this SV and a normalized output-token unembedding row." }
    };

    const SELECTIVE_METRICS = {
      tail_selectivity_score: { label: "Tail selectivity score", short: "tail score", kind: "decimal", help: "The stronger polarity's q99.9 robust-z × √(top-0.1% energy share) × minimum-support weight." },
      selected_tail_q999_robust_z: { label: "Selected-tail q99.9 robust z", short: "q99.9 robust z", kind: "decimal", help: "The score-driving tail's 99.9th percentile after median/MAD normalization." },
      selected_tail_top0_1pct_energy_share: { label: "Selected-tail top-0.1% energy", short: "top-0.1% energy", kind: "percent", help: "Share of one-sided squared robust-z energy carried by the strongest 0.1% of sampled tokens." },
      selected_tail_doc_rate_z5: { label: "Windows above z=5 · low first", short: "windows >5z", kind: "percent", direction: "asc", help: "Fraction of sampled windows whose score-driving polarity peak exceeds robust z=5. Lower values are rarer across windows." },
      effective_support_fraction: { label: "Effective support · low first", short: "effective support", kind: "percent", direction: "asc", help: "M2²/(N×M4). Smaller values mean the centered variance is carried by fewer tokens or heavier tails." },
      stable_excess_kurtosis: { label: "Excess kurtosis", short: "excess kurtosis", kind: "number", help: "Exact fourth-moment tail weight minus three; high values can reflect selectivity or artifacts." },
      selected_tail_top_context_largest_center_token_share: { label: "Largest center-token share · low first", short: "largest token share", kind: "percent", direction: "asc", help: "Largest lexical share among retained score-driving contexts. Lower values indicate more diverse activating tokens." },
      selected_tail_top_context_effective_center_tokens: { label: "Effective center tokens", short: "effective tokens", kind: "number", help: "Entropy-derived effective count of center tokens in retained score-driving contexts." },
      mean_abs_cosine: { label: "Mean |cosine|", short: "mean |cos|", kind: "decimal", help: "The original broad-activity score, included to compare both lenses." },
      max_abs_unembed_token_cosine: { label: "Max |token cosine|", short: "max |token cos|", kind: "decimal", help: "Strongest absolute output-token cosine; high values can indicate lexical anchoring." }
    };

    const MODE_CONFIG = {
      broad: {
        label: "Broad activity",
        title: "Broad-activity ranking",
        primaryRank: "rank_global_mean_abs_cosine",
        defaultMetric: "mean_abs_cosine",
        metrics: BROAD_METRICS,
        caption: "Primary rank is mean absolute cosine over all scanned tokens."
      },
      selective: {
        label: "Selective tails",
        title: "Tail-selectivity ranking",
        primaryRank: "rank_global_tail_selectivity",
        defaultMetric: "tail_selectivity_score",
        metrics: SELECTIVE_METRICS,
        caption: "Primary rank is a robust heavy-tail score; it is not a direct rarity or semantic-cleanliness rank."
      }
    };

    const LABELS = {
      candidate: "Candidate", layer: "Layer", sv_index_0: "SV index (zero-based)",
      singular_value: "Singular value", singular_value_over_sv0: "Singular value / SV0", n_tokens: "Tokens", n_documents: "Documents",
      mean_activation: "Mean activation", mean_abs_activation: "Mean |activation|", rms_activation: "RMS activation", std_activation: "Activation std",
      positive_rate: "Positive rate", max_activation: "Maximum activation", min_activation: "Minimum activation", mean_abs_cosine: "Mean |cosine|",
      rms_cosine: "RMS cosine", top1_abs_rate: "Top-1 absolute rate", top5_abs_rate: "Top-5 absolute rate",
      doc_top5_presence_rate: "Document top-5 presence", mean_document_peak_abs: "Mean document peak |act|",
      mean_layer_residual_norm: "Mean layer residual norm", sigma_weighted_mean_abs: "σ-weighted mean |act|",
      sigma_weighted_std: "σ-weighted activation std", dynamicity_std_over_abs_mean: "Dynamicity (std / mean |act|)",
      rank_global_mean_abs_cosine: "Global rank · mean |cosine|", rank_global_top1_rate: "Global rank · top-1 rate",
      rank_global_top5_abs_rate: "Global rank · top-5 rate", rank_global_sigma_weighted_mean_abs: "Global rank · σ-weighted mean |act|",
      rank_global_std_activation: "Global rank · activation std", rank_layer_mean_abs_activation: "Layer rank · mean |activation|",
      rank_layer_mean_abs_cosine: "Layer rank · mean |cosine|", rank_layer_top1_rate: "Layer rank · top-1 rate",
      rank_layer_top5_abs_rate: "Layer rank · top-5 rate", rank_layer_std_activation: "Layer rank · activation std",
      nearest_unembed_token_id: "Nearest token ID", nearest_unembed_token: "Nearest raw token", nearest_unembed_decoded: "Nearest decoded token",
      nearest_unembed_cosine: "Nearest token cosine", nearest_unembed_margin: "Nearest top-1 margin",
      farthest_unembed_token_id: "Farthest token ID", farthest_unembed_token: "Farthest raw token", farthest_unembed_decoded: "Farthest decoded token",
      farthest_unembed_cosine: "Farthest token cosine", farthest_unembed_margin: "Farthest top-1 margin",
      max_abs_unembed_token_cosine: "Maximum |token cosine|", nearest_unembed_token_z: "Nearest token z-score",
      farthest_unembed_token_z: "Farthest token z-score", unembedding_cosine_mean: "Vocabulary cosine mean",
      unembedding_cosine_std: "Vocabulary cosine std", rank_global_max_abs_unembed_token_cosine: "Global rank · max |token cosine|",
      token_sample_n: "Robust-stat token sample", median_activation: "Median activation", mad_activation: "Median absolute deviation",
      robust_activation_scale: "Robust activation scale", stable_skewness: "Stable skewness", stable_kurtosis: "Stable kurtosis",
      stable_excess_kurtosis: "Stable excess kurtosis", effective_support_fraction: "Effective support fraction",
      tail_selectivity_score: "Tail selectivity score", selected_tail_polarity: "Score-driving tail",
      rank_global_tail_selectivity: "Global rank · tail selectivity", rank_layer_tail_selectivity: "Layer rank · tail selectivity",
      rank_global_excess_kurtosis: "Global rank · excess kurtosis", rank_layer_excess_kurtosis: "Layer rank · excess kurtosis",
      selected_tail_q999_robust_z: "Selected-tail q99.9 robust z", selected_tail_top0_1pct_energy_share: "Selected-tail top-0.1% energy share",
      selected_tail_doc_count_z5: "Selected-tail windows above z=5", selected_tail_doc_rate_z5: "Selected-tail window rate above z=5",
      selected_tail_top_context_effective_center_tokens: "Selected-tail effective center tokens",
      selected_tail_top_context_largest_center_token_share: "Selected-tail largest center-token share"
    };

    const state = {
      mode: DATA.default_mode || "broad",
      views: Object.fromEntries(Object.entries(DATA.modes).map(([mode, payload]) => [mode, {
        query: "",
        layer: "all",
        metric: MODE_CONFIG[mode].defaultMetric,
        limit: Math.min(50, payload.rankings.length),
        selected: payload.rankings[0]?.candidate || null
      }])),
      contextQuery: "",
      contextLimit: 6,
      dedupe: false,
      tokenLimit: Math.min(8, DATA.meta.unembedding_neighbors_per_side || 8),
      leftTokenLimit: Math.min(8, DATA.meta.left_singular?.embedded_token_neighbors_per_side || 8)
    };

    const candidateIndexes = Object.fromEntries(Object.entries(DATA.modes).map(([mode, payload]) => [mode, new Map(payload.rankings.map(row => [row.candidate, row]))]));
    const $ = selector => document.querySelector(selector);
    const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
    const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    const integer = new Intl.NumberFormat();
    const modeData = () => DATA.modes[state.mode];
    const modeConfig = () => MODE_CONFIG[state.mode];
    const viewState = () => state.views[state.mode];
    const byCandidate = () => candidateIndexes[state.mode];

    function decimal(value, digits = 3) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      const magnitude = Math.abs(number);
      if (magnitude !== 0 && (magnitude < 0.001 || magnitude >= 10000)) return number.toExponential(2);
      return number.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function signed(value, digits = 3) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      return `${number >= 0 ? "+" : ""}${decimal(number, digits)}`;
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

    function setup() {
      const initial = parseHash();
      if (initial.mode && DATA.modes[initial.mode]) state.mode = initial.mode;
      if (initial.candidate && byCandidate().has(initial.candidate)) {
        viewState().selected = initial.candidate;
        expandLimitToCandidate(initial.candidate);
      }
      configureModeControls();

      document.querySelectorAll(".mode-tab").forEach(button => button.addEventListener("click", () => switchMode(button.dataset.mode)));
      $("#candidateSearch").addEventListener("input", event => { viewState().query = event.target.value.trim().toLowerCase(); render(); });
      $("#layerFilter").addEventListener("change", event => { viewState().layer = event.target.value; render(); });
      $("#sortMetric").addEventListener("change", event => { viewState().metric = event.target.value; render(); });
      $("#rowLimit").addEventListener("change", event => { viewState().limit = Number(event.target.value); render(); });
      $("#resetFilters").addEventListener("click", resetFilters);
      addEventListener("hashchange", () => {
        const target = parseHash();
        if (!target.mode || !DATA.modes[target.mode] || !target.candidate) return;
        state.mode = target.mode;
        if (byCandidate().has(target.candidate)) {
          configureModeControls();
          revealCandidate(target.candidate, false);
        }
      });
      addEventListener("keydown", event => {
        const tag = document.activeElement?.tagName;
        if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(tag)) {
          event.preventDefault();
          $("#candidateSearch").focus();
        }
        if (event.key === "Escape" && document.activeElement === $("#candidateSearch")) {
          $("#candidateSearch").value = "";
          viewState().query = "";
          render();
          $("#candidateSearch").blur();
        }
      });

      render();
      syncHash();
    }

    function parseHash() {
      let raw = location.hash.slice(1);
      try { raw = decodeURIComponent(raw); } catch (_) {}
      if (!raw) return { mode: DATA.default_mode || "broad", candidate: null };
      const slash = raw.indexOf("/");
      if (slash > 0 && DATA.modes[raw.slice(0, slash)]) return { mode: raw.slice(0, slash), candidate: raw.slice(slash + 1) };
      return { mode: "broad", candidate: raw };
    }

    function configureModeControls() {
      const payload = modeData();
      const meta = payload.meta;
      const view = viewState();
      document.body.dataset.mode = state.mode;
      document.querySelectorAll(".mode-tab").forEach(button => button.setAttribute("aria-selected", String(button.dataset.mode === state.mode)));
      $("#datasetFacts").innerHTML = [
        [compact.format(meta.tokens), "tokens"],
        [integer.format(meta.documents), "sampled windows"],
        [integer.format(meta.layers.length), "layers"],
        [integer.format(meta.embedded_candidates), "in this lens"]
      ].map(([value, label]) => `<div class="fact"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join("");
      $("#rankingTitle").textContent = modeConfig().title;
      const layers = [...new Set(payload.rankings.map(row => Number(row.layer)))].sort((a, b) => a - b);
      $("#layerFilter").innerHTML = `<option value="all">All embedded layers</option>` + layers.map(layer => `<option value="${layer}">Layer ${String(layer).padStart(2, "0")}</option>`).join("");
      $("#sortMetric").innerHTML = Object.entries(modeConfig().metrics).map(([key, item]) => `<option value="${key}">${esc(item.label)}</option>`).join("");
      if (!modeConfig().metrics[view.metric]) view.metric = modeConfig().defaultMetric;
      const limits = [25, 50, 100, 250].filter(value => value < payload.rankings.length);
      limits.push(payload.rankings.length);
      $("#rowLimit").innerHTML = [...new Set(limits)].map(value => `<option value="${value}">${value === payload.rankings.length ? `All ${integer.format(value)}` : `Top ${value}`}</option>`).join("");
      if (view.limit > payload.rankings.length) view.limit = payload.rankings.length;
      $("#candidateSearch").value = view.query;
      $("#layerFilter").value = view.layer;
      $("#sortMetric").value = view.metric;
      $("#rowLimit").value = String(view.limit);
      $("#footerNote").innerHTML = `SV identifiers are zero-based (<code>SV00</code> is the first singular vector); all rank numbers remain one-based. Both lenses scan the same ${integer.format(meta.total_candidates)} directions over the same ${integer.format(meta.documents)} FineWeb windows. This file embeds the top <b>${integer.format(DATA.modes.broad.meta.embedded_candidates)}</b> broad and <b>${integer.format(DATA.modes.selective.meta.embedded_candidates)}</b> selective candidates (${integer.format(DATA.meta.embedded_candidate_union)} unique; ${integer.format(DATA.meta.embedded_candidate_overlap)} shared), with mode-specific contexts, right-vector token neighbors, and fingerprint-validated J·V left-vector geometry plus ${integer.format(DATA.meta.left_singular.embedded_token_neighbors_per_side)} U-token neighbors per side. The left enrichment uses zero transformer forward passes. Model: <code>${esc(meta.model)}</code>.`;
    }

    function switchMode(mode) {
      if (!DATA.modes[mode] || mode === state.mode) return;
      state.mode = mode;
      state.contextQuery = "";
      state.contextLimit = 6;
      configureModeControls();
      render();
      syncHash();
    }

    function resetFilters() {
      const view = viewState();
      view.query = "";
      view.layer = "all";
      view.metric = modeConfig().defaultMetric;
      view.limit = Math.min(50, modeData().rankings.length);
      $("#candidateSearch").value = "";
      $("#layerFilter").value = "all";
      $("#sortMetric").value = view.metric;
      $("#rowLimit").value = String(view.limit);
      render();
    }

    function filteredRows(applyLimit = true) {
      const view = viewState();
      const query = view.query;
      const rows = modeData().rankings.filter(row => {
        if (view.layer !== "all" && Number(row.layer) !== Number(view.layer)) return false;
        if (!query) return true;
        const haystack = `${row.candidate} layer ${row.layer} l${String(row.layer).padStart(2,"0")} sv ${row.sv_index_0} sv${String(row.sv_index_0).padStart(2,"0")}`.toLowerCase();
        return haystack.includes(query);
      }).sort(compareRows);
      return applyLimit ? rows.slice(0, view.limit) : rows;
    }

    function compareRows(a, b) {
      const view = viewState();
      const metric = modeConfig().metrics[view.metric];
      const direction = metric.direction === "asc" ? 1 : -1;
      const av = Number(a[view.metric]);
      const bv = Number(b[view.metric]);
      const metricDifference = Number.isFinite(av) && Number.isFinite(bv) ? (av - bv) * direction : 0;
      return metricDifference || Number(a[modeConfig().primaryRank]) - Number(b[modeConfig().primaryRank]);
    }

    function render() {
      const view = viewState();
      const previousSelection = view.selected;
      const allMatches = filteredRows(false);
      const rows = allMatches.slice(0, view.limit);
      if (rows.length && !rows.some(row => row.candidate === view.selected)) view.selected = rows[0].candidate;
      if (!rows.length) view.selected = null;
      if (view.selected !== previousSelection) syncHash();
      renderRankings(rows, allMatches.length);
      renderDetail(rows);
    }

    function renderRankings(rows, matchCount) {
      const view = viewState();
      const metric = modeConfig().metrics[view.metric];
      const values = rows.map(row => Number(row[view.metric])).filter(Number.isFinite);
      const minValue = values.length ? Math.min(...values) : 0;
      const maxValue = values.length ? Math.max(...values) : 0;
      $("#rankingCount").textContent = `${integer.format(rows.length)} / ${integer.format(matchCount)}`;
      $("#rankingCaption").textContent = `${metric.label} · ${metric.direction === "asc" ? "ascending" : "descending"}. ${modeConfig().caption} SV indices are zero-based.`;
      if (!rows.length) {
        $("#rankingList").innerHTML = `<div class="empty">No directions match these filters.</div>`;
        return;
      }
      $("#rankingList").innerHTML = rows.map(row => {
        const value = Number(row[view.metric]);
        const span = Math.max(maxValue - minValue, 1e-12);
        const normalized = metric.direction === "asc" ? (maxValue - value) / span : (value - minValue) / span;
        const width = values.length === 1 ? 100 : Math.max(2, Math.min(100, normalized * 100));
        const tail = state.mode === "selective" ? `<span class="tail-pill">${row.selected_tail_polarity === "positive" ? "+ high" : "− low"} tail</span>` : "";
        const diagnostic = state.mode === "selective" ? `<span class="rank-diagnostic">${percent(row.selected_tail_top_context_largest_center_token_share, 1)} largest center token · broad #${integer.format(row.rank_global_mean_abs_cosine)}</span>` : "";
        return `<button class="rank-row ${row.candidate === view.selected ? "active" : ""}" type="button" ${row.candidate === view.selected ? 'aria-current="true"' : ""} data-candidate="${esc(row.candidate)}">
          <span class="rank-number">#${integer.format(row[modeConfig().primaryRank])}</span>
          <span>
            <span class="rank-topline"><span class="candidate">${esc(row.candidate)}${tail}</span><span class="layer-note">L${String(row.layer).padStart(2,"0")} · SV${String(row.sv_index_0).padStart(2,"0")}</span></span>
            <span class="rank-measure"><span>${esc(metric.short)}</span><b>${formatMetric(value, metric.kind)}</b></span>
            <span class="mini-track" aria-hidden="true"><i style="width:${width.toFixed(2)}%"></i></span>
            ${diagnostic}
          </span>
        </button>`;
      }).join("");
      $("#rankingList").querySelectorAll(".rank-row").forEach(button => button.addEventListener("click", () => {
        selectCandidate(button.dataset.candidate, true);
        if (innerWidth <= 880) $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
      }));
    }

    function selectCandidate(candidate, updateHash = true) {
      if (!byCandidate().has(candidate)) return;
      viewState().selected = candidate;
      if (updateHash) syncHash();
      render();
    }

    function expandLimitToCandidate(candidate) {
      const view = viewState();
      const sorted = modeData().rankings.slice().sort(compareRows);
      const needed = sorted.findIndex(row => row.candidate === candidate) + 1;
      const choices = [...new Set([25, 50, 100, 250, modeData().rankings.length])].filter(value => value <= modeData().rankings.length).sort((a, b) => a - b);
      view.limit = choices.find(value => value >= needed) || modeData().rankings.length;
    }

    function revealCandidate(candidate, updateHash = true) {
      const view = viewState();
      view.query = "";
      view.layer = "all";
      $("#candidateSearch").value = "";
      $("#layerFilter").value = "all";
      expandLimitToCandidate(candidate);
      $("#rowLimit").value = String(view.limit);
      view.selected = candidate;
      if (updateHash) syncHash();
      render();
    }

    function syncHash() {
      const selected = viewState().selected;
      const nextHash = selected ? `#${state.mode}/${encodeURIComponent(selected)}` : "";
      if (location.hash === nextHash) return;
      try { history.replaceState(null, "", nextHash || `${location.pathname}${location.search}`); }
      catch (_) { location.hash = nextHash; }
    }

    function metricCard(label, value, title) {
      return `<div class="metric-card" title="${esc(title || "")}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
    }

    function leftMetric(label, value, title="") {
      return `<div class="left-metric" title="${esc(title)}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
    }

    function svCandidate(layer, sv0) {
      return Number.isFinite(Number(layer)) && Number.isFinite(Number(sv0))
        ? `L${String(layer).padStart(2,"0")}_SV${String(sv0).padStart(2,"0")}`
        : "—";
    }

    function leftSingularSection(row) {
      const data=DATA.left_singular?.[row.candidate];
      if(!data) return "";
      const previous=data.previous_layer_match;
      const next=data.next_layer_match;
      const previousText=previous ? `${svCandidate(previous.layer,previous.sv_index_0)} · |cos| ${decimal(previous.abs_cosine,4)}` : "first saved layer";
      const nextText=next ? `${svCandidate(next.layer,next.sv_index_0)} · |cos| ${decimal(next.abs_cosine,4)}` : "last saved layer";
      const overlap=data.right_left_token_overlap || {};
      const alignedOverlap=overlap.positive_to_positive?.count;
      const opposedOverlap=overlap.negative_to_negative?.count;
      const flippedOverlap=(Number(overlap.positive_to_negative?.count)||0)+(Number(overlap.negative_to_positive?.count)||0);
      const tokens=data.token_geometry;
      const tokenPanel=tokens ? `<details class="left-token-details" open><summary><div class="left-token-summary"><div><p class="eyebrow">Output-side vocabulary geometry · U</p><h4>Left-vector token directions</h4><p>Cosine similarity between normalized J·V and every normalized lm_head row. These are geometric neighbors, not probabilities or observed activations.</p></div><b>max |cos| ${decimal(tokens.max_abs,4)} · ${integer.format(tokens.vocab)} tokens</b></div></summary><div class="left-token-content"><div class="left-token-toolbar"><label class="control token-limit">Tokens per side<select id="leftTokenLimit"></select></label></div><div class="neighbor-grid"><section class="neighbor-column aligned"><div class="neighbor-head"><h4>+ Aligned with U</h4><span id="leftAlignedTokenCount"></span></div><div class="neighbor-list" id="leftAlignedTokenList"></div></section><section class="neighbor-column opposed"><div class="neighbor-head"><h4>− Opposed to U</h4><span id="leftOpposedTokenCount"></span></div><div class="neighbor-list" id="leftOpposedTokenList"></div></section></div></div></details>` : `<p class="left-note">Full-vocabulary U token geometry is unavailable for this direction.</p>`;
      return `<section class="left-section"><div class="left-heading"><div><p class="eyebrow">Paired SVD output direction</p><h3>Corresponding left singular vector</h3><p>The FineWeb scan ranks and activates the right vector V. The fingerprint-matched enrichment reconstructs the paired output direction directly as normalized J·V, checks it against the saved SVD U, and projects it into vocabulary space.</p></div><span class="left-badge">J·V reconstructed · basis exact</span></div>
        <div class="sv-map"><div class="sv-node"><span>Right singular vector · input side</span><b>V · ${esc(row.candidate)}</b><small>Projected against FineWeb residual states by both dashboard lenses</small></div><div class="sv-operator"><span>J-Lens transport</span><b>J · V = σU</b><small>σ ${decimal(data.singular_value,4)}</small></div><div class="sv-node left"><span>Left singular vector · output side</span><b>U · ${esc(row.candidate)}</b><small>Normalized direct J·V, checked against the scanner's saved U</small></div></div>
        <div class="left-metrics">${leftMetric("Singular value σ",decimal(data.singular_value,5),"Stored singular value for this V/U pair.")}${leftMetric("Top-64 σ² share",percent(data.saved_spectral_energy_fraction,2),"Share of squared singular-value mass within the saved top-64 bank.")}${leftMetric("Actual ||J·V||",decimal(data.actual_transport_gain,5),"Transport gain computed directly from the J-Lens and saved V.")}${leftMetric("Actual gain / σ",decimal(data.gain_over_stored_singular_value,7),"Agreement between direct transport gain and the stored singular value.")}${leftMetric("Source ||V||",decimal(data.source_v_norm,7),"Right-vector norm before reconstruction normalization.")}${leftMetric("Saved ||U||",decimal(data.saved_u_norm,7),"Saved SVD left-vector norm before comparison.")}${leftMetric("paired cos(U,V)",signed(data.paired_u_v_cosine,5),"Raw residual-coordinate cosine; descriptive, not the SVD pairing criterion.")}${leftMetric("cos(J·V, saved U)",decimal(data.transport_vs_stored_u_cosine,7),"Directional agreement between direct transport and saved SVD U.")}${leftMetric("σU relative error",decimal(data.transport_vs_sigma_stored_u_relative_error,7),"Relative error between direct J·V and stored σU.")}${leftMetric("Max other-U |cos|",decimal(data.max_other_u_abs_cosine,6),"Largest absolute cosine with another directly reconstructed J·V direction in this layer.")}</div>
        <div class="left-relations"><article class="left-relation"><h4>Closest other U in this layer</h4><b>${svCandidate(row.layer,data.max_other_u_sv0)} · cos ${signed(data.max_other_u_cosine,5)}</b><p>Nonzero overlap exposes approximation leakage between reconstructed transported directions.</p></article><article class="left-relation"><h4>Left-vector continuity across layers</h4><b>← ${previousText}</b><b>→ ${nextText}</b><p>Best absolute-cosine reconstructed-U match in each adjacent layer; signs may flip.</p></article><article class="left-relation"><h4>Right ↔ left token overlap · top ${integer.format(overlap.k)}</h4><b>same sign + ${integer.format(alignedOverlap)} / − ${integer.format(opposedOverlap)}</b><p>${integer.format(flippedOverlap)} cross-sign overlaps. Low overlap means U and V point toward different vocabulary neighborhoods.</p></article></div>
        <p class="left-note">U and V share residual coordinates but have different roles: V identifies an input direction the J-Lens responds to; U identifies the output direction that input becomes. Their sign is joint and arbitrary—flipping both leaves the singular pair unchanged.</p>${tokenPanel}</section>`;
    }

    function unembeddingSection(data, row) {
      if (!data) return "";
      return `<section class="token-space-section">
        <div class="token-space-heading">
          <div><p class="eyebrow">Input-side vocabulary geometry · V</p><h3>Right-vector token-space neighbors</h3><p>Cosine similarity between this right singular vector and each normalized output-token unembedding row. These are geometric neighbors—not observed activations, generated tokens, or proof of a semantic label.</p><span class="sign-note">The +/− orientation is arbitrary. The spectrum uses one shared scale across all ${integer.format(DATA.meta.unembedding_candidates)} directions.</span></div>
          <label class="control token-limit">Tokens per side<select id="tokenLimit"></select></label>
        </div>
        <div class="unembed-stats">
          <div class="unembed-stat"><span>Strongest |token cosine|</span><b>${decimal(data.max_abs, 4)}</b></div>
          <div class="unembed-stat"><span>Global token-likeness rank</span><b>#${integer.format(row.rank_global_max_abs_unembed_token_cosine)}</b></div>
          <div class="unembed-stat"><span>Endpoint z-scores</span><b>${signed(data.nearest_z, 2)} / ${signed(data.farthest_z, 2)}</b></div>
          <div class="unembed-stat"><span>Vocabulary rows compared</span><b>${integer.format(data.vocab)}</b></div>
        </div>
        <div class="unembed-spectrum">
          <div class="spectrum-topline"><span>vocabulary mean <b>${signed(data.mean, 4)}</b> · σ <b>${decimal(data.std, 4)}</b></span><span>top-1 gaps <b>aligned ${decimal(data.nearest_margin, 4)}</b> / <b>opposed ${decimal(data.farthest_margin, 4)}</b></span></div>
          <div class="spectrum-track" id="unembeddingSpectrum"></div>
          <div class="spectrum-labels"><span id="spectrumMin"></span><span>0 cosine</span><span id="spectrumMax"></span></div>
          <div class="spectrum-legend"><span><i class="legend-dot opposed"></i>opposed tokens</span><span><i class="legend-band"></i>vocab mean ±1σ</span><span><i class="legend-dot aligned"></i>aligned tokens</span></div>
        </div>
        <div class="neighbor-grid">
          <section class="neighbor-column aligned"><div class="neighbor-head"><h4>+ Aligned with V</h4><span id="alignedTokenCount"></span></div><div class="neighbor-list" id="alignedTokenList"></div></section>
          <section class="neighbor-column opposed"><div class="neighbor-head"><h4>− Opposed to V</h4><span id="opposedTokenCount"></span></div><div class="neighbor-list" id="opposedTokenList"></div></section>
        </div>
      </section>`;
    }

    function renderDetail(visibleRows) {
      const root = $("#detail");
      const view = viewState();
      if (!view.selected) {
        root.innerHTML = `<div class="empty">Choose a broader filter to inspect a direction.</div>`;
        return;
      }
      const row = byCandidate().get(view.selected);
      const selectedIndex = visibleRows.findIndex(item => item.candidate === view.selected);
      const lo = Math.min(Number(row.min_activation), 0);
      const hi = Math.max(Number(row.max_activation), 0);
      const spread = Math.max(hi - lo, 1e-9);
      const zeroPosition = (0 - lo) / spread * 100;
      const meanPosition = Math.max(0, Math.min(100, (Number(row.mean_activation) - lo) / spread * 100));
      const tokenData = DATA.unembedding[view.selected];
      const leftData = DATA.left_singular?.[view.selected];
      const isSelective = state.mode === "selective";
      const rankLabel = isSelective ? "Global tail-selectivity rank" : "Global mean |cosine| rank";
      const summary = isSelective
        ? `The singular direction at <b>zero-based SV index ${integer.format(row.sv_index_0)}</b> in layer ${row.layer}, viewed through its robust activation tails. It ranks <b>#${integer.format(row.rank_global_tail_selectivity)} by tail selectivity</b> and <b>#${integer.format(row.rank_global_mean_abs_cosine)} by broad activity</b> over ${integer.format(row.n_tokens)} FineWeb tokens. The score rewards a sharp, energy-concentrated tail with minimum support; it does not itself penalize frequent activation once five sampled windows clear z=5.`
        : `The singular direction at <b>zero-based SV index ${integer.format(row.sv_index_0)}</b> in layer ${row.layer}. It is the <b>#${integer.format(row.rank_global_mean_abs_cosine)} direction globally</b> by mean absolute cosine over ${integer.format(row.n_tokens)} FineWeb tokens.`;
      const heroMetrics = isSelective ? selectiveHeroMetrics(row) : broadHeroMetrics(row);

      root.innerHTML = `<section class="detail-hero">
        <div class="detail-nav">
          <span class="detail-rank">${rankLabel} #${integer.format(row[modeConfig().primaryRank])}</span>
          <div class="nav-buttons">
            <button class="icon-button" id="previousCandidate" type="button" aria-label="Previous visible direction" title="Previous visible direction" ${selectedIndex <= 0 ? "disabled" : ""}>←</button>
            <button class="icon-button" id="nextCandidate" type="button" aria-label="Next visible direction" title="Next visible direction" ${selectedIndex < 0 || selectedIndex >= visibleRows.length - 1 ? "disabled" : ""}>→</button>
            <button class="copy-button" id="copyCandidate" type="button">Copy ID</button>
          </div>
        </div>
        <div class="title-row"><h2 class="detail-title">${esc(row.candidate)}</h2><span class="id-chip">L${String(row.layer).padStart(2,"0")} / SV${String(row.sv_index_0).padStart(2,"0")}</span></div>
        <p class="detail-summary">${summary}</p>
        <div class="metric-grid">${heroMetrics}</div>
        ${isSelective ? tailProfile(row) : ""}
        <div class="profile">
          <div>
            <div class="profile-labels"><span>${signed(lo)}</span><span>activation range</span><span>${signed(hi)}</span></div>
            <div class="axis" aria-label="Activation range from ${signed(lo)} to ${signed(hi)}; mean ${signed(row.mean_activation)}"><i class="zero-marker" style="left:${zeroPosition.toFixed(2)}%" title="zero"></i><i class="mean-marker" style="left:${meanPosition.toFixed(2)}%" title="mean ${signed(row.mean_activation)}"></i></div>
          </div>
          <div class="profile-stat"><span>Mean<b>${signed(row.mean_activation)}</b></span><span>Positive tokens<b>${percent(row.positive_rate, Number(row.positive_rate) < .01 ? 2 : 1)}</b></span></div>
        </div>
        <details class="metric-details"><summary>Inspect all ${integer.format(Object.keys(row).length - 1)} ranking metrics</summary>${allMetricGroups(row)}</details>
      </section>
      ${leftSingularSection(row)}
      ${unembeddingSection(tokenData, row)}
      <section class="context-section">
        <div class="section-heading"><div><p class="eyebrow">Observed examples · ${esc(modeConfig().label)}</p><h3>${isSelective ? "Extreme tail contexts" : "Top activation contexts"}</h3><p>${isSelective ? "Contexts are ordered by raw projection extremes; the displayed tail z is derived from this scan's median and robust MAD scale. The score-driving tail is highlighted, while both orientations remain available for comparison." : "Contexts are ordered by signed projection activation. Each highlight marks the activating token; cosine is normalized by the layer residual norm."}</p><span class="sign-note">High (+) and low (−) are arbitrary SVD orientations, not sentiment labels. ${isSelective ? "High selectivity or kurtosis can still reflect lexical, formatting, or data artifacts." : ""}</span></div></div>
        <div class="context-controls">
          <label class="control context-search">Search these contexts<input id="contextSearch" type="search" value="${esc(state.contextQuery)}" placeholder="Text, token, domain…"></label>
          <label class="control">Per side<select id="contextLimit"></select></label>
          <label class="check"><input id="dedupeSources" type="checkbox" ${state.dedupe ? "checked" : ""}> Unique sources</label>
        </div>
        <div class="context-grid">
          <section class="context-column positive ${isSelective && row.selected_tail_polarity === "positive" ? "score-driver" : ""}"><div class="column-head"><h4>+ ${isSelective ? "High tail" : "Positive direction"}${isSelective && row.selected_tail_polarity === "positive" ? '<span class="tail-pill">score driver</span>' : ""}</h4><span id="positiveCount"></span></div><div class="context-list" id="positiveList"></div></section>
          <section class="context-column negative ${isSelective && row.selected_tail_polarity === "negative" ? "score-driver" : ""}"><div class="column-head"><h4>− ${isSelective ? "Low tail" : "Negative direction"}${isSelective && row.selected_tail_polarity === "negative" ? '<span class="tail-pill">score driver</span>' : ""}</h4><span id="negativeCount"></span></div><div class="context-list" id="negativeList"></div></section>
        </div>
      </section>`;

      const maxContexts = modeData().meta.contexts_per_polarity || Math.max(...Object.values(modeData().contexts[view.selected] || {}).map(items => items.length), 0);
      const contextLimits = [...new Set([6, 12, 24, maxContexts].filter(value => value > 0 && value <= maxContexts))].sort((a, b) => a - b);
      if (!contextLimits.includes(state.contextLimit)) state.contextLimit = contextLimits[0] || maxContexts;
      $("#contextLimit").innerHTML = contextLimits.map(value => `<option value="${value}">${value === maxContexts ? `All ${value}` : value}</option>`).join("");
      $("#contextLimit").value = String(state.contextLimit);
      $("#contextSearch").addEventListener("input", event => { state.contextQuery = event.target.value.trim().toLowerCase(); renderContexts(); });
      $("#contextLimit").addEventListener("change", event => { state.contextLimit = Number(event.target.value); renderContexts(); });
      $("#dedupeSources").addEventListener("change", event => { state.dedupe = event.target.checked; renderContexts(); });
      $("#previousCandidate").addEventListener("click", () => selectedIndex > 0 && selectCandidate(visibleRows[selectedIndex - 1].candidate));
      $("#nextCandidate").addEventListener("click", () => selectedIndex >= 0 && selectedIndex < visibleRows.length - 1 && selectCandidate(visibleRows[selectedIndex + 1].candidate));
      $("#copyCandidate").addEventListener("click", copyCandidate);
      if (tokenData) {
        const maxTokenNeighbors = Math.max(tokenData.nearest.length, tokenData.farthest.length);
        const tokenLimits = [...new Set([8, 16, maxTokenNeighbors].filter(value => value > 0 && value <= maxTokenNeighbors))].sort((a, b) => a - b);
        if (!tokenLimits.includes(state.tokenLimit)) state.tokenLimit = tokenLimits[0] || maxTokenNeighbors;
        $("#tokenLimit").innerHTML = tokenLimits.map(value => `<option value="${value}">${value === maxTokenNeighbors ? `All ${value}` : value}</option>`).join("");
        $("#tokenLimit").value = String(state.tokenLimit);
        $("#tokenLimit").addEventListener("change", event => { state.tokenLimit = Number(event.target.value); renderUnembedding(tokenData); });
        renderUnembedding(tokenData);
      }
      if (leftData?.token_geometry && $("#leftTokenLimit")) {
        const leftTokens=leftData.token_geometry;
        const maxLeftNeighbors=Math.max(leftTokens.nearest.length,leftTokens.farthest.length);
        const leftLimits=[...new Set([8,16,maxLeftNeighbors].filter(value=>value>0&&value<=maxLeftNeighbors))].sort((a,b)=>a-b);
        if(!leftLimits.includes(state.leftTokenLimit)) state.leftTokenLimit=leftLimits[0]||maxLeftNeighbors;
        $("#leftTokenLimit").innerHTML=leftLimits.map(value=>`<option value="${value}">${value===maxLeftNeighbors?`All ${value}`:value}</option>`).join("");
        $("#leftTokenLimit").value=String(state.leftTokenLimit);
        $("#leftTokenLimit").addEventListener("change",event=>{state.leftTokenLimit=Number(event.target.value);renderLeftUnembedding(leftTokens);});
        renderLeftUnembedding(leftTokens);
      }
      renderContexts();
    }

    function broadHeroMetrics(row) {
      return [
        metricCard("Mean |cosine|", decimal(row.mean_abs_cosine, 4), BROAD_METRICS.mean_abs_cosine.help),
        metricCard("Top-1 token share", percent(row.top1_abs_rate), BROAD_METRICS.top1_abs_rate.help),
        metricCard("Top-5 token share", percent(row.top5_abs_rate), BROAD_METRICS.top5_abs_rate.help),
        metricCard("Mean |activation|", decimal(row.mean_abs_activation), "Mean magnitude of the projection coefficient over tokens."),
        metricCard("Activation std", decimal(row.std_activation), BROAD_METRICS.std_activation.help),
        metricCard("Singular value", decimal(row.singular_value), BROAD_METRICS.singular_value.help)
      ].join("");
    }

    function selectiveHeroMetrics(row) {
      const polarity = row.selected_tail_polarity === "positive" ? "High (+)" : "Low (−)";
      return [
        metricCard("Tail selectivity score", decimal(row.tail_selectivity_score, 4), SELECTIVE_METRICS.tail_selectivity_score.help),
        metricCard("Score-driving tail", polarity, "The arbitrary SV orientation whose tail produced the larger score."),
        metricCard("Selected q99.9 robust z", decimal(row.selected_tail_q999_robust_z, 3), SELECTIVE_METRICS.selected_tail_q999_robust_z.help),
        metricCard("Top-0.1% tail energy", percent(row.selected_tail_top0_1pct_energy_share, 2), SELECTIVE_METRICS.selected_tail_top0_1pct_energy_share.help),
        metricCard("Windows above z=5", `${integer.format(row.selected_tail_doc_count_z5)} · ${percent(row.selected_tail_doc_rate_z5, 1)}`, SELECTIVE_METRICS.selected_tail_doc_rate_z5.help),
        metricCard("Effective support", percent(row.effective_support_fraction, 2), SELECTIVE_METRICS.effective_support_fraction.help)
      ].join("");
    }

    function tailProfile(row) {
      const sampleSize = row.token_sample_n || modeData().meta.token_sample_per_layer?.[row.layer];
      return `<section class="tail-profile">
        <div class="tail-profile-head"><div><h3>Two-sided tail profile</h3><p>Robust z uses the per-direction median and 1.4826 × MAD scale. Quantiles and energy shares use a systematic ${integer.format(sampleSize)}-token layer sample; sampled-window peak counts use all ${integer.format(row.n_tokens)} scanned tokens.</p></div></div>
        <div class="tail-profile-grid">${tailSide(row, "positive")}${tailSide(row, "negative")}</div>
      </section>`;
    }

    function tailSide(row, polarity) {
      const prefix = `${polarity}_`;
      const selected = row.selected_tail_polarity === polarity;
      const largestShare = Number(row[`${prefix}top_context_largest_center_token_share`]);
      const warning = largestShare >= .5 ? `<span class="artifact-warning">High lexical concentration—inspect for a repeated token.</span>` : "Lexical concentration is below 50% in the retained extremes.";
      return `<article class="tail-side ${selected ? "selected" : ""}">
        <div class="tail-side-head"><h4>${polarity === "positive" ? "+ High tail" : "− Low tail"}</h4><span>${selected ? "Score driver" : `tail score ${decimal(row[`${prefix}tail_selectivity_score`], 3)}`}</span></div>
        <div class="tail-stat-grid">
          ${tailStat("q99 robust z", decimal(row[`${prefix}q99_robust_z`], 3))}
          ${tailStat("q99.9 robust z", decimal(row[`${prefix}q999_robust_z`], 3))}
          ${tailStat("q99.99 robust z", decimal(row[`${prefix}q9999_robust_z`], 3))}
          ${tailStat("Maximum robust z", decimal(row[`${prefix}max_robust_z`], 3))}
          ${tailStat("Top-0.1% z² energy", percent(row[`${prefix}top0_1pct_energy_share`], 2))}
          ${tailStat("Windows above z=5", `${integer.format(row[`${prefix}doc_count_z5`])} · ${percent(row[`${prefix}doc_rate_z5`], 1)}`)}
        </div>
        <p class="tail-artifact-note"><strong>${integer.format(row[`${prefix}top_context_unique_documents`])}</strong> unique windows · <strong>${decimal(row[`${prefix}top_context_effective_center_tokens`], 2)}</strong> effective center tokens · largest token <strong>${percent(largestShare, 1)}</strong>. ${warning}</p>
      </article>`;
    }

    function tailStat(label, value) {
      return `<div class="tail-stat"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
    }

    function allMetricGroups(row) {
      const groups = new Map();
      Object.entries(row).filter(([key]) => key !== "candidate").forEach(([key, value]) => {
        let group = "Core activity";
        if (key.startsWith("rank_")) group = "Ranks";
        else if (key.includes("unembed")) group = "Token geometry";
        else if (key.includes("top_context")) group = "Context diversity";
        else if (key.startsWith("positive_")) group = "+ high tail";
        else if (key.startsWith("negative_")) group = "− low tail";
        else if (key.startsWith("abs_") || key.startsWith("selected_tail_") || ["token_sample_n", "median_activation", "mad_activation", "robust_activation_scale", "stable_skewness", "stable_kurtosis", "stable_excess_kurtosis", "effective_support_fraction", "tail_selectivity_score"].includes(key)) group = "Robust distribution";
        if (!groups.has(group)) groups.set(group, []);
        groups.get(group).push(`<div class="all-metric"><span>${esc(LABELS[key] || key.replaceAll("_", " "))}</span><b>${esc(formatAny(key, value))}</b></div>`);
      });
      return `<div class="metric-groups">${[...groups.entries()].map(([label, items]) => `<section class="metric-group"><h4>${esc(label)}</h4><div class="all-metrics">${items.join("")}</div></section>`).join("")}</div>`;
    }

    function formatAny(key, value) {
      if (value == null) return "—";
      if (typeof value === "string") return visibleToken(value);
      if (key.startsWith("rank_") || key.endsWith("_count_z3") || key.endsWith("_count_z5") || key.endsWith("_count_z8") || ["layer", "sv_index_0", "n_tokens", "n_documents", "token_sample_n"].includes(key)) return integer.format(value);
      if (/(?:rate|share|fraction|weight)$/.test(key) || key.includes("energy_share")) return percent(value, 2);
      return decimal(value, 5);
    }

    function copyCandidate() {
      const button = $("#copyCandidate");
      const done = () => { button.textContent = "Copied"; setTimeout(() => button.textContent = "Copy ID", 1200); };
      const selected = viewState().selected;
      if (navigator.clipboard?.writeText) navigator.clipboard.writeText(selected).then(done).catch(() => fallbackCopy(selected, done));
      else fallbackCopy(selected, done);
    }

    function fallbackCopy(text, done) {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
      done();
    }

    function hostname(url) {
      try { return new URL(url).hostname.replace(/^www\./, ""); }
      catch (_) { return "unknown source"; }
    }

    function visibleToken(token) {
      return JSON.stringify(String(token ?? ""))
        .replaceAll("\\n", "↵")
        .replaceAll("\\t", "⇥");
    }

    function unembeddingTokenLabel(item) {
      const value = item.decoded || item.token;
      return value ? visibleToken(value) : `token #${item.id}`;
    }

    function renderUnembedding(data) {
      renderUnembeddingSpectrum(data);
      renderNeighborColumn("aligned", data.nearest, data);
      renderNeighborColumn("opposed", data.farthest, data);
    }

    function renderLeftUnembedding(data) {
      renderNeighborColumn("aligned",data.nearest,data,state.leftTokenLimit,"#leftAlignedTokenList","#leftAlignedTokenCount");
      renderNeighborColumn("opposed",data.farthest,data,state.leftTokenLimit,"#leftOpposedTokenList","#leftOpposedTokenCount");
    }

    function renderUnembeddingSpectrum(data) {
      const track = $("#unembeddingSpectrum");
      if (!track) return;
      const domain = Math.max(Number(DATA.meta.unembedding_domain_max), Number(data.max_abs), 1e-9);
      const position = value => Math.max(0, Math.min(100, (Number(value) + domain) / (2 * domain) * 100));
      const bandStart = position(Number(data.mean) - Number(data.std));
      const bandEnd = position(Number(data.mean) + Number(data.std));
      track.replaceChildren();
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", `Token cosine spectrum from ${signed(-domain, 4)} to ${signed(domain, 4)}. Vocabulary mean ${signed(data.mean, 4)} and standard deviation ${decimal(data.std, 4)}.`);

      const band = document.createElement("i");
      band.className = "spectrum-band";
      band.style.left = `${bandStart.toFixed(3)}%`;
      band.style.width = `${Math.max(0, bandEnd - bandStart).toFixed(3)}%`;
      const zero = document.createElement("i");
      zero.className = "spectrum-zero";
      zero.style.left = "50%";
      zero.title = "zero cosine";
      const mean = document.createElement("i");
      mean.className = "spectrum-mean";
      mean.style.left = `${position(data.mean).toFixed(3)}%`;
      mean.title = `vocabulary mean ${signed(data.mean, 4)}`;
      track.append(band, zero, mean);

      [["opposed", data.farthest], ["aligned", data.nearest]].forEach(([side, items]) => {
        items.forEach((item, index) => {
          const dot = document.createElement("i");
          const z = (Number(item.cosine) - Number(data.mean)) / Math.max(Number(data.std), 1e-9);
          dot.className = `spectrum-dot ${side}`;
          dot.style.left = `${position(item.cosine).toFixed(3)}%`;
          dot.style.top = `${9 + (index % 5) * 9}px`;
          dot.style.opacity = String(Math.max(.48, 1 - index * .016));
          dot.title = `${unembeddingTokenLabel(item)} · cosine ${signed(item.cosine, 4)} · ${signed(z, 2)}σ`;
          dot.setAttribute("aria-hidden", "true");
          track.append(dot);
        });
      });
      $("#spectrumMin").textContent = signed(-domain, 4);
      $("#spectrumMax").textContent = signed(domain, 4);
    }

    function renderNeighborColumn(side, items, data, limit=state.tokenLimit, listSelector=`#${side}TokenList`, countSelector=`#${side}TokenCount`) {
      const list = $(listSelector);
      if (!list) return;
      const shown = items.slice(0, limit);
      $(countSelector).textContent = `${shown.length} of ${items.length}`;
      list.replaceChildren();
      shown.forEach((item, index) => {
        const row = document.createElement("article");
        row.className = "neighbor-row";
        row.title = `Decoded: ${item.decoded || "(empty)"} · ${item.token ? `tokenizer spelling: ${String(item.token)}` : "tokenizer spelling unavailable"} · token ID ${item.id}`;

        const main = document.createElement("div");
        main.className = "neighbor-main";
        const rank = document.createElement("span");
        rank.className = "neighbor-rank";
        rank.textContent = `#${index + 1}`;
        const name = document.createElement("span");
        name.className = "neighbor-name";
        const decoded = document.createElement("span");
        decoded.className = "neighbor-token";
        decoded.textContent = unembeddingTokenLabel(item);
        const raw = document.createElement("span");
        raw.className = "neighbor-raw";
        raw.textContent = `${item.token ? `raw ${visibleToken(item.token)}` : "raw unavailable"} · id ${item.id}`;
        name.append(decoded, raw);
        const z = (Number(item.cosine) - Number(data.mean)) / Math.max(Number(data.std), 1e-9);
        const values = document.createElement("span");
        values.className = "neighbor-values";
        values.innerHTML = `<span>cos<b class="cosine">${esc(signed(item.cosine, 4))}</b></span><span>z<b>${esc(signed(z, 2))}σ</b></span>${item.dot_product==null?"":`<span>dot<b>${esc(signed(item.dot_product,4))}</b></span>`}`;
        main.append(rank, name, values);

        const bar = document.createElement("span");
        bar.className = "neighbor-bar";
        const fill = document.createElement("i");
        fill.style.width = `${Math.max(1, Math.min(100, Math.abs(Number(item.cosine)) / Math.max(Number(data.max_abs), 1e-9) * 100)).toFixed(2)}%`;
        bar.append(fill);
        row.append(main, bar);
        list.append(row);
      });
    }

    function filterContexts(items) {
      const query = state.contextQuery;
      const seen = new Set();
      return items.filter(item => {
        if (query && !`${item.context} ${item.token} ${hostname(item.url || "")} ${item.date || ""}`.toLowerCase().includes(query)) return false;
        const key = item.url || `document:${item.document}`;
        if (state.dedupe && seen.has(key)) return false;
        if (state.dedupe) seen.add(key);
        return true;
      });
    }

    function renderContexts() {
      const selected = viewState().selected;
      const grouped = modeData().contexts[selected] || { positive: [], negative: [] };
      const row = byCandidate().get(selected);
      renderContextColumn("positive", filterContexts(grouped.positive), row);
      renderContextColumn("negative", filterContexts(grouped.negative), row);
    }

    function renderContextColumn(polarity, filtered, row) {
      const list = $(`#${polarity}List`);
      const shown = filtered.slice(0, state.contextLimit);
      $(`#${polarity}Count`).textContent = `${shown.length} of ${filtered.length}`;
      list.replaceChildren();
      if (!shown.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No contexts match this search.";
        list.append(empty);
        return;
      }
      shown.forEach(item => list.append(contextCard(item, polarity, row)));
    }

    function contextCard(item, polarity, row) {
      const card = document.createElement("article");
      card.className = `context-card ${polarity}`;

      const meta = document.createElement("div");
      meta.className = "context-meta";
      const rank = document.createElement("span");
      rank.className = "context-rank";
      rank.textContent = `${polarity === "positive" ? "+" : "−"} context #${item.rank}`;
      const values = document.createElement("span");
      values.className = "context-values";
      const activation = document.createElement("span");
      activation.innerHTML = `activation <b class="activation">${esc(signed(item.activation))}</b>`;
      const cosine = document.createElement("span");
      cosine.innerHTML = `cos <b>${esc(signed(item.cosine, 4))}</b>`;
      values.append(activation, cosine);
      if (state.mode === "selective" && row) {
        const scale = Math.max(Number(row.robust_activation_scale), 1e-12);
        const centered = (Number(item.activation) - Number(row.median_activation)) / scale;
        const tailZ = polarity === "positive" ? centered : -centered;
        const robust = document.createElement("span");
        robust.innerHTML = `tail z <b>${esc(decimal(tailZ, 2))}</b>`;
        values.append(robust);
      }
      meta.append(rank, values);

      const copy = document.createElement("p");
      copy.className = "context-copy";
      appendMarkedText(copy, item.context || "");

      const source = document.createElement("div");
      source.className = "source-row";
      const info = document.createElement("span");
      info.className = "source-info";
      const token = document.createElement("span");
      token.className = "token-chip";
      token.title = `Raw token: ${String(item.token ?? "")}`;
      token.textContent = visibleToken(item.token);
      const details = [hostname(item.url || ""), item.date ? String(item.date).slice(0, 10) : null, item.document != null ? `doc ${item.document}` : null].filter(Boolean).join(" · ");
      info.append(token, document.createTextNode(`  ${details}`));
      source.append(info);
      if (/^https?:\/\//i.test(item.url || "")) {
        const link = document.createElement("a");
        link.className = "source-link";
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "source ↗";
        source.append(link);
      }
      card.append(meta, copy, source);
      return card;
    }

    function appendMarkedText(target, text) {
      let cursor = 0;
      while (cursor < text.length) {
        const start = text.indexOf("⟦", cursor);
        if (start < 0) {
          target.append(document.createTextNode(text.slice(cursor)));
          break;
        }
        const end = text.indexOf("⟧", start + 1);
        if (end < 0) {
          target.append(document.createTextNode(text.slice(cursor)));
          break;
        }
        target.append(document.createTextNode(text.slice(cursor, start)));
        const mark = document.createElement("mark");
        mark.textContent = text.slice(start + 1, end);
        target.append(mark);
        cursor = end + 1;
      }
      if (!text.length) target.textContent = "(empty context)";
    }

    setup();
  </script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    selectivity_data_dir = args.selectivity_data_dir.expanduser().resolve()
    left_data_dir = args.left_data_dir.expanduser().resolve()
    output = (args.output or data_dir / "report.html").expanduser().resolve()
    payload = build_payload(
        data_dir,
        selectivity_data_dir,
        left_data_dir,
        args.top,
        args.selectivity_top,
    )
    html = HTML_TEMPLATE.replace("__DASHBOARD_PAYLOAD__", safe_script_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    broad = payload["modes"]["broad"]
    selective = payload["modes"]["selective"]
    contexts_by_mode = {
        mode: sum(
            len(group[polarity])
            for group in mode_payload["contexts"].values()
            for polarity in ("positive", "negative")
        )
        for mode, mode_payload in payload["modes"].items()
    }
    embedded_neighbors = sum(
        len(group[side])
        for group in payload["unembedding"].values()
        for side in ("nearest", "farthest")
    )
    embedded_left_neighbors = sum(
        len(items)
        for record in payload["left_singular"].values()
        for side, items in (record.get("token_geometry") or {}).items()
        if side in ("nearest", "farthest")
    )
    print(f"Wrote {output}")
    print(
        f"Embedded {broad['meta']['embedded_candidates']:,} broad and "
        f"{selective['meta']['embedded_candidates']:,} selective candidates "
        f"({payload['meta']['embedded_candidate_union']:,} unique), "
        f"{embedded_neighbors:,} right-token neighbors, "
        f"{len(payload['left_singular']):,} paired left vectors with "
        f"{embedded_left_neighbors:,} left-token neighbors, and "
        f"{contexts_by_mode['broad'] + contexts_by_mode['selective']:,} contexts "
        f"({contexts_by_mode['broad']:,} broad + {contexts_by_mode['selective']:,} selective)"
    )
    print(f"Output size: {output.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
