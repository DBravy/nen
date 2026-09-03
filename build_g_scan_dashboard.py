#!/usr/bin/env python3
"""Build the self-contained dashboard for the checked-in G-direction scan.

The G scan uses the same ranking and unembedding schemas as the FineWeb
singular-vector dashboard, so this builder intentionally reuses that page.
The large context and direction-bank artifacts are ignored by git for this
scan. Context panels are included when top_contexts.jsonl is present; panels
that require unavailable paired left-vector artifacts are omitted cleanly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_unrealized_words_dashboard as shared


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "g_scan_sparse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the G scan outputs (default: g_scan_sparse)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: DATA_DIR/report.html)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=250,
        help="Number of broad-activity candidates to embed; 0 embeds all",
    )
    parser.add_argument(
        "--selectivity-top",
        type=int,
        default=250,
        help="Number of tail-selectivity candidates to embed; 0 embeds all",
    )
    return parser.parse_args()


def build_mode_payload(
    data_dir: Path,
    metadata: dict[str, Any],
    rankings_filename: str,
    rank_key: str,
    top_n: int,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rankings_path = data_dir / rankings_filename
    if not rankings_path.is_file():
        raise SystemExit(f"Missing required input: {rankings_path}")

    rankings, total_candidates = shared.load_rankings(
        rankings_path, top_n, rank_key
    )
    display_rankings: list[dict[str, Any]] = []
    display_contexts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rankings:
        source_candidate = str(row["candidate"])
        expected_source = (
            f"L{int(row['layer']):02d}_SV{int(row['sv_index_0']) + 1:02d}"
        )
        if source_candidate != expected_source:
            raise SystemExit(
                f"Unexpected source candidate {source_candidate!r}; "
                f"expected {expected_source!r} from layer and sv_index_0"
            )
        display_candidate = shared.zero_based_candidate(row)
        display_rankings.append(shared.display_row(row, mode))
        display_contexts[display_candidate] = {"positive": [], "negative": []}

    return (
        {
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
                "contexts_per_polarity": 0,
                "contexts_available": False,
                "source_contexts_per_polarity": metadata.get(
                    "top_contexts_per_polarity"
                ),
                "total_candidates": total_candidates,
                "embedded_candidates": len(rankings),
                "total_contexts": 0,
                "display_sv_numbering": "zero_based",
                "token_sample_per_layer": metadata.get(
                    "token_sample_actual_per_layer", {}
                ),
                "min_tail_docs_for_full_score_weight": metadata.get(
                    "min_tail_docs_for_full_score_weight"
                ),
            },
            "rankings": display_rankings,
            "contexts": display_contexts,
        },
        rankings,
    )


def build_payload(
    data_dir: Path, top_n: int, selectivity_top_n: int
) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing required input: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    broad, broad_source_rows = build_mode_payload(
        data_dir,
        metadata,
        "sv_rankings.csv",
        shared.BROAD_RANK,
        top_n,
        "broad",
    )
    selective, selective_source_rows = build_mode_payload(
        data_dir,
        metadata,
        "selectivity_rankings.csv",
        shared.SELECTIVITY_RANK,
        selectivity_top_n,
        "selective",
    )

    source_rows: dict[str, dict[str, Any]] = {}
    for row in broad_source_rows + selective_source_rows:
        source_rows.setdefault(str(row["candidate"]), row)
    source_candidates = list(source_rows)

    contexts_path = data_dir / "top_contexts.jsonl"
    contexts_available = contexts_path.is_file()
    if contexts_available:
        source_contexts, total_contexts = shared.load_contexts(
            contexts_path, source_candidates
        )
        expected_contexts = int(metadata.get("top_contexts_per_polarity", 0))
        incomplete = [
            f"{candidate}/{polarity} ({len(source_contexts[candidate][polarity])})"
            for candidate in source_candidates
            for polarity in ("positive", "negative")
            if expected_contexts
            and len(source_contexts[candidate][polarity]) != expected_contexts
        ]
        if incomplete:
            raise SystemExit(
                "Incomplete context sets for "
                f"{len(incomplete)} candidate/polarity pairs: "
                + ", ".join(incomplete[:8])
            )

        for mode_payload, mode_source_rows in (
            (broad, broad_source_rows),
            (selective, selective_source_rows),
        ):
            mode_payload["contexts"] = {
                shared.zero_based_candidate(row): source_contexts[str(row["candidate"])]
                for row in mode_source_rows
            }
            mode_payload["meta"].update(
                contexts_per_polarity=expected_contexts,
                contexts_available=True,
                total_contexts=total_contexts,
            )

    neighbors_path = data_dir / "unembedding_neighbors.jsonl"
    if not neighbors_path.is_file():
        raise SystemExit(f"Missing required input: {neighbors_path}")
    source_unembedding, unembedding_meta = shared.load_unembedding_neighbors(
        neighbors_path, source_candidates
    )
    display_unembedding = {
        shared.zero_based_candidate(source_rows[source_candidate]): source_unembedding[
            source_candidate
        ]
        for source_candidate in source_candidates
    }

    broad_candidates = {row["candidate"] for row in broad["rankings"]}
    selective_candidates = {row["candidate"] for row in selective["rankings"]}
    return {
        "default_mode": "broad",
        "meta": {
            "page": {
                "document_title": "FineWeb G Direction Atlas",
                "eyebrow": "Interpretability workbench · gated-gain directions",
                "title": "G Direction",
                "title_emphasis": "Atlas",
                "subtitle": (
                    "Explore the gated-gain matrix G's saved eigenvector bank through "
                    "broad-activity and selective-tail rankings, then inspect each "
                    "direction's activation statistics and vocabulary geometry."
                ),
            },
            "model": broad["meta"]["model"],
            "dataset": broad["meta"]["dataset"],
            "dataset_config": broad["meta"]["dataset_config"],
            "direction_source": metadata.get("direction_source"),
            "contexts_available": contexts_available,
            "display_sv_numbering": "zero_based",
            "unembedding_candidates": unembedding_meta["total"],
            "unembedding_space": unembedding_meta["space"],
            "unembedding_vocab_rows": unembedding_meta["vocab"],
            "unembedding_neighbors_per_side": unembedding_meta["per_side"],
            "unembedding_domain_max": unembedding_meta["domain_max"],
            "embedded_candidate_union": len(display_unembedding),
            "embedded_candidate_overlap": len(
                broad_candidates & selective_candidates
            ),
            "left_singular": {
                "available": False,
                "reason": "No paired left-vector artifacts are in g_scan_sparse.",
            },
        },
        "modes": {"broad": broad, "selective": selective},
        "unembedding": display_unembedding,
        "left_singular": {},
    }


def specialize_template(template: str) -> str:
    replacements = {
        "<title>FineWeb Singular Vector Atlas</title>": (
            "<title>FineWeb G Direction Atlas</title>"
        ),
        '<div class="brand-mark" aria-hidden="true">SV</div>': (
            '<div class="brand-mark" aria-hidden="true">G</div>'
        ),
        'aria-label="Ranked singular vectors"': 'aria-label="Ranked G directions"',
        "The singular direction at": "The G-bank direction at",
        "Singular value for this direction within its layer.": (
            "Scanner-supplied scale associated with this saved G direction."
        ),
        "Singular value": "Direction scale",
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    return template


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output = (args.output or data_dir / "report.html").expanduser().resolve()
    payload = build_payload(data_dir, args.top, args.selectivity_top)
    template = specialize_template(shared.HTML_TEMPLATE)
    html = template.replace(
        "__DASHBOARD_PAYLOAD__", shared.safe_script_json(payload)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    broad = payload["modes"]["broad"]
    selective = payload["modes"]["selective"]
    embedded_neighbors = sum(
        len(group[side])
        for group in payload["unembedding"].values()
        for side in ("nearest", "farthest")
    )
    print(f"Wrote {output}")
    print(
        f"Embedded {broad['meta']['embedded_candidates']:,} broad and "
        f"{selective['meta']['embedded_candidates']:,} selective candidates "
        f"({payload['meta']['embedded_candidate_union']:,} unique) with "
        f"{embedded_neighbors:,} token neighbors"
    )
    if payload["meta"]["contexts_available"]:
        embedded_contexts = sum(
            len(items)
            for mode_payload in payload["modes"].values()
            for grouped in mode_payload["contexts"].values()
            for items in grouped.values()
        )
        print(f"Embedded {embedded_contexts:,} activation contexts")
    print("Paired-left-vector panel omitted: artifacts are unavailable")


if __name__ == "__main__":
    main()
