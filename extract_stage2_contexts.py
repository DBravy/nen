#!/usr/bin/env python3
"""
extract_stage2_contexts.py

Select Stage-2 candidates and package their contexts, using the rotated
banks as the per-layer null bar.

Selection rule: a direction qualifies if its tail_selectivity_score exceeds
the q-th quantile (default 0.99) of the pooled rotated-bank scores at the
same layer. The rule is applied identically to every condition, including
the rotated banks themselves — rotated "hits" are the illusion-control arm
of the blinded judging set.

For each selected candidate, the script streams that run's
top_contexts.jsonl and keeps the top-N contexts of the candidate's selected
tail polarity plus M contexts of the opposite polarity (within-candidate
contrast). Contexts use the pre-marked `context_marked` field.

Outputs in --out:
  candidates.csv       one row per selected candidate: condition, run,
                       candidate, layer, sv_rank, singular_value, score,
                       layer bar, excess, kurtosis, doc support, polarity,
                       lexical concentration, blind_id.  (Small; shareable.)
  contexts_blind.jsonl one line per candidate keyed only by blind_id, with
                       its contexts. No condition, name, rank, or score.
  key.csv              blind_id -> condition/run/candidate. Keep this away
                       from whoever judges.
  review.md            (--markdown) blinded human/LLM-readable version.
  bank_stats.txt       per-layer Spearman(sv_rank, score) within the svd
                       bank — the sigma-linkage check — plus band summaries.

Example:
    python extract_stage2_contexts.py \
        --svd scan_svd_r0 \
        --rotated scan_rot_r0 scan_rot_r1 \
        --random scan_rand_r0 \
        --out stage2_set --markdown

Then share stage2_set/candidates.csv and stage2_set/review.md (or
contexts_blind.jsonl); keep key.csv local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

NEEDED = [
    "candidate",
    "layer",
    "sv_rank_1based",
    "singular_value",
    "tail_selectivity_score",
    "selected_tail_polarity",
    "stable_excess_kurtosis",
    "positive_doc_count_z5",
    "negative_doc_count_z5",
]
LEX_COL = "selected_tail_top_context_largest_center_token_share"
CTX_KEEP = [
    "context_marked",
    "token",
    "activation",
    "cosine_activation",
    "document_index",
    "token_position",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Select above-bar candidates and extract their contexts.",
    )
    p.add_argument("--svd", nargs="+", required=True)
    p.add_argument("--rotated", nargs="+", required=True)
    p.add_argument("--random", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--bar-quantile", type=float, default=0.99)
    p.add_argument(
        "--min-doc-z5",
        type=int,
        default=0,
        help="Require at least this many z>5 documents on the selected polarity.",
    )
    p.add_argument("--max-per-condition", type=int, default=80)
    p.add_argument("--contexts", type=int, default=12, help="Selected-polarity contexts kept.")
    p.add_argument("--opposite", type=int, default=4, help="Opposite-polarity contexts kept.")
    p.add_argument("--layers", default=None, help="Optional layer filter, e.g. 5-18 or 2,8,14.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markdown", action="store_true")
    return p.parse_args()


def parse_layer_filter(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def read_rankings(run_dir: Path) -> list[dict]:
    for name in ("selectivity_rankings.csv", "sv_rankings.csv"):
        path = run_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and "tail_selectivity_score" not in rows[0]:
                raise SystemExit(
                    f"{path} lacks tail_selectivity_score — rescan {run_dir} with the "
                    "current scanner before extracting."
                )
            return rows
    raise FileNotFoundError(f"No rankings CSV in {run_dir}")


def fnum(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def ranks_with_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sv = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return math.nan
    rx, ry = ranks_with_ties(x), ranks_with_ties(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_filter = parse_layer_filter(args.layers)
    rng = random.Random(args.seed)

    # ---- Per-layer bars from pooled rotated scores -------------------------
    by_layer_rot: dict[int, list[float]] = {}
    for d in args.rotated:
        for row in read_rankings(Path(d)):
            layer = int(float(row["layer"]))
            by_layer_rot.setdefault(layer, []).append(fnum(row, "tail_selectivity_score"))
    bars = {
        layer: float(np.quantile(np.asarray(v, dtype=np.float64), args.bar_quantile))
        for layer, v in by_layer_rot.items()
        if v
    }
    print(f"[bars] q{args.bar_quantile:.2f} from {sum(len(v) for v in by_layer_rot.values())} "
          f"rotated directions over {len(bars)} layers")

    # ---- Select candidates per condition -----------------------------------
    conditions = [("svd", d) for d in args.svd]
    conditions += [("rotated", d) for d in args.rotated]
    conditions += [("random", d) for d in args.random]

    selected: list[dict] = []
    svd_rows_all: list[dict] = []
    for label, d in conditions:
        run_dir = Path(d)
        rows = read_rankings(run_dir)
        if label == "svd":
            svd_rows_all.extend(rows)
        picks: list[dict] = []
        for row in rows:
            layer = int(float(row["layer"]))
            if layer_filter is not None and layer not in layer_filter:
                continue
            bar = bars.get(layer)
            if bar is None:
                continue
            score = fnum(row, "tail_selectivity_score")
            if not (score > bar):
                continue
            pol = (row.get("selected_tail_polarity") or "positive").strip()
            doc_col = f"{pol}_doc_count_z5"
            if args.min_doc_z5 > 0 and fnum(row, doc_col, 0.0) < args.min_doc_z5:
                continue
            picks.append(
                {
                    "condition": label,
                    "run": str(run_dir),
                    "candidate": row["candidate"],
                    "layer": layer,
                    "sv_rank_1based": int(float(row.get("sv_rank_1based", 0) or 0)),
                    "singular_value": fnum(row, "singular_value"),
                    "tail_selectivity_score": score,
                    "layer_bar": bar,
                    "excess_over_bar": score - bar,
                    "selected_tail_polarity": pol,
                    "stable_excess_kurtosis": fnum(row, "stable_excess_kurtosis"),
                    "doc_count_z5_selected": fnum(row, doc_col, math.nan),
                    "doc_count_z5_opposite": fnum(
                        row,
                        f"{'negative' if pol == 'positive' else 'positive'}_doc_count_z5",
                        math.nan,
                    ),
                    "lexical_concentration": fnum(row, LEX_COL),
                }
            )
        picks.sort(key=lambda r: r["excess_over_bar"], reverse=True)
        kept = picks[: args.max_per_condition]
        print(f"[select] {label:>7} {run_dir}: {len(picks)} above bar, keeping {len(kept)}")
        selected.extend(kept)

    if not selected:
        print("Nothing above the bar; lower --bar-quantile or check inputs.")
        sys.exit(0)

    # Blind ids.
    for rec in selected:
        raw = f"{args.seed}:{rec['condition']}:{rec['run']}:{rec['candidate']}"
        rec["blind_id"] = hashlib.sha1(raw.encode()).hexdigest()[:8]

    # ---- Pull contexts (one streaming pass per run) ------------------------
    wanted: dict[str, dict[str, dict]] = {}
    for rec in selected:
        wanted.setdefault(rec["run"], {})[rec["candidate"]] = rec
        rec["contexts"] = []
        rec["opposite_contexts"] = []

    for run, cands in wanted.items():
        path = Path(run) / "top_contexts.jsonl"
        if not path.exists():
            print(f"[warn] {path} missing; contexts skipped for {len(cands)} candidates",
                  file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = cands.get(obj.get("candidate"))
                if rec is None:
                    continue
                pol = obj.get("polarity")
                rank = int(obj.get("rank_within_polarity", 10**9))
                ctx = {k: obj.get(k) for k in CTX_KEEP}
                ctx["rank"] = rank
                if pol == rec["selected_tail_polarity"] and rank <= args.contexts:
                    rec["contexts"].append(ctx)
                elif pol != rec["selected_tail_polarity"] and rank <= args.opposite:
                    rec["opposite_contexts"].append(ctx)

    for rec in selected:
        rec["contexts"].sort(key=lambda c: c["rank"])
        rec["opposite_contexts"].sort(key=lambda c: c["rank"])
        if not rec["contexts"]:
            print(f"[warn] no contexts found for {rec['candidate']} ({rec['run']})",
                  file=sys.stderr)

    # ---- Sigma-linkage within the svd bank ---------------------------------
    stats_lines: list[str] = []
    if svd_rows_all:
        stats_lines.append("Spearman(sv_rank_1based, tail_selectivity_score) within svd bank")
        stats_lines.append("(negative = higher-sigma axes are more selective)")
        by_layer: dict[int, list[tuple[float, float]]] = {}
        for row in svd_rows_all:
            by_layer.setdefault(int(float(row["layer"])), []).append(
                (fnum(row, "sv_rank_1based"), fnum(row, "tail_selectivity_score"))
            )
        band_pairs = {"early(0-4)": [], "mid(5-18)": [], "late(19+)": []}
        for layer in sorted(by_layer):
            pairs = by_layer[layer]
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho = spearman(x, y)
            stats_lines.append(f"  layer {layer:>3}: rho = {rho:+.3f}  (n={len(pairs)})")
            band = ("early(0-4)" if layer <= 4 else "mid(5-18)" if layer <= 18 else "late(19+)")
            band_pairs[band].extend(pairs)
        stats_lines.append("")
        for band, pairs in band_pairs.items():
            if pairs:
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                stats_lines.append(f"  {band}: rho = {spearman(x, y):+.3f}  (n={len(pairs)})")
        (out_dir / "bank_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
        print("\n".join(stats_lines))

    # ---- Write outputs -----------------------------------------------------
    meta_cols = [
        "blind_id", "condition", "run", "candidate", "layer", "sv_rank_1based",
        "singular_value", "tail_selectivity_score", "layer_bar", "excess_over_bar",
        "selected_tail_polarity", "stable_excess_kurtosis",
        "doc_count_z5_selected", "doc_count_z5_opposite", "lexical_concentration",
    ]
    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_cols, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(selected, key=lambda r: (r["condition"], -r["excess_over_bar"])):
            w.writerow(rec)

    with (out_dir / "key.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "condition", "run", "candidate"])
        for rec in selected:
            w.writerow([rec["blind_id"], rec["condition"], rec["run"], rec["candidate"]])

    blinded = list(selected)
    rng.shuffle(blinded)
    with (out_dir / "contexts_blind.jsonl").open("w", encoding="utf-8") as f:
        for rec in blinded:
            f.write(json.dumps({
                "blind_id": rec["blind_id"],
                "layer": rec["layer"],
                "polarity": rec["selected_tail_polarity"],
                "contexts": rec["contexts"],
                "opposite_contexts": rec["opposite_contexts"],
            }, ensure_ascii=False) + "\n")

    if args.markdown:
        lines = ["# Stage-2 blinded review set", ""]
        for rec in blinded:
            lines.append(f"## {rec['blind_id']}  (layer {rec['layer']}, {rec['selected_tail_polarity']} tail)")
            lines.append("")
            for c in rec["contexts"]:
                lines.append(f"- a={c['activation']:.3g} :: {c['context_marked']}")
            if rec["opposite_contexts"]:
                lines.append("")
                lines.append("  opposite polarity:")
                for c in rec["opposite_contexts"]:
                    lines.append(f"  - a={c['activation']:.3g} :: {c['context_marked']}")
            lines.append("")
        (out_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")

    n_by = {}
    for rec in selected:
        n_by[rec["condition"]] = n_by.get(rec["condition"], 0) + 1
    print(f"\n[out] {out_dir}/candidates.csv, key.csv, contexts_blind.jsonl"
          + (", review.md" if args.markdown else ""))
    print("[selected]", ", ".join(f"{k}={v}" for k, v in sorted(n_by.items())))


if __name__ == "__main__":
    main()
