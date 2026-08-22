#!/usr/bin/env python3
"""Build a self-contained browser dashboard for the FineWeb SV scan.

The source context file is intentionally large. This script keeps the highest
ranked candidates, prunes each context to the fields used by the UI, and embeds
the result in one HTML file that can be opened directly from disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "unrealized_words_fineweb"
PRIMARY_RANK = "rank_global_mean_abs_cosine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing sv_rankings.csv, top_contexts.jsonl, and metadata.json",
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
        help="Number of globally top-ranked SVs to embed; 0 embeds all (default: 250)",
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


def load_rankings(path: Path, top_n: int) -> tuple[list[dict[str, Any]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "candidate" not in reader.fieldnames or PRIMARY_RANK not in reader.fieldnames:
            raise SystemExit(f"{path} is missing candidate or {PRIMARY_RANK}")
        rows: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, 2):
            try:
                row = {key: (value if key == "candidate" else number(value)) for key, value in raw.items()}
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric field on {path}:{row_number}: {exc}") from exc
            rows.append(row)

    rows.sort(key=lambda row: int(row[PRIMARY_RANK]))
    total = len(rows)
    if top_n < 0:
        raise SystemExit("--top must be 0 or greater")
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


def load_contexts(path: Path, candidates: set[str]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], int]:
    contexts = {candidate: {"positive": [], "negative": []} for candidate in candidates}
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
            if candidate not in contexts:
                continue
            polarity = raw.get("polarity")
            if polarity not in ("positive", "negative"):
                raise SystemExit(f"Unexpected polarity {polarity!r} on {path}:{line_number}")
            contexts[candidate][polarity].append(compact_context(raw))

    for by_polarity in contexts.values():
        for values in by_polarity.values():
            values.sort(key=lambda item: int(item.get("rank") or 0))
    return contexts, source_total


def safe_script_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_payload(data_dir: Path, top_n: int) -> dict[str, Any]:
    rankings_path = data_dir / "sv_rankings.csv"
    contexts_path = data_dir / "top_contexts.jsonl"
    metadata_path = data_dir / "metadata.json"
    for path in (rankings_path, contexts_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rankings, total_candidates = load_rankings(rankings_path, top_n)
    contexts, total_contexts = load_contexts(contexts_path, {row["candidate"] for row in rankings})

    expected = int(metadata.get("top_contexts_per_polarity", 0))
    incomplete: list[str] = []
    for candidate, by_polarity in contexts.items():
        if expected and any(len(by_polarity[polarity]) != expected for polarity in ("positive", "negative")):
            incomplete.append(candidate)
    if incomplete:
        preview = ", ".join(incomplete[:8])
        raise SystemExit(f"Incomplete context sets for {len(incomplete)} candidates: {preview}")

    return {
        "meta": {
            "model": metadata.get("model"),
            "dataset": metadata.get("dataset"),
            "dataset_config": metadata.get("dataset_config"),
            "documents": metadata.get("documents_processed"),
            "tokens": metadata.get("content_tokens_processed"),
            "layers": metadata.get("layers", []),
            "k": metadata.get("k"),
            "primary_sort": metadata.get("primary_csv_sort", PRIMARY_RANK),
            "contexts_per_polarity": expected,
            "total_candidates": total_candidates,
            "embedded_candidates": len(rankings),
            "total_contexts": total_contexts,
        },
        "rankings": rankings,
        "contexts": contexts,
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
      --shadow: 0 14px 40px rgba(42, 49, 44, .08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --serif: Iowan Old Style, Baskerville, Georgia, serif;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; color: var(--ink); background: var(--paper); font: 14px/1.5 var(--sans); }
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
    .rank-number { padding-top: 2px; color: var(--muted); font: 11px/1.2 var(--mono); text-align: right; }
    .rank-row.active .rank-number { color: var(--green); font-weight: 800; }
    .rank-topline { display: flex; align-items: baseline; justify-content: space-between; gap: 9px; }
    .candidate { font: 700 13px/1.3 var(--mono); letter-spacing: .01em; }
    .layer-note { color: var(--muted); font-size: 10px; white-space: nowrap; }
    .rank-measure { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 6px; color: var(--muted); font-size: 10px; }
    .rank-measure b { color: var(--ink-2); font: 600 10px/1 var(--mono); }
    .mini-track { display: block; height: 3px; margin-top: 7px; overflow: hidden; background: #e5e0d5; }
    .mini-track i { display: block; height: 100%; background: var(--green); }
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
    .detail-summary { max-width: 820px; margin: 10px 0 0; color: var(--muted); }
    .detail-summary b { color: var(--ink-2); }
    .metric-grid { display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr)); gap: 1px; margin-top: 24px; border: 1px solid var(--line); background: var(--line); }
    .metric-card { min-width: 0; background: rgba(255,253,248,.86); padding: 12px; }
    .metric-card span { display: block; min-height: 28px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .metric-card b { display: block; margin-top: 4px; overflow: hidden; color: var(--ink); font: 650 clamp(14px, 1.5vw, 18px)/1.1 var(--mono); text-overflow: ellipsis; }

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
    .all-metric { display: flex; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px dotted var(--line); }
    .all-metric span { min-width: 0; color: var(--muted); font-size: 10px; }
    .all-metric b { flex: 0 0 auto; font: 600 10px/1.4 var(--mono); }

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
    .column-head h4 { margin: 0; font: 600 13px/1.2 var(--mono); }
    .column-head span { color: var(--muted); font-size: 10px; }
    .context-list { display: grid; gap: 8px; }
    .context-card { overflow: hidden; border: 1px solid var(--line); background: var(--surface); }
    .context-meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; border-bottom: 1px solid #ebe7de; background: var(--surface-2); padding: 8px 11px; }
    .context-rank { color: var(--muted); font: 650 9px/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; }
    .context-values { display: flex; gap: 10px; color: var(--muted); font: 9px/1.2 var(--mono); }
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
      .all-metrics { grid-template-columns: repeat(2, 1fr); }
      .toolbar-inner { grid-template-columns: minmax(210px, 1.3fr) repeat(3, minmax(125px, .55fr)) auto; }
    }
    @media (max-width: 880px) {
      .masthead-inner { align-items: flex-start; flex-direction: column; min-height: 0; }
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
      .all-metrics { grid-template-columns: 1fr; }
      .context-grid { grid-template-columns: 1fr; }
      .context-controls { grid-template-columns: 1fr 1fr; }
      .context-controls .context-search { grid-column: 1 / -1; }
      .detail-hero, .context-section { padding: 20px 16px; }
      .section-heading { align-items: flex-start; flex-direction: column; }
    }
    @media print {
      .toolbar, .ranking-panel, .nav-buttons, .context-controls, .footer { display: none !important; }
      .masthead { color: var(--ink); background: white; border-color: var(--ink); }
      .masthead * { color: var(--ink) !important; }
      .masthead-inner { min-height: 0; padding: 18px 0; }
      .dashboard-grid { display: block; }
      .panel { box-shadow: none; }
      .context-card { break-inside: avoid; }
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
          <p class="subtitle">Browse the strongest cross-layer directions and read the text contexts that most activate each side of the singular vector.</p>
        </div>
      </div>
      <div class="dataset-facts" id="datasetFacts" aria-label="Dataset summary"></div>
    </div>
  </header>

  <section class="toolbar" aria-label="Ranking controls">
    <div class="toolbar-inner">
      <label class="control search-control">Find a direction
        <span class="search-wrap"><input id="candidateSearch" type="search" placeholder="Candidate ID, layer, or SV rank…" autocomplete="off"><kbd class="shortcut">/</kbd></span>
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
          <div class="panel-head-row"><h2>Ranked directions</h2><span class="count" id="rankingCount"></span></div>
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

    const METRICS = {
      mean_abs_cosine: { label: "Mean |cosine|", short: "mean |cos|", kind: "decimal", help: "Mean absolute activation divided by residual-stream norm; the default cross-layer score." },
      top1_abs_rate: { label: "Top-1 token share", short: "top-1 share", kind: "percent", help: "Fraction of tokens where this direction has the largest absolute activation in the layer's top-k bank." },
      top5_abs_rate: { label: "Top-5 token share", short: "top-5 share", kind: "percent", help: "Fraction of tokens where this direction is among the five largest absolute activations." },
      sigma_weighted_mean_abs: { label: "σ × mean |activation|", short: "σ × mean |act|", kind: "number", help: "Singular value multiplied by mean absolute activation." },
      std_activation: { label: "Activation variability", short: "activation std", kind: "number", help: "Standard deviation across token activations." },
      singular_value: { label: "Singular value", short: "singular value", kind: "number", help: "Singular value for this direction within its layer." },
      dynamicity_std_over_abs_mean: { label: "Dynamicity", short: "dynamicity", kind: "decimal", help: "Activation standard deviation divided by mean absolute activation." }
    };

    const LABELS = {
      candidate: "Candidate", layer: "Layer", sv_index_0: "SV index (0-based)", sv_rank_1based: "SV rank (within layer)",
      singular_value: "Singular value", singular_value_over_sv1: "Singular value / SV1", n_tokens: "Tokens", n_documents: "Documents",
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
      rank_layer_top5_abs_rate: "Layer rank · top-5 rate", rank_layer_std_activation: "Layer rank · activation std"
    };

    const state = {
      query: "",
      layer: "all",
      metric: "mean_abs_cosine",
      limit: Math.min(50, DATA.rankings.length),
      selected: null,
      contextQuery: "",
        contextLimit: Math.min(6, DATA.meta.contexts_per_polarity || 6),
      dedupe: false
    };

    const byCandidate = new Map(DATA.rankings.map(row => [row.candidate, row]));
    const $ = selector => document.querySelector(selector);
    const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
    const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    const integer = new Intl.NumberFormat();

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
      return kind === "percent" ? percent(value) : decimal(value, kind === "decimal" ? 4 : 3);
    }

    function ordinal(value) {
      const n = Number(value);
      const mod100 = n % 100;
      if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
      return `${n}${({1:"st",2:"nd",3:"rd"})[n % 10] || "th"}`;
    }

    function setup() {
      const meta = DATA.meta;
      $("#datasetFacts").innerHTML = [
        [compact.format(meta.tokens), "tokens"],
        [integer.format(meta.documents), "documents"],
        [integer.format(meta.layers.length), "layers"],
        [integer.format(meta.total_candidates), "directions"]
      ].map(([value, label]) => `<div class="fact"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join("");

      const layers = [...new Set(DATA.rankings.map(row => Number(row.layer)))].sort((a, b) => a - b);
      $("#layerFilter").innerHTML = `<option value="all">All embedded layers</option>` + layers.map(layer => `<option value="${layer}">Layer ${String(layer).padStart(2, "0")}</option>`).join("");
      $("#sortMetric").innerHTML = Object.entries(METRICS).map(([key, item]) => `<option value="${key}">${esc(item.label)}</option>`).join("");
      const limits = [25, 50, 100, 250].filter(value => value < DATA.rankings.length);
      limits.push(DATA.rankings.length);
      $("#rowLimit").innerHTML = [...new Set(limits)].map(value => `<option value="${value}">${value === DATA.rankings.length ? `All ${integer.format(value)}` : `Top ${value}`}</option>`).join("");

      const hashCandidate = decodeURIComponent(location.hash.slice(1));
      state.selected = byCandidate.has(hashCandidate) ? hashCandidate : DATA.rankings[0]?.candidate;
      if (byCandidate.has(hashCandidate)) expandLimitToCandidate(hashCandidate);
      $("#rowLimit").value = String(state.limit);

      $("#candidateSearch").addEventListener("input", event => { state.query = event.target.value.trim().toLowerCase(); render(); });
      $("#layerFilter").addEventListener("change", event => { state.layer = event.target.value; render(); });
      $("#sortMetric").addEventListener("change", event => { state.metric = event.target.value; render(); });
      $("#rowLimit").addEventListener("change", event => { state.limit = Number(event.target.value); render(); });
      $("#resetFilters").addEventListener("click", resetFilters);
      addEventListener("hashchange", () => {
        const candidate = decodeURIComponent(location.hash.slice(1));
        if (byCandidate.has(candidate) && candidate !== state.selected) revealCandidate(candidate);
      });
      addEventListener("keydown", event => {
        const tag = document.activeElement?.tagName;
        if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(tag)) {
          event.preventDefault();
          $("#candidateSearch").focus();
        }
        if (event.key === "Escape" && document.activeElement === $("#candidateSearch")) {
          $("#candidateSearch").value = "";
          state.query = "";
          render();
          $("#candidateSearch").blur();
        }
      });

      $("#footerNote").innerHTML = `Self-contained snapshot of the top <b>${integer.format(meta.embedded_candidates)}</b> of ${integer.format(meta.total_candidates)} candidates from <code>sv_rankings.csv</code>, with ${integer.format(meta.contexts_per_polarity)} contexts per polarity from <code>top_contexts.jsonl</code>. Model: <code>${esc(meta.model)}</code>.`;
      render();
    }

    function resetFilters() {
      state.query = "";
      state.layer = "all";
      state.metric = "mean_abs_cosine";
      state.limit = Math.min(50, DATA.rankings.length);
      $("#candidateSearch").value = "";
      $("#layerFilter").value = "all";
      $("#sortMetric").value = state.metric;
      $("#rowLimit").value = String(state.limit);
      render();
    }

    function filteredRows(applyLimit = true) {
      const query = state.query;
      const rows = DATA.rankings.filter(row => {
        if (state.layer !== "all" && Number(row.layer) !== Number(state.layer)) return false;
        if (!query) return true;
        const haystack = `${row.candidate} layer ${row.layer} l${String(row.layer).padStart(2,"0")} sv ${row.sv_rank_1based} sv${String(row.sv_rank_1based).padStart(2,"0")}`.toLowerCase();
        return haystack.includes(query);
      }).sort((a, b) => Number(b[state.metric]) - Number(a[state.metric]) || Number(a[PRIMARY_RANK]) - Number(b[PRIMARY_RANK]));
      return applyLimit ? rows.slice(0, state.limit) : rows;
    }

    const PRIMARY_RANK = "rank_global_mean_abs_cosine";

    function render() {
      const previousSelection = state.selected;
      const allMatches = filteredRows(false);
      const rows = allMatches.slice(0, state.limit);
      if (rows.length && !rows.some(row => row.candidate === state.selected)) state.selected = rows[0].candidate;
      if (!rows.length) state.selected = null;
      if (state.selected !== previousSelection) syncHash();
      renderRankings(rows, allMatches.length);
      renderDetail(rows);
    }

    function renderRankings(rows, matchCount) {
      const metric = METRICS[state.metric];
      const maxValue = Math.max(...rows.map(row => Number(row[state.metric]) || 0), 0);
      $("#rankingCount").textContent = `${integer.format(rows.length)} / ${integer.format(matchCount)}`;
      $("#rankingCaption").textContent = `${metric.label} · descending. Global rank always refers to mean |cosine|.`;
      if (!rows.length) {
        $("#rankingList").innerHTML = `<div class="empty">No directions match these filters.</div>`;
        return;
      }
      $("#rankingList").innerHTML = rows.map(row => {
        const value = Number(row[state.metric]);
        const width = maxValue > 0 ? Math.max(2, value / maxValue * 100) : 0;
        return `<button class="rank-row ${row.candidate === state.selected ? "active" : ""}" type="button" ${row.candidate === state.selected ? 'aria-current="true"' : ""} data-candidate="${esc(row.candidate)}">
          <span class="rank-number">#${integer.format(row[PRIMARY_RANK])}</span>
          <span>
            <span class="rank-topline"><span class="candidate">${esc(row.candidate)}</span><span class="layer-note">L${String(row.layer).padStart(2,"0")} · SV${String(row.sv_rank_1based).padStart(2,"0")}</span></span>
            <span class="rank-measure"><span>${esc(metric.short)}</span><b>${formatMetric(value, metric.kind)}</b></span>
            <span class="mini-track" aria-hidden="true"><i style="width:${width.toFixed(2)}%"></i></span>
          </span>
        </button>`;
      }).join("");
      $("#rankingList").querySelectorAll(".rank-row").forEach(button => button.addEventListener("click", () => {
        selectCandidate(button.dataset.candidate, true);
        if (innerWidth <= 880) $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
      }));
    }

    function selectCandidate(candidate, updateHash = true) {
      if (!byCandidate.has(candidate)) return;
      state.selected = candidate;
      if (updateHash) syncHash();
      render();
    }

    function expandLimitToCandidate(candidate) {
      const sorted = DATA.rankings.slice().sort((a, b) => Number(b[state.metric]) - Number(a[state.metric]) || Number(a[PRIMARY_RANK]) - Number(b[PRIMARY_RANK]));
      const needed = sorted.findIndex(row => row.candidate === candidate) + 1;
      const choices = [...new Set([25, 50, 100, 250, DATA.rankings.length])].filter(value => value <= DATA.rankings.length).sort((a, b) => a - b);
      state.limit = choices.find(value => value >= needed) || DATA.rankings.length;
    }

    function revealCandidate(candidate) {
      state.query = "";
      state.layer = "all";
      $("#candidateSearch").value = "";
      $("#layerFilter").value = "all";
      expandLimitToCandidate(candidate);
      $("#rowLimit").value = String(state.limit);
      state.selected = candidate;
      render();
    }

    function syncHash() {
      const nextHash = state.selected ? `#${encodeURIComponent(state.selected)}` : "";
      if (location.hash === nextHash) return;
      try { history.replaceState(null, "", nextHash || `${location.pathname}${location.search}`); }
      catch (_) { location.hash = nextHash; }
    }

    function metricCard(label, value, title) {
      return `<div class="metric-card" title="${esc(title || "")}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
    }

    function renderDetail(visibleRows) {
      const root = $("#detail");
      if (!state.selected) {
        root.innerHTML = `<div class="empty">Choose a broader filter to inspect a direction.</div>`;
        return;
      }
      const row = byCandidate.get(state.selected);
      const selectedIndex = visibleRows.findIndex(item => item.candidate === state.selected);
      const lo = Math.min(Number(row.min_activation), 0);
      const hi = Math.max(Number(row.max_activation), 0);
      const spread = Math.max(hi - lo, 1e-9);
      const zeroPosition = (0 - lo) / spread * 100;
      const meanPosition = Math.max(0, Math.min(100, (Number(row.mean_activation) - lo) / spread * 100));
      const hiddenKeys = new Set(["candidate"]);
      const allMetrics = Object.entries(row).filter(([key]) => !hiddenKeys.has(key)).map(([key, value]) => `<div class="all-metric"><span>${esc(LABELS[key] || key.replaceAll("_", " "))}</span><b>${esc(formatAny(key, value))}</b></div>`).join("");

      root.innerHTML = `<section class="detail-hero">
        <div class="detail-nav">
          <span class="detail-rank">Global mean |cosine| rank #${integer.format(row[PRIMARY_RANK])}</span>
          <div class="nav-buttons">
            <button class="icon-button" id="previousCandidate" type="button" aria-label="Previous visible direction" title="Previous visible direction" ${selectedIndex <= 0 ? "disabled" : ""}>←</button>
            <button class="icon-button" id="nextCandidate" type="button" aria-label="Next visible direction" title="Next visible direction" ${selectedIndex < 0 || selectedIndex >= visibleRows.length - 1 ? "disabled" : ""}>→</button>
            <button class="copy-button" id="copyCandidate" type="button">Copy ID</button>
          </div>
        </div>
        <div class="title-row"><h2 class="detail-title">${esc(row.candidate)}</h2><span class="id-chip">L${String(row.layer).padStart(2,"0")} / SV${String(row.sv_rank_1based).padStart(2,"0")}</span></div>
        <p class="detail-summary">The <b>${ordinal(row.sv_rank_1based)} singular direction</b> in layer ${row.layer}. It is the <b>#${integer.format(row[PRIMARY_RANK])} direction globally</b> by mean absolute cosine over ${integer.format(row.n_tokens)} FineWeb tokens.</p>
        <div class="metric-grid">
          ${metricCard("Mean |cosine|", decimal(row.mean_abs_cosine, 4), METRICS.mean_abs_cosine.help)}
          ${metricCard("Top-1 token share", percent(row.top1_abs_rate), METRICS.top1_abs_rate.help)}
          ${metricCard("Top-5 token share", percent(row.top5_abs_rate), METRICS.top5_abs_rate.help)}
          ${metricCard("Mean |activation|", decimal(row.mean_abs_activation), "Mean magnitude of the projection coefficient over tokens.")}
          ${metricCard("Activation std", decimal(row.std_activation), METRICS.std_activation.help)}
          ${metricCard("Singular value", decimal(row.singular_value), METRICS.singular_value.help)}
        </div>
        <div class="profile">
          <div>
            <div class="profile-labels"><span>${signed(lo)}</span><span>activation range</span><span>${signed(hi)}</span></div>
            <div class="axis" aria-label="Activation range from ${signed(lo)} to ${signed(hi)}; mean ${signed(row.mean_activation)}"><i class="zero-marker" style="left:${zeroPosition.toFixed(2)}%" title="zero"></i><i class="mean-marker" style="left:${meanPosition.toFixed(2)}%" title="mean ${signed(row.mean_activation)}"></i></div>
          </div>
          <div class="profile-stat"><span>Mean<b>${signed(row.mean_activation)}</b></span><span>Positive tokens<b>${percent(row.positive_rate, Number(row.positive_rate) < .01 ? 2 : 1)}</b></span></div>
        </div>
        <details class="metric-details"><summary>Inspect all ranking metrics</summary><div class="all-metrics">${allMetrics}</div></details>
      </section>
      <section class="context-section">
        <div class="section-heading"><div><p class="eyebrow">Observed examples</p><h3>Top activation contexts</h3><p>Contexts are ordered by signed projection activation. Each highlight marks the activating token; cosine is normalized by the layer residual norm.</p><span class="sign-note">Positive and negative are arbitrary SVD orientations, not sentiment labels.</span></div></div>
        <div class="context-controls">
          <label class="control context-search">Search these contexts<input id="contextSearch" type="search" value="${esc(state.contextQuery)}" placeholder="Text, token, domain…"></label>
          <label class="control">Per side<select id="contextLimit"></select></label>
          <label class="check"><input id="dedupeSources" type="checkbox" ${state.dedupe ? "checked" : ""}> Unique sources</label>
        </div>
        <div class="context-grid">
          <section class="context-column positive"><div class="column-head"><h4>+ Positive direction</h4><span id="positiveCount"></span></div><div class="context-list" id="positiveList"></div></section>
          <section class="context-column negative"><div class="column-head"><h4>− Negative direction</h4><span id="negativeCount"></span></div><div class="context-list" id="negativeList"></div></section>
        </div>
      </section>`;

      const maxContexts = DATA.meta.contexts_per_polarity || Math.max(...Object.values(DATA.contexts[state.selected] || {}).map(items => items.length), 0);
      const contextLimits = [...new Set([6, 12, maxContexts].filter(value => value > 0 && value <= maxContexts))].sort((a, b) => a - b);
      if (!contextLimits.includes(state.contextLimit)) state.contextLimit = contextLimits[0] || maxContexts;
      $("#contextLimit").innerHTML = contextLimits.map(value => `<option value="${value}">${value === maxContexts ? `All ${value}` : value}</option>`).join("");
      $("#contextLimit").value = String(state.contextLimit);
      $("#contextSearch").addEventListener("input", event => { state.contextQuery = event.target.value.trim().toLowerCase(); renderContexts(); });
      $("#contextLimit").addEventListener("change", event => { state.contextLimit = Number(event.target.value); renderContexts(); });
      $("#dedupeSources").addEventListener("change", event => { state.dedupe = event.target.checked; renderContexts(); });
      $("#previousCandidate").addEventListener("click", () => selectedIndex > 0 && selectCandidate(visibleRows[selectedIndex - 1].candidate));
      $("#nextCandidate").addEventListener("click", () => selectedIndex >= 0 && selectedIndex < visibleRows.length - 1 && selectCandidate(visibleRows[selectedIndex + 1].candidate));
      $("#copyCandidate").addEventListener("click", copyCandidate);
      renderContexts();
    }

    function formatAny(key, value) {
      if (value == null) return "—";
      if (key.startsWith("rank_") || ["layer", "sv_index_0", "sv_rank_1based", "n_tokens", "n_documents"].includes(key)) return integer.format(value);
      if (["positive_rate", "top1_abs_rate", "top5_abs_rate", "doc_top5_presence_rate"].includes(key)) return percent(value, 2);
      return decimal(value, 5);
    }

    function copyCandidate() {
      const button = $("#copyCandidate");
      const done = () => { button.textContent = "Copied"; setTimeout(() => button.textContent = "Copy ID", 1200); };
      if (navigator.clipboard?.writeText) navigator.clipboard.writeText(state.selected).then(done).catch(() => fallbackCopy(state.selected, done));
      else fallbackCopy(state.selected, done);
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
      const grouped = DATA.contexts[state.selected] || { positive: [], negative: [] };
      renderContextColumn("positive", filterContexts(grouped.positive));
      renderContextColumn("negative", filterContexts(grouped.negative));
    }

    function renderContextColumn(polarity, filtered) {
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
      shown.forEach(item => list.append(contextCard(item, polarity)));
    }

    function contextCard(item, polarity) {
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
    output = (args.output or data_dir / "report.html").expanduser().resolve()
    payload = build_payload(data_dir, args.top)
    html = HTML_TEMPLATE.replace("__DASHBOARD_PAYLOAD__", safe_script_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    meta = payload["meta"]
    embedded_contexts = sum(
        len(group[polarity])
        for group in payload["contexts"].values()
        for polarity in ("positive", "negative")
    )
    print(f"Wrote {output}")
    print(f"Embedded {meta['embedded_candidates']:,} of {meta['total_candidates']:,} candidates and {embedded_contexts:,} contexts")
    print(f"Output size: {output.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
