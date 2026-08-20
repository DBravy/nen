#!/usr/bin/env python
"""Targeted characterization of candidate J-lens singular directions.

This is the follow-up to explore_jlens_svd_concepts_v2.py.  It does NOT search
for new modes.  Instead, it takes a frozen set of candidate (layer, SV-rank)
directions from a previous SVD run and asks what those directions are doing on
an activation corpus (FineWeb discovery by default).

Default candidates are the channels that survived the first diverse-corpus run:
    L03/SV12
    L05/SV04, L06/SV04
    L09/SV03, L10/SV03, L11/SV03, L12/SV03

For each candidate it produces:
  * dozens of record-diverse +V and -V extrema
  * a local token-by-token activation trace around every extremum
  * distribution statistics / percentiles
  * nuisance diagnostics:
      - residual-norm correlation
      - normalized-position correlation
      - token-frequency correlation
      - token-identity R^2 (how much variance is explainable by token ID alone)
      - coarse token-type R^2
      - between-document R^2
  * family analysis for L05-06/SV04 and L09-12/SV03:
      - sign alignment from V cosine
      - cross-layer activation correlations on exactly the same tokens
      - family-average z-score extrema and traces

The goal is falsification first: can a mundane lexical/formatting/position/norm
variable explain the direction?  If not, the +/- contexts become evidence for
inferring a latent semantic/computational variable.

Typical usage
-------------
  python characterize_jlens_candidates.py

Defaults assume these directories beside the script:
  diverse_corpora/fineweb_discovery/index.jsonl
  svd_fineweb_discovery/directions/LXX.npz

Useful options
--------------
  --extremes 40            # 40 positive + 40 negative contexts per candidate
  --trace-window 12        # +/- 12 tokens around the extremum
  --terminal-extremes 8    # concise terminal preview; full set goes to text file

Outputs
-------
  candidate_characterization/
    characterization.txt   # BEST FILE TO SEND BACK TO CHATGPT
    summary.csv
    extremes.csv
    family_summary.csv
    family_extremes.csv
    meta.json

No generations are sampled. Exact stored corpus text is replayed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import jlens
from jlens import ActivationRecorder


@dataclass(frozen=True)
class Candidate:
    layer: int
    sv_rank: int

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}_SV{self.sv_rank:02d}"

    @property
    def label(self) -> str:
        return f"L{self.layer:02d}/SV{self.sv_rank:02d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--readouts", default=str(here / "diverse_corpora" / "fineweb_discovery"))
    p.add_argument("--directions", default=str(here / "svd_fineweb_discovery" / "directions"),
                   help="directory containing LXX.npz from explore_jlens_svd_concepts_v2.py")
    p.add_argument("--out", default=str(here / "candidate_characterization"))
    p.add_argument("--candidates", default="3:12,5:4,6:4,9:3,10:3,11:3,12:3",
                   help="comma-separated layer:svrank pairs")
    p.add_argument(
        "--families",
        default="sv04_early=5:4,6:4;sv03_mid=9:3,10:3,11:3,12:3",
        help="semicolon-separated name=layer:rank,... groups; blank disables family analysis",
    )
    p.add_argument("--max-records", type=int, default=0, help="0 = all records")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--include-special", action="store_true")
    p.add_argument("--extremes", type=int, default=30,
                   help="record-diverse extrema per sign written to characterization.txt")
    p.add_argument("--trace-window", type=int, default=12)
    p.add_argument("--context-window", type=int, default=24)
    p.add_argument("--terminal-extremes", type=int, default=8,
                   help="extrema per sign printed to terminal; full set is saved")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_candidates(s: str) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[tuple[int, int]] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split(":", 1)
            c = Candidate(int(a), int(b))
        except Exception as e:
            raise SystemExit(f"bad candidate {part!r}; expected layer:svrank") from e
        if c.sv_rank < 1:
            raise SystemExit(f"SV ranks are 1-based; got {part!r}")
        if (c.layer, c.sv_rank) not in seen:
            out.append(c)
            seen.add((c.layer, c.sv_rank))
    if not out:
        raise SystemExit("no candidates selected")
    return out


def parse_families(s: str) -> dict[str, list[Candidate]]:
    ans: dict[str, list[Candidate]] = {}
    if not s.strip():
        return ans
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"bad family {chunk!r}; expected name=L:SV,L:SV")
        name, members = chunk.split("=", 1)
        ans[name.strip()] = parse_candidates(members)
    return ans


def load_records(readouts: Path, max_records: int) -> list[dict[str, Any]]:
    p = readouts / "index.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("text")]
    if max_records > 0:
        rows = rows[:max_records]
    if not rows:
        raise SystemExit(f"no usable records in {p}")
    return rows


def load_directions(direction_dir: Path, candidates: list[Candidate]) -> tuple[dict[Candidate, np.ndarray], dict[Candidate, float]]:
    V: dict[Candidate, np.ndarray] = {}
    S: dict[Candidate, float] = {}
    cache: dict[int, Any] = {}
    for c in candidates:
        if c.layer not in cache:
            p = direction_dir / f"L{c.layer:02d}.npz"
            if not p.exists():
                raise SystemExit(f"missing {p}; point --directions at the v2 SVD directions folder")
            cache[c.layer] = np.load(p)
        z = cache[c.layer]
        vmat = z["V"]
        ss = z["S"]
        j = c.sv_rank - 1
        if j >= vmat.shape[1]:
            raise SystemExit(f"{c.label}: requested rank exceeds saved k={vmat.shape[1]}")
        v = np.asarray(vmat[:, j], dtype=np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        V[c] = v
        S[c] = float(ss[j])
    return V, S


def safe_tok(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )


def token_category(s: str) -> str:
    if "\n" in s or "\r" in s:
        return "newline"
    stripped = s.strip()
    if not stripped:
        return "whitespace"
    if stripped.isdigit():
        return "numeric"
    if any(ch.isalpha() for ch in stripped):
        return "alpha"
    if all((not ch.isalnum()) and (not ch.isspace()) for ch in stripped):
        return "punct"
    return "other"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]; y = y[m]
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / den)


def categorical_r2(values: np.ndarray, labels: list[Any]) -> float:
    """Fraction of variance explained by replacing each sample with its category mean."""
    x = np.asarray(values, dtype=np.float64)
    mu = float(x.mean())
    total = float(np.sum((x - mu) ** 2))
    if total <= 1e-30:
        return 0.0
    sums: dict[Any, float] = defaultdict(float)
    counts: dict[Any, int] = defaultdict(int)
    for v, lab in zip(x.tolist(), labels):
        sums[lab] += float(v)
        counts[lab] += 1
    between = 0.0
    for lab, n in counts.items():
        m = sums[lab] / n
        between += n * (m - mu) ** 2
    return float(max(0.0, min(1.0, between / total)))


def record_diverse_extrema(per_record: list[np.ndarray], n: int, positive: bool) -> list[tuple[float, int, int]]:
    """One extremum per record, then global top/bottom N."""
    rows: list[tuple[float, int, int]] = []
    for ri, a in enumerate(per_record):
        if a.size == 0:
            continue
        pi = int(np.argmax(a) if positive else np.argmin(a))
        rows.append((float(a[pi]), ri, pi))
    rows.sort(key=lambda t: t[0], reverse=positive)
    return rows[:n]


def context_string(tokens: list[str], pos: int, window: int) -> str:
    lo = max(0, pos - window)
    hi = min(len(tokens), pos + window + 1)
    parts = []
    for i in range(lo, hi):
        t = safe_tok(tokens[i])
        parts.append(f"⟦{t}⟧" if i == pos else t)
    return "".join(parts)


def trace_string(tokens: list[str], vals: np.ndarray, pos: int, mean: float, std: float, window: int) -> str:
    lo = max(0, pos - window)
    hi = min(len(tokens), pos + window + 1)
    std = max(std, 1e-12)
    parts = []
    for i in range(lo, hi):
        z = (float(vals[i]) - mean) / std
        body = f"{safe_tok(tokens[i])}{{{z:+.2f}}}"
        parts.append(f"⟦{body}⟧" if i == pos else body)
    return " ".join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    candidates = parse_candidates(args.candidates)
    families = parse_families(args.families)
    # Family members must also be collected even if omitted from --candidates.
    all_candidates = list(candidates)
    seen = {(c.layer, c.sv_rank) for c in all_candidates}
    for members in families.values():
        for c in members:
            if (c.layer, c.sv_rank) not in seen:
                all_candidates.append(c); seen.add((c.layer, c.sv_rank))

    readouts = Path(args.readouts).expanduser()
    direction_dir = Path(args.directions).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(readouts, args.max_records)
    V, sigma = load_directions(direction_dir, all_candidates)
    layers = sorted({c.layer for c in all_candidates})

    print(f"[setup] records={len(records)} candidates={','.join(c.label for c in candidates)}")
    print(f"[setup] layers={layers}")
    for name, members in families.items():
        print(f"[family] {name}: " + " -> ".join(c.label for c in members))

    print(f"[load] {args.model}")
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype="auto", device_map="cuda"
    )
    model = jlens.from_hf(hf, tok)
    if next(iter(V.values())).shape[0] != model.d_model:
        raise SystemExit(f"direction d={next(iter(V.values())).shape[0]} != model d={model.d_model}")

    # GPU vectors grouped by layer: [d_model, n_candidates_at_layer]
    by_layer: dict[int, list[Candidate]] = defaultdict(list)
    for c in all_candidates:
        by_layer[c.layer].append(c)
    V_gpu: dict[int, torch.Tensor] = {}
    for layer, cs in by_layer.items():
        V_gpu[layer] = torch.from_numpy(np.stack([V[c] for c in cs], axis=1)).to("cuda")

    special_ids = set(getattr(tok, "all_special_ids", []) or [])

    # Per record we keep only what is needed for later contextualization.
    record_data: list[dict[str, Any]] = []
    per_candidate: dict[Candidate, list[np.ndarray]] = {c: [] for c in all_candidates}
    per_norm: dict[Candidate, list[np.ndarray]] = {c: [] for c in all_candidates}
    token_freq: Counter[int] = Counter()

    print("[act] replaying frozen corpus; collecting candidate projections")
    for ri, rec in enumerate(records, 1):
        pid = str(rec.get("pid", f"record_{ri:04d}"))
        input_ids = model.encode(str(rec["text"]), max_length=args.max_seq_len)
        ids_all = input_ids[0].detach().cpu().tolist()
        keep = np.ones(len(ids_all), dtype=bool)
        if not args.include_special and special_ids:
            keep &= np.array([int(t) not in special_ids for t in ids_all], dtype=bool)
        keep_idx = np.flatnonzero(keep)
        if len(keep_idx) == 0:
            continue

        ids = [int(ids_all[i]) for i in keep_idx.tolist()]
        tokens = [tok.decode([t]) for t in ids]
        for t in ids:
            token_freq[t] += 1

        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)

        data = {
            "pid": pid,
            "ids": ids,
            "tokens": tokens,
            "orig_pos": keep_idx.astype(np.int32),
            "n": len(ids),
        }
        record_data.append(data)

        for layer, cs in by_layer.items():
            h = recorder.activations[layer][0].detach()[keep_idx].float()
            M = V_gpu[layer].to(dtype=h.dtype, device=h.device)
            P = (h @ M).detach().cpu().numpy().astype(np.float32)
            norms = h.norm(dim=1).detach().cpu().numpy().astype(np.float32)
            for j, c in enumerate(cs):
                per_candidate[c].append(P[:, j].copy())
                per_norm[c].append(norms.copy())
            del h, P, norms
        del recorder

        if ri <= 5 or ri % 25 == 0 or ri == len(records):
            print(f"[act] {ri:03d}/{len(records)} {pid:30s} kept={len(ids):4d}")

    if not record_data:
        raise SystemExit("no activation records collected")

    # Shared metadata arrays across all candidates.
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in record_data])
    all_posfrac = np.concatenate([
        np.linspace(0.0, 1.0, int(r["n"]), dtype=np.float32) if int(r["n"]) > 1 else np.zeros(1, dtype=np.float32)
        for r in record_data
    ])
    all_abpos = np.concatenate([np.arange(int(r["n"]), dtype=np.float32) for r in record_data])
    all_token_strs = [t for r in record_data for t in r["tokens"]]
    all_types = [token_category(t) for t in all_token_strs]
    logfreq = np.log1p(np.asarray([token_freq[int(t)] for t in all_ids], dtype=np.float64))
    record_labels = [ri for ri, r in enumerate(record_data) for _ in range(int(r["n"]))]

    summary_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    report: list[str] = []
    report.append("TARGETED J-LENS CANDIDATE CHARACTERIZATION")
    report.append(f"records={len(record_data)}  positions={len(all_ids)}")
    report.append(f"readouts={readouts}")
    report.append(f"directions={direction_dir}\n")

    candidate_flat: dict[Candidate, np.ndarray] = {}
    candidate_z_flat: dict[Candidate, np.ndarray] = {}

    for c in candidates:
        vals = np.concatenate(per_candidate[c]).astype(np.float64)
        norms = np.concatenate(per_norm[c]).astype(np.float64)
        candidate_flat[c] = vals
        mean = float(vals.mean())
        std = float(vals.std())
        zvals = (vals - mean) / max(std, 1e-12)
        candidate_z_flat[c] = zvals

        q = np.quantile(vals, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
        row = {
            "candidate": c.label,
            "layer": c.layer,
            "sv_rank": c.sv_rank,
            "sigma": sigma[c],
            "mean": mean,
            "std": std,
            "q001": float(q[0]), "q01": float(q[1]), "q05": float(q[2]),
            "median": float(q[3]), "q95": float(q[4]), "q99": float(q[5]), "q999": float(q[6]),
            "corr_resid_norm": pearson(vals, norms),
            "corr_position_frac": pearson(vals, all_posfrac),
            "corr_position_abs": pearson(vals, all_abpos),
            "corr_log_token_frequency": pearson(vals, logfreq),
            "token_id_r2": categorical_r2(vals, all_ids.tolist()),
            "token_type_r2": categorical_r2(vals, all_types),
            "document_r2": categorical_r2(vals, record_labels),
        }
        summary_rows.append(row)

        header = (
            f"\n=== {c.label} ===\n"
            f"sigma={sigma[c]:.4g} mean={mean:+.4f} std={std:.4f}\n"
            f"nuisance: resid_norm_r={row['corr_resid_norm']:+.3f}  "
            f"posfrac_r={row['corr_position_frac']:+.3f}  "
            f"logtokfreq_r={row['corr_log_token_frequency']:+.3f}  "
            f"tokenID_R2={row['token_id_r2']:.3f}  tokenType_R2={row['token_type_r2']:.3f}  "
            f"document_R2={row['document_r2']:.3f}\n"
            f"quantiles: .1%={q[0]:+.3f} 1%={q[1]:+.3f} 5%={q[2]:+.3f} "
            f"50%={q[3]:+.3f} 95%={q[4]:+.3f} 99%={q[5]:+.3f} 99.9%={q[6]:+.3f}"
        )
        report.append(header)
        print(header)

        pos_ext = record_diverse_extrema(per_candidate[c], args.extremes, True)
        neg_ext = record_diverse_extrema(per_candidate[c], args.extremes, False)

        for sign_name, ext in (("POSITIVE", pos_ext), ("NEGATIVE", neg_ext)):
            report.append(f"\n-- {sign_name} EXTREMA (max one per document) --")
            term_lines: list[str] = []
            for rank, (val, ri, pi) in enumerate(ext, 1):
                rd = record_data[ri]
                z = (val - mean) / max(std, 1e-12)
                ctx = context_string(rd["tokens"], pi, args.context_window)
                tr = trace_string(rd["tokens"], per_candidate[c][ri], pi, mean, std, args.trace_window)
                line = f"{rank:02d}. z={z:+.2f} raw={val:+.3f} | {rd['pid']} @{int(rd['orig_pos'][pi])} | {ctx}"
                report.append(line)
                report.append("    trace_z: " + tr)
                extreme_rows.append({
                    "candidate": c.label, "sign": sign_name.lower(), "rank": rank,
                    "z": z, "raw": val, "pid": rd["pid"],
                    "position": int(rd["orig_pos"][pi]), "kept_position": pi,
                    "token": safe_tok(rd["tokens"][pi]), "context": ctx, "trace_z": tr,
                })
                if rank <= args.terminal_extremes:
                    term_lines.append(line)
            print(f"[{c.label} {sign_name.lower()}]")
            for line in term_lines:
                print("  " + line)

    # Family analysis: align signs from source-side V geometry, then compare the
    # actual activation trajectories and aggregate them into a family signal.
    family_summary_rows: list[dict[str, Any]] = []
    family_extreme_rows: list[dict[str, Any]] = []
    report.append("\n\n================ FAMILY ANALYSIS ================")
    print("\n================ FAMILY ANALYSIS ================")

    for name, members in families.items():
        # Only members we actually collected.
        members = [c for c in members if c in per_candidate]
        if len(members) < 2:
            continue
        ref = members[0]
        signs: dict[Candidate, float] = {ref: 1.0}
        vcos: dict[Candidate, float] = {ref: 1.0}
        for c in members[1:]:
            cos = float(np.dot(V[ref], V[c]))
            signs[c] = 1.0 if cos >= 0 else -1.0
            vcos[c] = abs(cos)

        aligned_z: dict[Candidate, np.ndarray] = {}
        for c in members:
            vals = np.concatenate(per_candidate[c]).astype(np.float64) * signs[c]
            aligned_z[c] = (vals - vals.mean()) / max(float(vals.std()), 1e-12)

        corr = np.eye(len(members), dtype=np.float64)
        for i, a in enumerate(members):
            for j, b in enumerate(members):
                if j > i:
                    corr[i, j] = corr[j, i] = pearson(aligned_z[a], aligned_z[b])

        agg = np.mean(np.stack([aligned_z[c] for c in members], axis=1), axis=1)
        # Normalize aggregate itself for readable traces/extrema.
        agg = (agg - agg.mean()) / max(float(agg.std()), 1e-12)

        family_summary_rows.append({
            "family": name,
            "members": ",".join(c.label for c in members),
            "ref": ref.label,
            "mean_pair_activation_corr": float(np.mean(corr[np.triu_indices(len(members), 1)])),
            "min_pair_activation_corr": float(np.min(corr[np.triu_indices(len(members), 1)])),
            "mean_vcos_to_ref": float(np.mean([vcos[c] for c in members])),
        })

        lines = [f"\n=== FAMILY {name}: " + " -> ".join(c.label for c in members) + " ==="]
        lines.append("sign alignment to " + ref.label + ": " + ", ".join(
            f"{c.label}:sign={int(signs[c]):+d},|Vcos|={vcos[c]:.3f}" for c in members
        ))
        lines.append("activation correlation matrix (sign-aligned):")
        lines.append("             " + " ".join(f"{c.label:>10s}" for c in members))
        for i, c in enumerate(members):
            lines.append(f"{c.label:>11s} " + " ".join(f"{corr[i,j]:10.3f}" for j in range(len(members))))
        report.extend(lines)
        print("\n".join(lines))

        # Split flat family aggregate back into per-record arrays.
        agg_records: list[np.ndarray] = []
        off = 0
        for rd in record_data:
            n = int(rd["n"])
            agg_records.append(agg[off:off+n])
            off += n

        for sign_name, ext in (("POSITIVE", record_diverse_extrema(agg_records, args.extremes, True)),
                               ("NEGATIVE", record_diverse_extrema(agg_records, args.extremes, False))):
            report.append(f"\n-- FAMILY {name} {sign_name} EXTREMA --")
            term_lines: list[str] = []
            for rank, (val, ri, pi) in enumerate(ext, 1):
                rd = record_data[ri]
                ctx = context_string(rd["tokens"], pi, args.context_window)
                # Aggregate trace is already z-scaled.
                tr = trace_string(rd["tokens"], agg_records[ri], pi, 0.0, 1.0, args.trace_window)
                member_here = []
                for c in members:
                    # Per-record values, sign-aligned then standardized globally.
                    raw_arr = per_candidate[c][ri].astype(np.float64) * signs[c]
                    all_aligned_raw = np.concatenate(per_candidate[c]).astype(np.float64) * signs[c]
                    mz = (float(raw_arr[pi]) - float(all_aligned_raw.mean())) / max(float(all_aligned_raw.std()), 1e-12)
                    member_here.append(f"{c.label}:{mz:+.2f}")
                line = (
                    f"{rank:02d}. family_z={val:+.2f} [{', '.join(member_here)}] | "
                    f"{rd['pid']} @{int(rd['orig_pos'][pi])} | {ctx}"
                )
                report.append(line)
                report.append("    family_trace_z: " + tr)
                family_extreme_rows.append({
                    "family": name, "sign": sign_name.lower(), "rank": rank,
                    "family_z": val, "pid": rd["pid"], "position": int(rd["orig_pos"][pi]),
                    "token": safe_tok(rd["tokens"][pi]), "member_z": ",".join(member_here),
                    "context": ctx, "trace_z": tr,
                })
                if rank <= args.terminal_extremes:
                    term_lines.append(line)
            print(f"[family {name} {sign_name.lower()}]")
            for line in term_lines:
                print("  " + line)

    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "extremes.csv", extreme_rows)
    write_csv(out / "family_summary.csv", family_summary_rows)
    write_csv(out / "family_extremes.csv", family_extreme_rows)
    (out / "characterization.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "model": args.model,
        "model_revision": args.model_revision,
        "readouts": str(readouts),
        "directions": str(direction_dir),
        "records": len(record_data),
        "positions": int(len(all_ids)),
        "candidates": [c.label for c in candidates],
        "families": {k: [c.label for c in v] for k, v in families.items()},
        "extremes_per_sign": args.extremes,
        "trace_window": args.trace_window,
        "context_window": args.context_window,
    }, indent=2), encoding="utf-8")

    print("\n[done]")
    print(f"  {out / 'characterization.txt'}  <-- send this back")
    print(f"  {out / 'summary.csv'}")
    print(f"  {out / 'extremes.csv'}")
    print(f"  {out / 'family_summary.csv'}")
    print(f"  {out / 'family_extremes.csv'}")


if __name__ == "__main__":
    main()
