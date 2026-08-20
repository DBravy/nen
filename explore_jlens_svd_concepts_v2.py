#!/usr/bin/env python
"""Balanced J-lens SVD explorer for latent / weakly lexicalized directions.

This is a second-pass version of explore_jlens_svd_concepts.py, designed around
what the first gpt-oss-20b run revealed.

Main changes from v1
--------------------
1. RECORD-BALANCED activation usage
   A 400-token generation no longer counts ~30x more than a 13-token prompt.
   Every transcript receives equal mass. Usage is decomposed into:
      * within-record variance  : changes while moving through one transcript
      * between-record variance : stable shifts between different transcripts
      * total balanced variance : within + between

2. GROUP-SPECIFIC usage
   We separately report prompt-only, raw-generation, and chat-generation usage.
   This makes formatting/generation channels much easier to recognize.

3. BETTER lexicality null
   In addition to isotropic random final-space directions, every layer gets a
   matched null made from random directions INSIDE that layer's top-J U subspace.
   Thus "nonlexical" asks whether an actual singular mode is unusually diffuse
   even relative to other directions in the geometry used by that J operator.

4. CROSS-LAYER continuity
   Adjacent layers are compared using sign-invariant cosine overlap for both V
   (source-side) and U (transported final-side) singular directions. This finds
   persistent channels and handles arbitrary SVD sign flips.

5. DIVERSE extreme contexts
   Top +V/-V examples are chosen at most once per transcript before global
   ranking, preventing a long generation from occupying every example slot.

6. LAYER-NORMALIZED candidate ranking
   The global list no longer ranks raw sigma values across depth. Gain and usage
   are ranked within each layer; matched-subspace nonlexicality is the third
   factor. Raw sigma / transported signal are still preserved as diagnostics.

For J = U diag(S) V^T:
  V[:,i]  = source/intermediate-layer singular direction
  U[:,i]  = transported final-layer singular direction
  S[i]    = downstream gain

Designed to live beside collect_readouts.py and reuse ./readouts/index.jsonl.
No responses are regenerated: exact stored transcript text is replayed.

Typical usage
-------------
  python explore_jlens_svd_concepts_v2.py

Useful comparison:
  python explore_jlens_svd_concepts_v2.py --operator raw   --out svd_v2_raw
  python explore_jlens_svd_concepts_v2.py --operator delta --out svd_v2_delta

Outputs
-------
  <out>/report.html
  <out>/candidates.csv
  <out>/directions.csv
  <out>/layer_summary.csv
  <out>/mode_links.csv
  <out>/meta.json
  <out>/directions/LXX.npz

The terminal summary is intentionally verbose enough to paste back into ChatGPT.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import jlens
from jlens import ActivationRecorder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--readouts", default=str(here / "readouts"))
    p.add_argument("--out", default=str(here / "svd_concepts_v2"))
    p.add_argument("--layers", default="",
                   help="comma-separated fitted layer ids; default = all fitted source layers")
    p.add_argument("--only", default="",
                   help="comma-separated pids/base-pids to include")
    p.add_argument("--exclude", default="",
                   help="comma-separated pids/base-pids to exclude")
    p.add_argument("--max-records", type=int, default=0,
                   help="0 = all selected records")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--position-mode", choices=["all", "last"], default="all")
    p.add_argument("--include-special", action="store_true")

    p.add_argument("--k", type=int, default=32)
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--svd-iters", type=int, default=2)
    p.add_argument("--exact-svd", action="store_true")
    p.add_argument("--operator", choices=["raw", "delta"], default="raw")

    p.add_argument("--decode-topk", type=int, default=8)
    p.add_argument("--context-topk", type=int, default=6)
    p.add_argument("--context-window", type=int, default=8)
    p.add_argument("--isotropic-baseline", type=int, default=128,
                   help="global isotropic final-space lexicality null")
    p.add_argument("--subspace-baseline", type=int, default=128,
                   help="per-layer random directions in span(top-k U), the primary lexicality null")
    p.add_argument("--lex-batch", type=int, default=32,
                   help="batch size when decoding random lexicality null directions")

    p.add_argument("--terminal-top", type=int, default=18)
    p.add_argument("--terminal-persistent", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_set(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def pid_matches(pid: str, names: set[str]) -> bool:
    if not names:
        return False
    base = pid.split("__")[0]
    return pid in names or base in names


def load_records(readouts: Path, only: set[str], exclude: set[str], max_records: int) -> list[dict[str, Any]]:
    path = readouts / "index.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; point --readouts at collect_readouts.py output")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if r.get("text")]
    if only:
        recs = [r for r in recs if pid_matches(str(r.get("pid", "")), only)]
    if exclude:
        recs = [r for r in recs if not pid_matches(str(r.get("pid", "")), exclude)]
    if max_records > 0:
        recs = recs[:max_records]
    if not recs:
        raise SystemExit("no readout records selected")
    return recs


def record_group(rec: dict[str, Any]) -> str:
    """Coarse group that can be inferred reliably from collect_readouts metadata."""
    if not bool(rec.get("generated", False)):
        return "prompt"
    fmt = str(rec.get("format", ""))
    if fmt == "raw":
        return "raw_gen"
    if fmt == "chat":
        return "chat_gen"
    return "generated"


def topk_svd(A: torch.Tensor, k: int, oversample: int, niter: int, exact: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U,S,V as columns, A ~= U diag(S) V^T."""
    d0, d1 = A.shape
    k = min(k, d0, d1)
    if exact:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k].T.contiguous()
    q = min(k + max(0, oversample), d0, d1)
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
    order = torch.argsort(S, descending=True)[:k]
    return U[:, order], S[order], V[:, order]


@torch.no_grad()
def lexical_stats(model, tok, dirs: torch.Tensor, decode_topk: int) -> dict[str, Any]:
    """Full lexical stats + top token strings for rows of dirs [K,d]."""
    logits = model.unembed(dirs).float()
    n_vocab = logits.shape[-1]
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
    z = (logits - mean) / std
    peak_abs_z = z.abs().amax(dim=-1)

    tk = min(decode_topk, n_vocab)
    pos_vals, pos_ids = logits.topk(tk, dim=-1)
    neg_vals, neg_ids = (-logits).topk(tk, dim=-1)

    top2 = logits.topk(min(2, n_vocab), dim=-1).values
    margin_z = ((top2[:, 0] - top2[:, 1]) / std.squeeze(-1)
                if n_vocab >= 2 else torch.zeros(logits.shape[0], device=logits.device))

    lse = torch.logsumexp(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    entropy = lse - (p * logits).sum(dim=-1)
    entropy_norm = entropy / math.log(max(2, n_vocab))
    top_mass = torch.exp(torch.logsumexp(pos_vals, dim=-1) - lse)

    def decode_rows(ids: torch.Tensor) -> list[list[str]]:
        return [[tok.decode([int(t)]) for t in row] for row in ids.detach().cpu()]

    out = {
        "peak_abs_z": peak_abs_z.detach().cpu().numpy(),
        "top1_margin_z": margin_z.detach().cpu().numpy(),
        "entropy_norm": entropy_norm.detach().cpu().numpy(),
        "topk_mass": top_mass.detach().cpu().numpy(),
        "pos_tokens": decode_rows(pos_ids),
        "neg_tokens": decode_rows(neg_ids),
        "pos_logits": pos_vals.detach().cpu().numpy(),
        "neg_logits": neg_vals.detach().cpu().numpy(),
    }
    del logits, p, z
    return out


@torch.no_grad()
def lexical_peak_z_batched(model, dirs: torch.Tensor, batch_size: int) -> np.ndarray:
    """Peak absolute vocabulary z-score only; memory-friendly for null distributions."""
    vals: list[np.ndarray] = []
    for a in range(0, dirs.shape[0], max(1, batch_size)):
        x = dirs[a:a + max(1, batch_size)]
        logits = model.unembed(x).float()
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
        peak = ((logits - mean) / std).abs().amax(dim=-1)
        vals.append(peak.detach().cpu().numpy().astype(np.float32))
        del logits, mean, std, peak
    return np.concatenate(vals) if vals else np.empty((0,), dtype=np.float32)


def percentile_nonlex(observed: np.ndarray, null_peaks: np.ndarray) -> np.ndarray:
    """P(null peak >= observed). High means unusually diffuse / weakly lexical."""
    if len(null_peaks) == 0:
        return np.full_like(observed, np.nan, dtype=np.float32)
    return np.array([(null_peaks >= x).mean() for x in observed], dtype=np.float32)


def nonlex_z(observed: np.ndarray, null_peaks: np.ndarray) -> np.ndarray:
    """Positive means lower lexical peak than null mean, in null SD units."""
    if len(null_peaks) < 2:
        return np.full_like(observed, np.nan, dtype=np.float32)
    sd = float(np.std(null_peaks))
    if sd < 1e-8:
        return np.zeros_like(observed, dtype=np.float32)
    return ((float(np.mean(null_peaks)) - observed) / sd).astype(np.float32)


def safe_token_text(s: str) -> str:
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def make_context(token_strs: list[str], pos: int, window: int) -> str:
    a = max(0, pos - window)
    b = min(len(token_strs), pos + window + 1)
    pieces = []
    for i in range(a, b):
        s = safe_token_text(token_strs[i])
        pieces.append(f"⟦{s}⟧" if i == pos else s)
    return "".join(pieces)


def rank01_desc(x: np.ndarray) -> np.ndarray:
    """Empirical [0,1] rank; 1 = largest. Ties are not specially averaged."""
    if len(x) <= 1:
        return np.ones_like(x, dtype=np.float32)
    order = np.argsort(np.argsort(x))
    return order.astype(np.float32) / (len(x) - 1)


def weighted_record_coeff_stats(P: np.ndarray, record_infos: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Equal-record decomposition: total variance = mean within-var + between mean-var."""
    means = []
    vars_ = []
    rms2s = []
    for info in record_infos:
        a, b = int(info["start"]), int(info["end"])
        x = P[a:b]
        means.append(x.mean(axis=0))
        vars_.append(x.var(axis=0))
        rms2s.append(np.mean(x * x, axis=0))
    M = np.stack(means)
    W = np.stack(vars_)
    R2 = np.stack(rms2s)
    within_var = W.mean(axis=0)
    between_var = M.var(axis=0)
    total_var = np.maximum(0.0, within_var + between_var)
    return {
        "mean": M.mean(axis=0),
        "std": np.sqrt(total_var),
        "within_std": np.sqrt(np.maximum(0.0, within_var)),
        "between_std": np.sqrt(np.maximum(0.0, between_var)),
        "within_frac": within_var / np.maximum(total_var, 1e-30),
        "rms": np.sqrt(np.maximum(0.0, R2.mean(axis=0))),
    }


def balanced_isotropic_stats(record_h_means: list[torch.Tensor], record_h_norm2: list[float], d_model: int) -> dict[str, float]:
    """Equal-record isotropic scale, split into within- and between-record components."""
    H = torch.stack(record_h_means).double()  # [R,d]
    e_norm2 = float(np.mean(record_h_norm2))
    global_mean = H.mean(dim=0)
    mean_mean_norm2 = float(H.square().sum(dim=1).mean().item())
    global_mean_norm2 = float(global_mean.square().sum().item())
    within_trace = max(0.0, e_norm2 - mean_mean_norm2)
    between_trace = max(0.0, mean_mean_norm2 - global_mean_norm2)
    total_trace = within_trace + between_trace
    return {
        "total": math.sqrt(total_trace / d_model) if total_trace > 0 else float("nan"),
        "within": math.sqrt(within_trace / d_model) if within_trace > 0 else float("nan"),
        "between": math.sqrt(between_trace / d_model) if between_trace > 0 else float("nan"),
    }


def group_record_indices(record_infos: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, info in enumerate(record_infos):
        out.setdefault(str(info["group"]), []).append(i)
    return out


def subset_record_infos(record_infos: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    """Return infos with slices remapped into a newly concatenated P subset."""
    out = []
    cursor = 0
    for i in indices:
        src = record_infos[i]
        n = int(src["end"]) - int(src["start"])
        x = dict(src)
        x["start"] = cursor
        x["end"] = cursor + n
        out.append(x)
        cursor += n
    return out


def diverse_context_indices(c: np.ndarray, record_infos: list[dict[str, Any]], topk: int, positive: bool) -> list[int]:
    """One extremum per transcript, then top transcripts globally."""
    choices: list[tuple[float, int]] = []
    for info in record_infos:
        a, b = int(info["start"]), int(info["end"])
        local = c[a:b]
        if len(local) == 0:
            continue
        rel = int(np.argmax(local) if positive else np.argmin(local))
        ix = a + rel
        choices.append((float(c[ix]), ix))
    choices.sort(key=lambda z: z[0], reverse=positive)
    return [ix for _, ix in choices[:topk]]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x: Any, n: int = 3) -> str:
    try:
        return f"{float(x):.{n}g}"
    except Exception:
        return str(x)


def build_html_report(path: Path, rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    top = sorted(rows, key=lambda r: (float(r["candidate_score"]), float(r.get("continuity_best", 0))), reverse=True)[:100]

    def esc(x: Any) -> str:
        return html.escape(str(x))

    cards = []
    for r in top:
        group_bits = []
        for g in meta.get("groups", []):
            key = f"usage_{g}"
            if key in r and np.isfinite(float(r[key])):
                group_bits.append(f"{g}={fmt(r[key])}")
        cards.append(f"""
        <section class="card">
          <h3>L{int(r['layer'])} · SV{int(r['sv_rank'])} <span class="score">score {fmt(r['candidate_score'])}</span></h3>
          <div class="metrics">
            <b>σ</b> {fmt(r['sigma'])} &nbsp;
            <b>balanced usage×iso</b> {fmt(r['usage_ratio'])} &nbsp;
            <b>within frac</b> {fmt(r['within_frac'])} &nbsp;
            <b>subspace nonlex p</b> {fmt(r['nonlex_subspace_p'])} &nbsp;
            <b>subspace nonlex z</b> {fmt(r['nonlex_subspace_z'])} &nbsp;
            <b>continuity</b> {fmt(r['continuity_best'])}
          </div>
          <div class="muted">group usage×iso: {esc(' | '.join(group_bits))}</div>
          <div class="grid">
            <div><b>+U tokens</b><pre>{esc(r['pos_tokens'])}</pre></div>
            <div><b>−U tokens</b><pre>{esc(r['neg_tokens'])}</pre></div>
          </div>
          <div class="grid">
            <div><b>diverse +V contexts</b><pre>{esc(r['pos_contexts'])}</pre></div>
            <div><b>diverse −V contexts</b><pre>{esc(r['neg_contexts'])}</pre></div>
          </div>
        </section>""")

    layer_table = "".join(
        f"<tr><td>{int(r['layer'])}</td><td>{fmt(r['top_sigma'])}</td><td>{fmt(r['captured_frob_frac'])}</td>"
        f"<td>{fmt(r['sv_triplet_rel_resid'])}</td><td>{fmt(r.get('isotropic_total_std'))}</td>"
        f"<td>{fmt(r.get('subspace_null_peak_median'))}</td></tr>"
        for r in layer_rows
    )

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Balanced J-lens SVD explorer</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width:1500px; margin:30px auto; padding:0 24px; background:#111; color:#eee; }}
h1,h2,h3 {{ color:#fff; }} .muted {{ color:#aaa; }}
.card {{ border:1px solid #333; border-radius:12px; padding:16px; margin:16px 0; background:#171717; }}
.score {{ font-size:.8em; color:#bbb; font-weight:normal; }} .metrics {{ margin-bottom:8px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#0c0c0c; border-radius:8px; padding:10px; color:#ddd; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid #333; padding:7px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
</style></head><body>
<h1>Balanced J-lens SVD explorer</h1>
<p class="muted">operator={esc(meta['operator'])}; model={esc(meta['model'])}; k={meta['k']}; records={meta['n_records']}; raw positions={meta['n_positions']}.</p>
<p><b>Primary score:</b> within-layer gain rank × balanced-usage rank × matched-subspace nonlexicality. Transcript length does not determine usage weight.</p>
<h2>Layer diagnostics</h2>
<table><thead><tr><th>layer</th><th>top σ</th><th>top-k energy</th><th>SVD residual</th><th>balanced iso std</th><th>subspace null median peak-z</th></tr></thead><tbody>{layer_table}</tbody></table>
<h2>Top candidates</h2>
{''.join(cards)}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    readouts = Path(args.readouts).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    (out / "directions").mkdir(parents=True, exist_ok=True)

    records = load_records(readouts, parse_str_set(args.only), parse_str_set(args.exclude), args.max_records)
    group_counts = Counter(record_group(r) for r in records)
    print(f"[records] {len(records)} transcripts from {readouts / 'index.jsonl'}")
    print("[records] groups: " + "  ".join(f"{g}={n}" for g, n in sorted(group_counts.items())))

    print(f"[load] {args.model}")
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype="auto", device_map="cuda")
    print(f"[load] model allocated {torch.cuda.memory_allocated()/1e9:.1f} GB")
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    if lens.d_model != model.d_model:
        raise SystemExit(f"lens d_model={lens.d_model} != model d_model={model.d_model}")

    layers = parse_int_list(args.layers) if args.layers else list(lens.source_layers)
    bad = sorted(set(layers) - set(lens.source_layers))
    if bad:
        raise SystemExit(f"layers not present in fitted lens: {bad}; available={lens.source_layers}")
    print(f"[lens] {lens}; analyzing layers {layers}")

    # Global isotropic lexicality null: retained as a diagnostic, no longer primary.
    iso_random_peaks = np.empty((0,), dtype=np.float32)
    if args.isotropic_baseline > 0:
        print(f"[lex] isotropic null: {args.isotropic_baseline} final-space directions")
        rd = torch.randn(args.isotropic_baseline, model.d_model, device=model.input_device, dtype=torch.float32)
        rd = torch.nn.functional.normalize(rd, dim=-1)
        iso_random_peaks = lexical_peak_z_batched(model, rd, args.lex_batch)
        del rd
        torch.cuda.empty_cache()
        print(f"[lex] isotropic peak-z median={np.median(iso_random_peaks):.2f}  p10={np.percentile(iso_random_peaks,10):.2f}  p90={np.percentile(iso_random_peaks,90):.2f}")

    # SVD + matched lexicality null for each layer.
    svd: dict[int, dict[str, Any]] = {}
    layer_rows: list[dict[str, Any]] = []
    eye = None
    for li, layer in enumerate(layers, 1):
        t0 = time.time()
        J_cpu = lens.jacobians[layer]
        A = J_cpu.to("cuda", dtype=torch.float32)
        if args.operator == "delta":
            if eye is None or eye.shape[0] != A.shape[0]:
                eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            A = A - eye

        fro2 = float((A * A).sum().item())
        U, S, V = topk_svd(A, args.k, args.oversample, args.svd_iters, args.exact_svd)
        rhs = U * S.unsqueeze(0)
        resid = float(torch.linalg.vector_norm(A @ V - rhs) / torch.linalg.vector_norm(rhs).clamp_min(1e-12))
        captured = float(S.square().sum().item() / max(fro2, 1e-30))

        lex = lexical_stats(model, tok, U.T.contiguous(), args.decode_topk)
        nonlex_iso_p = percentile_nonlex(lex["peak_abs_z"], iso_random_peaks)
        nonlex_iso_z = nonlex_z(lex["peak_abs_z"], iso_random_peaks)

        sub_peaks = np.empty((0,), dtype=np.float32)
        if args.subspace_baseline > 0:
            C = torch.randn(args.subspace_baseline, U.shape[1], device=U.device, dtype=torch.float32)
            C = torch.nn.functional.normalize(C, dim=-1)
            sub_dirs = C @ U.T
            sub_peaks = lexical_peak_z_batched(model, sub_dirs, args.lex_batch)
            del C, sub_dirs
        nonlex_sub_p = percentile_nonlex(lex["peak_abs_z"], sub_peaks)
        nonlex_sub_z = nonlex_z(lex["peak_abs_z"], sub_peaks)

        svd[layer] = {
            "U": U.detach().cpu().float(),
            "S": S.detach().cpu().float(),
            "V": V.detach().cpu().float(),
            "V_gpu": V.detach(),
            "lex": lex,
            "nonlex_iso_p": nonlex_iso_p,
            "nonlex_iso_z": nonlex_iso_z,
            "nonlex_sub_p": nonlex_sub_p,
            "nonlex_sub_z": nonlex_sub_z,
            "sub_peaks": sub_peaks,
            "captured_frob_frac": captured,
            "operator_fro2": fro2,
            "sv_triplet_rel_resid": resid,
        }
        layer_rows.append({
            "layer": layer,
            "top_sigma": float(S[0].item()),
            "captured_frob_frac": captured,
            "operator_fro2": fro2,
            "sv_triplet_rel_resid": resid,
            "isotropic_total_std": float("nan"),
            "isotropic_within_std": float("nan"),
            "isotropic_between_std": float("nan"),
            "subspace_null_peak_median": float(np.median(sub_peaks)) if len(sub_peaks) else float("nan"),
            "subspace_null_peak_p10": float(np.percentile(sub_peaks, 10)) if len(sub_peaks) else float("nan"),
            "subspace_null_peak_p90": float(np.percentile(sub_peaks, 90)) if len(sub_peaks) else float("nan"),
            "wall_s": round(time.time() - t0, 2),
        })
        print(f"[svd] L{layer:02d} {li:02d}/{len(layers)} topσ={float(S[0]):.3g} energy={captured:.3f} resid={resid:.2e} "
              f"subnull_med={np.median(sub_peaks):.2f}" if len(sub_peaks) else
              f"[svd] L{layer:02d} {li:02d}/{len(layers)} topσ={float(S[0]):.3g} energy={captured:.3f} resid={resid:.2e}")
        del A, U, S, V, rhs
        torch.cuda.empty_cache()

    # Cross-layer continuity before dropping U/V.
    links: list[dict[str, Any]] = []
    continuity: dict[tuple[int, int], dict[str, float | int]] = {}
    for ai in range(len(layers) - 1):
        la, lb = layers[ai], layers[ai + 1]
        Ua, Va = svd[la]["U"], svd[la]["V"]
        Ub, Vb = svd[lb]["U"], svd[lb]["V"]
        mu = torch.abs(Ua.T @ Ub).numpy()
        mv = torch.abs(Va.T @ Vb).numpy()
        joint = np.sqrt(np.clip(mu * mv, 0, 1))

        for j in range(joint.shape[0]):
            q = int(np.argmax(joint[j]))
            continuity[(la, j + 1)] = {
                **continuity.get((la, j + 1), {}),
                "next_layer": lb, "next_sv": q + 1,
                "next_u_cos": float(mu[j, q]), "next_v_cos": float(mv[j, q]),
                "next_joint": float(joint[j, q]),
            }
        for q in range(joint.shape[1]):
            j = int(np.argmax(joint[:, q]))
            continuity[(lb, q + 1)] = {
                **continuity.get((lb, q + 1), {}),
                "prev_layer": la, "prev_sv": j + 1,
                "prev_u_cos": float(mu[j, q]), "prev_v_cos": float(mv[j, q]),
                "prev_joint": float(joint[j, q]),
            }
        # All best links from la -> lb, convenient separate table.
        for j in range(joint.shape[0]):
            q = int(np.argmax(joint[j]))
            links.append({
                "layer": la, "sv_rank": j + 1, "next_layer": lb, "next_sv_rank": q + 1,
                "u_cos": float(mu[j, q]), "v_cos": float(mv[j, q]), "joint_cos": float(joint[j, q]),
            })

    # Replay transcripts. Store projections and per-record hidden moments.
    proj_chunks: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    record_h_means: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    record_h_norm2: dict[int, list[float]] = {l: [] for l in layers}
    sample_meta: list[dict[str, Any]] = []
    record_infos: list[dict[str, Any]] = []
    special_ids = set(getattr(tok, "all_special_ids", []) or [])

    print("[act] replaying exact stored transcript text (no generation); usage will be equal-weight per transcript")
    for ri, rec in enumerate(records, 1):
        pid = str(rec.get("pid", f"record_{ri}"))
        group = record_group(rec)
        text = str(rec["text"])
        input_ids = model.encode(text, max_length=args.max_seq_len)
        ids = input_ids[0].detach().cpu().tolist()
        token_strs = [tok.decode([int(t)]) for t in ids]

        if args.position_mode == "last":
            keep = np.zeros(len(ids), dtype=bool)
            keep[-1] = True
        else:
            keep = np.ones(len(ids), dtype=bool)
        if not args.include_special and special_ids:
            keep &= np.array([int(t) not in special_ids for t in ids], dtype=bool)
        keep_idx = np.flatnonzero(keep)
        if len(keep_idx) == 0:
            print(f"[act] {pid}: no positions after filtering; skipped")
            continue

        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)

        start = len(sample_meta)
        for pos in keep_idx.tolist():
            sample_meta.append({
                "pid": pid, "group": group, "pos": int(pos),
                "token_id": int(ids[pos]), "token": safe_token_text(token_strs[pos]),
                "context": make_context(token_strs, pos, args.context_window),
            })
        end = len(sample_meta)
        record_infos.append({"pid": pid, "group": group, "start": start, "end": end, "n": end - start})

        for layer in layers:
            h = recorder.activations[layer][0].detach()[keep_idx].float()
            coeff = h @ svd[layer]["V_gpu"]
            proj_chunks[layer].append(coeff.detach().cpu().numpy().astype(np.float32))
            record_h_means[layer].append(h.mean(dim=0).detach().cpu().double())
            record_h_norm2[layer].append(float(h.square().sum(dim=1).mean().item()))
            del h, coeff

        del recorder
        print(f"[act] {ri:02d}/{len(records)} {pid:28s} group={group:8s} kept={len(keep_idx):4d} raw_total={len(sample_meta):5d}")

    if not record_infos:
        raise SystemExit("no activation samples collected")

    # Only successfully processed records count from here.
    groups = sorted({str(x["group"]) for x in record_infos})
    group_indices = group_record_indices(record_infos)
    raw_pos_counts = Counter()
    for info in record_infos:
        raw_pos_counts[str(info["group"])] += int(info["n"])
    print("[balance] raw positions: " + "  ".join(f"{g}={raw_pos_counts[g]}" for g in groups))
    print("[balance] effective mass after balancing: each transcript=1.0; " +
          "  ".join(f"{g}={len(group_indices[g])}" for g in groups))

    rows: list[dict[str, Any]] = []
    layer_row_by_id = {int(r["layer"]): r for r in layer_rows}

    for layer in layers:
        P = np.concatenate(proj_chunks[layer], axis=0)
        if P.shape[0] != len(sample_meta):
            raise RuntimeError(f"metadata/projection mismatch at L{layer}: {P.shape[0]} vs {len(sample_meta)}")

        stats = weighted_record_coeff_stats(P, record_infos)
        iso = balanced_isotropic_stats(record_h_means[layer], record_h_norm2[layer], model.d_model)
        usage = stats["std"] / max(iso["total"], 1e-12)
        usage_within = stats["within_std"] / max(iso["within"], 1e-12)
        usage_between = stats["between_std"] / max(iso["between"], 1e-12)

        group_usage: dict[str, np.ndarray] = {}
        for g in groups:
            idxs = group_indices[g]
            # Concatenate group records, then remap their slices.
            Ps = [P[int(record_infos[i]["start"]):int(record_infos[i]["end"])] for i in idxs]
            Pg = np.concatenate(Ps, axis=0)
            infos_g = subset_record_infos(record_infos, idxs)
            gs = weighted_record_coeff_stats(Pg, infos_g)
            hmeans_g = [record_h_means[layer][i] for i in idxs]
            hnorms_g = [record_h_norm2[layer][i] for i in idxs]
            giso = balanced_isotropic_stats(hmeans_g, hnorms_g, model.d_model)
            group_usage[g] = gs["std"] / max(giso["total"], 1e-12)

        S = svd[layer]["S"].numpy()
        transported_std = S * stats["std"]
        lex = svd[layer]["lex"]
        sub_p = svd[layer]["nonlex_sub_p"]
        sub_z = svd[layer]["nonlex_sub_z"]

        # Everything scale-sensitive is ranked WITHIN layer.
        r_gain = rank01_desc(S)
        r_usage = rank01_desc(usage)
        if np.isfinite(sub_p).all():
            r_nonlex = np.clip(sub_p, 0, 1)
        else:
            r_nonlex = 1.0 - rank01_desc(lex["peak_abs_z"])
        score = np.cbrt(np.clip(r_gain, 1e-6, 1) * np.clip(r_usage, 1e-6, 1) * np.clip(r_nonlex, 1e-6, 1))

        layer_row_by_id[layer]["isotropic_total_std"] = iso["total"]
        layer_row_by_id[layer]["isotropic_within_std"] = iso["within"]
        layer_row_by_id[layer]["isotropic_between_std"] = iso["between"]

        save = {
            "U": svd[layer]["U"].numpy(), "S": S, "V": svd[layer]["V"].numpy(),
            "proj_mean_balanced": stats["mean"], "proj_std_balanced": stats["std"],
            "proj_within_std": stats["within_std"], "proj_between_std": stats["between_std"],
            "within_frac": stats["within_frac"], "proj_rms_balanced": stats["rms"],
            "usage_ratio": usage, "usage_within_ratio": usage_within, "usage_between_ratio": usage_between,
            "transported_std_balanced": transported_std,
            "lex_peak_abs_z": lex["peak_abs_z"],
            "lex_top1_margin_z": lex["top1_margin_z"],
            "lex_entropy_norm": lex["entropy_norm"], "lex_topk_mass": lex["topk_mass"],
            "nonlex_isotropic_p": svd[layer]["nonlex_iso_p"],
            "nonlex_isotropic_z": svd[layer]["nonlex_iso_z"],
            "nonlex_subspace_p": sub_p, "nonlex_subspace_z": sub_z,
            "candidate_score": score,
        }
        for g in groups:
            save[f"usage_{g}"] = group_usage[g]
        np.savez_compressed(out / "directions" / f"L{layer:02d}.npz", **save)

        for j in range(P.shape[1]):
            c = P[:, j]
            pos_idx = diverse_context_indices(c, record_infos, args.context_topk, positive=True)
            neg_idx = diverse_context_indices(c, record_infos, args.context_topk, positive=False)

            def contexts(indices: list[int]) -> str:
                return "\n".join(
                    f"{c[ix]:+.3f} | {sample_meta[ix]['pid']} [{sample_meta[ix]['group']}] @{sample_meta[ix]['pos']} | {sample_meta[ix]['context']}"
                    for ix in indices
                )

            pos_tok = " | ".join(
                f"{safe_token_text(t)} ({float(v):+.2f})"
                for t, v in zip(lex["pos_tokens"][j], lex["pos_logits"][j])
            )
            neg_tok = " | ".join(
                f"{safe_token_text(t)} ({float(v):+.2f})"
                for t, v in zip(lex["neg_tokens"][j], lex["neg_logits"][j])
            )

            cont = continuity.get((layer, j + 1), {})
            prev_joint = float(cont.get("prev_joint", float("nan")))
            next_joint = float(cont.get("next_joint", float("nan")))
            finite_joint = [x for x in (prev_joint, next_joint) if np.isfinite(x)]
            continuity_best = max(finite_joint) if finite_joint else float("nan")

            row = {
                "layer": layer, "sv_rank": j + 1, "sigma": float(S[j]),
                "sigma_sq_frac_of_operator": float(S[j] ** 2 / max(float(svd[layer]["operator_fro2"]), 1e-30)),
                "coeff_mean_balanced": float(stats["mean"][j]),
                "coeff_std_balanced": float(stats["std"][j]),
                "coeff_within_std": float(stats["within_std"][j]),
                "coeff_between_std": float(stats["between_std"][j]),
                "within_frac": float(stats["within_frac"][j]),
                "isotropic_total_std": float(iso["total"]),
                "usage_ratio": float(usage[j]),
                "usage_within_ratio": float(usage_within[j]),
                "usage_between_ratio": float(usage_between[j]),
                "transported_std_balanced": float(transported_std[j]),
                "lex_peak_abs_z": float(lex["peak_abs_z"][j]),
                "lex_top1_margin_z": float(lex["top1_margin_z"][j]),
                "lex_entropy_norm": float(lex["entropy_norm"][j]),
                "lex_topk_mass": float(lex["topk_mass"][j]),
                "nonlex_isotropic_p": float(svd[layer]["nonlex_iso_p"][j]),
                "nonlex_isotropic_z": float(svd[layer]["nonlex_iso_z"][j]),
                "nonlex_subspace_p": float(sub_p[j]),
                "nonlex_subspace_z": float(sub_z[j]),
                "candidate_score": float(score[j]),
                "prev_layer": cont.get("prev_layer", ""), "prev_sv": cont.get("prev_sv", ""),
                "prev_u_cos": cont.get("prev_u_cos", float("nan")), "prev_v_cos": cont.get("prev_v_cos", float("nan")),
                "prev_joint": prev_joint,
                "next_layer": cont.get("next_layer", ""), "next_sv": cont.get("next_sv", ""),
                "next_u_cos": cont.get("next_u_cos", float("nan")), "next_v_cos": cont.get("next_v_cos", float("nan")),
                "next_joint": next_joint, "continuity_best": continuity_best,
                "pos_tokens": pos_tok, "neg_tokens": neg_tok,
                "pos_contexts": contexts(pos_idx), "neg_contexts": contexts(neg_idx),
            }
            for g in groups:
                row[f"usage_{g}"] = float(group_usage[g][j])
            rows.append(row)

        best_j = int(np.argmax(score))
        print(f"[metric] L{layer:02d} iso={iso['total']:.3g} best=SV{best_j+1:02d} score={score[best_j]:.3f} "
              f"usage={usage[best_j]:.2f} within={stats['within_frac'][best_j]:.2f} "
              f"subP={sub_p[best_j]:.2f} subZ={sub_z[best_j]:+.2f}")

    # Free GPU copies.
    for layer in layers:
        svd[layer].pop("V_gpu", None)
    torch.cuda.empty_cache()

    group_fields = [f"usage_{g}" for g in groups]
    fieldnames = [
        "layer", "sv_rank", "sigma", "sigma_sq_frac_of_operator",
        "coeff_mean_balanced", "coeff_std_balanced", "coeff_within_std", "coeff_between_std", "within_frac",
        "isotropic_total_std", "usage_ratio", "usage_within_ratio", "usage_between_ratio",
        *group_fields, "transported_std_balanced",
        "lex_peak_abs_z", "lex_top1_margin_z", "lex_entropy_norm", "lex_topk_mass",
        "nonlex_isotropic_p", "nonlex_isotropic_z", "nonlex_subspace_p", "nonlex_subspace_z",
        "candidate_score",
        "prev_layer", "prev_sv", "prev_u_cos", "prev_v_cos", "prev_joint",
        "next_layer", "next_sv", "next_u_cos", "next_v_cos", "next_joint", "continuity_best",
        "pos_tokens", "neg_tokens", "pos_contexts", "neg_contexts",
    ]
    write_csv(out / "directions.csv", rows, fieldnames)
    candidate_rows = sorted(rows, key=lambda r: (float(r["candidate_score"]), float(r.get("continuity_best", -1))), reverse=True)
    write_csv(out / "candidates.csv", candidate_rows, fieldnames)

    layer_fields = [
        "layer", "top_sigma", "captured_frob_frac", "sv_triplet_rel_resid",
        "isotropic_total_std", "isotropic_within_std", "isotropic_between_std",
        "subspace_null_peak_median", "subspace_null_peak_p10", "subspace_null_peak_p90", "wall_s",
    ]
    write_csv(out / "layer_summary.csv", layer_rows, layer_fields)
    write_csv(out / "mode_links.csv", links,
              ["layer", "sv_rank", "next_layer", "next_sv_rank", "u_cos", "v_cos", "joint_cos"])

    meta = {
        "model": args.model, "model_revision": args.model_revision,
        "lens_repo": args.lens_repo, "lens_file": args.lens_file,
        "operator": args.operator, "layers": layers, "k": args.k,
        "exact_svd": args.exact_svd, "oversample": args.oversample, "svd_iters": args.svd_iters,
        "n_records": len(record_infos), "record_pids": [x["pid"] for x in record_infos],
        "groups": groups, "group_record_counts": {g: len(group_indices[g]) for g in groups},
        "group_raw_position_counts": dict(raw_pos_counts), "n_positions": len(sample_meta),
        "position_mode": args.position_mode, "include_special": args.include_special,
        "isotropic_baseline_n": int(len(iso_random_peaks)),
        "isotropic_peak_z": {
            "median": float(np.median(iso_random_peaks)) if len(iso_random_peaks) else None,
            "p10": float(np.percentile(iso_random_peaks, 10)) if len(iso_random_peaks) else None,
            "p90": float(np.percentile(iso_random_peaks, 90)) if len(iso_random_peaks) else None,
        },
        "subspace_baseline_n": args.subspace_baseline,
        "usage_definition": "equal transcript weight; total variance decomposed into mean within-transcript variance + variance of transcript means",
        "candidate_score_definition": "geometric mean of within-layer sigma rank, within-layer balanced usage rank, and matched top-U-subspace nonlexical p",
        "continuity_definition": "sqrt(abs(U_l^T U_next) * abs(V_l^T V_next)); best match independently for each adjacent layer",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    build_html_report(out / "report.html", rows, layer_rows, meta)

    print("\n[done] outputs:")
    print(f"  {out / 'report.html'}")
    print(f"  {out / 'candidates.csv'}")
    print(f"  {out / 'directions.csv'}")
    print(f"  {out / 'mode_links.csv'}")
    print(f"  {out / 'directions'}/*.npz")

    print("\nTop balanced exploratory candidates:")
    for r in candidate_rows[:args.terminal_top]:
        grp = ",".join(f"{g}:{float(r[f'usage_{g}']):.2f}" for g in groups)
        cont = float(r["continuity_best"])
        cont_s = f"{cont:.2f}" if np.isfinite(cont) else "na"
        print(f"  L{int(r['layer']):02d} SV{int(r['sv_rank']):02d} score={r['candidate_score']:.3f} "
              f"σ={r['sigma']:.3g} use={r['usage_ratio']:.2f} within={r['within_frac']:.2f} "
              f"subP={r['nonlex_subspace_p']:.2f} subZ={r['nonlex_subspace_z']:+.2f} cont={cont_s} "
              f"groups[{grp}] +U=[{r['pos_tokens'][:110]}]")

    print("\nBest candidate in each layer:")
    for layer in layers:
        lr = [r for r in rows if int(r["layer"]) == layer]
        r = max(lr, key=lambda x: float(x["candidate_score"]))
        grp = ",".join(f"{g}:{float(r[f'usage_{g}']):.2f}" for g in groups)
        print(f"  L{layer:02d} SV{int(r['sv_rank']):02d} score={r['candidate_score']:.3f} "
              f"use={r['usage_ratio']:.2f} within={r['within_frac']:.2f} "
              f"subP={r['nonlex_subspace_p']:.2f} subZ={r['nonlex_subspace_z']:+.2f} groups[{grp}]")

    persistent = [r for r in rows if np.isfinite(float(r["continuity_best"]))]
    persistent.sort(key=lambda r: float(r["continuity_best"]), reverse=True)
    print("\nMost persistent adjacent-layer modes:")
    for r in persistent[:args.terminal_persistent]:
        prev = (f"L{int(r['prev_layer']):02d}/SV{int(r['prev_sv']):02d}:{float(r['prev_joint']):.2f}"
                if r["prev_layer"] != "" and np.isfinite(float(r["prev_joint"])) else "-")
        nxt = (f"L{int(r['next_layer']):02d}/SV{int(r['next_sv']):02d}:{float(r['next_joint']):.2f}"
               if r["next_layer"] != "" and np.isfinite(float(r["next_joint"])) else "-")
        print(f"  L{int(r['layer']):02d} SV{int(r['sv_rank']):02d} cont={float(r['continuity_best']):.3f} "
              f"prev={prev} next={nxt} score={float(r['candidate_score']):.3f} +U=[{r['pos_tokens'][:90]}]")


if __name__ == "__main__":
    main()
