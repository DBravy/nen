#!/usr/bin/env python
"""Explore dominant J-lens singular directions that are used but weakly lexicalized.

Designed to live next to collect_readouts.py and reuse its ./readouts/index.jsonl.
It does NOT regenerate any responses. It reloads the exact stored transcript text,
runs a hooks-visible forward pass, and measures how strongly real residual-stream
activations project onto the right singular vectors of each fitted J-lens matrix.

For each layer, with J = U diag(S) V^T:
  * V[:, i] is the intermediate-layer direction (the side you asked about).
  * U[:, i] is its transported final-layer direction.
  * S[i] is downstream gain.
  * coeff_std = std(h_l @ V[:, i]) measures how much activations vary along it.
  * transported_std = S[i] * coeff_std is the RMS-ish downstream signal carried
    by that mode (using centered activation coefficients).
  * lexical stats decode U[:, i] through the model's own final norm + lm_head and
    ask whether one vocabulary token dominates that direction.
  * top +/- activation contexts show where the model actually occupies V[:, i].

Outputs:
  <out>/directions.csv              one row per (layer, singular direction)
  <out>/candidates.csv              same rows, sorted by exploratory candidate score
  <out>/layer_summary.csv           SVD quality / energy diagnostics
  <out>/report.html                 self-contained browsable report
  <out>/meta.json                   run settings + lexical random baseline
  <out>/directions/LXX.npz          U, S, V and activation statistics for later work

Typical usage:
  python explore_jlens_svd_concepts.py

A faster first pass:
  python explore_jlens_svd_concepts.py --k 24 --random-baseline 64 --max-records 12

Focus on selected layers / prompts:
  python explore_jlens_svd_concepts.py --layers 8,12,16,20 --only covert_animal,covert_emotion

Inspect directions changed by the downstream stack rather than raw passthrough:
  python explore_jlens_svd_concepts.py --operator delta

Notes:
  * Default SVD is randomized (torch.svd_lowrank), because exact full SVD of every
    d_model x d_model matrix is unnecessarily expensive for exploration.
  * Singular-vector signs are arbitrary, so BOTH +U/-U token associations and
    +V/-V activation contexts are reported.
  * "Nonlexical" here is deliberately operational, not metaphysical: low lexical
    peak relative to isotropic random final-space directions. A causal intervention
    is still needed before calling a direction a concept.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
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
    p.add_argument("--readouts", default=str(here / "readouts"),
                   help="directory made by collect_readouts.py")
    p.add_argument("--out", default=str(here / "svd_concepts"))
    p.add_argument("--layers", default="",
                   help="comma-separated fitted layer ids; default = every fitted source layer")
    p.add_argument("--only", default="",
                   help="comma-separated readout pids/base-pids to use; default = all index records")
    p.add_argument("--exclude", default="",
                   help="comma-separated readout pids/base-pids to exclude")
    p.add_argument("--max-records", type=int, default=0,
                   help="0 = all selected records")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--position-mode", choices=["all", "last"], default="all")
    p.add_argument("--include-special", action="store_true",
                   help="include BOS/Harmony/control tokens in activation-usage statistics")
    p.add_argument("--k", type=int, default=32,
                   help="number of dominant singular directions per layer")
    p.add_argument("--oversample", type=int, default=16,
                   help="extra randomized-SVD dimensions")
    p.add_argument("--svd-iters", type=int, default=2)
    p.add_argument("--exact-svd", action="store_true",
                   help="use full torch.linalg.svd instead of randomized top-k SVD")
    p.add_argument("--operator", choices=["raw", "delta"], default="raw",
                   help="raw: SVD(J); delta: SVD(J-I), useful for removing residual passthrough")
    p.add_argument("--decode-topk", type=int, default=8)
    p.add_argument("--context-topk", type=int, default=6)
    p.add_argument("--context-window", type=int, default=8)
    p.add_argument("--random-baseline", type=int, default=128,
                   help="isotropic final-space directions used to calibrate lexical peak")
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


def topk_svd(A: torch.Tensor, k: int, oversample: int, niter: int, exact: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U[k], S[k], V[k] as columns, with A ~= U diag(S) V^T."""
    d0, d1 = A.shape
    k = min(k, d0, d1)
    if exact:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k].T.contiguous()
    q = min(k + max(0, oversample), d0, d1)
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
    # svd_lowrank normally returns descending values, but sort explicitly.
    order = torch.argsort(S, descending=True)[:k]
    return U[:, order], S[order], V[:, order]


@torch.no_grad()
def lexical_stats(model, tok, dirs: torch.Tensor, decode_topk: int) -> dict[str, Any]:
    """Decode rows of dirs [K,d] through jlens' exact model.unembed path."""
    logits = model.unembed(dirs).float()  # [K, vocab]
    n_vocab = logits.shape[-1]
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
    z = (logits - mean) / std
    peak_abs_z = z.abs().amax(dim=-1)

    tk = min(decode_topk, n_vocab)
    pos_vals, pos_ids = logits.topk(tk, dim=-1)
    neg_vals, neg_ids = (-logits).topk(tk, dim=-1)  # highest tokens for -direction

    top2 = logits.topk(min(2, n_vocab), dim=-1).values
    if n_vocab >= 2:
        margin_z = (top2[:, 0] - top2[:, 1]) / std.squeeze(-1)
    else:
        margin_z = torch.zeros(logits.shape[0], device=logits.device)

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
        "pos_ids": pos_ids.detach().cpu().numpy(),
        "neg_ids": neg_ids.detach().cpu().numpy(),
        "pos_tokens": decode_rows(pos_ids),
        "neg_tokens": decode_rows(neg_ids),
        "pos_logits": pos_vals.detach().cpu().numpy(),
        "neg_logits": neg_vals.detach().cpu().numpy(),
    }
    del logits, p, z
    return out


def percentile_nonlex(observed: np.ndarray, random_peaks: np.ndarray) -> np.ndarray:
    """High = observed lexical peak is unusually LOW relative to random directions."""
    # P(random peak >= observed peak). 1.0 means especially non-peaky/nonlexical.
    return np.array([(random_peaks >= x).mean() for x in observed], dtype=np.float32)


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
    """Empirical [0,1] rank; 1 = largest."""
    if len(x) <= 1:
        return np.ones_like(x, dtype=np.float32)
    order = np.argsort(np.argsort(x))
    return order.astype(np.float32) / (len(x) - 1)


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
    top = sorted(rows, key=lambda r: float(r["candidate_score"]), reverse=True)[:80]
    layer_map = {int(r["layer"]): r for r in layer_rows}

    def esc(x: Any) -> str:
        return html.escape(str(x))

    cards = []
    for r in top:
        layer = int(r["layer"])
        cards.append(f"""
        <section class="card">
          <h3>L{layer} · SV{int(r['sv_rank'])} <span class="score">score {fmt(r['candidate_score'])}</span></h3>
          <div class="metrics">
            <b>σ</b> {fmt(r['sigma'])} &nbsp; <b>usage×iso</b> {fmt(r['usage_ratio'])} &nbsp;
            <b>σ·std</b> {fmt(r['transported_std'])} &nbsp; <b>nonlex p</b> {fmt(r['nonlex_random_p'])} &nbsp;
            <b>lex peak z</b> {fmt(r['lex_peak_abs_z'])}
          </div>
          <div class="grid">
            <div><b>+U tokens</b><pre>{esc(r['pos_tokens'])}</pre></div>
            <div><b>−U tokens</b><pre>{esc(r['neg_tokens'])}</pre></div>
          </div>
          <div class="grid">
            <div><b>highest +V activation contexts</b><pre>{esc(r['pos_contexts'])}</pre></div>
            <div><b>highest −V activation contexts</b><pre>{esc(r['neg_contexts'])}</pre></div>
          </div>
        </section>""")

    layer_table = "".join(
        f"<tr><td>{int(r['layer'])}</td><td>{fmt(r['top_sigma'])}</td><td>{fmt(r['captured_frob_frac'])}</td>"
        f"<td>{fmt(r['sv_triplet_rel_resid'])}</td><td>{int(r['n_activation_samples'])}</td></tr>"
        for r in layer_rows
    )

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>J-lens SVD concept explorer</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 1500px; margin: 30px auto; padding: 0 24px; background:#111; color:#eee; }}
h1,h2,h3 {{ color:#fff; }} .muted {{ color:#aaa; }}
.card {{ border:1px solid #333; border-radius:12px; padding:16px; margin:16px 0; background:#171717; }}
.score {{ font-size:.8em; color:#bbb; font-weight:normal; }}
.metrics {{ color:#ddd; margin-bottom:12px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#0c0c0c; border-radius:8px; padding:10px; color:#ddd; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid #333; padding:7px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
code {{ color:#ddd; }}
</style></head><body>
<h1>J-lens SVD concept explorer</h1>
<p class="muted">operator={esc(meta['operator'])}; model={esc(meta['model'])}; k={meta['k']}; records={meta['n_records']}; positions={meta['n_positions']}.</p>
<p><b>Interpretation:</b> high candidate score means a top singular mode has high downstream signal on real activations and a relatively weak single-token lexical peak. It is a search heuristic, not evidence by itself that the mode is a concept.</p>
<h2>Layer diagnostics</h2>
<table><thead><tr><th>layer</th><th>top σ</th><th>top-k Frobenius energy</th><th>SVD residual</th><th>activation samples</th></tr></thead><tbody>{layer_table}</tbody></table>
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

    only = parse_str_set(args.only)
    exclude = parse_str_set(args.exclude)
    records = load_records(readouts, only, exclude, args.max_records)
    print(f"[records] {len(records)} transcripts from {readouts / 'index.jsonl'}")

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

    # --- lexical random baseline in final-layer residual space -----------------
    random_peaks = np.empty((0,), dtype=np.float32)
    if args.random_baseline > 0:
        print(f"[lex] calibrating with {args.random_baseline} isotropic random final-space directions")
        rd = torch.randn(args.random_baseline, model.d_model, device=model.input_device, dtype=torch.float32)
        rd = torch.nn.functional.normalize(rd, dim=-1)
        random_peaks = lexical_stats(model, tok, rd, min(2, args.decode_topk))["peak_abs_z"].astype(np.float32)
        del rd
        torch.cuda.empty_cache()
        print(f"[lex] random peak-z median={np.median(random_peaks):.2f}  p10={np.percentile(random_peaks,10):.2f}  p90={np.percentile(random_peaks,90):.2f}")

    # --- SVD each fitted J ------------------------------------------------------
    svd: dict[int, dict[str, Any]] = {}
    layer_rows: list[dict[str, Any]] = []
    eye = None
    for li, layer in enumerate(layers, 1):
        t0 = time.time()
        J_cpu = lens.jacobians[layer]  # jlens loads these fp32 on CPU
        A = J_cpu.to("cuda", dtype=torch.float32)
        if args.operator == "delta":
            if eye is None or eye.shape[0] != A.shape[0]:
                eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            A = A - eye

        fro2 = float((A * A).sum().item())
        U, S, V = topk_svd(A, args.k, args.oversample, args.svd_iters, args.exact_svd)
        # A @ v_i should equal sigma_i u_i. This catches a convention/transposition mistake.
        rhs = U * S.unsqueeze(0)
        resid = float(torch.linalg.vector_norm(A @ V - rhs) / torch.linalg.vector_norm(rhs).clamp_min(1e-12))
        captured = float((S.square().sum().item()) / max(fro2, 1e-30))

        # Decode final-side singular directions NOW; don't keep vocab logits.
        lex = lexical_stats(model, tok, U.T.contiguous(), args.decode_topk)
        if len(random_peaks):
            nonlex_p = percentile_nonlex(lex["peak_abs_z"], random_peaks)
        else:
            nonlex_p = np.full(len(S), np.nan, dtype=np.float32)

        svd[layer] = {
            "U": U.detach().cpu().float(),
            "S": S.detach().cpu().float(),
            "V": V.detach().cpu().float(),
            "V_gpu": V.detach(),
            "lex": lex,
            "nonlex_p": nonlex_p,
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
            "n_activation_samples": 0,
            "wall_s": round(time.time() - t0, 2),
        })
        del A, U, S, V, rhs
        torch.cuda.empty_cache()
        print(f"[svd] L{layer:02d} {li:02d}/{len(layers)}  topσ={layer_rows[-1]['top_sigma']:.3g}  top-k-energy={captured:.3f}  resid={resid:.2e}  {layer_rows[-1]['wall_s']:.1f}s")

    # --- replay stored transcripts and project actual residuals onto V ----------
    proj_chunks: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    sum_h: dict[int, torch.Tensor] = {l: torch.zeros(model.d_model, dtype=torch.float64) for l in layers}
    sum_h_norm2: dict[int, float] = {l: 0.0 for l in layers}
    n_h: dict[int, int] = {l: 0 for l in layers}
    sample_meta: list[dict[str, Any]] = []
    special_ids = set(getattr(tok, "all_special_ids", []) or [])

    print("[act] replaying stored transcript text (no generation)")
    for ri, rec in enumerate(records, 1):
        pid = str(rec.get("pid", f"record_{ri}"))
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

        # Metadata is identical for every layer, so append once.
        base_index = len(sample_meta)
        for pos in keep_idx.tolist():
            sample_meta.append({
                "pid": pid,
                "pos": int(pos),
                "token_id": int(ids[pos]),
                "token": safe_token_text(token_strs[pos]),
                "context": make_context(token_strs, pos, args.context_window),
            })

        for layer in layers:
            h = recorder.activations[layer][0].detach()[keep_idx].float()  # [N,d]
            Vg = svd[layer]["V_gpu"]
            coeff = h @ Vg  # [N,k]
            proj_chunks[layer].append(coeff.detach().cpu().numpy().astype(np.float32))
            sum_h[layer] += h.sum(dim=0).detach().cpu().double()
            sum_h_norm2[layer] += float(h.square().sum().item())
            n_h[layer] += int(h.shape[0])
            del h, coeff

        del recorder
        print(f"[act] {ri:02d}/{len(records)} {pid:28s} kept={len(keep_idx):4d} total_samples={len(sample_meta):5d}")

    if not sample_meta:
        raise SystemExit("no activation samples collected")

    # --- combine metrics, save arrays, construct candidate rows -----------------
    rows: list[dict[str, Any]] = []
    layer_row_by_id = {int(r["layer"]): r for r in layer_rows}

    for layer in layers:
        P = np.concatenate(proj_chunks[layer], axis=0)  # [N,k]
        if P.shape[0] != len(sample_meta):
            raise RuntimeError(f"metadata/projection mismatch at L{layer}: {P.shape[0]} vs {len(sample_meta)}")
        mean = P.mean(axis=0)
        std = P.std(axis=0)
        rms = np.sqrt(np.mean(P * P, axis=0))

        n = n_h[layer]
        mean_h = sum_h[layer] / max(n, 1)
        e_norm2 = sum_h_norm2[layer] / max(n, 1)
        trace_cov = max(0.0, e_norm2 - float(mean_h.square().sum().item()))
        isotropic_std = math.sqrt(trace_cov / model.d_model) if trace_cov > 0 else float("nan")
        usage_ratio = std / max(isotropic_std, 1e-12)

        S = svd[layer]["S"].numpy()
        transported_std = S * std
        transported_rms = S * rms
        lex = svd[layer]["lex"]
        nonlex_p = svd[layer]["nonlex_p"]

        # Three independent desiderata: gain, actual activation usage, and nonlexicality.
        # Keep a within-layer score now; a global score is added after all layers are assembled.
        r_gain = rank01_desc(S)
        r_usage = rank01_desc(usage_ratio)
        if np.isfinite(nonlex_p).all():
            r_nonlex = nonlex_p
        else:
            r_nonlex = 1.0 - rank01_desc(lex["peak_abs_z"])
        candidate_score_layer = np.cbrt(
            np.clip(r_gain, 1e-6, 1) * np.clip(r_usage, 1e-6, 1) * np.clip(r_nonlex, 1e-6, 1)
        )

        # Save raw directions/statistics before stringifying contexts.
        np.savez_compressed(
            out / "directions" / f"L{layer:02d}.npz",
            U=svd[layer]["U"].numpy(),
            S=S,
            V=svd[layer]["V"].numpy(),
            proj_mean=mean,
            proj_std=std,
            proj_rms=rms,
            usage_ratio=usage_ratio,
            transported_std=transported_std,
            transported_rms=transported_rms,
            lex_peak_abs_z=lex["peak_abs_z"],
            lex_top1_margin_z=lex["top1_margin_z"],
            lex_entropy_norm=lex["entropy_norm"],
            lex_topk_mass=lex["topk_mass"],
            nonlex_random_p=nonlex_p,
            candidate_score_layer=candidate_score_layer,
        )

        layer_row_by_id[layer]["n_activation_samples"] = int(P.shape[0])
        layer_row_by_id[layer]["isotropic_activation_std"] = isotropic_std

        for j in range(P.shape[1]):
            c = P[:, j]
            m = min(args.context_topk, len(c))
            pos_idx = np.argpartition(c, -m)[-m:]
            pos_idx = pos_idx[np.argsort(c[pos_idx])[::-1]]
            neg_idx = np.argpartition(c, m - 1)[:m]
            neg_idx = neg_idx[np.argsort(c[neg_idx])]

            def contexts(indices: np.ndarray) -> str:
                parts = []
                for ix in indices.tolist():
                    sm = sample_meta[ix]
                    parts.append(f"{c[ix]:+.3f} | {sm['pid']}@{sm['pos']} | {sm['context']}")
                return "\n".join(parts)

            pos_tok = " | ".join(
                f"{safe_token_text(t)} ({float(v):+.2f})"
                for t, v in zip(lex["pos_tokens"][j], lex["pos_logits"][j])
            )
            neg_tok = " | ".join(
                f"{safe_token_text(t)} ({float(v):+.2f})"
                for t, v in zip(lex["neg_tokens"][j], lex["neg_logits"][j])
            )

            rows.append({
                "layer": layer,
                "sv_rank": j + 1,
                "sigma": float(S[j]),
                "sigma_sq_frac_of_operator": float(S[j] ** 2 / max(float(svd[layer]["operator_fro2"]), 1e-30)),
                "coeff_mean": float(mean[j]),
                "coeff_std": float(std[j]),
                "coeff_rms": float(rms[j]),
                "isotropic_activation_std": float(isotropic_std),
                "usage_ratio": float(usage_ratio[j]),
                "transported_std": float(transported_std[j]),
                "transported_rms": float(transported_rms[j]),
                "lex_peak_abs_z": float(lex["peak_abs_z"][j]),
                "lex_top1_margin_z": float(lex["top1_margin_z"][j]),
                "lex_entropy_norm": float(lex["entropy_norm"][j]),
                "lex_topk_mass": float(lex["topk_mass"][j]),
                "nonlex_random_p": float(nonlex_p[j]) if np.isfinite(nonlex_p[j]) else float("nan"),
                "candidate_score_layer": float(candidate_score_layer[j]),
                "candidate_score": float("nan"),  # filled globally after all layers
                "pos_tokens": pos_tok,
                "neg_tokens": neg_tok,
                "pos_contexts": contexts(pos_idx),
                "neg_contexts": contexts(neg_idx),
            })

        print(f"[metric] L{layer:02d} isotropic_std={isotropic_std:.3g}  best_layer_candidate={candidate_score_layer.max():.3f}")

    # Add a cross-layer score using the same three independent desiderata.
    gains = np.array([float(r["sigma"]) for r in rows], dtype=np.float64)
    usages = np.array([float(r["usage_ratio"]) for r in rows], dtype=np.float64)
    peaks = np.array([float(r["lex_peak_abs_z"]) for r in rows], dtype=np.float64)
    nonlexes = np.array([float(r["nonlex_random_p"]) for r in rows], dtype=np.float64)
    rg = rank01_desc(gains)
    ru = rank01_desc(usages)
    if np.isfinite(nonlexes).all():
        rn = nonlexes
    else:
        rn = 1.0 - rank01_desc(peaks)
    global_scores = np.cbrt(np.clip(rg, 1e-6, 1) * np.clip(ru, 1e-6, 1) * np.clip(rn, 1e-6, 1))
    for r, score in zip(rows, global_scores):
        r["candidate_score"] = float(score)

    # Drop GPU V copies before writing/reporting.
    for layer in layers:
        svd[layer].pop("V_gpu", None)
    torch.cuda.empty_cache()

    fieldnames = [
        "layer", "sv_rank", "sigma", "sigma_sq_frac_of_operator",
        "coeff_mean", "coeff_std", "coeff_rms", "isotropic_activation_std", "usage_ratio",
        "transported_std", "transported_rms",
        "lex_peak_abs_z", "lex_top1_margin_z", "lex_entropy_norm", "lex_topk_mass",
        "nonlex_random_p", "candidate_score_layer", "candidate_score",
        "pos_tokens", "neg_tokens", "pos_contexts", "neg_contexts",
    ]
    write_csv(out / "directions.csv", rows, fieldnames)
    candidate_rows = sorted(rows, key=lambda r: float(r["candidate_score"]), reverse=True)
    write_csv(out / "candidates.csv", candidate_rows, fieldnames)

    layer_fields = ["layer", "top_sigma", "captured_frob_frac", "sv_triplet_rel_resid",
                    "n_activation_samples", "isotropic_activation_std", "wall_s"]
    write_csv(out / "layer_summary.csv", layer_rows, layer_fields)

    meta = {
        "model": args.model,
        "model_revision": args.model_revision,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "operator": args.operator,
        "layers": layers,
        "k": args.k,
        "exact_svd": args.exact_svd,
        "oversample": args.oversample,
        "svd_iters": args.svd_iters,
        "n_records": len(records),
        "record_pids": [r.get("pid") for r in records],
        "n_positions": len(sample_meta),
        "position_mode": args.position_mode,
        "include_special": args.include_special,
        "random_baseline_n": int(len(random_peaks)),
        "random_lex_peak_z": {
            "mean": float(random_peaks.mean()) if len(random_peaks) else None,
            "median": float(np.median(random_peaks)) if len(random_peaks) else None,
            "p10": float(np.percentile(random_peaks, 10)) if len(random_peaks) else None,
            "p90": float(np.percentile(random_peaks, 90)) if len(random_peaks) else None,
        },
        "candidate_score_definition": "global geometric mean of gain rank (sigma), activation-usage rank (usage_ratio), and random-calibrated nonlexicality; candidate_score_layer is the analogous within-layer score",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    build_html_report(out / "report.html", rows, layer_rows, meta)

    print("\n[done] outputs:")
    print(f"  {out / 'report.html'}")
    print(f"  {out / 'candidates.csv'}")
    print(f"  {out / 'directions.csv'}")
    print(f"  {out / 'directions'}/*.npz")
    print("\nTop exploratory candidates:")
    for r in candidate_rows[:12]:
        print(f"  L{int(r['layer']):02d} SV{int(r['sv_rank']):02d} score={r['candidate_score']:.3f} "
              f"sigma={r['sigma']:.3g} usage×iso={r['usage_ratio']:.2f} "
              f"nonlex_p={r['nonlex_random_p']:.2f}  +U=[{r['pos_tokens'][:100]}]")


if __name__ == "__main__":
    main()
