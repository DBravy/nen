#!/usr/bin/env python3
"""
harvest_stratified_contexts.py  (Stage 2, detection test — part 1 of 2)

For each hypothesized direction, run a HELD-OUT corpus shard through the
model, record the direction's signed activation at every content token, and
sample contexts stratified across the full activation range plus
token-matched distractors. Writes a blinded judging file and a separate
truth file for score_hypotheses.py.

Why: the blinded reading only saw the extremes, and rotated mixtures have
coherent extremes too. A hypothesis is validated only if it predicts the
direction's activation off-tail on text it was never derived from.

Inputs
  --candidates   stage2_set/candidates.csv  (from extract_stage2_contexts.py)
  --hypotheses   blinded_hypotheses.md      (committed hypotheses; used to pick conf >= --min-conf)
  Direction vectors are read from <run>/directions/LXX.npz for each candidate's run.
  Peak tokens (for distractors) are read from <run>/top_contexts.jsonl.

Corpus: streamed like the scanner but with a DIFFERENT --seed (and optional
--skip-docs) so the documents are disjoint from the 2,000 used to derive the
hypotheses. Or pass --input-jsonl with {"text": ...} lines.

Strata on the polarity-signed activation s (per direction, exact quantiles
over all harvested content tokens):
  top1        s >= q99
  high        q90 <= s < q99
  mid         q40 <= s < q60
  low         s < q20
  distractor  token string is one of the direction's peak tokens AND s < q60
Ground truth for scoring: positives = top1+high, negatives = mid+low+distractor.

Layer convention: --hs-offset 1 means lens layer L == hidden_states[L+1]
(output of block L), which matches a lens fitted on layers 0..n_layers-2.
The script checks this by comparing harvested RMS activation per direction
to the scanner's rms_activation column and warns if the ratio is far from 1;
if it is, try --hs-offset 0.

Example:
  python harvest_stratified_contexts.py \
      --candidates stage2_set/candidates.csv --hypotheses blinded_hypotheses.md --min-conf 2 \
      --model openai/gpt-oss-20b --seed 1 --n-docs 600 --out stage2_pred

Outputs in --out:
  judging_items.jsonl   {"blind_id","layer","items":[{"item_id","context_marked"}]}  (blind)
  truth.jsonl           per-item stratum + activations, plus a meta line per candidate
  harvest_meta.json     rms self-check table and run settings
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

STRATA = ("top1", "high", "mid", "low", "distractor")


# ------------------------------------------------------------------ CLI ----


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--candidates", required=True)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--min-conf", type=int, default=2)
    p.add_argument("--conditions", nargs="+", default=["svd", "rotated", "random"])
    p.add_argument("--out", required=True)
    # model / corpus
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--hs-offset", type=int, default=1, help="lens layer L -> hidden_states[L+offset]")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--input-jsonl", default=None, help="Local corpus with {'text':...} lines.")
    p.add_argument("--seed", type=int, default=1, help="Use a DIFFERENT seed from the scan.")
    p.add_argument("--skip-docs", type=int, default=0)
    p.add_argument("--n-docs", type=int, default=600)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--char-crop", type=int, default=20000)
    p.add_argument("--skip-first", type=int, default=0, help="Skip this many content positions after BOS.")
    p.add_argument("--context-radius", type=int, default=20)
    # sampling
    p.add_argument("--per-stratum", type=int, default=10)
    p.add_argument("--n-distractors", type=int, default=10)
    p.add_argument("--n-peak-tokens", type=int, default=5)
    p.add_argument("--peak-rank-max", type=int, default=32)
    p.add_argument("--sample-seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------- candidate setup ----


def load_hypothesis_conf(path: Path) -> dict[str, int]:
    pat = re.compile(r"^([0-9a-f]{8}) L\d+[+-] \| .* \| ([0-3]) \| \w+\s*$")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    if not out:
        raise SystemExit(f"No hypothesis lines parsed from {path}")
    return out


def parse_candidate_name(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"L(\d+)_SV(\d+)", name)
    if not m:
        raise ValueError(f"unexpected candidate name {name}")
    return int(m.group(1)), int(m.group(2)) - 1  # layer, sv_index_0


def select_candidates(args) -> list[dict]:
    conf = load_hypothesis_conf(Path(args.hypotheses))
    rows = list(csv.DictReader(open(args.candidates, encoding="utf-8", newline="")))
    sel = []
    for r in rows:
        bid = r["blind_id"]
        if conf.get(bid, -1) < args.min_conf:
            continue
        if r["condition"] not in args.conditions:
            continue
        layer, idx = parse_candidate_name(r["candidate"])
        sel.append(
            {
                "blind_id": bid,
                "candidate": r["candidate"],
                "run": r["run"],
                "layer": layer,
                "sv_index_0": idx,
                "polarity": (r.get("selected_tail_polarity") or "positive").strip(),
                "sign": 1.0 if (r.get("selected_tail_polarity") or "positive").strip() == "positive" else -1.0,
                "conf": conf[bid],
            }
        )
    if not sel:
        raise SystemExit("No candidates selected; check --min-conf / --conditions.")
    return sel


def load_direction(run: str, layer: int, idx: int) -> np.ndarray:
    path = Path(run) / "directions" / f"L{layer:02d}.npz"
    z = np.load(path)
    V = np.asarray(z["V"], dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f"{path}: V must be 2D")
    if V.shape[0] < V.shape[1]:
        V = V.T  # -> [d_model, k]
    return V[:, idx].copy()


def load_peak_tokens(cands: list[dict], n_peak: int, rank_max: int) -> None:
    by_run: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        by_run[c["run"]].append(c)
    for run, cs in by_run.items():
        path = Path(run) / "top_contexts.jsonl"
        wanted = {(c["candidate"], c["polarity"]): collections.Counter() for c in cs}
        if not path.exists():
            print(f"[warn] {path} missing; no peak tokens for {len(cs)} candidates", file=sys.stderr)
            for c in cs:
                c["peak_tokens"] = []
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (o.get("candidate"), o.get("polarity"))
                if key in wanted and int(o.get("rank_within_polarity", 10**9)) <= rank_max:
                    tok = o.get("token")
                    if tok is not None:
                        wanted[key][tok] += 1
        for c in cs:
            c["peak_tokens"] = [t for t, _ in wanted[(c["candidate"], c["polarity"])].most_common(n_peak)]


# ------------------------------------------------------- corpus streaming ---


def stream_texts(args):
    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = o.get("text")
                if t:
                    yield t, {k: v for k, v in o.items() if k != "text"}
        return
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t, {k: ex[k] for k in ("id", "url") if k in ex}


def make_windows(args, tokenizer):
    """Yield (token_ids list incl. BOS, source_meta)."""
    rng = random.Random(args.seed)
    bos = tokenizer.bos_token_id
    n_out = 0
    skipped = 0
    for text, meta in stream_texts(args):
        if skipped < args.skip_docs:
            skipped += 1
            continue
        text = text[: args.char_crop]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        room = args.max_seq_len - (1 if bos is not None else 0)
        if len(ids) < 8:
            continue
        if len(ids) > room:
            start = rng.randint(0, len(ids) - room)
            ids = ids[start : start + room]
        if bos is not None:
            ids = [bos] + ids
        yield ids, meta
        n_out += 1
        if n_out >= args.n_docs:
            break


# ---------------------------------------------------- activation collection --


def collect_activations(args, cands: list[dict]):
    """Forward pass over held-out windows. Returns (windows, metas, acts, tokpos)
    where acts is [n_tokens, n_cands] raw projections and tokpos is a list of
    (window_index, position)."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device)
    model.eval()

    layers = sorted({c["layer"] for c in cands})
    V_by_layer: dict[int, torch.Tensor] = {}
    cols_by_layer: dict[int, list[int]] = {}
    for L in layers:
        idxs = [i for i, c in enumerate(cands) if c["layer"] == L]
        mat = np.stack([load_direction(c["run"], c["layer"], c["sv_index_0"]) for c in (cands[i] for i in idxs)], axis=1)
        V_by_layer[L] = torch.from_numpy(mat).to(args.device, dtype=torch.float32)
        cols_by_layer[L] = idxs

    special = set(tokenizer.all_special_ids)
    windows, metas, act_chunks, tokpos = [], [], [], []
    n_cands = len(cands)
    with torch.no_grad():
        for w_i, (ids, meta) in enumerate(make_windows(args, tokenizer)):
            inp = torch.tensor([ids], device=args.device)
            out = model(input_ids=inp, output_hidden_states=True)
            hs = out.hidden_states
            keep = [p for p in range(len(ids)) if ids[p] not in special and p >= 1 + args.skip_first]
            if not keep:
                continue
            A = np.zeros((len(keep), n_cands), dtype=np.float32)
            kp = torch.tensor(keep, device=args.device)
            for L in layers:
                H = hs[L + args.hs_offset][0].index_select(0, kp).to(torch.float32)
                proj = (H @ V_by_layer[L]).cpu().numpy()
                A[:, cols_by_layer[L]] = proj
            act_chunks.append(A)
            tokpos.extend((w_i, p) for p in keep)
            windows.append(ids)
            metas.append(meta)
            if (w_i + 1) % 50 == 0:
                print(f"[harvest] windows={w_i+1} tokens={sum(len(a) for a in act_chunks):,}", flush=True)
    acts = np.concatenate(act_chunks, axis=0) if act_chunks else np.zeros((0, n_cands), np.float32)
    return tokenizer, windows, metas, acts, tokpos


# --------------------------------------------------------- offline steps ---


def render_context(tokenizer, ids: list[int], pos: int, radius: int) -> tuple[str, str]:
    left = tokenizer.decode(ids[max(0, pos - radius) : pos])
    center = tokenizer.decode([ids[pos]])
    right = tokenizer.decode(ids[pos + 1 : pos + 1 + radius])
    return center, f"{left}⟦{center}⟧{right}"


def build_items(cands, acts, tokpos, windows, tokenizer, per_stratum, n_distractors, radius, seed):
    """Stratify each direction's signed activations and sample contexts."""
    rng = random.Random(seed)
    n_tok = acts.shape[0]
    # decode each token string once (needed for distractor matching)
    tok_cache: dict[int, str] = {}

    def tok_str(tid: int) -> str:
        if tid not in tok_cache:
            tok_cache[tid] = tokenizer.decode([tid])
        return tok_cache[tid]

    token_strs = [tok_str(windows[w][p]) for (w, p) in tokpos]
    results = []
    for j, c in enumerate(cands):
        s = acts[:, j] * c["sign"]
        q = {k: float(np.quantile(s, v)) for k, v in (("q20", 0.2), ("q40", 0.4), ("q60", 0.6), ("q90", 0.9), ("q99", 0.99))}
        pools = {
            "top1": np.where(s >= q["q99"])[0],
            "high": np.where((s >= q["q90"]) & (s < q["q99"]))[0],
            "mid": np.where((s >= q["q40"]) & (s < q["q60"]))[0],
            "low": np.where(s < q["q20"])[0],
        }
        peak = set(c.get("peak_tokens", []))
        used: set[int] = set()
        items = []
        for stratum in ("top1", "high", "mid", "low"):
            pool = [int(i) for i in pools[stratum] if int(i) not in used]
            rng.shuffle(pool)
            for i in pool[:per_stratum]:
                used.add(i)
                items.append((stratum, i))
        if peak:
            dpool = [i for i in range(n_tok) if i not in used and s[i] < q["q60"] and token_strs[i] in peak]
            rng.shuffle(dpool)
            for i in dpool[:n_distractors]:
                used.add(i)
                items.append(("distractor", i))
        rng.shuffle(items)
        records = []
        for k, (stratum, i) in enumerate(items):
            w, p = tokpos[i]
            center, marked = render_context(tokenizer, windows[w], p, radius)
            records.append(
                {
                    "item_id": f"{c['blind_id']}-{k:03d}",
                    "stratum": stratum,
                    "signed_activation": float(s[i]),
                    "raw_activation": float(acts[i, j]),
                    "token": center,
                    "window_index": int(w),
                    "position": int(p),
                    "context_marked": marked,
                }
            )
        results.append(
            {
                "cand": c,
                "quantiles": q,
                "rms": float(np.sqrt(np.mean(acts[:, j] ** 2))) if n_tok else math.nan,
                "n_tokens": int(n_tok),
                "items": records,
            }
        )
    return results


def rms_check(results) -> list[dict]:
    rows = []
    cache: dict[str, dict[str, float]] = {}
    for r in results:
        c = r["cand"]
        run = c["run"]
        if run not in cache:
            cache[run] = {}
            for name in ("selectivity_rankings.csv", "sv_rankings.csv"):
                path = Path(run) / name
                if path.exists():
                    for row in csv.DictReader(open(path, encoding="utf-8", newline="")):
                        try:
                            cache[run][row["candidate"]] = float(row["rms_activation"])
                        except (KeyError, ValueError):
                            pass
                    break
        ref = cache[run].get(c["candidate"], math.nan)
        ratio = r["rms"] / ref if ref and not math.isnan(ref) else math.nan
        rows.append({"blind_id": c["blind_id"], "candidate": c["candidate"], "rms_harvest": r["rms"], "rms_scan": ref, "ratio": ratio})
    ratios = [x["ratio"] for x in rows if not math.isnan(x["ratio"])]
    if ratios:
        med = float(np.median(ratios))
        print(f"[rms-check] median harvest/scan rms ratio = {med:.3f} over {len(ratios)} directions")
        if not (0.6 <= med <= 1.6):
            print("[rms-check] WARNING: ratio far from 1 — likely a layer-convention mismatch; try the other --hs-offset", file=sys.stderr)
    return rows


def write_outputs(out_dir: Path, results, args, rms_rows) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "judging_items.jsonl").open("w", encoding="utf-8") as fj, \
         (out_dir / "truth.jsonl").open("w", encoding="utf-8") as ft:
        for r in results:
            c = r["cand"]
            fj.write(json.dumps({
                "blind_id": c["blind_id"],
                "layer": c["layer"],
                "items": [{"item_id": it["item_id"], "context_marked": it["context_marked"]} for it in r["items"]],
            }, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "blind_id": c["blind_id"], "meta": {
                    "candidate": c["candidate"], "run": c["run"], "polarity": c["polarity"],
                    "peak_tokens": c.get("peak_tokens", []), "quantiles": r["quantiles"],
                    "rms": r["rms"], "n_tokens": r["n_tokens"],
                    "n_by_stratum": dict(collections.Counter(it["stratum"] for it in r["items"])),
                }}, ensure_ascii=False) + "\n")
            for it in r["items"]:
                ft.write(json.dumps({"blind_id": c["blind_id"], **it}, ensure_ascii=False) + "\n")
    meta = {"args": {k: v for k, v in vars(args).items()}, "rms_check": rms_rows, "n_candidates": len(results)}
    (out_dir / "harvest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[out] {out_dir}/judging_items.jsonl (blind), truth.jsonl, harvest_meta.json")


def main() -> None:
    args = parse_args()
    cands = select_candidates(args)
    print(f"[select] {len(cands)} candidates with conf >= {args.min_conf}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(collections.Counter(c['run'] for c in cands).items())))
    load_peak_tokens(cands, args.n_peak_tokens, args.peak_rank_max)
    tokenizer, windows, metas, acts, tokpos = collect_activations(args, cands)
    print(f"[harvest] done: {len(windows)} windows, {acts.shape[0]:,} content tokens")
    results = build_items(cands, acts, tokpos, windows, tokenizer,
                          args.per_stratum, args.n_distractors, args.context_radius, args.sample_seed)
    rms_rows = rms_check(results)
    write_outputs(Path(args.out), results, args, rms_rows)


if __name__ == "__main__":
    main()
