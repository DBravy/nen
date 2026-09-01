#!/usr/bin/env python3
"""
score_hypotheses.py  (Stage 2, detection test — part 2 of 2)

Judge each committed hypothesis against its held-out contexts (from
harvest_stratified_contexts.py), blind to activations and conditions, then
score how well the judge's ratings predict the true activations.

Per candidate:
  auc             judge rating separates positives (top1+high strata) from
                  negatives (mid+low+distractor)          [primary]
  spearman        rank correlation of rating with signed activation, all items
  distractor_auc  positives vs token-matched distractors only (sense vs string)
  lexical_auc     baseline: indicator(marked token in peak tokens) as the
                  predictor — what a pure string-matcher would score

Judges:
  --judge anthropic   Anthropic Messages API (env ANTHROPIC_API_KEY; pip install anthropic)
  --judge openai      OpenAI chat API      (env OPENAI_API_KEY;    pip install openai)
  --judge manual      writes manual_labels.csv for a human to fill (rating 0-10),
                      then reads it back on the next run
  --judge mock        PIPELINE TEST ONLY: lexical rule + noise. Never report.

Ratings are cached in <out>/ratings.jsonl, so re-runs skip judged items.

Blinding: run without --unblind to judge and score. Add --unblind
stage2_set/candidates.csv only for the final report, which compares
conditions (svd vs rotated vs random) with a one-sided Mann-Whitney and
lists finalists.

Example:
  python score_hypotheses.py --items stage2_pred/judging_items.jsonl \
      --truth stage2_pred/truth.jsonl --hypotheses blinded_hypotheses.md \
      --judge anthropic --judge-model claude-sonnet-5 --out stage2_pred/scores
  python score_hypotheses.py ... --unblind stage2_set/candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

_NORM = NormalDist()

PROMPT_HEADER = (
    "You are evaluating a hypothesis about what a direction inside a language model detects.\n"
    "Hypothesis: {hyp}\n\n"
    "Below are numbered text snippets. In each, one token is marked with ⟦ ⟧; that is the exact "
    "position where the direction is measured. Using ONLY the hypothesis, rate for each snippet how "
    "strongly the direction should be active at the marked token, from 0 (should be inactive) to 10 "
    "(should be maximally active). Judge the marked token in its context, not the snippet overall.\n"
    "Respond with a single JSON object mapping snippet number to integer rating, e.g. {{\"1\": 7, \"2\": 0}}. "
    "No other text.\n\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--items", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", default="anthropic", choices=["anthropic", "openai", "manual", "mock"])
    p.add_argument("--judge-model", default="claude-sonnet-5")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unblind", default=None, help="candidates.csv; enables per-condition report.")
    p.add_argument("--finalist-auc", type=float, default=0.80)
    p.add_argument("--finalist-distractor-auc", type=float, default=0.70)
    return p.parse_args()


# --------------------------------------------------------------- loading ---


def load_hypotheses(path: Path) -> dict[str, str]:
    pat = re.compile(r"^([0-9a-f]{8}) L\d+[+-] \| (.*) \| [0-3] \| \w+\s*$")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def load_items(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["blind_id"]] = o
    return out


def load_truth(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    meta, items = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if "meta" in o:
            meta[o["blind_id"]] = o["meta"]
        else:
            items[o["item_id"]] = o
    return meta, items


def load_ratings(path: Path) -> dict[str, float]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                r = float(o["rating"]) if o.get("rating") is not None else math.nan
                if not math.isnan(r):
                    out[o["item_id"]] = r  # NaN = failed batch; leave uncached
    return out


# ---------------------------------------------------------------- judges ---


def _balanced_objects(text: str):
    start = text.find("{")
    while start != -1:
        depth = 0
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return
        yield text[start : end + 1]
        start = text.find("{", end + 1)


def parse_rating_json(text: str, n: int) -> dict[int, float] | None:
    """Accept {"1": 7, ...}, {"snippet 1": "7", ...}, or [7, 0, ...]."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    obj = None
    for cand in [text, *_balanced_objects(text)]:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            obj = parsed
            break
    if obj is None:
        m = re.search(r"\[[\d\s,.\-]+\]", text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        return None
    out: dict[int, float] = {}
    if isinstance(obj, list):
        for i, v in enumerate(obj, 1):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(obj, dict):
        for k, v in obj.items():
            km = re.search(r"\d+", str(k))
            if not km:
                continue
            if isinstance(v, dict):  # e.g. {"rating": 7, "reason": ...}
                v = v.get("rating", v.get("score"))
            try:
                out[int(km.group(0))] = float(v)
            except (TypeError, ValueError):
                continue
    return out if len(out) >= max(1, int(0.5 * n)) else None


def call_anthropic(model: str, prompt: str, max_tokens: int = 8000) -> str:
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        system="You output only a JSON object mapping snippet numbers to integer ratings 0-10. No prose, no code fences.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
    if not text.strip():
        kinds = [getattr(b, "type", "?") for b in msg.content]
        text = f"<<EMPTY stop_reason={getattr(msg, 'stop_reason', '?')} blocks={kinds}>>"
    return text


def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI()
    r = client.chat.completions.create(model=model,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


def judge_batch(args, hyp: str, batch: list[dict], mock_ctx: dict | None) -> dict[str, float]:
    """Return item_id -> rating for one batch."""
    if args.judge == "mock":
        rng = random.Random(hash((args.seed, batch[0]["item_id"])) & 0xFFFFFFFF)
        peaks = set(mock_ctx.get("peak_tokens", [])) if mock_ctx else set()
        out = {}
        for it in batch:
            m = re.search(r"⟦(.*?)⟧", it["context_marked"])
            tok = m.group(1) if m else ""
            base = 8.0 if tok in peaks else 2.0
            out[it["item_id"]] = max(0.0, min(10.0, base + rng.gauss(0, 1.5)))
        return out

    return _judge_recursive(args, hyp, batch)


def _judge_once(args, hyp: str, batch: list[dict]) -> tuple[dict[int, float] | None, bool]:
    """One API call. Returns (parsed ratings by 1-based index or None, refused flag)."""
    prompt = PROMPT_HEADER.format(hyp=hyp)
    for k, it in enumerate(batch, 1):
        prompt += f"{k}. {it['context_marked'].replace(chr(10), ' ')}\n"
    raw_log = Path(args.out) / "judge_raw.log"
    for attempt in range(args.max_retries):
        try:
            text = call_anthropic(args.judge_model, prompt) if args.judge == "anthropic" else call_openai(args.judge_model, prompt)
        except Exception as exc:  # network / rate limit
            wait = 2 ** attempt
            print(f"[judge] API error ({exc}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        with raw_log.open("a", encoding="utf-8") as fl:
            fl.write(json.dumps({"batch_first": batch[0]["item_id"], "n": len(batch), "attempt": attempt, "raw": text}, ensure_ascii=False) + "\n")
        if text.startswith("<<EMPTY stop_reason=refusal"):
            return None, True  # deterministic; do not retry
        parsed = parse_rating_json(text, len(batch))
        if parsed is not None:
            return parsed, False
        print(f"[judge] unparseable (attempt {attempt+1}, n={len(batch)}); excerpt: {text[:200].replace(chr(10),' ')!r}", file=sys.stderr)
    return None, False


def _judge_recursive(args, hyp: str, batch: list[dict]) -> dict[str, float]:
    parsed, refused = _judge_once(args, hyp, batch)
    if parsed is not None:
        return {it["item_id"]: parsed.get(k, math.nan) for k, it in enumerate(batch, 1)}
    if len(batch) == 1:
        it = batch[0]
        with (Path(args.out) / "judge_refused.log").open("a", encoding="utf-8") as fl:
            fl.write(json.dumps({"item_id": it["item_id"], "refused": refused, "context_marked": it["context_marked"]}, ensure_ascii=False) + "\n")
        print(f"[judge] item {it['item_id']} {'refused' if refused else 'unparseable'} -> NaN", file=sys.stderr)
        return {it["item_id"]: math.nan}
    mid = len(batch) // 2
    print(f"[judge] batch of {len(batch)} {'refused' if refused else 'unparseable'}; bisecting", file=sys.stderr)
    out = _judge_recursive(args, hyp, batch[:mid])
    out.update(_judge_recursive(args, hyp, batch[mid:]))
    return out


def run_manual(out_dir: Path, items: dict[str, dict], hyps: dict[str, str], ratings: dict[str, float]) -> dict[str, float]:
    path = out_dir / "manual_labels.csv"
    if path.exists():
        for row in csv.DictReader(open(path, encoding="utf-8", newline="")):
            try:
                if row.get("rating", "").strip() != "":
                    ratings[row["item_id"]] = float(row["rating"])
            except ValueError:
                pass
        print(f"[manual] read {len(ratings)} ratings from {path}")
        return ratings
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "hypothesis", "item_id", "context_marked", "rating"])
        for bid, o in items.items():
            for it in o["items"]:
                w.writerow([bid, hyps.get(bid, ""), it["item_id"], it["context_marked"].replace("\n", " "), ""])
    print(f"[manual] wrote {path}; fill the 'rating' column (0-10) and re-run.")
    sys.exit(0)


# --------------------------------------------------------------- scoring ---


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


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    allv = np.concatenate([pos, neg])
    r = ranks_with_ties(allv)
    u = r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return math.nan
    rx, ry = ranks_with_ties(x), ranks_with_ties(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (rx.std() * ry.std()))


def mann_whitney_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    allv = np.concatenate([x, y])
    r = ranks_with_ties(allv)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie = float(((counts.astype(float) ** 3) - counts).sum()) / (n * (n - 1))
    var = n1 * n2 / 12.0 * ((n + 1) - tie)
    rbc = 2.0 * u1 / (n1 * n2) - 1.0
    if var <= 0:
        return 0.5, rbc
    z = (u1 - n1 * n2 / 2.0 - 0.5) / math.sqrt(var)
    return 1.0 - _NORM.cdf(z), rbc


POS_STRATA = {"top1", "high"}
NEG_STRATA = {"mid", "low", "distractor"}


def score_candidate(bid: str, truth_items: list[dict], ratings: dict[str, float], meta: dict) -> dict:
    rat = np.array([ratings.get(t["item_id"], math.nan) for t in truth_items])
    s = np.array([t["signed_activation"] for t in truth_items])
    strata = np.array([t["stratum"] for t in truth_items])
    peaks = set(meta.get("peak_tokens", []))
    lex = np.array([1.0 if t["token"] in peaks else 0.0 for t in truth_items])
    ok = ~np.isnan(rat)
    pos = np.array([st in POS_STRATA for st in strata]) & ok
    neg = np.array([st in NEG_STRATA for st in strata]) & ok
    dis = (strata == "distractor") & ok
    return {
        "blind_id": bid,
        "n_items": int(ok.sum()),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "n_distractor": int(dis.sum()),
        "auc": auc(rat[pos], rat[neg]),
        "spearman": spearman(rat, s),
        "distractor_auc": auc(rat[pos], rat[dis]) if dis.sum() >= 3 else math.nan,
        "lexical_auc": auc(lex[pos], lex[neg]),
        "mean_rating_pos": float(np.nanmean(rat[pos])) if pos.sum() else math.nan,
        "mean_rating_mid_low": float(np.nanmean(rat[neg & ~dis])) if (neg & ~dis).sum() else math.nan,
        "mean_rating_distractor": float(np.nanmean(rat[dis])) if dis.sum() else math.nan,
    }


# ------------------------------------------------------------------ main ---


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyps = load_hypotheses(Path(args.hypotheses))
    items = load_items(Path(args.items))
    meta, truth = load_truth(Path(args.truth))
    ratings_path = out_dir / "ratings.jsonl"
    ratings = load_ratings(ratings_path)
    if args.judge == "mock":
        print("[WARNING] mock judge: lexical rule + noise, for pipeline testing only.", file=sys.stderr)

    missing = {bid for bid in items if bid not in hyps}
    if missing:
        print(f"[warn] {len(missing)} candidates have no hypothesis text; skipped: {sorted(missing)[:5]}...", file=sys.stderr)

    if args.judge == "manual":
        ratings = run_manual(out_dir, {b: o for b, o in items.items() if b in hyps}, hyps, ratings)
    else:
        rng = random.Random(args.seed)
        todo = 0
        with ratings_path.open("a", encoding="utf-8") as fr:
            for bid, o in items.items():
                if bid not in hyps:
                    continue
                pending = [it for it in o["items"] if it["item_id"] not in ratings]
                if not pending:
                    continue
                rng.shuffle(pending)
                for i in range(0, len(pending), args.batch_size):
                    batch = pending[i : i + args.batch_size]
                    got = judge_batch(args, hyps[bid], batch, meta.get(bid))
                    for iid, r in got.items():
                        ratings[iid] = r
                        fr.write(json.dumps({"item_id": iid, "rating": r}) + "\n")
                    fr.flush()
                    todo += len(batch)
                    print(f"[judge] {bid}: {min(i+args.batch_size, len(pending))}/{len(pending)}", flush=True)
        print(f"[judge] rated {todo} new items; total cached {len(ratings)}")

    # Score.
    by_bid: dict[str, list[dict]] = {}
    for t in truth.values():
        by_bid.setdefault(t["blind_id"], []).append(t)
    scores = [score_candidate(bid, by_bid[bid], ratings, meta.get(bid, {})) for bid in items if bid in hyps and bid in by_bid]
    scores.sort(key=lambda r: (-(r["auc"] if not math.isnan(r["auc"]) else -1)))
    keys = list(scores[0].keys()) if scores else []
    with (out_dir / "hypothesis_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in scores:
            w.writerow(r)
    aucs = np.array([r["auc"] for r in scores], float)
    print(f"\n[scores] {len(scores)} candidates | median AUC={np.nanmedian(aucs):.3f} | "
          f">= {args.finalist_auc}: {int(np.nansum(aucs >= args.finalist_auc))}")
    print(f"[out] {out_dir}/hypothesis_scores.csv")

    if not args.unblind:
        print("\nBlind mode: rerun with --unblind candidates.csv for the per-condition report.")
        return

    # Unblinded report.
    cond = {}
    for row in csv.DictReader(open(args.unblind, encoding="utf-8", newline="")):
        cond[row["blind_id"]] = (row["condition"], row["candidate"], row.get("sv_rank_1based", ""))
    for r in scores:
        c = cond.get(r["blind_id"], ("?", "?", ""))
        r["condition"], r["candidate"], r["sv_rank"] = c
    lines = ["", "=" * 72, "Detection test — per-condition report", "=" * 72]
    by_cond: dict[str, list[dict]] = {}
    for r in scores:
        by_cond.setdefault(r["condition"], []).append(r)
    for metric in ("auc", "distractor_auc", "spearman", "lexical_auc"):
        lines.append(f"\n{metric}:")
        for c in ("svd", "rotated", "random"):
            if c in by_cond:
                v = np.array([r[metric] for r in by_cond[c]], float)
                lines.append(f"  {c:>8}: n={len(v):>3} median={np.nanmedian(v):.3f}  q25={np.nanpercentile(v,25):.3f}  q75={np.nanpercentile(v,75):.3f}")
        if "svd" in by_cond and "rotated" in by_cond:
            p, rbc = mann_whitney_greater(np.array([r[metric] for r in by_cond['svd']], float),
                                          np.array([r[metric] for r in by_cond['rotated']], float))
            lines.append(f"  svd > rotated: rbc={rbc:+.3f}  p={p:.2e}")
    lines.append("\nFinalists (svd, AUC >= %.2f and distractor AUC >= %.2f or n/a):" % (args.finalist_auc, args.finalist_distractor_auc))
    for r in by_cond.get("svd", []):
        d_ok = math.isnan(r["distractor_auc"]) or r["distractor_auc"] >= args.finalist_distractor_auc
        if not math.isnan(r["auc"]) and r["auc"] >= args.finalist_auc and d_ok:
            lines.append(f"  {r['candidate']:>10} rank={r['sv_rank']:>3} auc={r['auc']:.3f} distr={r['distractor_auc']:.3f} "
                         f"lex={r['lexical_auc']:.3f} | {hyps[r['blind_id']][:60]}")
    lines.append("\nRead: AUC ~0.5 = coherent tail, no off-tail prediction (illusion signature).")
    lines.append("      AUC high but lexical_auc equally high = the hypothesis is a string match.")
    lines.append("      AUC high, distractor_auc high, lexical_auc lower = sense/condition-level feature.")
    report = "\n".join(lines)
    print(report)
    (out_dir / "detection_report.txt").write_text(report + "\n", encoding="utf-8")
    with (out_dir / "hypothesis_scores_unblinded.csv").open("w", encoding="utf-8", newline="") as f:
        keys = list(scores[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in scores:
            w.writerow(r)
    print(f"[out] {out_dir}/detection_report.txt, hypothesis_scores_unblinded.csv")


if __name__ == "__main__":
    main()