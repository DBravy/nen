#!/usr/bin/env python3
"""
stratified_sense_test.py

Genre-stratified sense separation. CPU-only; reuses the sense_* harvest.

The question: does a J-lens SVD axis separate word senses only *between*
genres (because genre is all it carries), while the residual stream separates
them *within* genre? Operationally, for the deontic-vs-epistemic "may"
contrast:

  1. Score every occurrence along a stratifier axis (default L07_SV46, the
     genre axis: technical/structural nouns vs asking/quotative text).
  2. Split occurrences into strata (median split or terciles), by default at
     the *document* level (genre is a document property; this also keeps
     bootstrap clusters intact).
  3. Within each stratum, compute de-vs-ep AUC for:
       * the stratifier axis itself        (prediction: collapses toward 0.5)
       * other J axes, e.g. L07_SV39       (prediction: collapses if its weak
                                            pooled separation was genre-borne)
       * the contrast-mean probe, global   (prediction: stays high)
       * the probe re-trained per stratum  (the clean within-genre ceiling)
       * the global probe with the
         stratifier direction projected out
       * a random-direction null           (calibrates "near 0.5" at this n)
  4. Report the deontic share per stratum (the size of the confound) and
     document-cluster bootstrap CIs throughout.

The asymmetric prediction: axis rows collapse, probe rows hold. Note that the
stratifier axis is range-restricted within its own strata partly by
construction; the argument therefore rests on the *probe under the identical
restriction* staying high, and on non-stratifier axes (SV39) collapsing too.
The random-null row tells you what "no separation" looks like at each
stratum's sample size.

Inputs (all already produced by test_sense_selectivity.py):
  --out DIR                harvest/analyze output dir (e.g. sense_may). The
                           script auto-discovers the residual arrays and the
                           per-occurrence scores_*_oK.csv (which fixes row
                           order, labels, docs, variants).
  --directions-dir DIR     the SVD bank dir (scan_svd_r0/directions).

Before trusting anything, the script re-computes H @ v for the axis named in
the scores CSV and checks it against the CSV's svd_axis column. If that
alignment check fails, nothing else is reported.

Example (the headline run):

    python stratified_sense_test.py \
        --out sense_may --directions-dir scan_svd_r0/directions \
        --stratify-axis L07_SV46 --test-axes L07_SV39 L07_SV46 \
        --offsets 0 --n-strata 2 --plots

Then, if counts allow, --n-strata 3; and a second run with
--stratify-axis L07_SV39 is possible but less interpretable (SV39 mixes
token-detection with whatever else it carries).

If auto-discovery of the residual array fails or is ambiguous, pass
--resid PATH[:KEY] (npz key or torch dict key). If the array is 3-D
[n_occ, n_offsets, d], also pass --offset-index for the slice to use.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

SENSES = ["deontic", "epistemic", "month", "optative", "other", "uncertain"]


# =============================================================================
# Small helpers (conventions copied from test_sense_selectivity.py)
# =============================================================================


def parse_axis(name: str) -> tuple[int, int]:
    """'L07_SV46' -> (7, 46) with 1-based rank."""
    m = re.fullmatch(r"L(\d+)_SV(\d+)", name.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad axis name {name!r}; expected like L07_SV46")
    return int(m.group(1)), int(m.group(2))


def load_bank_V(directions_dir: Path, layer: int) -> np.ndarray:
    z = np.load(directions_dir / f"L{layer:02d}.npz")
    V = np.asarray(z["V"], dtype=np.float32)
    if V.ndim != 2:
        raise ValueError("V must be 2D")
    if V.shape[0] < V.shape[1]:
        V = V.T
    return V  # [d_model, k]


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


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
    """P(score_pos > score_neg), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    allv = np.concatenate([pos, neg]).astype(np.float64)
    r = ranks_with_ties(allv)
    n1, n2 = len(pos), len(neg)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n2))


def cluster_bootstrap_auc(
    scores: np.ndarray,
    is_pos: np.ndarray,
    is_neg: np.ndarray,
    docs: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI for AUC by resampling documents with replacement."""
    if n_boot <= 0:
        return math.nan, math.nan
    keep = (is_pos | is_neg) & np.isfinite(scores)
    if keep.sum() == 0:
        return math.nan, math.nan
    s, ip, ing, d = scores[keep], is_pos[keep], is_neg[keep], docs[keep]
    uniq = np.unique(d)
    by_doc = {u: np.where(d == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_doc[u] for u in pick])
        a = s[idx][ip[idx]]
        b = s[idx][ing[idx]]
        if len(a) == 0 or len(b) == 0:
            continue
        vals.append(auc(a, b))
    if len(vals) < max(20, n_boot // 10):
        return math.nan, math.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =============================================================================
# Loading: per-occurrence metadata (scores CSV) and residual arrays
# =============================================================================


def load_meta(out: Path, offset: int, meta_csv: str | None) -> dict:
    if meta_csv:
        path = Path(meta_csv)
        if not path.exists():
            raise SystemExit(f"[meta] {path} not found")
    else:
        cands = sorted(out.glob(f"scores_*_o{offset}.csv"))
        if not cands:
            raise SystemExit(
                f"[meta] no scores_*_o{offset}.csv in {out}. Run the analyze step of "
                f"test_sense_selectivity.py first (it writes per-occurrence CSVs whose row "
                f"order matches the harvested residuals), or pass --meta-csv."
            )
        path = cands[0]
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[meta] {path} is empty")
    need = {"id", "document_index", "sense", "variant", "valid", "svd_axis"}
    missing = need - set(rows[0].keys())
    if missing:
        raise SystemExit(f"[meta] {path} lacks columns {sorted(missing)}")
    m = re.search(r"scores_(L\d+_SV\d+)_o(\d+)\.csv$", path.name)
    axis_in_name = m.group(1) if m else None

    def _f(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return math.nan

    meta = {
        "path": path,
        "axis_in_name": axis_in_name,
        "ids": [r["id"] for r in rows],
        "docs": np.array([r["document_index"] for r in rows], dtype=object),
        "sense": np.array([r["sense"].strip() for r in rows], dtype=object),
        "variant": np.array([r["variant"] for r in rows], dtype=object),
        "valid": np.array([r["valid"].strip() in ("1", "True", "true") for r in rows], dtype=bool),
        "svd_axis": np.array([_f(r["svd_axis"]) for r in rows], dtype=np.float64),
        "context": [r.get("context_marked", "") for r in rows],
    }
    print(f"[meta] {path.name}: n={len(rows)}  (row order defines residual alignment)")
    return meta


def _iter_candidate_arrays(out: Path):
    """Yield (name, array) for every 2-D/3-D array stored under out."""
    for p in sorted(out.rglob("*")):
        if p.suffix == ".npz":
            try:
                z = np.load(p, allow_pickle=False)
            except Exception:
                continue
            for k in z.files:
                try:
                    arr = z[k]
                except Exception:
                    continue
                if arr.ndim in (2, 3):
                    yield f"{p}:{k}", arr
        elif p.suffix == ".npy":
            try:
                arr = np.load(p, mmap_mode="r")
            except Exception:
                continue
            if arr.ndim in (2, 3):
                yield str(p), arr
        elif p.suffix in (".pt", ".pth"):
            try:
                import torch  # noqa: PLC0415

                obj = torch.load(p, map_location="cpu")
            except Exception:
                continue
            items = obj.items() if isinstance(obj, dict) else [("", obj)]
            for k, v in items:
                if hasattr(v, "ndim") and v.ndim in (2, 3):
                    yield f"{p}:{k}" if k else str(p), np.asarray(v.float().numpy(), dtype=np.float32)


def _name_score(name: str, layer: int, offset: int) -> int:
    s = name.lower()
    pts = 0
    if re.search(rf"l0*{layer}(?!\d)", s):
        pts += 2
    if re.search(rf"(?:^|[^a-z\d])o(?:ff(?:set)?)?_?0*{offset}(?!\d)", s):
        pts += 2
    if any(kw in s for kw in ("resid", "hidden", "acts", "h_occ", "occ")):
        pts += 1
    return pts


def load_residuals(
    out: Path, layer: int, offset: int, n_rows: int, d_model: int,
    resid: str | None, offset_index: int | None,
) -> np.ndarray:
    """Locate/load the [n_occ, d_model] residual array matching the meta CSV rows."""
    if resid:
        path_s, _, key = resid.partition(":")
        p = Path(path_s)
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=False)
            if not key:
                if len(z.files) == 1:
                    key = z.files[0]
                else:
                    raise SystemExit(f"[resid] {p} has keys {z.files}; pass --resid {p}:KEY")
            arr = z[key]
        elif p.suffix == ".npy":
            arr = np.load(p)
        else:
            import torch  # noqa: PLC0415

            obj = torch.load(p, map_location="cpu")
            arr = np.asarray((obj[key] if key else obj).float().numpy(), dtype=np.float32)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if offset_index is None:
                raise SystemExit(
                    f"[resid] {resid} is 3-D {arr.shape}; pass --offset-index for the offset slice"
                )
            arr = arr[:, offset_index, :]
        if arr.shape != (n_rows, d_model):
            raise SystemExit(f"[resid] {resid} has shape {arr.shape}, expected ({n_rows},{d_model})")
        print(f"[resid] using {resid}  shape={arr.shape}")
        return np.asarray(arr, dtype=np.float32)

    cands = []
    for name, arr in _iter_candidate_arrays(out):
        if arr.ndim == 2 and arr.shape == (n_rows, d_model):
            cands.append((_name_score(name, layer, offset), name, arr))
    if not cands:
        raise SystemExit(
            f"[resid] no 2-D array of shape ({n_rows},{d_model}) found under {out}. "
            f"Pass --resid PATH[:KEY] pointing at the harvested residuals for "
            f"layer {layer}, offset {offset}."
        )
    cands.sort(key=lambda t: -t[0])
    best = [c for c in cands if c[0] == cands[0][0]]
    if len(best) > 1 or cands[0][0] == 0:
        print(f"[resid] candidates for layer {layer} offset {offset}:")
        for sc, name, arr in cands[:12]:
            print(f"    score={sc}  {name}  shape={arr.shape}")
        raise SystemExit("[resid] ambiguous; pass --resid PATH[:KEY] explicitly")
    sc, name, arr = best[0]
    print(f"[resid] auto-discovered {name}  shape={arr.shape}  (match score {sc})")
    return np.asarray(arr, dtype=np.float32)


def verify_alignment(H: np.ndarray, meta: dict, directions_dir: Path) -> None:
    """Recompute H @ v for the axis named in the scores CSV and compare to its column."""
    axis = meta["axis_in_name"]
    if axis is None:
        print("[verify] WARNING: could not parse an axis name from the meta CSV filename; "
              "skipping the alignment check. Consider --meta-csv with a scores_L*_SV*_o*.csv.")
        return
    layer, rank = parse_axis(axis)
    V = load_bank_V(directions_dir, layer)
    ref = meta["svd_axis"]
    ok = np.isfinite(ref)
    best_err = math.inf
    for v in (V[:, rank - 1], unit(V[:, rank - 1])):
        s = H @ v.astype(np.float32)
        err = float(np.max(np.abs(s[ok] - ref[ok])))
        best_err = min(best_err, err)
    scale = float(np.std(ref[ok])) + 1e-9
    if best_err > max(0.01, 0.01 * scale):
        raise SystemExit(
            f"[verify] ALIGNMENT CHECK FAILED for {axis}: max |H@v - csv.svd_axis| = "
            f"{best_err:.4f} (score sd {scale:.2f}). The residual array's row order does not "
            f"match the scores CSV. Fix the --resid choice before trusting any result."
        )
    print(f"[verify] alignment OK: H @ v_{axis} matches csv svd_axis (max err {best_err:.5f})")


# =============================================================================
# Probes
# =============================================================================


def doc_folds(docs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2-fold assignment by document over masked rows; -1 elsewhere."""
    uniq = np.unique(docs[mask])
    perm = rng.permutation(len(uniq))
    fold_of = {d: int(perm[i] % 2) for i, d in enumerate(uniq)}
    return np.array([fold_of.get(d, -1) if m else -1 for d, m in zip(docs, mask)], dtype=int)


def cv_contrast_mean(
    H: np.ndarray, ia: np.ndarray, ib: np.ndarray, docs: np.ndarray,
    rng: np.random.Generator, v_orth: np.ndarray | None = None, min_per_fold: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold contrast-mean scores over ia|ib rows (NaN elsewhere).

    If v_orth is given, the stratifier direction is projected out of each
    fold's probe before scoring. Returns (scores, cos-with-v_orth per fold).
    """
    mask = ia | ib
    fold = doc_folds(docs, mask, rng)
    s = np.full(len(H), np.nan, dtype=np.float32)
    coss: list[float] = []
    for k in (0, 1):
        tr = (fold == (1 - k))
        te = (fold == k)
        if (tr & ia).sum() < min_per_fold or (tr & ib).sum() < min_per_fold:
            continue
        w = H[tr & ia].mean(0) - H[tr & ib].mean(0)
        w = unit(w)
        if v_orth is not None:
            coss.append(float(abs(w @ v_orth)))
            w = unit(w - (w @ v_orth) * v_orth)
        s[te] = H[te] @ w
    return s, coss


# =============================================================================
# The stratified analysis
# =============================================================================


def fmt_auc(val: float, lo: float, hi: float, n_a: int, n_b: int, floor: int) -> str:
    if math.isnan(val):
        return f"{'---':>22}"
    ci = f"[{lo:.2f},{hi:.2f}]" if not math.isnan(lo) else "[  --  ]"
    flag = "!" if min(n_a, n_b) < floor else " "
    return f"{val:6.3f} {ci:>11}{flag}"


def analyze_offset(
    H: np.ndarray, meta: dict, args, v_strat: np.ndarray, strat_name: str,
    test_axes: list[tuple[str, np.ndarray]], rng: np.random.Generator,
    lines: list[str], rows_out: list[dict], offset: int,
) -> None:
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    a, b = args.contrast
    labels, docs, valid, variant = meta["sense"], meta["docs"], meta["valid"], meta["variant"]
    ia = valid & (labels == a)
    ib = valid & (labels == b)

    emit("-" * 100)
    emit(f"offset {offset}   contrast: {a} (n={int(ia.sum())}) vs {b} (n={int(ib.sum())})   "
         f"stratifier: {strat_name}   strata: {args.n_strata} ({args.strata_unit}-level)")
    emit("-" * 100)

    # --- same-token restriction --------------------------------------------
    if args.same_token:
        cmask = ia | ib
        vs, counts = np.unique(variant[cmask], return_counts=True)
        per = {v: (int((ia & (variant == v)).sum()), int((ib & (variant == v)).sum())) for v in vs}
        best_v = max(per, key=lambda v: min(per[v]))
        dropped_a = int(ia.sum()) - per[best_v][0]
        dropped_b = int(ib.sum()) - per[best_v][1]
        ia = ia & (variant == best_v)
        ib = ib & (variant == best_v)
        emit(f"same-token restriction: keeping variant {best_v!r} "
             f"({a} n={per[best_v][0]}, {b} n={per[best_v][1]}; dropped {dropped_a}+{dropped_b}). "
             f"variant table: " + ", ".join(f"{v!r}:{per[v][0]}/{per[v][1]}" for v in vs))
    cmask = ia | ib
    if ia.sum() < args.min_per_class or ib.sum() < args.min_per_class:
        emit(f"[skip] fewer than {args.min_per_class} per class after restriction")
        return

    # --- stratify -----------------------------------------------------------
    s_strat = H @ v_strat.astype(np.float32)
    qs = [i / args.n_strata for i in range(1, args.n_strata)]
    if args.strata_unit == "doc":
        uniq = np.unique(docs[cmask])
        doc_mean = {u: float(s_strat[cmask & (docs == u)].mean()) for u in uniq}
        edges = np.quantile(np.array([doc_mean[u] for u in uniq]), qs) if qs else np.array([])
        doc_str = {u: int(np.searchsorted(edges, doc_mean[u], side="right")) for u in uniq}
        stratum = np.array([doc_str.get(d, -1) if m else -1 for d, m in zip(docs, cmask)], dtype=int)
    else:
        edges = np.quantile(s_strat[cmask], qs) if qs else np.array([])
        stratum = np.where(cmask, np.searchsorted(edges, s_strat, side="right"), -1)

    # --- probes -------------------------------------------------------------
    s_pg, cos_pg = cv_contrast_mean(H, ia, ib, docs, rng)
    s_pr, _ = cv_contrast_mean(H, ia, ib, docs, rng, v_orth=v_strat)
    s_pw = np.full(len(H), np.nan, dtype=np.float32)
    for k in range(args.n_strata):
        sm = stratum == k
        sk, _ = cv_contrast_mean(H, ia & sm, ib & sm, docs, rng)
        s_pw[sm] = sk[sm]
    if cos_pg:
        emit(f"|cos(global probe, {strat_name})| = {np.mean(cos_pg):.3f}   "
             f"(how much of the sense probe *is* the stratifier direction)")

    # --- composition --------------------------------------------------------
    emit()
    emit(f"{'stratum':<10}{'range of ' + strat_name + ' score':<34}{'n_' + a[:3]:>7}{'n_' + b[:3]:>7}"
         f"{'docs':>7}{a[:3] + ' share':>12}")
    for k in range(args.n_strata):
        sm = stratum == k
        na, nb = int((ia & sm).sum()), int((ib & sm).sum())
        nd = len(np.unique(docs[sm])) if sm.any() else 0
        rng_lo = float(s_strat[sm].min()) if sm.any() else math.nan
        rng_hi = float(s_strat[sm].max()) if sm.any() else math.nan
        share = na / max(na + nb, 1)
        emit(f"{'S' + str(k + 1):<10}{f'[{rng_lo:+.1f}, {rng_hi:+.1f}]':<34}{na:>7}{nb:>7}{nd:>7}{share:>11.1%}")
    emit(f"(the {a[:3]}-share gradient across strata is the size of the genre-sense confound)")

    # --- directions to evaluate --------------------------------------------
    dirs: list[tuple[str, np.ndarray]] = []
    dirs.append((f"{strat_name} (stratifier)", s_strat))
    for name, v in test_axes:
        if name == strat_name:
            continue
        dirs.append((name, H @ v.astype(np.float32)))
    dirs.append(("probe (global CV)", s_pg))
    dirs.append((f"probe (global CV, minus {strat_name})", s_pr))
    dirs.append(("probe (within-stratum CV)", s_pw))

    # --- table --------------------------------------------------------------
    emit()
    hdr = f"{'direction':<42}{'pooled':>22}" + "".join(f"{'S' + str(k + 1):>22}" for k in range(args.n_strata))
    emit(hdr)
    emit(f"{'':<42}" + f"{'(AUC ' + a[:2] + '-vs-' + b[:2] + ' [95% CI])':>22}" * (args.n_strata + 1))
    for name, s in dirs:
        cells = []
        for k in [-1] + list(range(args.n_strata)):
            sm = cmask if k == -1 else (stratum == k)
            sa, sb = ia & sm, ib & sm
            ok = np.isfinite(s)
            na, nb = int((sa & ok).sum()), int((sb & ok).sum())
            val = auc(s[sa & ok], s[sb & ok])
            lo, hi = cluster_bootstrap_auc(s, sa, sb, docs, args.bootstrap, rng)
            cells.append(fmt_auc(val, lo, hi, na, nb, args.min_per_class))
            rows_out.append(dict(offset=offset, stratifier=strat_name, direction=name,
                                 stratum=("pooled" if k == -1 else f"S{k + 1}"),
                                 n_a=na, n_b=nb, auc=val, ci_lo=lo, ci_hi=hi))
        note = "  <- pooled = stratum-matched aggregate" if name.startswith("probe (within") else ""
        emit(f"{name:<42}" + "".join(cells) + note)

    # --- random null per stratum -------------------------------------------
    if args.n_random > 0:
        cells = []
        for k in [-1] + list(range(args.n_strata)):
            sm = cmask if k == -1 else (stratum == k)
            sa, sb = ia & sm, ib & sm
            seps = []
            for _ in range(args.n_random):
                g = unit(rng.standard_normal(H.shape[1]).astype(np.float32))
                sg = H[sm] @ g
                seps.append(abs(auc(sg[sa[sm]], sg[sb[sm]]) - 0.5))
            seps = np.asarray(seps)
            cells.append(f"{'med ' + format(0.5 + np.median(seps), '.3f'):>11}"
                         f"{'p95 ' + format(0.5 + np.percentile(seps, 95), '.3f'):>11}")
        emit(f"{'random null (|AUC-.5| -> AUC scale)':<42}" + "".join(cells))

    # --- qualitative spot check --------------------------------------------
    if args.show_contexts > 0:
        emit()
        emit("random contexts per stratum (eyeball the genre reading; not cherry-picked):")
        for k in range(args.n_strata):
            idx = np.where(stratum == k)[0]
            if len(idx) == 0:
                continue
            for i in rng.choice(idx, size=min(args.show_contexts, len(idx)), replace=False):
                txt = meta["context"][i][:150]
                emit(f"  S{k + 1} [{labels[i][:3]}] {txt}")

    if args.plots:
        make_plot(Path(args.out), offset, args, dirs, ia, ib, cmask, stratum, strat_name)


def make_plot(out: Path, offset: int, args, dirs, ia, ib, cmask, stratum, strat_name) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable ({exc})", file=sys.stderr)
        return
    groups = ["pooled"] + [f"S{k + 1}" for k in range(args.n_strata)]
    fig, ax = plt.subplots(figsize=(1.9 + 2.1 * len(groups), 4.2))
    width = 0.8 / len(dirs)
    for j, (name, s) in enumerate(dirs):
        vals = []
        for k in [-1] + list(range(args.n_strata)):
            sm = cmask if k == -1 else (stratum == k)
            ok = np.isfinite(s)
            v = auc(s[ia & sm & ok], s[ib & sm & ok])
            vals.append(abs(v - 0.5) + 0.5 if not math.isnan(v) else math.nan)
        x = np.arange(len(groups)) + (j - len(dirs) / 2 + 0.5) * width
        ax.bar(x, vals, width=width * 0.95, label=name)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel(f"|AUC| {args.contrast[0]} vs {args.contrast[1]} (0.5 = chance)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(f"Sense separation within {strat_name} strata, offset {offset}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = out / f"stratified_{strat_name}_o{offset}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"[plot] {p}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="sense_* dir from test_sense_selectivity.py")
    p.add_argument("--directions-dir", required=True, help="scan_svd_r0/directions")
    p.add_argument("--stratify-axis", default="L07_SV46")
    p.add_argument("--test-axes", nargs="*", default=["L07_SV39", "L07_SV46"],
                   help="axes evaluated within strata (same layer as the stratifier)")
    p.add_argument("--contrast", nargs=2, default=["deontic", "epistemic"])
    p.add_argument("--offsets", nargs="+", type=int, default=[0])
    p.add_argument("--n-strata", type=int, default=2, choices=(2, 3))
    p.add_argument("--strata-unit", choices=("doc", "occ"), default="doc",
                   help="stratify documents by their mean stratifier score (default), or occurrences directly")
    p.add_argument("--same-token", dest="same_token", action="store_true", default=True)
    p.add_argument("--no-same-token", dest="same_token", action="store_false",
                   help="disable restriction of both classes to one shared token variant")
    p.add_argument("--min-per-class", type=int, default=20,
                   help="floor per class per stratum; cells below it are flagged with '!'")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--n-random", type=int, default=200)
    p.add_argument("--show-contexts", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resid", default=None, help="PATH[:KEY] of the residual array, if auto-discovery fails")
    p.add_argument("--offset-index", type=int, default=None, help="slice index if --resid is 3-D")
    p.add_argument("--meta-csv", default=None, help="explicit scores_*_oK.csv (row order source)")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    if args.resid and len(args.offsets) > 1:
        raise SystemExit("--resid points at one array; run one offset at a time with it")

    out = Path(args.out)
    ddir = Path(args.directions_dir)
    rng = np.random.default_rng(args.seed)

    sl, sr = parse_axis(args.stratify_axis)
    layers = {sl} | {parse_axis(t)[0] for t in args.test_axes}
    if len(layers) != 1:
        raise SystemExit(f"stratifier and test axes must share one layer (got layers {sorted(layers)}); "
                         f"the residual array is per-layer. Run other layers separately.")
    V = load_bank_V(ddir, sl)
    d_model = V.shape[0]
    v_strat = unit(V[:, sr - 1])
    test_axes = []
    for t in dict.fromkeys(args.test_axes):
        _, r = parse_axis(t)
        if r > V.shape[1]:
            raise SystemExit(f"{t}: rank {r} exceeds bank k={V.shape[1]}")
        test_axes.append((t, unit(V[:, r - 1])))

    lines: list[str] = []
    rows_out: list[dict] = []
    hdr = (f"Stratified sense test: stratifier={args.stratify_axis}  test axes={[t for t, _ in test_axes]}  "
           f"contrast={args.contrast}  strata={args.n_strata} ({args.strata_unit})  seed={args.seed}")
    print("=" * 100)
    print(hdr)
    print("=" * 100)
    lines += ["=" * 100, hdr, "=" * 100]

    for o in args.offsets:
        meta = load_meta(out, o, args.meta_csv)
        H = load_residuals(out, sl, o, len(meta["ids"]), d_model, args.resid, args.offset_index)
        if not args.skip_verify:
            verify_alignment(H, meta, ddir)
        analyze_offset(H, meta, args, v_strat, args.stratify_axis, test_axes, rng, lines, rows_out, o)

    tail = [
        "",
        "Reading the result: the asymmetric prediction is that axis rows collapse toward the random-null",
        "band within strata while both probe rows stay near the pooled ceiling. The stratifier's own",
        "within-stratum AUC is partly range-restricted by construction; the load-bearing comparisons are",
        "(a) the probe under the identical restriction, (b) non-stratifier axes like SV39, and (c) the",
        "'minus stratifier' probe, which shows the sense direction is not the genre direction. If the",
        "within-stratum probe holds >=~0.8 while every axis sits inside the null band, the J bank's sense",
        "separation was genre-mediated and the sense-specific signal lives outside these axes.",
    ]
    for t in tail:
        print(t)
    lines += tail

    (out / "stratified_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "stratified_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["offset", "stratifier", "direction", "stratum",
                                          "n_a", "n_b", "auc", "ci_lo", "ci_hi"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\n[out] {out / 'stratified_report.txt'} and {out / 'stratified_results.csv'}")


if __name__ == "__main__":
    main()
