#!/usr/bin/env python3
"""
compare_stage1.py

Judge-free Stage-1 comparison of scan_sparse_jlens_directions.py runs across
three conditions:

  --svd      run(s) scanned with the J-lens right-SV bank
  --rotated  run(s) scanned with Haar-rotated within-subspace control banks
  --random   run(s) scanned with random full-space control banks (optional)

Each argument is one or more scanner OUT directories (containing
sv_rankings.csv and metadata.json). Multiple directories per condition are
treated as replicates and pooled, with per-replicate medians also reported.

What it computes, per layer and pooled:

  1. Primary distributional test on tail_selectivity_score:
     one-sided Mann-Whitney (svd stochastically greater than control),
     tie-corrected normal approximation with continuity correction,
     rank-biserial effect size, Benjamini-Hochberg q-values across layers,
     and a Stouffer combination across layers.
  2. Excess yield: the fraction of J-SV directions exceeding the rotated
     null's 90th/95th/99th percentile (per layer), with Wilson 95% CIs.
     Under the null these sit at 10/5/1%.
  3. Secondary metrics (pooled only): stable_excess_kurtosis,
     abs_top0_1pct_energy_share, abs_q999_robust_z,
     effective_support_fraction (lower = sparser, orientation flipped).
  4. Subspace-energy invariant check (svd vs rotated): for orthonormal bases
     of the same subspace scanned on the same tokens,
     sum_i rms_activation_i^2 is invariant per layer. Ratios far from 1
     indicate a bank/layer mismatch or a broken pairing, not physics.

Caveats printed with the results: all conditions share the same corpus (by
design; that is the paired construction), and directions within a bank share
tokens, so p-values are approximate. Treat effect sizes and excess-yield
numbers as the primary readout; p-values as a sanity screen.

Usage:

    python compare_stage1.py \
        --svd scan_svd \
        --rotated scan_rot_r0 scan_rot_r1 \
        --random scan_rand_r0 \
        --out stage1_report \
        [--plots]

Outputs: OUT/stage1_per_layer.csv, OUT/stage1_summary.txt (also printed),
and OUT/plots/*.png when --plots is given and matplotlib is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

PRIMARY = "tail_selectivity_score"
# (metric, orientation): "greater" means larger = more selective.
SECONDARY = [
    ("stable_excess_kurtosis", "greater"),
    ("abs_top0_1pct_energy_share", "greater"),
    ("abs_q999_robust_z", "greater"),
    ("effective_support_fraction", "less"),
]
META_KEYS = [
    "dataset",
    "dataset_config",
    "split",
    "seed",
    "documents_processed",
    "content_tokens_processed",
    "max_seq_len",
    "k",
]
_NORM = NormalDist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Stage-1 distributional comparison of direction-bank scans.",
    )
    p.add_argument("--svd", nargs="+", required=True, help="J-SV scan OUT dir(s).")
    p.add_argument("--rotated", nargs="+", required=True, help="Rotated-bank OUT dir(s).")
    p.add_argument("--random", nargs="*", default=[], help="Random-bank OUT dir(s).")
    p.add_argument("--out", required=True, help="Report output directory.")
    p.add_argument(
        "--yield-quantiles",
        nargs="+",
        type=float,
        default=[0.90, 0.95, 0.99],
        help="Rotated-null quantiles used as excess-yield bars.",
    )
    p.add_argument("--plots", action="store_true", help="Write ECDF/trace PNGs.")
    return p.parse_args()


# ----------------------------- Statistics ----------------------------------


def mann_whitney_greater(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """One-sided MW test that x is stochastically greater than y.

    Tie-corrected normal approximation with continuity correction.
    Returns U1, z, p, rank-biserial effect (2U/(n1 n2) - 1), medians.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    out = {
        "n1": float(n1),
        "n2": float(n2),
        "median_x": float(np.median(x)) if n1 else math.nan,
        "median_y": float(np.median(y)) if n2 else math.nan,
    }
    if n1 == 0 or n2 == 0:
        out.update({"U1": math.nan, "z": math.nan, "p": math.nan, "rbc": math.nan})
        return out

    allv = np.concatenate([x, y])
    n = n1 + n2
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    _, counts = np.unique(allv, return_counts=True)
    tie_term = float(((counts.astype(np.float64) ** 3) - counts).sum()) / (n * (n - 1))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term)
    rbc = 2.0 * u1 / (n1 * n2) - 1.0
    if var <= 0:
        out.update({"U1": u1, "z": 0.0, "p": 0.5, "rbc": rbc})
        return out
    z = (u1 - mu - 0.5) / math.sqrt(var)
    p = 1.0 - _NORM.cdf(z)
    out.update({"U1": u1, "z": z, "p": p, "rbc": rbc})
    return out


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    idx = sorted(range(m), key=lambda i: (math.isnan(pvals[i]), pvals[i]))
    adj = [math.nan] * m
    prev = 1.0
    for rank_from_end, i in enumerate(reversed(idx)):
        rank = m - rank_from_end
        p = pvals[i]
        if math.isnan(p):
            continue
        prev = min(prev, p * m / rank)
        adj[i] = prev
    return adj


def stouffer(pvals: list[float], weights: list[float]) -> tuple[float, float]:
    zs, ws = [], []
    for p, w in zip(pvals, weights):
        if math.isnan(p) or w <= 0:
            continue
        p = min(max(p, 1e-300), 1.0 - 1e-16)
        zs.append(_NORM.inv_cdf(1.0 - p))
        ws.append(w)
    if not zs:
        return math.nan, math.nan
    zs_arr = np.asarray(zs)
    ws_arr = np.asarray(ws)
    z = float((ws_arr * zs_arr).sum() / math.sqrt(float((ws_arr**2).sum())))
    return z, 1.0 - _NORM.cdf(z)


def wilson_ci(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = hits / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


# ------------------------------- Loading ------------------------------------


def read_run(run_dir: Path) -> tuple[list[dict], dict]:
    csv_path = run_dir / "sv_rankings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    meta_path = run_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    return rows, meta


def fnum(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def load_condition(
    dirs: list[str], label: str, metrics: list[str]
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict], dict[int, list[tuple[str, float]]]]:
    """Return per-layer metric arrays (replicates pooled), run metadata list,
    and per-layer per-replicate primary medians."""
    per_layer: dict[int, dict[str, list[float]]] = {}
    metas: list[dict] = []
    rep_medians: dict[int, list[tuple[str, float]]] = {}
    for d in dirs:
        run_dir = Path(d)
        rows, meta = read_run(run_dir)
        meta["_run_dir"] = str(run_dir)
        meta["_condition"] = label
        metas.append(meta)
        by_layer_primary: dict[int, list[float]] = {}
        missing = [m for m in metrics if rows and m not in rows[0]]
        if missing:
            print(f"[warn] {run_dir}: missing columns {missing}", file=sys.stderr)
        for row in rows:
            try:
                layer = int(float(row["layer"]))
            except (KeyError, ValueError):
                continue
            slot = per_layer.setdefault(layer, {m: [] for m in metrics})
            for m in metrics:
                slot.setdefault(m, []).append(fnum(row, m))
            by_layer_primary.setdefault(layer, []).append(fnum(row, PRIMARY))
        for layer, vals in by_layer_primary.items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            med = float(np.median(arr)) if arr.size else math.nan
            rep_medians.setdefault(layer, []).append((str(run_dir), med))
    pooled = {
        layer: {m: np.asarray(vals, dtype=np.float64) for m, vals in slot.items()}
        for layer, slot in per_layer.items()
    }
    return pooled, metas, rep_medians


def check_comparability(all_metas: list[dict]) -> list[str]:
    notes: list[str] = []
    if not all_metas:
        return notes
    ref = all_metas[0]
    for meta in all_metas[1:]:
        for key in META_KEYS:
            a, b = ref.get(key), meta.get(key)
            if a != b:
                notes.append(
                    f"metadata mismatch '{key}': {ref.get('_run_dir')}={a!r} vs "
                    f"{meta.get('_run_dir')}={b!r}"
                )
    return notes


def clean(a: np.ndarray) -> np.ndarray:
    return a[~np.isnan(a)]


# ------------------------------- Analysis ------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [PRIMARY] + [m for m, _ in SECONDARY] + ["rms_activation"]
    svd, svd_meta, svd_rep = load_condition(args.svd, "svd", metrics)
    rot, rot_meta, rot_rep = load_condition(args.rotated, "rotated", metrics)
    rnd, rnd_meta, rnd_rep = ({}, [], {})
    if args.random:
        rnd, rnd_meta, rnd_rep = load_condition(args.random, "random", metrics)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    notes = check_comparability(svd_meta + rot_meta + rnd_meta)
    emit("=" * 78)
    emit("Stage 1: distributional comparison of direction banks")
    emit("=" * 78)
    for meta in svd_meta + rot_meta + rnd_meta:
        emit(
            f"  [{meta['_condition']}] {meta['_run_dir']}  "
            f"docs={meta.get('documents_processed')} tokens={meta.get('content_tokens_processed')} "
            f"seed={meta.get('seed')} k={meta.get('k')}"
        )
    for n in notes:
        emit(f"  [WARN] {n}")
    if notes:
        emit("  [WARN] Conditions are not corpus-matched; pairing is broken and the")
        emit("         energy invariant below is only approximate.")
    emit()

    layers = sorted(set(svd) & set(rot))
    if not layers:
        emit("No overlapping layers between svd and rotated runs; nothing to compare.")
        sys.exit(1)
    dropped = sorted((set(svd) | set(rot)) - set(layers))
    if dropped:
        emit(f"[warn] layers present in only one condition, skipped: {dropped}")

    # ---- 1. Primary per-layer tests -------------------------------------
    per_layer_rows: list[dict] = []
    p_rot: list[float] = []
    weights: list[float] = []
    emit("-" * 78)
    emit(f"Primary metric: {PRIMARY}  (one-sided MW: svd > control)")
    emit("-" * 78)
    header = (
        f"{'layer':>5} {'n_svd':>5} {'n_rot':>5} {'med_svd':>10} {'med_rot':>10} "
        f"{'rbc':>7} {'p(rot)':>9} {'q(rot)':>9} {'med_rnd':>10} {'rbc_rnd':>8} {'p(rnd)':>9}"
    )
    emit(header)
    stats_rot: dict[int, dict[str, float]] = {}
    stats_rnd: dict[int, dict[str, float]] = {}
    for layer in layers:
        xs = clean(svd[layer][PRIMARY])
        yr = clean(rot[layer][PRIMARY])
        st = mann_whitney_greater(xs, yr)
        stats_rot[layer] = st
        p_rot.append(st["p"])
        weights.append(math.sqrt(st["n1"] + st["n2"]))
        if layer in rnd:
            stats_rnd[layer] = mann_whitney_greater(xs, clean(rnd[layer][PRIMARY]))

    q_rot = bh_adjust(p_rot)
    for layer, q in zip(layers, q_rot):
        st = stats_rot[layer]
        sn = stats_rnd.get(layer)
        emit(
            f"{layer:>5} {int(st['n1']):>5} {int(st['n2']):>5} "
            f"{st['median_x']:>10.4g} {st['median_y']:>10.4g} "
            f"{st['rbc']:>7.3f} {st['p']:>9.2e} {q:>9.2e} "
            + (
                f"{sn['median_y']:>10.4g} {sn['rbc']:>8.3f} {sn['p']:>9.2e}"
                if sn
                else f"{'-':>10} {'-':>8} {'-':>9}"
            )
        )
        row = {
            "layer": layer,
            "n_svd": int(st["n1"]),
            "n_rotated": int(st["n2"]),
            "median_svd": st["median_x"],
            "median_rotated": st["median_y"],
            "rbc_vs_rotated": st["rbc"],
            "p_vs_rotated": st["p"],
            "q_vs_rotated": q,
        }
        if sn:
            row.update(
                {
                    "n_random": int(sn["n2"]),
                    "median_random": sn["median_y"],
                    "rbc_vs_random": sn["rbc"],
                    "p_vs_random": sn["p"],
                }
            )
        per_layer_rows.append(row)

    z_comb, p_comb = stouffer(p_rot, weights)
    xs_all = clean(np.concatenate([svd[l][PRIMARY] for l in layers]))
    yr_all = clean(np.concatenate([rot[l][PRIMARY] for l in layers]))
    pooled_rot = mann_whitney_greater(xs_all, yr_all)
    emit()
    emit(
        f"Pooled (all layers): n_svd={int(pooled_rot['n1'])} n_rot={int(pooled_rot['n2'])}  "
        f"median {pooled_rot['median_x']:.4g} vs {pooled_rot['median_y']:.4g}  "
        f"rbc={pooled_rot['rbc']:.3f}  p={pooled_rot['p']:.2e}"
    )
    emit(f"Stouffer across layers (svd > rotated): z={z_comb:.2f}  p={p_comb:.2e}")
    if rnd:
        rl = [l for l in layers if l in rnd]
        if rl:
            yn_all = clean(np.concatenate([rnd[l][PRIMARY] for l in rl]))
            xn_all = clean(np.concatenate([svd[l][PRIMARY] for l in rl]))
            pooled_rnd = mann_whitney_greater(xn_all, yn_all)
            emit(
                f"Pooled vs random: rbc={pooled_rnd['rbc']:.3f}  p={pooled_rnd['p']:.2e}  "
                f"(median {pooled_rnd['median_x']:.4g} vs {pooled_rnd['median_y']:.4g})"
            )
    emit()

    # ---- 2. Excess yield over rotated-null quantile bars ------------------
    emit("-" * 78)
    emit("Excess yield: fraction of J-SV directions above the rotated null's")
    emit("per-layer quantile bar (expected under null: 1 - q). Wilson 95% CI.")
    emit("-" * 78)
    for qv in args.yield_quantiles:
        hits = 0
        total = 0
        hits_rnd = 0
        total_rnd = 0
        for layer in layers:
            yr = clean(rot[layer][PRIMARY])
            if yr.size == 0:
                continue
            bar = float(np.quantile(yr, qv))
            xs = clean(svd[layer][PRIMARY])
            hits += int((xs > bar).sum())
            total += xs.size
            if layer in rnd:
                xr = clean(rnd[layer][PRIMARY])
                hits_rnd += int((xr > bar).sum())
                total_rnd += xr.size
        lo, hi = wilson_ci(hits, total)
        frac = hits / total if total else math.nan
        line = (
            f"  bar=q{qv:.2f}: svd yield {hits}/{total} = {frac:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  (null expectation {1 - qv:.3f})"
        )
        if total_rnd:
            line += f"  |  random-bank yield {hits_rnd / total_rnd:.3f}"
        emit(line)
        for row in per_layer_rows:
            layer = row["layer"]
            yr = clean(rot[layer][PRIMARY])
            xs = clean(svd[layer][PRIMARY])
            if yr.size and xs.size:
                bar = float(np.quantile(yr, qv))
                row[f"svd_yield_above_rot_q{int(round(qv * 100))}"] = float(
                    (xs > bar).mean()
                )
    emit()

    # ---- 3. Secondary metrics (pooled) ------------------------------------
    emit("-" * 78)
    emit("Secondary metrics, pooled across layers (one-sided MW, svd more-selective")
    emit("than rotated; 'less'-oriented metrics are sign-flipped before testing)")
    emit("-" * 78)
    for metric, orient in SECONDARY:
        try:
            xs_m = clean(np.concatenate([svd[l][metric] for l in layers]))
            yr_m = clean(np.concatenate([rot[l][metric] for l in layers]))
        except KeyError:
            emit(f"  {metric:<34} [missing column, skipped]")
            continue
        if xs_m.size == 0 or yr_m.size == 0:
            emit(f"  {metric:<34} [no data]")
            continue
        sgn = 1.0 if orient == "greater" else -1.0
        st = mann_whitney_greater(sgn * xs_m, sgn * yr_m)
        emit(
            f"  {metric:<34} median {np.median(xs_m):>10.4g} vs {np.median(yr_m):>10.4g}  "
            f"rbc={st['rbc']:>6.3f}  p={st['p']:.2e}"
            + ("  (lower=better)" if orient == "less" else "")
        )
    emit()

    # ---- 4. Subspace-energy invariant (svd vs rotated) ---------------------
    emit("-" * 78)
    emit("Subspace-energy invariant: sum_i rms_activation_i^2, svd / rotated.")
    emit("For orthonormal bases of the same subspace on the same tokens this is 1.")
    emit("-" * 78)
    bad = 0
    for layer in layers:
        try:
            e_svd = float(np.nansum(np.square(svd[layer]["rms_activation"])))
            e_rot = float(np.nansum(np.square(rot[layer]["rms_activation"])))
        except KeyError:
            emit("  [rms_activation column missing; check skipped]")
            break
        n_rep_s = max(1, len(args.svd))
        n_rep_r = max(1, len(args.rotated))
        ratio = (e_svd / n_rep_s) / max(e_rot / n_rep_r, 1e-30)
        flag = ""
        if not (0.95 <= ratio <= 1.05):
            flag = "  <-- MISMATCH: check layers/banks/corpus pairing"
            bad += 1
        emit(f"  layer {layer:>3}: ratio = {ratio:.4f}{flag}")
        for row in per_layer_rows:
            if row["layer"] == layer:
                row["energy_ratio_svd_over_rotated"] = ratio
    if bad:
        emit(f"  [WARN] {bad} layer(s) violate the invariant; results there are suspect.")
    emit()

    # ---- 5. Per-replicate medians -----------------------------------------
    if len(args.svd) > 1 or len(args.rotated) > 1 or len(args.random) > 1:
        emit("-" * 78)
        emit(f"Per-replicate pooled medians of {PRIMARY} (draw-to-draw variance):")
        emit("-" * 78)
        for label, rep_map in (("svd", svd_rep), ("rotated", rot_rep), ("random", rnd_rep)):
            runs: dict[str, list[float]] = {}
            for layer, entries in rep_map.items():
                for run, med in entries:
                    runs.setdefault(run, []).append(med)
            for run, meds in sorted(runs.items()):
                arr = np.asarray([m for m in meds if not math.isnan(m)])
                if arr.size:
                    emit(f"  [{label}] {run}: median-of-layer-medians = {np.median(arr):.4g}")
        emit()

    # ---- Interpretation guide ----------------------------------------------
    emit("-" * 78)
    emit("Reading the result:")
    emit("  svd >> rotated (positive rbc, low q, yield >> nominal):")
    emit("      the eigen-axes carry structure beyond the subspace -> Stage 2 on")
    emit("      selectivity-matched candidates.")
    emit("  svd ~= rotated >> random:")
    emit("      the top-k subspace is special, its axes are not; the SVD is")
    emit("      scaffolding. Reframe claims to the subspace.")
    emit("  svd ~= rotated ~= random:")
    emit("      tail selectivity of this kind is generic under the selection")
    emit("      pipeline; basis claims are unsupported.")
    emit("Caveat: all conditions share one corpus (intentionally paired) and")
    emit("directions share tokens, so p-values are approximate; lean on effect")
    emit("sizes and excess yield.")

    # ---- Write outputs -----------------------------------------------------
    per_layer_csv = out_dir / "stage1_per_layer.csv"
    if per_layer_rows:
        keys: list[str] = []
        for row in per_layer_rows:
            for kk in row:
                if kk not in keys:
                    keys.append(kk)
        with per_layer_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in per_layer_rows:
                w.writerow(row)
    with (out_dir / "stage1_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[out] {per_layer_csv}")
    print(f"[out] {out_dir / 'stage1_summary.txt'}")

    if args.plots:
        make_plots(out_dir, layers, svd, rot, rnd)


def make_plots(
    out_dir: Path,
    layers: list[int],
    svd: dict[int, dict[str, np.ndarray]],
    rot: dict[int, dict[str, np.ndarray]],
    rnd: dict[int, dict[str, np.ndarray]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable ({exc}); skipping plots", file=sys.stderr)
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    def ecdf(ax, vals: np.ndarray, label: str) -> None:
        v = np.sort(vals[~np.isnan(vals)])
        if v.size == 0:
            return
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", label=label)

    # Pooled ECDF.
    fig, ax = plt.subplots(figsize=(6, 4))
    ecdf(ax, np.concatenate([svd[l][PRIMARY] for l in layers]), "J-SV")
    ecdf(ax, np.concatenate([rot[l][PRIMARY] for l in layers]), "rotated")
    rl = [l for l in layers if l in rnd]
    if rl:
        ecdf(ax, np.concatenate([rnd[l][PRIMARY] for l in rl]), "random")
    ax.set_xlabel(PRIMARY)
    ax.set_ylabel("ECDF")
    ax.set_title("Pooled across layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ecdf_pooled.png", dpi=150)
    plt.close(fig)

    # Median trace by layer.
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, data, style in (("J-SV", svd, "-o"), ("rotated", rot, "-s"), ("random", rnd, "-^")):
        ls = [l for l in layers if l in data]
        if not ls:
            continue
        meds = [float(np.nanmedian(data[l][PRIMARY])) for l in ls]
        ax.plot(ls, meds, style, label=cond, markersize=4)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"median {PRIMARY}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "median_by_layer.png", dpi=150)
    plt.close(fig)

    # Per-layer ECDF grid.
    ncols = 4
    nrows = int(math.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    for idx, layer in enumerate(layers):
        ax = axes[idx // ncols][idx % ncols]
        ecdf(ax, svd[layer][PRIMARY], "J-SV")
        ecdf(ax, rot[layer][PRIMARY], "rot")
        if layer in rnd:
            ecdf(ax, rnd[layer][PRIMARY], "rand")
        ax.set_title(f"L{layer}", fontsize=9)
        ax.tick_params(labelsize=7)
    for idx in range(len(layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "ecdf_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"[out] {plots_dir}/ecdf_pooled.png, median_by_layer.png, ecdf_by_layer.png")


if __name__ == "__main__":
    main()
