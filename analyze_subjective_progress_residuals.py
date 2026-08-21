#!/usr/bin/env python3
"""
Analyze targeted residual-stream windows around annotated subjective-progress events.

This script consumes the output of collect_subjective_progress_residuals.py and
performs the two discovery analyses that should precede causal steering:

1) BROAD SV SCREEN
   For every layer x J-lens right singular vector already collected, test whether
   its activation changes around subjective-progress events more than around the
   matched same-rollout controls.

2) DIRECT PROGRESS-DIRECTION DISCOVERY
   Do not assume the signal is an SV.  At every layer, form an event-specific
   residual transition

       d_i = mean(h_post) - mean(h_pre)

   and subtract the mean matched-control transition.  The resulting vector is a
   candidate subjective-progress displacement for that event.  Learn the common
   direction with equal rollout weighting and evaluate it with leave-one-rollout-
   out (LORO) cross-validation.

The incorrect subjective-progress events are a target condition, not label noise.
A direction is more interesting if it generalizes to BOTH objectively correct and
objectively incorrect progress events.

Expected input
--------------
<residual-data>/
  meta.json
  windows.jsonl
  windows/*.npz
  direction_bank.npz

Typical usage
-------------
    python3 analyze_subjective_progress_residuals.py \
        --data subjective_progress_residual_data \
        --directions-dir task_gaming_jlens/directions \
        --out subjective_progress_discovery

Important defaults
------------------
* pre  = t=-5..-1
* post = t=0..3
* SV ranking metric = delta = mean(post)-mean(pre)
* minimum matched controls per event = 3
* inference unit = rollout, not event
* direct-direction validation = leave-one-rollout-out

Outputs
-------
<out>/
  analysis_config.json
  summary.json
  summary.md

  sv_event_effects.csv
      Event-level event-minus-control effects for every layer x SV.
  sv_candidate_statistics.csv
      Grouped statistics for every scalar metric / layer / SV.
  sv_rankings.csv
  sv_rankings.json
      Ranked SV candidates for --sv-rank-metric.

  direct_event_effects.npz
      event_ids, run_ids, statuses, layers, effect_vectors [E,L,D].
  direct_event_scores.csv
      Leave-one-rollout-out projection and cosine score per event x layer.
  direct_layer_statistics.csv
  direct_layer_rankings.csv
  direct_layer_rankings.json
      Ranked residual-space layers using cross-validated direction scores.
  direct_status_transfer.csv
      Correct-derived -> incorrect and incorrect-derived -> correct transfer.
  progress_directions.npz
      Full-data equal-rollout-weighted unit directions [L,D] for later steering,
      plus raw means and correct/incorrect descriptive directions.

Interpretation
--------------
The script is a DISCOVERY analysis.  LORO cross-validation asks whether the
progress direction generalizes to unseen trajectories within this campaign, but
choosing the best layer/direction still uses this dataset.  Final causal steering
should be evaluated on new held-out behavioral rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"
STATUSES = ("correct", "incorrect", "ambiguous")
SV_METRICS = ("delta", "anchor", "peak_response", "trough_response")


# -----------------------------------------------------------------------------
# CLI / I/O
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data", type=Path, default=Path("subjective_progress_residual_data"))
    p.add_argument(
        "--directions-dir",
        type=Path,
        default=Path("task_gaming_jlens/directions"),
        help="Original LXX.npz files containing V; used only to compare learned directions with SVs.",
    )
    p.add_argument("--out", type=Path, default=Path("subjective_progress_discovery"))

    p.add_argument("--pre-start", type=int, default=-5)
    p.add_argument("--pre-end", type=int, default=-1)
    p.add_argument("--post-start", type=int, default=0)
    p.add_argument("--post-end", type=int, default=3)
    p.add_argument("--baseline-start", type=int, default=-15)
    p.add_argument("--baseline-end", type=int, default=-5)
    p.add_argument("--peak-start", type=int, default=-3)
    p.add_argument("--peak-end", type=int, default=5)

    p.add_argument(
        "--min-controls",
        type=int,
        default=3,
        help="Minimum complete matched controls required for an event/metric.",
    )
    p.add_argument(
        "--sv-rank-metric",
        choices=SV_METRICS,
        default="delta",
        help="Scalar event-minus-control metric used for final SV ranking.",
    )
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument(
        "--max-sv-rank-output",
        type=int,
        default=0,
        help="0 writes every candidate; positive N limits sv_rankings.* only.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace, collection_meta: Mapping[str, Any]) -> None:
    if args.min_controls < 1:
        raise SystemExit("--min-controls must be >= 1")
    if args.bootstrap_samples < 1 or args.permutations < 1:
        raise SystemExit("bootstrap/permutation counts must be >= 1")
    intervals = [
        ("pre", args.pre_start, args.pre_end),
        ("post", args.post_start, args.post_end),
        ("baseline", args.baseline_start, args.baseline_end),
        ("peak", args.peak_start, args.peak_end),
    ]
    window = int(collection_meta.get("window", 0))
    for name, start, end in intervals:
        if start > end:
            raise SystemExit(f"{name} starts after it ends: {start}>{end}")
        if start < -window or end > window:
            raise SystemExit(
                f"{name} interval [{start},{end}] exceeds collected window +/-{window}"
            )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON {path}:{line_number}: {exc}") from exc
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(row.get(key), float) and not math.isfinite(float(row[key]))
                    else row.get(key, "")
                    for key in fields
                }
            )


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Loaded windows
# -----------------------------------------------------------------------------


class Window:
    def __init__(self, data_root: Path, meta: Mapping[str, Any]):
        self.meta = dict(meta)
        self.path = data_root / str(meta["npz"])
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with np.load(self.path) as z:
            self.layers = np.asarray(z["layers"], dtype=np.int32)
            self.relative_tokens = np.asarray(z["relative_tokens"], dtype=np.int32)
            self.valid_mask = np.asarray(z["valid_mask"], dtype=bool)
            self.residuals = np.asarray(z["residuals"], dtype=np.float32)
            self.sv = np.asarray(z["sv_activations"], dtype=np.float32)

        if self.residuals.ndim != 3:
            raise ValueError(f"{self.path}: residuals should be [L,T,D], got {self.residuals.shape}")
        if self.sv.ndim != 3:
            raise ValueError(f"{self.path}: sv_activations should be [L,T,K], got {self.sv.shape}")
        if self.residuals.shape[:2] != (len(self.layers), len(self.relative_tokens)):
            raise ValueError(f"{self.path}: inconsistent residual axes")
        if self.sv.shape[:2] != (len(self.layers), len(self.relative_tokens)):
            raise ValueError(f"{self.path}: inconsistent SV axes")

    def indices(self, start: int, end: int) -> np.ndarray | None:
        wanted = list(range(start, end + 1))
        lookup = {int(t): i for i, t in enumerate(self.relative_tokens)}
        if any(t not in lookup for t in wanted):
            return None
        idx = np.asarray([lookup[t] for t in wanted], dtype=np.int32)
        if not np.all(self.valid_mask[idx]):
            return None
        return idx

    def mean_residual(self, start: int, end: int) -> np.ndarray | None:
        idx = self.indices(start, end)
        if idx is None:
            return None
        x = self.residuals[:, idx, :]
        if not np.all(np.isfinite(x)):
            return None
        return np.mean(x, axis=1, dtype=np.float64).astype(np.float32)

    def residual_delta(self, args: argparse.Namespace) -> np.ndarray | None:
        pre = self.mean_residual(args.pre_start, args.pre_end)
        post = self.mean_residual(args.post_start, args.post_end)
        if pre is None or post is None:
            return None
        return post - pre

    def sv_metric(self, metric: str, args: argparse.Namespace) -> np.ndarray | None:
        if self.sv.shape[2] == 0:
            return None
        if metric == "delta":
            pre_idx = self.indices(args.pre_start, args.pre_end)
            post_idx = self.indices(args.post_start, args.post_end)
            if pre_idx is None or post_idx is None:
                return None
            pre = np.mean(self.sv[:, pre_idx, :], axis=1)
            post = np.mean(self.sv[:, post_idx, :], axis=1)
            out = post - pre
        elif metric == "anchor":
            idx = self.indices(0, 0)
            if idx is None:
                return None
            out = self.sv[:, idx[0], :]
        else:
            base_idx = self.indices(args.baseline_start, args.baseline_end)
            peak_idx = self.indices(args.peak_start, args.peak_end)
            if base_idx is None or peak_idx is None:
                return None
            baseline = np.mean(self.sv[:, base_idx, :], axis=1)
            if metric == "peak_response":
                extreme = np.max(self.sv[:, peak_idx, :], axis=1)
            elif metric == "trough_response":
                extreme = np.min(self.sv[:, peak_idx, :], axis=1)
            else:
                raise KeyError(metric)
            out = extreme - baseline
        return out.astype(np.float32) if np.all(np.isfinite(out)) else None


# -----------------------------------------------------------------------------
# Statistics with rollout as independent unit
# -----------------------------------------------------------------------------


def group_rollout_means(
    values: np.ndarray,
    run_ids: Sequence[str],
    statuses: Sequence[str],
    status: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """values [E]; return rollout means and corresponding run names."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, run_id, st in zip(values, run_ids, statuses):
        if status != "all" and st != status:
            continue
        if np.isfinite(value):
            buckets[str(run_id)].append(float(value))
    names = np.asarray(sorted(buckets), dtype=object)
    means = np.asarray([np.mean(buckets[str(r)]) for r in names], dtype=np.float64)
    return means, names


def descriptive_group_stats(
    values: np.ndarray,
    run_ids: Sequence[str],
    statuses: Sequence[str],
    status: str,
) -> dict[str, Any]:
    event_vals = np.asarray(
        [v for v, st in zip(values, statuses) if np.isfinite(v) and (status == "all" or st == status)],
        dtype=np.float64,
    )
    rv, _ = group_rollout_means(values, run_ids, statuses, status)
    if rv.size == 0:
        return {
            "n_events": int(event_vals.size),
            "n_rollouts": 0,
            "event_weighted_mean": math.nan,
            "rollout_weighted_mean": math.nan,
            "rollout_sd": math.nan,
            "rollout_se": math.nan,
            "standardized_rollout_effect": math.nan,
            "rollout_consistency": math.nan,
            "event_consistency": math.nan,
        }
    mean = float(np.mean(rv))
    sd = float(np.std(rv, ddof=1)) if rv.size > 1 else math.nan
    se = sd / math.sqrt(rv.size) if finite(sd) else math.nan
    standardized = mean / sd if finite(sd) and sd > 1e-12 else math.nan
    sign = 1.0 if mean >= 0 else -1.0
    return {
        "n_events": int(event_vals.size),
        "n_rollouts": int(rv.size),
        "event_weighted_mean": float(np.mean(event_vals)) if event_vals.size else math.nan,
        "rollout_weighted_mean": mean,
        "rollout_sd": sd,
        "rollout_se": se,
        "standardized_rollout_effect": standardized,
        "rollout_consistency": float(np.mean(sign * rv > 0)),
        "event_consistency": float(np.mean(sign * event_vals > 0)) if event_vals.size else math.nan,
    }


def inference_from_rollout_values(
    rv: np.ndarray,
    bootstrap_samples: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    rv = np.asarray(rv, dtype=np.float64)
    rv = rv[np.isfinite(rv)]
    if rv.size == 0:
        return {
            "bootstrap_ci_low": math.nan,
            "bootstrap_ci_high": math.nan,
            "permutation_p_two_sided": math.nan,
            "permutation_p_directional": math.nan,
        }
    observed = float(np.mean(rv))
    bi = rng.integers(0, rv.size, size=(bootstrap_samples, rv.size))
    bmeans = np.mean(rv[bi], axis=1)
    lo, hi = np.quantile(bmeans, [0.025, 0.975])

    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, rv.size))
    null = np.mean(signs * rv[None, :], axis=1)
    p2 = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1)
    orient = 1.0 if observed >= 0 else -1.0
    pd = (1 + int(np.sum(orient * null >= abs(observed)))) / (permutations + 1)
    return {
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "permutation_p_two_sided": float(p2),
        "permutation_p_directional": float(pd),
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out.tolist()
    order_local = np.argsort(p[finite_idx])
    ordered_idx = finite_idx[order_local]
    m = len(ordered_idx)
    running = 0.0
    for rank, idx in enumerate(ordered_idx):
        adj = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def preservation_ratio(all_mean: float, correct_mean: float, incorrect_mean: float) -> float:
    if not all(finite(x) for x in (all_mean, correct_mean, incorrect_mean)):
        return 0.0
    orient = 1.0 if all_mean >= 0 else -1.0
    c = max(0.0, orient * correct_mean)
    i = max(0.0, orient * incorrect_mean)
    denom = max(abs(all_mean), 1e-12)
    return float(np.clip(min(c, i) / denom, 0.0, 1.0))


def ranking_score(std_effect: float, consistency: float, preservation: float) -> float:
    if not finite(std_effect):
        return 0.0
    consistency = consistency if finite(consistency) else 0.0
    return float(
        abs(std_effect)
        * (0.5 + 0.5 * np.clip(consistency, 0.0, 1.0))
        * (0.25 + 0.75 * np.clip(preservation, 0.0, 1.0))
    )


# -----------------------------------------------------------------------------
# Direction helpers
# -----------------------------------------------------------------------------


def normalize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=-1)
    denom = np.maximum(norms, 1e-12)[..., None]
    return (x / denom).astype(np.float32), norms.astype(np.float32)


def equal_rollout_vector_mean(
    effects: np.ndarray,
    run_ids: Sequence[str],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """effects [E,L,D] -> equal-rollout mean [L,D]."""
    E = effects.shape[0]
    if mask is None:
        mask = np.ones(E, dtype=bool)
    run_vectors = []
    for run_id in sorted(set(str(r) for r in run_ids)):
        idx = np.asarray([str(r) == run_id for r in run_ids], dtype=bool) & mask
        if np.any(idx):
            run_vectors.append(np.mean(effects[idx], axis=0, dtype=np.float64))
    if not run_vectors:
        return np.full(effects.shape[1:], np.nan, dtype=np.float32)
    return np.mean(np.stack(run_vectors), axis=0, dtype=np.float64).astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a,b [L,D] -> [L]."""
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 1e-12)
    return out


def event_scores_against_direction(effect: np.ndarray, unit_dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """effect [L,D], unit_dirs [L,D] -> projection, cosine [L]."""
    projection = np.sum(effect * unit_dirs, axis=-1, dtype=np.float64)
    norms = np.linalg.norm(effect, axis=-1)
    cosine = np.full(effect.shape[0], np.nan, dtype=np.float64)
    np.divide(projection, norms, out=cosine, where=norms > 1e-12)
    return projection, cosine


def load_v_bank(directions_dir: Path, layers: np.ndarray, k: int, hidden_size: int) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    if k <= 0:
        return result
    for layer in layers:
        p = directions_dir / f"L{int(layer):02d}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            if "V" not in z:
                continue
            V = np.asarray(z["V"], dtype=np.float32)
        if V.ndim != 2 or V.shape[0] != hidden_size:
            continue
        result[int(layer)] = V[:, : min(k, V.shape[1])]
    return result


# -----------------------------------------------------------------------------
# Main data construction
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    directions_dir = args.directions_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta_path = data_root / "meta.json"
    index_path = data_root / "windows.jsonl"
    if not meta_path.exists() or not index_path.exists():
        raise SystemExit(f"Expected meta.json and windows.jsonl under {data_root}")
    collection_meta = read_json(meta_path)
    validate_args(args, collection_meta)

    index_rows = read_jsonl(index_path)
    if not index_rows:
        raise SystemExit(f"No windows in {index_path}")

    print(f"[load] windows={len(index_rows)} from {data_root}")
    windows: dict[str, Window] = {}
    for i, row in enumerate(index_rows, 1):
        wid = str(row["window_id"])
        if wid in windows:
            raise ValueError(f"duplicate window_id {wid}")
        windows[wid] = Window(data_root, row)
        if i % 100 == 0:
            print(f"  loaded {i}/{len(index_rows)}")

    first = next(iter(windows.values()))
    layers = first.layers.copy()
    L = len(layers)
    D = int(first.residuals.shape[2])
    K = int(first.sv.shape[2])
    for w in windows.values():
        if not np.array_equal(w.layers, layers):
            raise ValueError(f"{w.path}: layers differ from first window")
        if w.residuals.shape[2] != D or w.sv.shape[2] != K:
            raise ValueError(f"{w.path}: hidden/SV dimensions differ")

    event_windows: dict[str, Window] = {}
    control_windows: dict[str, list[Window]] = defaultdict(list)
    for w in windows.values():
        event_id = str(w.meta["event_id"])
        if w.meta.get("role") == "event":
            if event_id in event_windows:
                raise ValueError(f"multiple event windows for {event_id}")
            event_windows[event_id] = w
        elif w.meta.get("role") == "control":
            control_windows[event_id].append(w)

    event_ids = sorted(event_windows)
    print(
        f"[data] events={len(event_ids)} controls={sum(len(v) for v in control_windows.values())} "
        f"layers={L} hidden={D} sv_k={K}"
    )

    # ------------------------------------------------------------------
    # Build event-level scalar SV effects and residual-vector effects.
    # ------------------------------------------------------------------
    sv_effects_by_metric = {
        metric: np.full((len(event_ids), L, K), np.nan, dtype=np.float32)
        for metric in SV_METRICS
    }
    residual_effects = np.full((len(event_ids), L, D), np.nan, dtype=np.float32)
    run_ids: list[str] = []
    statuses: list[str] = []
    environments: list[str] = []
    conditions: list[str] = []
    event_types: list[str] = []
    direct_n_controls: list[int] = []
    metric_n_controls: dict[str, list[int]] = {m: [] for m in SV_METRICS}

    for ei, event_id in enumerate(event_ids):
        ew = event_windows[event_id]
        controls = sorted(
            control_windows.get(event_id, []),
            key=lambda w: int(w.meta.get("control_index") or 0),
        )
        run_id = str(ew.meta.get("run_id", ""))
        status = str(ew.meta.get("objective_status", ""))
        run_ids.append(run_id)
        statuses.append(status)
        environments.append(str(ew.meta.get("environment", "")))
        conditions.append(str(ew.meta.get("condition", "")))
        event_types.append(str(ew.meta.get("event_type", "")))

        if any(str(c.meta.get("run_id", "")) != run_id for c in controls):
            raise ValueError(f"{event_id}: control from another rollout")

        # Direct vector effect: event transition - mean matched control transition.
        e_delta = ew.residual_delta(args)
        c_deltas = [c.residual_delta(args) for c in controls]
        c_deltas = [x for x in c_deltas if x is not None]
        direct_n_controls.append(len(c_deltas))
        if e_delta is not None and len(c_deltas) >= args.min_controls:
            residual_effects[ei] = e_delta - np.mean(np.stack(c_deltas), axis=0)

        # Scalar SV effects for every requested metric.
        for metric in SV_METRICS:
            e_metric = ew.sv_metric(metric, args)
            c_metrics = [c.sv_metric(metric, args) for c in controls]
            c_metrics = [x for x in c_metrics if x is not None]
            metric_n_controls[metric].append(len(c_metrics))
            if e_metric is not None and len(c_metrics) >= args.min_controls:
                sv_effects_by_metric[metric][ei] = e_metric - np.mean(np.stack(c_metrics), axis=0)

    run_ids_arr = np.asarray(run_ids, dtype=object)
    statuses_arr = np.asarray(statuses, dtype=object)
    valid_direct = np.all(np.isfinite(residual_effects), axis=(1, 2))
    direct_idx = np.flatnonzero(valid_direct)
    if direct_idx.size == 0:
        raise SystemExit("No events have complete residual pre/post windows plus enough controls")

    print(
        f"[direct] eligible events={direct_idx.size}/{len(event_ids)} across "
        f"{len(set(run_ids_arr[direct_idx]))} rollouts"
    )

    # Save direct raw event effects for transparent downstream re-analysis.
    np.savez_compressed(
        out / "direct_event_effects.npz",
        event_ids=np.asarray(event_ids, dtype=str),
        run_ids=np.asarray(run_ids, dtype=str),
        objective_status=np.asarray(statuses, dtype=str),
        environments=np.asarray(environments, dtype=str),
        conditions=np.asarray(conditions, dtype=str),
        event_types=np.asarray(event_types, dtype=str),
        layers=layers.astype(np.int16),
        valid_event_mask=valid_direct.astype(bool),
        effect_vectors=residual_effects.astype(np.float32),
    )

    # ------------------------------------------------------------------
    # BROAD SV SCREEN
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    sv_effect_rows: list[dict[str, Any]] = []
    sv_stat_rows: list[dict[str, Any]] = []
    sv_rankings: list[dict[str, Any]] = []

    if K > 0:
        # Event-level table includes all scalar metrics for each candidate.
        for ei, event_id in enumerate(event_ids):
            for li, layer in enumerate(layers):
                for sj in range(K):
                    row = {
                        "event_id": event_id,
                        "run_id": run_ids[ei],
                        "environment": environments[ei],
                        "condition": conditions[ei],
                        "objective_status": statuses[ei],
                        "event_type": event_types[ei],
                        "layer": int(layer),
                        "sv_rank": sj + 1,
                    }
                    for metric in SV_METRICS:
                        v = float(sv_effects_by_metric[metric][ei, li, sj])
                        row[f"{metric}_contrast"] = v if math.isfinite(v) else math.nan
                        row[f"{metric}_n_controls"] = metric_n_controls[metric][ei]
                    sv_effect_rows.append(row)
        write_csv(out / "sv_event_effects.csv", sv_effect_rows)

        # Candidate-level grouped statistics for each metric.
        for metric in SV_METRICS:
            effects = sv_effects_by_metric[metric]
            metric_rows_start = len(sv_stat_rows)
            for li, layer in enumerate(layers):
                for sj in range(K):
                    vals = effects[:, li, sj].astype(np.float64)
                    all_stats = descriptive_group_stats(vals, run_ids, statuses, "all")
                    corr_stats = descriptive_group_stats(vals, run_ids, statuses, "correct")
                    inc_stats = descriptive_group_stats(vals, run_ids, statuses, "incorrect")
                    amb_stats = descriptive_group_stats(vals, run_ids, statuses, "ambiguous")
                    rv, _ = group_rollout_means(vals, run_ids, statuses, "all")
                    inf = inference_from_rollout_values(
                        rv, args.bootstrap_samples, args.permutations, rng
                    )
                    pres = preservation_ratio(
                        all_stats["rollout_weighted_mean"],
                        corr_stats["rollout_weighted_mean"],
                        inc_stats["rollout_weighted_mean"],
                    )
                    score = ranking_score(
                        all_stats["standardized_rollout_effect"],
                        all_stats["rollout_consistency"],
                        pres,
                    )
                    sv_stat_rows.append(
                        {
                            "metric": metric,
                            "layer": int(layer),
                            "sv_rank": sj + 1,
                            **all_stats,
                            **inf,
                            "correct_n_events": corr_stats["n_events"],
                            "correct_n_rollouts": corr_stats["n_rollouts"],
                            "correct_rollout_weighted_mean": corr_stats["rollout_weighted_mean"],
                            "correct_rollout_consistency": corr_stats["rollout_consistency"],
                            "incorrect_n_events": inc_stats["n_events"],
                            "incorrect_n_rollouts": inc_stats["n_rollouts"],
                            "incorrect_rollout_weighted_mean": inc_stats["rollout_weighted_mean"],
                            "incorrect_rollout_consistency": inc_stats["rollout_consistency"],
                            "ambiguous_n_events": amb_stats["n_events"],
                            "ambiguous_rollout_weighted_mean": amb_stats["rollout_weighted_mean"],
                            "status_preservation_ratio": pres,
                            "rank_score": score,
                        }
                    )

            # Holm correction is metric-specific across every L x SV candidate.
            metric_rows = sv_stat_rows[metric_rows_start:]
            adjusted = holm_adjust([r["permutation_p_two_sided"] for r in metric_rows])
            for row, p_adj in zip(metric_rows, adjusted):
                row["permutation_p_holm"] = p_adj

        write_csv(out / "sv_candidate_statistics.csv", sv_stat_rows)
        selected = [r for r in sv_stat_rows if r["metric"] == args.sv_rank_metric]
        selected.sort(
            key=lambda r: (
                -float(r["rank_score"]),
                float(r["permutation_p_holm"]) if finite(r["permutation_p_holm"]) else 2.0,
                int(r["layer"]),
                int(r["sv_rank"]),
            )
        )
        if args.max_sv_rank_output > 0:
            selected = selected[: args.max_sv_rank_output]
        for rank, row in enumerate(selected, 1):
            sv_rankings.append({"rank": rank, **row})
        write_csv(out / "sv_rankings.csv", sv_rankings)
        write_json(out / "sv_rankings.json", sv_rankings)
        print(f"[sv] ranked {len(selected)} candidates on metric={args.sv_rank_metric}")
    else:
        (out / "sv_event_effects.csv").write_text("", encoding="utf-8")
        (out / "sv_candidate_statistics.csv").write_text("", encoding="utf-8")
        (out / "sv_rankings.csv").write_text("", encoding="utf-8")
        write_json(out / "sv_rankings.json", [])
        print("[sv] no SV activations in collection; skipping broad screen")

    # ------------------------------------------------------------------
    # DIRECT DIRECTION DISCOVERY + LORO VALIDATION
    # ------------------------------------------------------------------
    effects = residual_effects[valid_direct]
    d_event_ids = np.asarray(event_ids, dtype=object)[valid_direct]
    d_run_ids = run_ids_arr[valid_direct]
    d_status = statuses_arr[valid_direct]
    d_env = np.asarray(environments, dtype=object)[valid_direct]
    d_cond = np.asarray(conditions, dtype=object)[valid_direct]
    d_event_type = np.asarray(event_types, dtype=object)[valid_direct]

    unique_runs = sorted(set(str(r) for r in d_run_ids))
    if len(unique_runs) < 3:
        raise SystemExit("Need at least three event-bearing rollouts for LORO discovery")

    # Equal-rollout full-data direction per layer, used ONLY as the final candidate
    # direction to carry into future causal experiments. Ranking uses held-out scores.
    raw_full = equal_rollout_vector_mean(effects, d_run_ids)
    unit_full, full_norms = normalize_rows(raw_full)

    corr_mask = d_status == "correct"
    inc_mask = d_status == "incorrect"
    raw_correct = equal_rollout_vector_mean(effects, d_run_ids, corr_mask)
    raw_incorrect = equal_rollout_vector_mean(effects, d_run_ids, inc_mask)
    unit_correct, correct_norms = normalize_rows(raw_correct)
    unit_incorrect, incorrect_norms = normalize_rows(raw_incorrect)
    status_direction_cosine = cosine_rows(unit_correct, unit_incorrect)

    # LORO scores: train a direction without the held-out rollout, then score every
    # event in that rollout. Equal rollout weighting is preserved in every fold.
    cv_projection = np.full((effects.shape[0], L), np.nan, dtype=np.float64)
    cv_cosine = np.full((effects.shape[0], L), np.nan, dtype=np.float64)
    fold_train_runs: dict[str, int] = {}

    for heldout in unique_runs:
        train_mask = d_run_ids != heldout
        test_idx = np.flatnonzero(d_run_ids == heldout)
        train_raw = equal_rollout_vector_mean(effects[train_mask], d_run_ids[train_mask])
        train_unit, train_norm = normalize_rows(train_raw)
        if np.any(train_norm <= 1e-12):
            print(f"[warn] near-zero training direction in LORO fold heldout={heldout}")
        fold_train_runs[heldout] = len(set(str(r) for r in d_run_ids[train_mask]))
        for ei in test_idx:
            proj, cos = event_scores_against_direction(effects[ei], train_unit)
            cv_projection[ei] = proj
            cv_cosine[ei] = cos

    # Event-level LORO table.
    direct_score_rows: list[dict[str, Any]] = []
    for ei in range(effects.shape[0]):
        for li, layer in enumerate(layers):
            direct_score_rows.append(
                {
                    "event_id": str(d_event_ids[ei]),
                    "run_id": str(d_run_ids[ei]),
                    "environment": str(d_env[ei]),
                    "condition": str(d_cond[ei]),
                    "objective_status": str(d_status[ei]),
                    "event_type": str(d_event_type[ei]),
                    "layer": int(layer),
                    "train_rollouts_in_fold": fold_train_runs[str(d_run_ids[ei])],
                    "cv_projection": float(cv_projection[ei, li]),
                    "cv_cosine": float(cv_cosine[ei, li]),
                    "effect_norm": float(np.linalg.norm(effects[ei, li])),
                }
            )
    write_csv(out / "direct_event_scores.csv", direct_score_rows)

    # Cross-status transfer: derive direction entirely from one objective-status
    # class and score the other.  This is descriptive but directly targets the
    # subjective-progress-vs-objective-correctness dissociation.
    status_transfer_rows: list[dict[str, Any]] = []
    correct_on_incorrect_cos = np.full(L, np.nan)
    incorrect_on_correct_cos = np.full(L, np.nan)
    correct_on_incorrect_proj = np.full(L, np.nan)
    incorrect_on_correct_proj = np.full(L, np.nan)

    for li, layer in enumerate(layers):
        if np.any(inc_mask) and correct_norms[li] > 1e-12:
            projs, coss = [], []
            for e in effects[inc_mask]:
                p, c = event_scores_against_direction(e, unit_correct)
                projs.append(p[li]); coss.append(c[li])
            correct_on_incorrect_proj[li] = float(np.mean(projs))
            correct_on_incorrect_cos[li] = float(np.mean(coss))
        if np.any(corr_mask) and incorrect_norms[li] > 1e-12:
            projs, coss = [], []
            for e in effects[corr_mask]:
                p, c = event_scores_against_direction(e, unit_incorrect)
                projs.append(p[li]); coss.append(c[li])
            incorrect_on_correct_proj[li] = float(np.mean(projs))
            incorrect_on_correct_cos[li] = float(np.mean(coss))

        status_transfer_rows.append(
            {
                "layer": int(layer),
                "correct_direction_norm": float(correct_norms[li]),
                "incorrect_direction_norm": float(incorrect_norms[li]),
                "correct_vs_incorrect_direction_cosine": float(status_direction_cosine[li]),
                "correct_direction_on_incorrect_mean_projection": float(correct_on_incorrect_proj[li]),
                "correct_direction_on_incorrect_mean_cosine": float(correct_on_incorrect_cos[li]),
                "incorrect_direction_on_correct_mean_projection": float(incorrect_on_correct_proj[li]),
                "incorrect_direction_on_correct_mean_cosine": float(incorrect_on_correct_cos[li]),
            }
        )
    write_csv(out / "direct_status_transfer.csv", status_transfer_rows)

    # Compare full learned progress direction with the existing SV bank.
    v_bank = load_v_bank(directions_dir, layers, K, D)
    nearest_sv_rank = np.full(L, -1, dtype=np.int32)
    nearest_sv_cosine = np.full(L, np.nan, dtype=np.float64)
    nearest_sv_abs_cosine = np.full(L, np.nan, dtype=np.float64)
    for li, layer in enumerate(layers):
        V = v_bank.get(int(layer))
        if V is None or V.shape[1] == 0:
            continue
        vn = V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12)
        cos = unit_full[li].astype(np.float64) @ vn.astype(np.float64)
        j = int(np.argmax(np.abs(cos)))
        nearest_sv_rank[li] = j + 1
        nearest_sv_cosine[li] = float(cos[j])
        nearest_sv_abs_cosine[li] = abs(float(cos[j]))

    # Layer statistics based on CV cosine. Projection stats are included as a
    # secondary scale-sensitive readout.
    direct_layer_rows: list[dict[str, Any]] = []
    for li, layer in enumerate(layers):
        cos_vals = cv_cosine[:, li]
        proj_vals = cv_projection[:, li]
        all_cos = descriptive_group_stats(cos_vals, d_run_ids, d_status, "all")
        corr_cos = descriptive_group_stats(cos_vals, d_run_ids, d_status, "correct")
        inc_cos = descriptive_group_stats(cos_vals, d_run_ids, d_status, "incorrect")
        all_proj = descriptive_group_stats(proj_vals, d_run_ids, d_status, "all")
        rv, _ = group_rollout_means(cos_vals, d_run_ids, d_status, "all")
        inf = inference_from_rollout_values(rv, args.bootstrap_samples, args.permutations, rng)
        pres = preservation_ratio(
            all_cos["rollout_weighted_mean"],
            corr_cos["rollout_weighted_mean"],
            inc_cos["rollout_weighted_mean"],
        )
        score = ranking_score(
            all_cos["standardized_rollout_effect"],
            all_cos["rollout_consistency"],
            pres,
        )
        direct_layer_rows.append(
            {
                "layer": int(layer),
                "direction_raw_norm": float(full_norms[li]),
                "cv_cosine_n_events": all_cos["n_events"],
                "cv_cosine_n_rollouts": all_cos["n_rollouts"],
                "cv_cosine_event_weighted_mean": all_cos["event_weighted_mean"],
                "cv_cosine_rollout_weighted_mean": all_cos["rollout_weighted_mean"],
                "cv_cosine_rollout_sd": all_cos["rollout_sd"],
                "cv_cosine_rollout_se": all_cos["rollout_se"],
                "cv_cosine_standardized_rollout_effect": all_cos["standardized_rollout_effect"],
                "cv_cosine_rollout_consistency": all_cos["rollout_consistency"],
                "cv_cosine_event_consistency": all_cos["event_consistency"],
                "bootstrap_ci_low": inf["bootstrap_ci_low"],
                "bootstrap_ci_high": inf["bootstrap_ci_high"],
                "permutation_p_two_sided": inf["permutation_p_two_sided"],
                "permutation_p_directional": inf["permutation_p_directional"],
                "correct_cv_cosine_n_events": corr_cos["n_events"],
                "correct_cv_cosine_n_rollouts": corr_cos["n_rollouts"],
                "correct_cv_cosine_rollout_weighted_mean": corr_cos["rollout_weighted_mean"],
                "correct_cv_cosine_rollout_consistency": corr_cos["rollout_consistency"],
                "incorrect_cv_cosine_n_events": inc_cos["n_events"],
                "incorrect_cv_cosine_n_rollouts": inc_cos["n_rollouts"],
                "incorrect_cv_cosine_rollout_weighted_mean": inc_cos["rollout_weighted_mean"],
                "incorrect_cv_cosine_rollout_consistency": inc_cos["rollout_consistency"],
                "cv_projection_rollout_weighted_mean": all_proj["rollout_weighted_mean"],
                "cv_projection_standardized_rollout_effect": all_proj["standardized_rollout_effect"],
                "status_preservation_ratio": pres,
                "correct_vs_incorrect_direction_cosine": float(status_direction_cosine[li]),
                "correct_direction_on_incorrect_mean_cosine": float(correct_on_incorrect_cos[li]),
                "incorrect_direction_on_correct_mean_cosine": float(incorrect_on_correct_cos[li]),
                "nearest_scanned_sv_rank": int(nearest_sv_rank[li]) if nearest_sv_rank[li] > 0 else None,
                "nearest_scanned_sv_cosine": float(nearest_sv_cosine[li]),
                "nearest_scanned_sv_abs_cosine": float(nearest_sv_abs_cosine[li]),
                "rank_score": score,
            }
        )

    adjusted = holm_adjust([r["permutation_p_two_sided"] for r in direct_layer_rows])
    for row, p_adj in zip(direct_layer_rows, adjusted):
        row["permutation_p_holm"] = p_adj
    write_csv(out / "direct_layer_statistics.csv", direct_layer_rows)

    direct_rankings = sorted(
        direct_layer_rows,
        key=lambda r: (
            -float(r["rank_score"]),
            float(r["permutation_p_holm"]) if finite(r["permutation_p_holm"]) else 2.0,
            int(r["layer"]),
        ),
    )
    direct_rankings = [{"rank": i + 1, **r} for i, r in enumerate(direct_rankings)]
    write_csv(out / "direct_layer_rankings.csv", direct_rankings)
    write_json(out / "direct_layer_rankings.json", direct_rankings)

    # Carry every layer direction forward; do not silently hard-code the top layer.
    np.savez_compressed(
        out / "progress_directions.npz",
        layers=layers.astype(np.int16),
        unit_directions=unit_full.astype(np.float32),
        raw_mean_directions=raw_full.astype(np.float32),
        raw_direction_norms=full_norms.astype(np.float32),
        correct_unit_directions=unit_correct.astype(np.float32),
        incorrect_unit_directions=unit_incorrect.astype(np.float32),
        correct_incorrect_direction_cosine=status_direction_cosine.astype(np.float32),
        nearest_scanned_sv_rank=nearest_sv_rank.astype(np.int32),
        nearest_scanned_sv_cosine=nearest_sv_cosine.astype(np.float32),
    )

    # ------------------------------------------------------------------
    # Compact human-readable summary
    # ------------------------------------------------------------------
    top_sv = sv_rankings[:10]
    top_direct = direct_rankings[:10]
    status_counts = {s: int(np.sum(statuses_arr == s)) for s in STATUSES}
    eligible_status_counts = {s: int(np.sum(d_status == s)) for s in STATUSES}

    summary = {
        "schema_version": SCHEMA_VERSION,
        "data": str(data_root),
        "directions_dir": str(directions_dir),
        "window_rows": len(index_rows),
        "events_total": len(event_ids),
        "events_direct_eligible": int(effects.shape[0]),
        "event_status_counts": status_counts,
        "direct_eligible_status_counts": eligible_status_counts,
        "direct_eligible_rollouts": len(unique_runs),
        "layers": [int(x) for x in layers],
        "hidden_size": D,
        "sv_k": K,
        "sv_rank_metric": args.sv_rank_metric,
        "intervals": {
            "pre": [args.pre_start, args.pre_end],
            "post": [args.post_start, args.post_end],
            "baseline": [args.baseline_start, args.baseline_end],
            "peak": [args.peak_start, args.peak_end],
        },
        "min_controls": args.min_controls,
        "top_sv_candidates": top_sv,
        "top_direct_layers": top_direct,
        "definitions": {
            "sv_effect": "event scalar metric minus mean matched-control scalar metric",
            "direct_event_effect": "(event post-pre residual transition) minus mean matched-control transition",
            "direct_direction": "equal-rollout mean direct_event_effect, L2 normalized",
            "direct_cv": "leave-one-rollout-out: direction estimated from all other rollouts",
            "primary_direct_score": "cosine(event effect, held-out-fold direction)",
        },
        "caution": (
            "This is discovery on the annotated campaign. LORO tests trajectory-level generalization "
            "within the campaign, but final causal steering should be evaluated on new rollouts."
        ),
    }
    write_json(out / "analysis_config.json", vars(args))
    write_json(out / "summary.json", summary)

    lines = [
        "# Subjective-progress discovery summary",
        "",
        f"- Events: **{len(event_ids)}** total; **{effects.shape[0]}** direct-analysis eligible across **{len(unique_runs)}** rollouts.",
        f"- Objective statuses among eligible events: correct={eligible_status_counts['correct']}, incorrect={eligible_status_counts['incorrect']}, ambiguous={eligible_status_counts['ambiguous']}.",
        f"- Model representation: {L} layers, hidden size {D}; scanned SVs/layer={K}.",
        f"- Direct transition: mean t={args.post_start}:{args.post_end} minus mean t={args.pre_start}:{args.pre_end}, then event minus matched controls.",
        "- Direct validation: leave-one-rollout-out; rollout is the independent unit.",
        "",
        "## Top direct residual-space layers",
        "",
        "| rank | layer | CV cosine | std effect | Holm p | rollout consistency | correct | incorrect | status-dir cosine | nearest SV | |cos| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_direct:
        lines.append(
            "| {rank} | {layer} | {mean:.4f} | {std:.3f} | {p:.4g} | {cons:.3f} | {corr:.4f} | {inc:.4f} | {statuscos:.3f} | {sv} | {svcos:.3f} |".format(
                rank=r["rank"],
                layer=r["layer"],
                mean=r["cv_cosine_rollout_weighted_mean"],
                std=r["cv_cosine_standardized_rollout_effect"] if finite(r["cv_cosine_standardized_rollout_effect"]) else float("nan"),
                p=r["permutation_p_holm"] if finite(r["permutation_p_holm"]) else float("nan"),
                cons=r["cv_cosine_rollout_consistency"] if finite(r["cv_cosine_rollout_consistency"]) else float("nan"),
                corr=r["correct_cv_cosine_rollout_weighted_mean"] if finite(r["correct_cv_cosine_rollout_weighted_mean"]) else float("nan"),
                inc=r["incorrect_cv_cosine_rollout_weighted_mean"] if finite(r["incorrect_cv_cosine_rollout_weighted_mean"]) else float("nan"),
                statuscos=r["correct_vs_incorrect_direction_cosine"] if finite(r["correct_vs_incorrect_direction_cosine"]) else float("nan"),
                sv=r["nearest_scanned_sv_rank"] if r["nearest_scanned_sv_rank"] is not None else "-",
                svcos=r["nearest_scanned_sv_abs_cosine"] if finite(r["nearest_scanned_sv_abs_cosine"]) else float("nan"),
            )
        )

    lines.extend([
        "",
        f"## Top SV candidates ({args.sv_rank_metric})",
        "",
    ])
    if top_sv:
        lines.extend([
            "| rank | layer | SV | effect | std effect | Holm p | consistency | correct | incorrect | preservation |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for r in top_sv:
            lines.append(
                "| {rank} | {layer} | {sv} | {effect:.4f} | {std:.3f} | {p:.4g} | {cons:.3f} | {corr:.4f} | {inc:.4f} | {pres:.3f} |".format(
                    rank=r["rank"], layer=r["layer"], sv=r["sv_rank"],
                    effect=r["rollout_weighted_mean"],
                    std=r["standardized_rollout_effect"] if finite(r["standardized_rollout_effect"]) else float("nan"),
                    p=r["permutation_p_holm"] if finite(r["permutation_p_holm"]) else float("nan"),
                    cons=r["rollout_consistency"] if finite(r["rollout_consistency"]) else float("nan"),
                    corr=r["correct_rollout_weighted_mean"] if finite(r["correct_rollout_weighted_mean"]) else float("nan"),
                    inc=r["incorrect_rollout_weighted_mean"] if finite(r["incorrect_rollout_weighted_mean"]) else float("nan"),
                    pres=r["status_preservation_ratio"],
                )
            )
    else:
        lines.append("No SV bank was collected.")

    lines.extend([
        "",
        "## What to inspect next",
        "",
        "The strongest dopamine-like candidate is not simply the largest effect. Prefer a layer/direction with positive held-out CV cosine across rollouts, preservation in both correct and incorrect subjective-progress events, and good correct↔incorrect cross-status transfer. The nearest-SV column shows whether the directly learned direction is approximately one existing J-lens singular vector or a mixture outside any single scanned SV.",
        "",
        "`progress_directions.npz` contains the full-data unit direction for every layer. Use a top-ranked direction only in a new causal intervention experiment; do not use the present campaign as the final behavioral test set.",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== discovery summary ===")
    print(f"direct eligible events : {effects.shape[0]}")
    print(f"direct eligible runs   : {len(unique_runs)}")
    if top_direct:
        r = top_direct[0]
        print(
            f"top direct layer       : L{int(r['layer']):02d} "
            f"CVcos={r['cv_cosine_rollout_weighted_mean']:.4f} "
            f"correct={r['correct_cv_cosine_rollout_weighted_mean']:.4f} "
            f"incorrect={r['incorrect_cv_cosine_rollout_weighted_mean']:.4f}"
        )
    if top_sv:
        r = top_sv[0]
        print(
            f"top SV                 : L{int(r['layer']):02d}/SV{int(r['sv_rank']):02d} "
            f"effect={r['rollout_weighted_mean']:.4f} metric={args.sv_rank_metric}"
        )
    print(f"outputs                : {out}")


if __name__ == "__main__":
    main()
