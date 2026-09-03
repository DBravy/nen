#!/usr/bin/env python3
"""
patch_second_moment.py

Finite-size patch experiment for the second-moment analysis produced by
workspace_second_moment.py. The question: do top-G directions (large context
VARIANCE of transport) produce large, context-conditional, decodable output
changes at finite nudge sizes, compared against top-Jbar directions (large
MEAN transport), energy-matched pairs, and random directions?

This script must sit NEXT TO workspace_second_moment.py; it imports the model
plumbing (load_model, Recorder, prompt/recipe handling, harvest loading) from
it so that the patch is applied to exactly the tensor the harvest
differentiated: the output of block l (Recorder's tap), position p, plus
eps * f, with everything downstream free to react.

Design notes baked in:
  * Selection vs evaluation prompts are disjoint by default (the harvest used
    prompts [0, 25); patching defaults to --prompt-offset 25), so a direction
    cannot look good merely by overfitting the selection set. Selection still
    uses the full-harvest S (per-prompt matrices were not stored), so "top-G"
    carries winner's-curse noise in its RANKING; the measurements on fresh
    prompts are unbiased for the fixed directions chosen.
  * Every batch chunk carries a zero-patch row; all deltas are measured
    against the in-batch zero row, which cancels batch-shape nondeterminism.
    Zero rows across chunks give a per-(context, position) noise floor.
  * Both signs at every scale. The antisymmetric half (D+ - D-)/2 is the
    empirical per-context linear response (it equals eps * J_c f to second
    order), so no JVP machinery is needed in the main loop; smoke mode
    verifies the equivalence once via a double-backward JVP.
  * Full-vocab delta-logit vectors are too big to store, so each record keeps
    norms, KL, the top-k logit moves, and a fixed seeded Gaussian sketch
    (Johnson-Lindenstrauss) of delta-logits and of delta-h_target. Cosines
    and norms among sketches approximate the real ones, which is all the
    decoding and template-stability analyses need.

Subcommands
  select   (CPU)  Read the harvest (second_moment.pt) and the lenses; build
                  per-layer direction sets:
                    topG      top eigenvectors of G_J
                    topJbar   top right singular vectors of the J-lens
                    gnotj     top-G components orthogonalized against the
                              lens's top-64 (the "G-only" channels)
                    pairH/pairL  energy-matched pairs: similar f'S_J f,
                              opposite gatedness
                    rand      random unit directions (null)
                  and, when the harvest has both arms:
                    topG_R    top eigenvectors of G_R
                    jpriv     high gamma under J, low under R (the variance
                              the relevance rules discard)
                    shared    high gamma under both arms
                  Writes directions.pt with per-direction metadata
                  (gamma, f'Sf, |Mbar f|^2, both arms where available).
  patch    (GPU)  The intervention loop. For each fresh prompt, position, and
                  selected layer: one clean forward, then batched patched
                  forwards over directions x signs x scales, recording
                  delta-logits / delta-h_target summaries per patch.
                  Checkpointed and resumable. --smoke runs one
                  (prompt, layer, position) and additionally checks
                  (a) the patch hook is transparent at delta = 0 (bitwise),
                  (b) upstream activations are untouched and the harvest tap
                      point is exactly what is being patched,
                  (c) a scale ladder locating the smallest alpha the bf16
                      forward supports at this layer (rungs below it inject
                      quantization, not signal),
                  (d) the double-backward JVP agrees with the finite
                      difference (guarded; skipped on OOM),
                  and prints a time estimate for the full grid.
  analyze  (CPU)  Response curves (antisymmetric and symmetric parts vs
                  alpha), departure-from-linearity scale per direction,
                  context-conditionality (does the small-alpha linear
                  response predict the large-alpha response, context by
                  context), moment cross-check against f'Sf, template
                  stability, and within- vs cross-context decodability.
                  Writes patch_report.txt plus response_curves.csv,
                  per_direction.csv, decoding.csv (and routing.csv when
                  router logging was active).

Usage
  python patch_second_moment.py select  --harvest sm9b --out sm9b_patch/directions.pt
  python patch_second_moment.py patch   --directions sm9b_patch/directions.pt \
                                        --out sm9b_patch --smoke
  python patch_second_moment.py patch   --directions sm9b_patch/directions.pt \
                                        --out sm9b_patch
  python patch_second_moment.py analyze --out sm9b_patch

Cost (L4 24 GB, Qwen3.5-9B bf16): patch is forwards only. The default grid,
30 prompts x 2 positions x 4 layers x (about 40 to 52 directions) x 2 signs
x 5 scales, packed at --patch-batch 8 with a zero row per chunk, is roughly
15k forwards of a 128-token batch: expect 1.5 to 3 h. Trim with --layers,
--sets, --alphas, or --n-prompts. patches.pt lands around 150 to 250 MB at
the default sketch sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    import workspace_second_moment as wsm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "patch_second_moment.py must sit next to workspace_second_moment.py "
        f"(import failed: {exc})")

BASE_SETS = ("topG", "topJbar", "gnotj", "pairH", "pairL", "rand")
R_SETS = ("topG_R", "jpriv", "shared")


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _unit_cols(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-30)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sketch_matrix(dim: int, width: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(dim, width, generator=g) / math.sqrt(dim)


# =============================================================================
# select
# =============================================================================


def _arm_objects(blob, lenses, arm: str, layer: int):
    """Return (S, M, G) as float64 numpy for one arm at one layer."""
    S = blob["S"][arm][layer].double().numpy()
    S = 0.5 * (S + S.T)
    M = lenses[arm].jac[layer].double().numpy()
    G = S - M.T @ M
    G = 0.5 * (G + G.T)
    return S, M, G


def _quad(S: np.ndarray, F: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk,ki->i", F.T, S, F)


def _gamma(S: np.ndarray, M: np.ndarray, F: np.ndarray):
    tot = _quad(S, F)
    inv = np.sum((M @ F) ** 2, axis=0)
    return (tot - inv) / np.maximum(inv, 1e-30), tot, inv


def cmd_select(args: argparse.Namespace) -> None:
    harvest = Path(args.harvest)
    blob, from_ckpt = wsm.load_harvest(harvest)
    cfg = blob["config"]
    arms = list(cfg["arms"])
    if "j" not in arms:
        raise SystemExit("select needs the gradient arm in the harvest (arms must include 'j')")
    have_r = "r" in arms and (args.r_sets != "off")

    # lens paths: CLI wins, else the harvest-recorded path (same rule as analyze)
    for arm in arms:
        if getattr(args, f"lens_path_{arm}", None) is None and cfg.get(f"lens_path_{arm}"):
            setattr(args, f"lens_path_{arm}", cfg[f"lens_path_{arm}"])
    lenses = wsm.resolve_lens_pair(args, "".join(arms))

    source_layers = [l for l in sorted(blob["S"]["j"]) if l in lenses["j"].jac]
    if args.layers:
        layers = [int(x) for x in args.layers]
        missing = [l for l in layers if l not in source_layers]
        if missing:
            raise SystemExit(f"layers {missing} not in harvest/lens (have {source_layers})")
    else:
        # spread four layers across the depth range: early, pre-crossover, hump, late
        fr = (0.08, 0.35, 0.60, 0.85)
        layers = sorted({source_layers[min(len(source_layers) - 1, int(round(f * (len(source_layers) - 1))))]
                         for f in fr})
    d = int(cfg["d_model"])
    rng = np.random.default_rng(args.seed)
    k_top, k_orth, n_pairs, n_rand = args.k_top, args.k_orth, args.n_pairs, args.n_rand

    print(f"[select] harvest={harvest} (checkpoint={from_ckpt}) arms={'+'.join(arms)} "
          f"d={d} layers={layers}")
    out_layers = {}
    for l in layers:
        S_J, M_J, G_J = _arm_objects(blob, lenses, "j", l)
        wG, VG = np.linalg.eigh(G_J)
        wG, VG = wG[::-1], VG[:, ::-1]
        wS, VS = np.linalg.eigh(S_J)
        wS, VS = wS[::-1], VS[:, ::-1]
        _, sv, Vt = np.linalg.svd(M_J, full_matrices=False)
        Vb = _unit_cols(Vt[:64].T)                     # lens top-64 input directions
        VG64 = _unit_cols(VG[:, :64])
        VS64 = _unit_cols(VS[:, :64])

        sets: dict[str, np.ndarray] = {}
        sets["topG"] = VG64[:, :k_top]
        sets["topJbar"] = Vb[:, :k_top]

        # G-not-Jbar: orthogonalize top-G against the lens's top-64, keep big residuals
        got, names_resid = [], []
        for j in range(VG64.shape[1]):
            f = VG64[:, j]
            q = f - Vb @ (Vb.T @ f)
            rn = float(np.linalg.norm(q))
            if rn >= args.min_residual:
                got.append(q / rn)
                names_resid.append((j + 1, rn))
            if len(got) == k_orth:
                break
        sets["gnotj"] = np.stack(got, axis=1) if got else np.zeros((d, 0))

        # energy-matched pairs with opposite gatedness, drawn from a candidate pool
        pool = np.concatenate([VS64, VG64, Vb], axis=1)
        keep = []
        for j in range(pool.shape[1]):
            f = pool[:, j]
            if all(abs(float(f @ pool[:, i])) < 0.999 for i in keep):
                keep.append(j)
        pool = pool[:, keep]
        g_pool, t_pool, _ = _gamma(S_J, M_J, pool)
        med = float(np.median(g_pool))
        hi = [i for i in range(pool.shape[1]) if g_pool[i] > med]
        lo = [i for i in range(pool.shape[1]) if g_pool[i] <= med]
        pairs, used_lo = [], set()
        hi = sorted(hi, key=lambda i: -g_pool[i])
        for ih in hi:
            best, best_cost = None, None
            for il in lo:
                if il in used_lo:
                    continue
                cost = abs(math.log(t_pool[ih] / max(t_pool[il], 1e-30)))
                if cost > math.log(args.pair_energy_tol):
                    continue
                if g_pool[ih] < args.pair_gamma_ratio * max(g_pool[il], 1e-3):
                    continue
                if best_cost is None or cost < best_cost:
                    best, best_cost = il, cost
            if best is not None:
                pairs.append((ih, best))
                used_lo.add(best)
            if len(pairs) == n_pairs:
                break
        sets["pairH"] = pool[:, [a for a, _ in pairs]] if pairs else np.zeros((d, 0))
        sets["pairL"] = pool[:, [b for _, b in pairs]] if pairs else np.zeros((d, 0))

        R = rng.standard_normal((d, n_rand))
        sets["rand"] = _unit_cols(R)

        if have_r:
            S_R, M_R, G_R = _arm_objects(blob, lenses, "r", l)
            wGr, VGr = np.linalg.eigh(G_R)
            sets["topG_R"] = _unit_cols(VGr[:, ::-1][:, :k_top])
            gJ64, _, _ = _gamma(S_J, M_J, VG64)
            gR64, _, _ = _gamma(S_R, M_R, VG64)
            score = (gJ64 + 1.0) / (gR64 + 1.0)
            order = np.argsort(-score)
            sets["jpriv"] = VG64[:, order[:k_orth]]
            both = np.minimum(gJ64, gR64)
            sets["shared"] = VG64[:, np.argsort(-both)[:k_orth]]

        # metadata per direction, both arms where available
        layer_rec = {"sets": {}}
        for name, V in sets.items():
            if V.shape[1] == 0:
                print(f"[select] L{l:02d} {name}: empty (skipped)")
                continue
            V = _unit_cols(V)
            gJ, tJ, iJ = _gamma(S_J, M_J, V)
            rec = {"V": torch.from_numpy(V.astype(np.float32)),
                   "names": [f"L{l:02d}:{name}:{j + 1:02d}" for j in range(V.shape[1])],
                   "gamma_J": gJ.tolist(), "fSf_J": tJ.tolist(), "inv_J": iJ.tolist()}
            if have_r:
                gR, tR, iR = _gamma(S_R, M_R, V)
                rec.update({"gamma_R": gR.tolist(), "fSf_R": tR.tolist(), "inv_R": iR.tolist()})
            layer_rec["sets"][name] = rec
            print(f"[select] L{l:02d} {name:8s} n={V.shape[1]:2d} "
                  f"med gamma_J={float(np.median(gJ)):8.2f} med f'Sf={float(np.median(tJ)):.3e}")
        out_layers[l] = layer_rec

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "harvest": str(harvest), "model": cfg["model"], "d_model": d,
        "target_layer": int(cfg["target_layer"]), "skip_first": int(cfg["skip_first"]),
        "max_seq_len": int(cfg["max_seq_len"]), "dataset": cfg["dataset"],
        "harvest_prompt_offset": int(cfg.get("prompt_offset", 0)),
        "harvest_n_prompts": int(cfg["n_prompts"]),
        "arms": "".join(arms), "have_r_sets": bool(have_r),
        "layers": [int(l) for l in layers], "seed": int(args.seed),
        "k_top": k_top, "k_orth": k_orth, "n_pairs": n_pairs, "n_rand": n_rand,
        "pair_energy_tol": args.pair_energy_tol, "pair_gamma_ratio": args.pair_gamma_ratio,
        "min_residual": args.min_residual,
        "lens_path_j": lenses["j"].path, "lens_path_r": lenses["r"].path if "r" in lenses else None,
    }
    _atomic_save({"config": config, "layers": out_layers}, out)
    print(f"[select] wrote {out}")
    print(f"Next: python {Path(__file__).name} patch --directions {out} "
          f"--out {out.parent} --smoke")


# =============================================================================
# patch
# =============================================================================


class PatchHook:
    """Adds a per-row delta to block l's output at one position."""

    def __init__(self, block):
        self.block = block
        self.delta = None            # [B, seq, d] on device, model dtype, or None
        self.handle = None

    def __enter__(self):
        def hook(_mod, _inp, out):
            if self.delta is None:
                return None
            if isinstance(out, tuple):
                return (out[0] + self.delta.to(out[0].dtype),) + out[1:]
            return out + self.delta.to(out.dtype)
        self.handle = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


class RouterLog:
    """Best-effort MoE router logging: top-k expert ids per hooked layer.

    Looks for block submodules that expose a router-style linear (named 'gate'
    or 'router') when the config declares num_experts_per_tok. Dense models
    (the Qwen3.5-9B default) simply produce no hooks and routing columns are
    omitted. Treat as experimental for anything but gpt-oss-style layouts.
    """

    def __init__(self, hf, blocks, layers):
        self.k = int(getattr(hf.config, "num_experts_per_tok", 0) or 0)
        self.mods, self.handles, self.tops = [], [], {}
        if self.k == 0:
            return
        for i in layers:
            mlp = getattr(blocks[i], "mlp", None)
            gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
            if gate is not None and hasattr(gate, "weight"):
                self.mods.append((i, gate))

    @property
    def active(self) -> bool:
        return bool(self.mods)

    def __enter__(self):
        self.tops = {}
        for i, gate in self.mods:
            def mk(idx):
                def hook(_m, _inp, out):
                    logits = out[0] if isinstance(out, tuple) else out
                    self.tops[idx] = logits.detach().topk(self.k, dim=-1).indices.cpu()
                return hook
            self.handles.append(gate.register_forward_hook(mk(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False

    def sets_at(self, row: int, pos: int):
        """Frozen set of (layer, expert) pairs at one position, one batch row."""
        out = set()
        for i, top in self.tops.items():
            t = top[row]
            t = t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t
            for e in t[pos].tolist():
                out.add((i, int(e)))
        return frozenset(out)


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp = torch.log_softmax(p_logits.float(), dim=-1)
    lq = torch.log_softmax(q_logits.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum())


def cmd_patch(args: argparse.Namespace) -> None:
    dirs_path = Path(args.directions)
    dirs = torch.load(dirs_path, map_location="cpu", weights_only=False)
    dcfg = dirs["config"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "patches_ckpt.pt"

    layers = [int(l) for l in (args.layers or dcfg["layers"])]
    set_names = args.sets or (list(BASE_SETS) + (list(R_SETS) if dcfg["have_r_sets"] else []))
    alphas = [float(a) for a in args.alphas]

    hf, tok, base, blocks = wsm.load_model(args.model or dcfg["model"], args.attn_implementation)
    if hf.config.hidden_size != dcfg["d_model"]:
        raise SystemExit(f"model hidden_size={hf.config.hidden_size} != directions d_model={dcfg['d_model']}")
    dev = next(hf.parameters()).device
    target_layer = int(dcfg["target_layer"])
    skip_first = int(dcfg["skip_first"])
    max_seq_len = int(dcfg["max_seq_len"])
    vocab = int(hf.get_output_embeddings().weight.shape[0])
    d = int(dcfg["d_model"])

    # fresh prompts by default: harvest used [offset, offset + n)
    default_offset = dcfg["harvest_prompt_offset"] + dcfg["harvest_n_prompts"]
    offset = args.prompt_offset if args.prompt_offset is not None else default_offset
    prompts = wsm.load_prompts(args.n_prompts, offset, args.min_chars, dcfg["dataset"])
    print(f"[patch] {len(prompts)} evaluation prompts from {dcfg['dataset']} offset={offset} "
          f"(harvest used [{dcfg['harvest_prompt_offset']}, {default_offset}))")

    # flat direction table for the chosen layers/sets
    table = {}      # layer -> (F [d, N] tensor on dev, meta rows)
    meta_rows = []  # global: one dict per (layer, dir)
    for l in layers:
        cols, metas = [], []
        for s in set_names:
            rec = dirs["layers"].get(l, {}).get("sets", {}).get(s)
            if rec is None:
                continue
            V = rec["V"]
            for j in range(V.shape[1]):
                metas.append({"layer": l, "set": s, "dir": len(metas), "name": rec["names"][j],
                              "gamma_J": rec["gamma_J"][j], "fSf_J": rec["fSf_J"][j],
                              "inv_J": rec["inv_J"][j],
                              "gamma_R": rec.get("gamma_R", [None] * V.shape[1])[j],
                              "fSf_R": rec.get("fSf_R", [None] * V.shape[1])[j]})
                cols.append(V[:, j])
        if not cols:
            raise SystemExit(f"no directions for layer {l} with sets {set_names}")
        F = torch.stack(cols, dim=1).to(dev)
        table[l] = (F, metas)
        meta_rows.extend(metas)
        print(f"[patch] L{l:02d}: {F.shape[1]} directions from sets "
              f"{sorted({m['set'] for m in metas})}")

    P_log = _sketch_matrix(args.sketch_dim, vocab, args.seed * 1000 + 7)
    P_h = _sketch_matrix(args.sketch_dim_h, d, args.seed * 1000 + 8)
    try:
        P_log_d, P_h_d = P_log.to(dev), P_h.to(dev)
    except RuntimeError:
        print("[patch][warn] sketch matrices stay on CPU (device memory)")
        P_log_d, P_h_d = P_log, P_h

    config = {
        "directions": str(dirs_path), "directions_hash": _file_hash(dirs_path),
        "model": args.model or dcfg["model"], "d_model": d, "vocab": vocab,
        "target_layer": target_layer, "skip_first": skip_first, "max_seq_len": max_seq_len,
        "dataset": dcfg["dataset"], "n_prompts": int(args.n_prompts), "prompt_offset": int(offset),
        "min_chars": int(args.min_chars), "layers": layers, "sets": set_names,
        "alphas": alphas, "positions_per_prompt": int(args.positions_per_prompt),
        "patch_batch": int(args.patch_batch), "topk_logits": int(args.topk_logits),
        "sketch_dim": int(args.sketch_dim), "sketch_dim_h": int(args.sketch_dim_h),
        "sketch_seed_log": args.seed * 1000 + 7, "sketch_seed_h": args.seed * 1000 + 8,
        "seed": int(args.seed), "log_routing": args.log_routing,
    }

    state = None
    if args.resume and ckpt_path.exists() and not args.smoke:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise SystemExit(f"checkpoint config differs; delete {ckpt_path} or match args")
        print(f"[resume] {state['n_done']} prompts done, resuming at index {state['next_idx']}")
    if state is None:
        state = {"records": [], "prompt_meta": [], "n_done": 0, "next_idx": 0, "config": config}

    router = RouterLog(hf, blocks, [i for i in range(len(blocks)) if i > min(layers)])
    use_routing = (args.log_routing == "on") or (args.log_routing == "auto" and router.active)
    print(f"[patch] router logging: {'on' if use_routing else 'off'}"
          + (f" ({len(router.mods)} routers, top-{router.k})" if router.active else " (no routers found)"))

    if args.smoke:
        smoke_patch(hf, tok, blocks, prompts[0], table, layers, target_layer,
                    skip_first, max_seq_len, alphas, dev)
        return

    t0 = time.time()
    for idx in range(state["next_idx"], len(prompts)):
        ids = wsm.encode(tok, prompts[idx], max_seq_len, dev)
        seq_len = int(ids.shape[1])
        try:
            valid = wsm.valid_positions(seq_len, skip_first)
        except ValueError as exc:
            print(f"[skip] prompt {idx}: {exc}")
            state["next_idx"] = idx + 1
            continue
        vp = valid[valid <= seq_len - 2].numpy()
        if vp.size == 0:
            state["next_idx"] = idx + 1
            continue
        qs = np.linspace(0.25, 0.9, args.positions_per_prompt)
        positions = sorted({int(vp[min(vp.size - 1, int(round(q * (vp.size - 1))))]) for q in qs})

        # clean forward: h_l norms for eps calibration
        with torch.no_grad(), wsm.Recorder(blocks, layers) as rec:
            hf(input_ids=ids, use_cache=False)
            h_norm = {l: {p: float(rec.acts[l][0, p].float().norm()) for p in positions}
                      for l in layers}

        t_prompt = time.time()
        n_rec_before = len(state["records"])
        for l in layers:
            F, metas = table[l]
            N = F.shape[1]
            for p in positions:
                eps0 = h_norm[l][p]
                jobs = [(j, a, s) for a in alphas for j in range(N) for s in (+1, -1)]
                zero_ref = None          # (logits_row, tgt_row) of the very first zero row
                with PatchHook(blocks[l]) as ph:
                    for c0 in range(0, len(jobs), args.patch_batch - 1):
                        chunk = jobs[c0:c0 + args.patch_batch - 1]
                        B = len(chunk) + 1
                        delta = torch.zeros(B, seq_len, d, device=dev)
                        for b, (j, a, s) in enumerate(chunk, start=1):
                            delta[b, p] = s * a * eps0 * F[:, j]
                        ph.delta = delta
                        ctx = router if use_routing else _null_ctx()
                        with torch.no_grad(), ctx, wsm.Recorder(blocks, [target_layer]) as rec:
                            logits = hf(input_ids=ids.expand(B, -1), use_cache=False).logits
                            tact = rec.acts[target_layer]
                        Lp = logits[:, p, :].float()
                        Tp = tact[:, p, :].float()
                        Tt = tact[:, p:, :].float()
                        if zero_ref is None:
                            zero_ref = (Lp[0].clone(), Tp[0].clone())
                        else:
                            # noise floor: this chunk's zero row vs the first one
                            dl = Lp[0] - zero_ref[0]
                            state["records"].append(_record(
                                idx, l, p, "-", -1, 0.0, 0, 0.0, dl, Tp[0] - zero_ref[1],
                                torch.zeros(1), 0.0, P_log_d, P_h_d, args.topk_logits, None))
                        r0 = router.sets_at(0, p) if use_routing else None
                        for b, (j, a, s) in enumerate(chunk, start=1):
                            dl = Lp[b] - Lp[0]
                            dh = Tp[b] - Tp[0]
                            tail = (Tt[b] - Tt[0]).norm()
                            kl = _kl(Lp[b], Lp[0])
                            flip = None
                            if use_routing:
                                flip = router.sets_at(b, p) != r0
                            m = metas[j]
                            state["records"].append(_record(
                                idx, l, p, m["set"], j, a, s, a * eps0, dl, dh, tail, kl,
                                P_log_d, P_h_d, args.topk_logits, flip))
                        ph.delta = None
        state["prompt_meta"].append({"idx": idx, "offset_idx": offset + idx,
                                     "seq_len": seq_len, "positions": positions,
                                     "h_norms": h_norm})
        state["n_done"] += 1
        state["next_idx"] = idx + 1
        print(f"[patch] prompt {idx + 1}/{len(prompts)} seq={seq_len} pos={positions} "
              f"records+={len(state['records']) - n_rec_before} {time.time() - t_prompt:.0f}s "
              f"elapsed={(time.time() - t0) / 60:.1f}m")
        if (idx + 1) % args.checkpoint_every == 0 or idx + 1 == len(prompts):
            _atomic_save(state, ckpt_path)
            print(f"[checkpoint] {ckpt_path} ({state['n_done']} prompts, "
                  f"{len(state['records'])} records)")

    final = _columnar(state, meta_rows)
    _atomic_save(final, out / "patches.pt")
    print(f"\n[done] {state['n_done']} prompts, {len(state['records'])} records "
          f"-> {out / 'patches.pt'}")
    print(f"Next: python {Path(__file__).name} analyze --out {out}")


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _record(prompt, layer, pos, set_name, dir_id, alpha, sign, eps,
            dlog, dh, tail, kl, P_log, P_h, topk, flip):
    dl_dev = dlog.to(P_log.device)
    dh_dev = dh.to(P_h.device)
    _, i = dlog.abs().topk(topk)
    tail_f = tail if isinstance(tail, float) else (tail.item() if tail.numel() == 1
                                                   else float(tail.norm()))
    return {
        "prompt": prompt, "layer": layer, "pos": pos, "set": set_name, "dir": dir_id,
        "alpha": alpha, "sign": sign, "eps": eps,
        "dlog_norm": float(dlog.norm()), "dh_norm": float(dh.norm()),
        "dh_tail_norm": float(tail_f), "kl": float(kl),
        "sk_log": (P_log @ dl_dev).half().cpu(),
        "sk_h": (P_h @ dh_dev).half().cpu(),
        "top_idx": i.cpu().to(torch.int32),
        "top_val": dlog[i].half().cpu(),
        "flip": (-1 if flip is None else int(flip)),
    }


def _columnar(state, meta_rows) -> dict:
    recs = state["records"]
    keys_f = ("alpha", "sign", "eps", "dlog_norm", "dh_norm", "dh_tail_norm", "kl")
    keys_i = ("prompt", "layer", "pos", "dir", "flip")
    out = {"config": state["config"], "prompt_meta": state["prompt_meta"],
           "direction_meta": meta_rows,
           "set": [r["set"] for r in recs]}
    for k in keys_f:
        out[k] = torch.tensor([float(r[k]) for r in recs], dtype=torch.float32)
    for k in keys_i:
        out[k] = torch.tensor([int(r[k]) for r in recs], dtype=torch.int32)
    out["sk_log"] = torch.stack([r["sk_log"] for r in recs])
    out["sk_h"] = torch.stack([r["sk_h"] for r in recs])
    out["top_idx"] = torch.stack([r["top_idx"] for r in recs])
    out["top_val"] = torch.stack([r["top_val"] for r in recs])
    return out


# =============================================================================
# patch smoke checks
# =============================================================================


def smoke_patch(hf, tok, blocks, prompt, table, layers, target_layer,
                skip_first, max_seq_len, alphas, dev):
    l = layers[0]
    F, metas = table[l]
    ids = wsm.encode(tok, prompt, max_seq_len, dev)
    seq_len = int(ids.shape[1])
    valid = wsm.valid_positions(seq_len, skip_first)
    p = int(valid[valid.numel() // 2])
    d = F.shape[0]
    print(f"[smoke] layer={l} target={target_layer} pos={p} seq={seq_len} "
          f"dirs={F.shape[1]}")

    # (a) hook transparency at delta = 0
    with torch.no_grad():
        ref = hf(input_ids=ids, use_cache=False).logits.float().cpu()
    with PatchHook(blocks[l]) as ph:
        ph.delta = torch.zeros(1, seq_len, d, device=dev)
        with torch.no_grad():
            alt = hf(input_ids=ids, use_cache=False).logits.float().cpu()
    diff = float((ref - alt).abs().max())
    print(f"[smoke] (a) zero-delta transparency: max|dlogits|={diff:.3e} <- should be exactly 0")

    # (b) tap point pre-patch, additive semantics at (l, p), downstream reaction.
    # Context entry order matters: `pre` registers its hook before PatchHook, so it
    # captures the residual BEFORE the delta (the tensor the harvest differentiated);
    # `post` registers after, so it sees the modified output.
    f0 = F[:, 0]
    with torch.no_grad(), wsm.Recorder(blocks, [l, target_layer]) as rec:
        hf(input_ids=ids, use_cache=False)
        up_clean = rec.acts[l][0].float().cpu()
        tgt_clean = rec.acts[target_layer][0].float().cpu()
    eps = 0.1 * float(up_clean[p].norm())
    with torch.no_grad(), wsm.Recorder(blocks, [l, target_layer]) as pre, \
            PatchHook(blocks[l]) as ph, wsm.Recorder(blocks, [l]) as post:
        ph.delta = torch.zeros(1, seq_len, d, device=dev)
        ph.delta[0, p] = eps * f0
        hf(input_ids=ids, use_cache=False)
        up_pre = pre.acts[l][0].float().cpu()
        up_post = post.acts[l][0].float().cpu()
        tgt_pat = pre.acts[target_layer][0].float().cpu()
    dd = up_post - up_pre
    off = dd.clone()
    off[p] = 0
    want = (eps * f0).float().cpu()
    print(f"[smoke] (b) harvest tap sees pre-patch state: max|dh_l|="
          f"{float((up_clean - up_pre).abs().max()):.3e} <- should be exactly 0")
    print(f"[smoke] (b) additive at (l,p): off-position max={float(off.abs().max()):.3e} "
          f"(should be 0), on-position |realized - intended| / |intended| = "
          f"{float((dd[p] - want).norm() / (want.norm() + 1e-12)):.2e} (bf16 rounding level)")
    print(f"[smoke] (b) downstream reacts: |dh_target[p]|={float((tgt_clean - tgt_pat)[p].norm()):.3e} "
          f"(patch |delta|={eps:.3e})")

    # (c) scale ladder: find the smallest alpha the bf16 forward supports here.
    # Adding a delta below the residual stream's bf16 mantissa resolution injects a
    # quantized, sign-asymmetric perturbation, so the antisym response stops scaling
    # with alpha. Deviations at the SMALL end of the ladder are that floor;
    # deviations at the large end are genuine nonlinearity.
    eps0 = float(up_clean[p].norm())

    def anti_at(a):
        outs = {}
        for sgn in (+1, -1):
            with PatchHook(blocks[l]) as ph:
                ph.delta = torch.zeros(1, seq_len, d, device=dev)
                ph.delta[0, p] = sgn * a * eps0 * f0
                with torch.no_grad(), wsm.Recorder(blocks, [target_layer]) as rec:
                    hf(input_ids=ids, use_cache=False)
                    outs[sgn] = rec.acts[target_layer][0, p].float().cpu()
        return 0.5 * (outs[1] - outs[-1])

    ladder = sorted({0.005, 0.01, 0.02, 0.05, 0.1, 0.2, min(alphas)})
    antis = {a: anti_at(a) for a in ladder}
    rec_a = None
    for a1, a2 in zip(ladder, ladder[1:]):
        pred = antis[a1] * (a2 / a1)
        dev_lin = float((antis[a2] - pred).norm() / (pred.norm() + 1e-12))
        if dev_lin < 0.15 and rec_a is None:
            rec_a = a1
        print(f"[smoke] (c) alpha {a1:g} -> {a2:g}: |antisym| {float(antis[a1].norm()):.3e} -> "
              f"{float(antis[a2].norm()):.3e}, scaling deviation {dev_lin:.1%}")
    if rec_a is None:
        print("[smoke] (c) no rung scaled cleanly; this layer may be strongly nonlinear "
              "at all tested scales")
    else:
        print(f"[smoke] (c) smallest reliable alpha at this layer: ~{rec_a:g}. Make it the "
              f"bottom of --alphas; rungs below it are bf16 quantization, not signal.")

    # (d) double-backward JVP vs finite difference, on a truncated sequence so the
    # create-graph backward fits next to the model. Both sides use the same input.
    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        jseq = min(seq_len, 64)
        idsj = ids[:, :jseq]
        pj = min(p, jseq - 2)
        with torch.no_grad(), wsm.Recorder(blocks, [l]) as rc:
            hf(input_ids=idsj, use_cache=False)
            epsj = float(rc.acts[l][0, pj].float().norm())
        aj = 0.05
        outs = {}
        for sgn in (+1, -1):
            with PatchHook(blocks[l]) as ph:
                ph.delta = torch.zeros(1, jseq, d, device=dev)
                ph.delta[0, pj] = sgn * aj * epsj * f0
                with torch.no_grad(), wsm.Recorder(blocks, [target_layer]) as rc:
                    hf(input_ids=idsj, use_cache=False)
                    outs[sgn] = rc.acts[target_layer][0, pj].float().cpu()
        fd = (0.5 * (outs[1] - outs[-1])) / (aj * epsj)
        with wsm.Recorder(blocks, [l, target_layer]) as rc, torch.enable_grad():
            hf(input_ids=idsj, use_cache=False)
            tact = rc.acts[target_layer]
            sact = rc.acts[l]
            u = torch.zeros_like(tact, requires_grad=True)
            g = torch.autograd.grad(tact, sact, grad_outputs=u, create_graph=True)[0]
            deltav = torch.zeros_like(sact)
            deltav[0, pj] = f0.to(deltav.dtype)
            jv = torch.autograd.grad((g * deltav).sum(), u, retain_graph=False)[0]
            jv_p = jv[0, pj].float().cpu()
        del g, jv, tact, sact, u, deltav
        cos = float((fd @ jv_p) / (fd.norm() * jv_p.norm() + 1e-12))
        rat = float(fd.norm() / (jv_p.norm() + 1e-12))
        print(f"[smoke] (d) JVP vs finite difference (seq truncated to {jseq}, alpha={aj:g}): "
              f"cos={cos:.4f} |FD|/|JVP|={rat:.3f} "
              f"<- cos~1, ratio~1: the FD antisym IS the per-context linear response")
    except (RuntimeError, SystemExit) as exc:
        print(f"[smoke] (d) JVP check skipped ({type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]})")
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # (e) time estimate
    t = time.time()
    with PatchHook(blocks[l]) as ph:
        ph.delta = torch.zeros(8, seq_len, d, device=dev)
        with torch.no_grad():
            hf(input_ids=ids.expand(8, -1), use_cache=False)
    per_fwd = time.time() - t
    print(f"[smoke] (e) one batch-8 forward: {per_fwd:.2f}s. Grid forwards scale as "
          f"n_prompts * positions * layers * ceil(dirs*2*alphas / (batch-1)); "
          f"see docstring for the default-grid estimate.")


# =============================================================================
# analyze
# =============================================================================


def _akey(x) -> float:
    """Canonical alpha value: alphas ride through float32 storage, so 0.05 comes
    back as 0.05000000074...; round so dict keys match the config's floats."""
    return round(float(x), 6)


def _antisym_pairs(P):
    """Index pairs (i+, i-) matched on (prompt, layer, pos, set, dir, alpha)."""
    key = {}
    sets = P["set"]
    for i in range(len(sets)):
        if sets[i] == "-":
            continue
        k = (int(P["prompt"][i]), int(P["layer"][i]), int(P["pos"][i]), sets[i],
             int(P["dir"][i]), _akey(P["alpha"][i]))
        key.setdefault(k, {})[int(P["sign"][i])] = i
    return [(k, v[1], v[-1]) for k, v in key.items() if 1 in v and -1 in v]


def cmd_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    P = torch.load(out / "patches.pt", map_location="cpu", weights_only=False)
    cfg = P["config"]
    dmeta = {(m["layer"], m["set"], m["dir"]): m for m in P["direction_meta"]}
    alphas = sorted(_akey(a) for a in cfg["alphas"])
    a0 = alphas[0]
    layers = cfg["layers"]
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"Finite-size patch analysis: {cfg['model']}  prompts={cfg['n_prompts']} "
         f"(offset {cfg['prompt_offset']})  layers={layers}  alphas={alphas}")
    emit(f"directions: {cfg['directions']} ({cfg['directions_hash']})")
    emit("=" * 78)

    sk_log = P["sk_log"].float()
    sk_h = P["sk_h"].float()
    pairs = _antisym_pairs(P)
    noise_mask = [i for i, s in enumerate(P["set"]) if s == "-"]
    noise = float(np.median([float(sk_log[i].norm()) for i in noise_mask])) if noise_mask else 0.0
    emit(f"\nZero-row noise floor (median |sketch dlogits| across {len(noise_mask)} "
         f"zero-vs-zero pairs): {noise:.4g}")
    emit("Records whose antisymmetric response sits below ~3x this floor are unreliable;")
    emit("per-direction SNR at the smallest alpha is in per_direction.csv.")

    # organize antisym data: (layer,set,dir) -> alpha -> list over (prompt,pos)
    A: dict = {}
    for (k, ip, im) in pairs:
        prompt, layer, pos, s, dir_id, alpha = k
        anti_l = 0.5 * (sk_log[ip] - sk_log[im])
        anti_h = 0.5 * (sk_h[ip] - sk_h[im])
        sym_l = 0.5 * (sk_log[ip] + sk_log[im])
        A.setdefault((layer, s, dir_id), {}).setdefault(alpha, []).append({
            "cp": (prompt, pos), "anti_l": anti_l, "anti_h": anti_h,
            "skp": sk_log[ip], "skm": sk_log[im],
            "anti_ln": float(anti_l.norm()), "anti_hn": float(anti_h.norm()),
            "sym_ln": float(sym_l.norm()),
            "eps": float(P["eps"][ip]),
            "raw_ln": float(P["dlog_norm"][ip]), "kl": float(P["kl"][ip]),
            "tail_n": 0.5 * (float(P["dh_tail_norm"][ip]) + float(P["dh_tail_norm"][im])),
            "flip": int(P["flip"][ip]),
        })

    import csv
    # ---- response curves per (layer, set, alpha) ------------------------------
    emit("\nResponse curves: median antisymmetric |dlogits| (sketch) per unit eps, by")
    emit("direction set and alpha; sym/anti is the nonlinear fraction (should grow with")
    emit("alpha, from ~0). Full curves in response_curves.csv.")
    curve_rows = []
    emit(f"{'layer':>5} {'set':>8} " + "".join(f" a={a:<8g}" for a in alphas) + "   sym/anti@max")
    for layer in layers:
        set_list = sorted({s for (l2, s, _) in A if l2 == layer})
        for s in set_list:
            entry = [v for (l2, s2, _), v in A.items() if l2 == layer and s2 == s]
            row = f"{layer:>5} {s:>8}"
            frac = float("nan")
            for a in alphas:
                vals = [e["anti_ln"] / max(e["eps"], 1e-30) for d_ in entry for e in d_.get(a, [])]
                med = float(np.median(vals)) if vals else float("nan")
                row += f" {med:9.3g}"
                curve_rows.append({"layer": layer, "set": s, "alpha": a,
                                   "med_anti_per_eps": med, "n": len(vals)})
                if a == alphas[-1] and vals:
                    sf = [e["sym_ln"] / max(e["anti_ln"], 1e-30) for d_ in entry for e in d_.get(a, [])]
                    frac = float(np.median(sf))
            emit(row + f"   {frac:.2f}")
    with (out / "response_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)

    # ---- per-direction statistics --------------------------------------------
    emit("\nPer-direction: g0 = median antisym |dh_target|/eps at the smallest alpha (the")
    emit("measured per-context linear gain); moment check g2/fSf compares mean g0^2 to the")
    emit("harvest's f'S f (same order expected, not equality: S sums future targets while")
    emit("patches read one position); tail^2/fSf does the same with the response summed")
    emit("over positions >= p (closer to S's future-summed convention, includes a small")
    emit("symmetric part); alpha* = first alpha whose antisym departs >25% from linear")
    emit("scaling (nan: linear through the whole tested range); corr_lin = per-context")
    emit("gain at the smallest alpha predicting the mid-alpha antisym (foreseeability in")
    emit("the linear regime); corr_sat = same predictor against the raw largest-alpha")
    emit("response (saturation predictability); stability = mean pairwise cos of antisym")
    emit("dlogit sketches across contexts (the output template).")
    dir_rows = []
    for (layer, s, dir_id), by_a in sorted(A.items()):
        m = dmeta.get((layer, s, dir_id), {})
        e0 = by_a.get(a0, [])
        if not e0:
            continue
        g0 = [e["anti_hn"] / max(e["eps"], 1e-30) for e in e0]
        g0_med = float(np.median(g0))
        g2 = float(np.mean(np.array(g0) ** 2))
        snr = (float("inf") if noise == 0
               else float(np.median([e["anti_ln"] for e in e0]) / noise))
        base = {e["cp"]: e["anti_ln"] / max(e["eps"], 1e-30) for e in e0}
        astar = float("nan")
        for a in alphas[1:]:
            ea = by_a.get(a, [])
            devs = [abs(e["anti_ln"] / max(e["eps"], 1e-30) / max(base.get(e["cp"], 1e-30), 1e-30) - 1.0)
                    for e in ea if e["cp"] in base]
            if devs and float(np.median(devs)) > 0.25 and math.isnan(astar):
                astar = a

        def _corr(pairs_xy):
            if len(pairs_xy) < 4:
                return float("nan")
            x = np.array([q[0] for q in pairs_xy])
            ydat = np.array([q[1] for q in pairs_xy])
            if x.std() == 0 or ydat.std() == 0:
                return float("nan")
            return float(np.corrcoef(x, ydat)[0, 1])

        a_mid = alphas[min(2, len(alphas) - 1)]
        corr_lin = _corr([(base[e["cp"]], e["anti_ln"] / max(e["eps"], 1e-30))
                          for e in by_a.get(a_mid, []) if e["cp"] in base])
        abig = alphas[-1]
        corr = _corr([(base[e["cp"]], e["raw_ln"]) for e in by_a.get(abig, [])
                      if e["cp"] in base])
        gt = [e["tail_n"] / max(e["eps"], 1e-30) for e in e0]
        gt2 = float(np.mean(np.array(gt) ** 2)) if gt else float("nan")
        sks = [e["anti_l"] / max(e["anti_ln"], 1e-30) for e in e0 if e["anti_ln"] > 3 * noise]
        stab = float("nan")
        if len(sks) >= 2:
            S_ = torch.stack(sks)
            C = (S_ @ S_.T).numpy()
            iu = np.triu_indices(len(sks), 1)
            stab = float(np.mean(C[iu]))
        flips = [e["flip"] for a in alphas for e in by_a.get(a, []) if e["flip"] >= 0]
        dir_rows.append({
            "layer": layer, "set": s, "dir": dir_id, "name": m.get("name", ""),
            "gamma_J": m.get("gamma_J"), "fSf_J": m.get("fSf_J"), "inv_J": m.get("inv_J"),
            "gamma_R": m.get("gamma_R"), "g0_med": g0_med, "g0sq_over_fSf":
                (g2 / m["fSf_J"] if m.get("fSf_J") else float("nan")),
            "gtail_sq_over_fSf": (gt2 / m["fSf_J"] if m.get("fSf_J") else float("nan")),
            "snr_a0": snr, "alpha_star": astar, "corr_lin": corr_lin, "corr_pred": corr,
            "template_stability": stab,
            "flip_frac": (float(np.mean(flips)) if flips else float("nan")),
        })
    if not dir_rows:
        emit("[warn] no per-direction rows survived; check alpha keys and SNR threshold")
    else:
        with (out / "per_direction.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dir_rows[0].keys()))
            w.writeheader()
            w.writerows(dir_rows)
    for layer in layers:
        emit(f"\n  L{layer:02d} set medians "
             f"(gamma_J | g0 | g0^2/fSf | tail^2/fSf | alpha* | corr_lin | corr_sat | stability):")
        for s in sorted({r["set"] for r in dir_rows if r["layer"] == layer}):
            rs = [r for r in dir_rows if r["layer"] == layer and r["set"] == s]
            def med(key):
                vals = [r[key] for r in rs if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
                return float(np.median(vals)) if vals else float("nan")
            emit(f"    {s:>8}: {med('gamma_J'):8.2f} | {med('g0_med'):9.3g} | "
                 f"{med('g0sq_over_fSf'):7.2f} | {med('gtail_sq_over_fSf'):7.2f} | "
                 f"{med('alpha_star'):6.3g} | {med('corr_lin'):6.2f} | "
                 f"{med('corr_pred'):6.2f} | {med('template_stability'):6.2f}")

    # ---- decodability ---------------------------------------------------------
    emit("\nDecodability (nearest-centroid on antisym dlogit sketches, chance = 1/K over")
    emit("all directions at the layer). within: match a +patch against the same context's")
    emit("sign-flipped -patch templates. cross: leave-one-context-out mean templates.")
    emit("The gating prediction: invariant directions decode cross-context; gated")
    emit("directions decode within-context but their cross-context templates wash out.")
    dec_rows = []
    for layer in layers:
        dirs_l = sorted({(s, d_) for (l2, s, d_) in A if l2 == layer})
        K = len(dirs_l)
        for a in alphas:
            per_cp: dict = {}
            for di, (s, d_) in enumerate(dirs_l):
                for e in A.get((layer, s, d_), {}).get(a, []):
                    if e["anti_ln"] <= 3 * noise:
                        continue
                    per_cp.setdefault(e["cp"], {})[di] = {
                        "q": e["skp"] / max(float(e["skp"].norm()), 1e-30),
                        "t": -e["skm"] / max(float(e["skm"].norm()), 1e-30),
                        "anti": e["anti_l"] / e["anti_ln"]}
            within = {di: [0, 0] for di in range(K)}
            for cp, vecs in per_cp.items():
                if len(vecs) < 2:
                    continue
                dims = sorted(vecs)
                T_ = torch.stack([vecs[di]["t"] for di in dims])
                for di in dims:
                    sim = T_ @ vecs[di]["q"]
                    pred = dims[int(sim.argmax())]
                    within[di][0] += int(pred == di)
                    within[di][1] += 1
            cross = {di: [0, 0] for di in range(K)}
            cps = list(per_cp.keys())
            for cp in cps:
                others = [c for c in cps if c != cp]
                if not others:
                    continue
                cents = []
                for di in range(K):
                    vs = [per_cp[c][di]["anti"] for c in others if di in per_cp[c]]
                    cents.append(torch.stack(vs).mean(0) if vs else None)
                for di, v in per_cp[cp].items():
                    sims = [(-2.0 if c is None else float(c @ v["anti"] / (c.norm() + 1e-30)))
                            for c in cents]
                    pred = int(np.argmax(sims))
                    cross[di][0] += int(pred == di)
                    cross[di][1] += 1
            for setname in sorted({s for s, _ in dirs_l}):
                idxs = [i for i, (s, _) in enumerate(dirs_l) if s == setname]
                win = [within[i] for i in idxs]
                crs = [cross[i] for i in idxs]
                wacc = (sum(w[0] for w in win) / max(sum(w[1] for w in win), 1))
                cacc = (sum(c[0] for c in crs) / max(sum(c[1] for c in crs), 1))
                dec_rows.append({"layer": layer, "alpha": a, "set": setname, "K": K,
                                 "within_acc": round(wacc, 4), "cross_acc": round(cacc, 4),
                                 "n_within": sum(w[1] for w in win), "n_cross": sum(c[1] for c in crs)})
    if dec_rows:
        with (out / "decoding.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dec_rows[0].keys()))
            w.writeheader()
            w.writerows(dec_rows)
    mid_a = alphas[len(alphas) // 2]
    emit(f"\n  decoding at alpha={mid_a} (within / cross, chance=1/K):")
    for r in dec_rows:
        if r["alpha"] == mid_a:
            emit(f"    L{r['layer']:02d} {r['set']:>8}: {r['within_acc']:.2f} / {r['cross_acc']:.2f} "
                 f"(K={r['K']}, n={r['n_within']}/{r['n_cross']})")

    # ---- routing --------------------------------------------------------------
    fl = P["flip"]
    if int((fl >= 0).sum()) > 0:
        emit("\nRouting flips: fraction of patches changing any downstream top-k expert set")
        emit("at the patched position (routing.csv has the per-set-per-alpha table). Split")
        emit("results by flip before trusting a linear reading of large-alpha responses.")
        rr = []
        set_arr = np.array(P["set"])
        layer_arr = P["layer"].numpy()
        alpha_arr = P["alpha"].numpy()
        fl_arr = fl.numpy()
        for layer in layers:
            for s in sorted(set(P["set"])):
                if s == "-":
                    continue
                for a in alphas:
                    m = (set_arr == s) & (layer_arr == layer) & \
                        (np.isclose(alpha_arr, a)) & (fl_arr >= 0)
                    if m.any():
                        rr.append({"layer": layer, "set": s, "alpha": a,
                                   "flip_frac": round(float(fl_arr[m].mean()), 4),
                                   "n": int(m.sum())})
        if rr:
            with (out / "routing.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
                w.writeheader()
                w.writerows(rr)
    else:
        emit("\nRouting: not logged (dense model or no routers detected).")

    emit("\nReading guide. The strong claim (variance is conditional signal) predicts, for")
    emit("topG and gnotj vs topJbar and pairL: larger g0, corr_pred near 1, high within_acc")
    emit("with LOW cross_acc and LOW template stability, and g0^2/fSf of the same order.")
    emit("Nonlinear early death (small alpha*) for G-sets with topJbar persisting would")
    emit("instead say the variance was a small-signal artifact. High flip_frac says the")
    emit("conditionality lives in discrete routing, below the Jacobian's resolution.")
    (out / "patch_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {out / 'patch_report.txt'}, response_curves.csv, per_direction.csv, "
          f"decoding.csv" + (", routing.csv" if int((fl >= 0).sum()) > 0 else ""))


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="CPU: build direction sets from the harvest")
    s.add_argument("--harvest", required=True, help="dir containing second_moment.pt")
    s.add_argument("--out", required=True, help="output directions.pt path")
    wsm.add_lens_args(s)
    s.add_argument("--layers", nargs="*", default=None,
                   help="default: four layers spread across the depth range")
    s.add_argument("--k-top", type=int, default=8)
    s.add_argument("--k-orth", type=int, default=4)
    s.add_argument("--n-pairs", type=int, default=6)
    s.add_argument("--n-rand", type=int, default=8)
    s.add_argument("--pair-energy-tol", type=float, default=1.6,
                   help="max f'Sf ratio inside a matched pair")
    s.add_argument("--pair-gamma-ratio", type=float, default=4.0,
                   help="min gamma_hi/gamma_lo inside a matched pair")
    s.add_argument("--min-residual", type=float, default=0.5,
                   help="min orthogonal-residual norm for gnotj directions")
    s.add_argument("--r-sets", default="auto", choices=["auto", "off"])
    s.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("patch", help="GPU: run the finite-size patch grid")
    t.add_argument("--directions", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default=None, help="default: from directions.pt")
    t.add_argument("--attn-implementation", default=None)
    t.add_argument("--n-prompts", type=int, default=30)
    t.add_argument("--prompt-offset", type=int, default=None,
                   help="default: first prompt after the harvest's selection set")
    t.add_argument("--min-chars", type=int, default=0)
    t.add_argument("--positions-per-prompt", type=int, default=2)
    t.add_argument("--layers", nargs="*", default=None, help="subset of the selected layers")
    t.add_argument("--sets", nargs="*", default=None, help="subset of direction sets")
    t.add_argument("--alphas", nargs="*", default=["0.01", "0.05", "0.2", "0.5", "1.0"],
                   help="patch scale as a fraction of the local residual norm")
    t.add_argument("--patch-batch", type=int, default=8,
                   help="rows per forward, including the zero row")
    t.add_argument("--topk-logits", type=int, default=32)
    t.add_argument("--sketch-dim", type=int, default=256)
    t.add_argument("--sketch-dim-h", type=int, default=128)
    t.add_argument("--log-routing", default="auto", choices=["auto", "on", "off"])
    t.add_argument("--checkpoint-every", type=int, default=3)
    t.add_argument("--no-resume", dest="resume", action="store_false")
    t.add_argument("--smoke", action="store_true",
                   help="one (prompt, layer, position) + hook/linearity/JVP checks")
    t.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="CPU: curves, conditionality, decoding, report")
    a.add_argument("--out", required=True, help="dir containing patches.pt")

    args = p.parse_args()
    torch.set_grad_enabled(False)
    if args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "patch":
        cmd_patch(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
