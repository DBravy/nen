#!/usr/bin/env python3
"""
workspace_second_moment.py

Two-arm second-moment analysis for the camilablank/workspace-lenses J-lens and
R-lens pairs (default model: Qwen/Qwen3.5-9B). For each arm a in {J, R} this
builds, on the model AS IT RUNS ON THIS MACHINE,

    S_a(l) = E[ mean_p  M_p(a,l)^T M_p(a,l) ]          (total gain)
    G_a(l) = S_a(l) - Mbar_a(l)^T Mbar_a(l)            (gated gain)

where M_p(J,l) is the future-summed per-position Jacobian from layer l to the
target layer, and M_p(R,l) is the SAME quantity read through the RelP backward
graph (LN-rule, identity-rule, half-rule stop-gradients; forward values are
unchanged, only the backward differs). Mbar_a is the downloaded lens for that
arm, or a local refit made with the `fit` subcommand. Everything (probes,
positions, prompts) follows the workspace-lenses recipe read from the lens
provenance: target = penultimate block, skip_first = 4, n = 25 prompts from
NeelNanda/pile-10k, so the subtraction that defines G is done with the lens's
own weighting.

The two arms share identical probes on identical forward inputs, so the harvest
also accumulates the cross second moment

    C(l) = E[ mean_p  R_p(l)^T J_p(l) ]

from which analyze computes, per direction f, the correlation between the
context fluctuations of gradient transport (J f) and relevance transport (R f).
That correlation is the direct test of the R-lens mechanism: if the LRP
stop-gradients remove context-erratic terms, R's gated fraction should sit
below J's at early layers, and the discarded J-variance should be the part
that does NOT correlate with R.

Never mix arms inside a subtraction: G_J pairs S_J with the J-lens, G_R pairs
S_R with the R-lens. The per-arm shared-probe consistency check verifies each
pairing; if it fails for an arm, refit that arm locally with `fit` and pass
the refit to analyze via --lens-path-j / --lens-path-r.

Subcommands
  harvest  (GPU)   both arms per prompt; checkpointed and resumable.
                   --smoke runs one prompt and additionally checks
                   (a) forward equality of the RelP-patched model,
                   (b) linearity of the patched backward in the cotangent,
                   (c) per-arm consistency cosines, (d) memory and a time
                   estimate for the full run.
  analyze  (CPU)   per-arm consistency, decomposition (gated fraction,
                   neg_mass), eigenbank export (S_bank_J, G_bank_J, S_bank_R,
                   G_bank_R, scanner LXX.npz format), gatedness of an existing
                   direction bank, and the cross-arm section: Jbar-vs-Rbar
                   alignment, subspace overlaps (incl. Rbar-in-G_J), and the
                   C-based gated correlation.
  fit      (GPU)   local refit of one arm's lens (one-hot cotangents, exact
                   rows) under the same recipe; only needed if that arm's
                   consistency check fails. Output loads anywhere a downloaded
                   lens does.
  compare  (CPU)   local refit vs downloaded lens for one arm: Frobenius
                   cosine, top-k right-singular subspace overlap, named axes.

Usage
  python workspace_second_moment.py harvest --out sm9b --smoke
  python workspace_second_moment.py harvest --out sm9b --probes 64 --probe-batch 2
  python workspace_second_moment.py analyze --out sm9b --k 64
  python workspace_second_moment.py fit --arm r --out local_r.pt        # only if needed
  python workspace_second_moment.py analyze --out sm9b --lens-path-r local_r.pt

Cost and memory (L4 24 GB, Qwen3.5-9B bf16 about 18.5 GB on device):
  harvest: per prompt, 2 forwards + 2 * ceil(probes/probe_batch) backwards.
  At --probes 64 --probe-batch 2 that is 64 backward passes per prompt; with
  25 prompts expect roughly 1.5-3 h total. probe_batch 2 is the safe default;
  if smoke shows headroom, 4 halves the passes. Host RAM: the gradient stash
  is probes*n_valid*d_model*2B per layer per arm (about 3.6 GB total at the
  defaults) plus the running accumulators (about 12.5 GB at float64 over ~30
  layers, 3 matrices; --acc-dtype float32 halves this).
  fit: ceil(d_model/dim_batch) backwards per prompt; at dim_batch 2 that is
  2048 passes per prompt, 25 prompts, so an overnight job per arm. Only run
  it for an arm whose consistency check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_LENS_REPO = "camilablank/workspace-lenses"
DEFAULT_LENS_FILE_J = "qwen3.5-9b/j-lens/lens.pt"
DEFAULT_LENS_FILE_R = "qwen3.5-9b/r-lens/lens.pt"
DEFAULT_DATASET = "NeelNanda/pile-10k"

ARM_NAMES = {"j": "J", "r": "R"}


# =============================================================================
# lens loading (workspace-lenses dict format, also produced by `fit`)
# =============================================================================


class Lens:
    def __init__(self, jac, target_layer, skip_first, n_prompts, d_model,
                 dataset_id, max_seq_len, provenance, path, anchor_err):
        self.jac = jac                      # {layer: fp32 [d_out, d_in]}, target row removed
        self.source_layers = sorted(jac)
        self.target_layer = target_layer
        self.skip_first = skip_first
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.dataset_id = dataset_id
        self.max_seq_len = max_seq_len
        self.provenance = provenance
        self.path = path
        self.anchor_err = anchor_err


def _prov_get(prov: dict, *keys, default=None):
    for k in keys:
        if k in prov and prov[k] is not None:
            return prov[k]
    return default


def load_lens(path: str) -> Lens:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(blob, dict) and "J" in blob and "source_layers" in blob):
        raise SystemExit(f"{path}: expected a dict with keys 'J' and 'source_layers' "
                         f"(workspace-lenses format); got {type(blob)} with "
                         f"keys {list(blob) if isinstance(blob, dict) else '?'}")
    J = blob["J"]
    layers = [int(x) for x in blob["source_layers"]]
    if len(J) != len(layers):
        raise SystemExit(f"{path}: len(J)={len(J)} != len(source_layers)={len(layers)}")
    prov = blob.get("provenance", {}) or {}
    if not isinstance(prov, dict):
        prov = {"raw": str(prov)}
    jac = {l: J[i].float() for i, l in enumerate(layers)}
    d = int(_prov_get(prov, "d_model", default=blob.get("d_model", next(iter(jac.values())).shape[0])))

    target = _prov_get(prov, "target_layer", "target")
    anchor_err = None
    if target is None:
        # Detect the identity anchor row instead.
        last = max(jac)
        err = float((jac[last] - torch.eye(d)).norm() / d ** 0.5)
        if err < 1e-2:
            target = last
    if target is not None:
        target = int(target)
        if target in jac:
            anchor_err = float((jac[target] - torch.eye(d)).norm() / d ** 0.5)
            del jac[target]

    return Lens(
        jac=jac,
        target_layer=target,
        skip_first=_prov_get(prov, "skip_first", "skip_first_n_positions"),
        n_prompts=int(_prov_get(prov, "n_prompts", default=blob.get("n_prompts", 0)) or 0) or None,
        d_model=d,
        dataset_id=_prov_get(prov, "dataset_id", "dataset"),
        max_seq_len=_prov_get(prov, "max_seq_len", "max_length", "max_seq_length"),
        provenance=prov,
        path=path,
        anchor_err=anchor_err,
    )


def resolve_lens_pair(args, arms: str):
    """Download or open the lens for each requested arm; print provenance."""
    out = {}
    for arm in arms:
        path = getattr(args, f"lens_path_{arm}", None)
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(args.lens_repo, getattr(args, f"lens_file_{arm}"))
        lens = load_lens(path)
        out[arm] = lens
        cfg = lens.provenance.get("config_json")
        print(f"[lens-{ARM_NAMES[arm]}] {path}")
        print(f"         d_model={lens.d_model} target_layer={lens.target_layer} "
              f"skip_first={lens.skip_first} n_prompts={lens.n_prompts} "
              f"dataset={lens.dataset_id} layers={lens.source_layers[0]}..{lens.source_layers[-1]} "
              f"({len(lens.source_layers)})"
              + (f" anchor_identity_err={lens.anchor_err:.2e}" if lens.anchor_err is not None else ""))
        if cfg:
            print(f"         config_json: {str(cfg)[:400]}")
    if len(out) == 2:
        LJ, LR = out["j"], out["r"]
        for f in ("target_layer", "skip_first", "d_model"):
            if getattr(LJ, f) != getattr(LR, f):
                raise SystemExit(f"J and R lenses disagree on {f}: "
                                 f"{getattr(LJ, f)} vs {getattr(LR, f)}; they are not a matched pair")
        if LJ.source_layers != LR.source_layers:
            print("[warn] J and R lenses fit different layer sets; using the intersection")
    return out


# =============================================================================
# RelP backward rules (LN-rule, identity-rule, half-rule)
# =============================================================================

STREAM_NORM_SUFFIXES = ("input_layernorm", "post_attention_layernorm",
                        "pre_feedforward_layernorm", "post_feedforward_layernorm")


def _relp_rmsnorm_forward(mod):
    """RMSNorm with the normalization factor detached (LN-rule).

    Mirrors the transformers Qwen-family RMSNorm forward exactly (fp32 upcast,
    weight applied after downcast) so forward values are unchanged.
    """
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))

    def forward(hidden_states):
        dt = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + eps).detach()          # LN-rule
        return mod.weight * h.to(dt)

    return forward


def _relp_swiglu_forward(mod):
    """Gated MLP with identity-rule on SiLU and half-rule on the product.

    forward value: down( SiLU(gate(x)) * up(x) ), unchanged.
    backward: dSiLU becomes sigmoid(z) (detached), and the multiplicative gate
    routes half the cotangent through each branch instead of the product rule.
    """

    def forward(x):
        z = mod.gate_proj(x)
        u = mod.up_proj(x)
        a = z * torch.sigmoid(z).detach()                # identity-rule
        p = 0.5 * (a * u.detach() + a.detach() * u)      # half-rule
        return mod.down_proj(p)

    return forward


class relp_rules:
    """Context manager that installs the RelP rules on a HF model in place.

    Patches only residual-stream RMSNorms (input_layernorm and
    post_attention_layernorm; q_norm/k_norm and the final norm are left alone)
    and gated MLPs (modules with gate_proj/up_proj/down_proj). Attention and
    plain linear layers are untouched, matching the dense-model 'full RelP'
    arm of the workspace-lenses card. Restores original forwards on exit.
    """

    def __init__(self, hf_model, verbose=False):
        self.model = hf_model
        self.verbose = verbose
        self.patched = []
        self.counts = (0, 0)

    def __enter__(self):
        n_norm = n_mlp = 0
        for name, mod in self.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            cls = type(mod).__name__
            if "RMSNorm" in cls and leaf in STREAM_NORM_SUFFIXES:
                if "Gemma" in cls:
                    raise SystemExit("Gemma-style (1+w) RMSNorm detected; this patcher "
                                     "implements the Qwen-family norm only")
                self._patch(mod, _relp_rmsnorm_forward(mod))
                n_norm += 1
            elif all(hasattr(mod, a) for a in ("gate_proj", "up_proj", "down_proj")):
                act = type(getattr(mod, "act_fn", None)).__name__
                if "SiLU" not in act:
                    print(f"[relp][warn] {name}: act_fn={act}, identity-rule assumes SiLU")
                self._patch(mod, _relp_swiglu_forward(mod))
                n_mlp += 1
        if n_norm == 0 or n_mlp == 0:
            raise SystemExit(f"RelP patcher matched norms={n_norm} mlps={n_mlp}; "
                             "module naming differs from the Qwen family, patcher needs adapting")
        self.counts = (n_norm, n_mlp)
        if self.verbose:
            print(f"[relp] patched {n_norm} residual-stream norms, {n_mlp} gated MLPs")
        return self

    def _patch(self, mod, fn):
        self.patched.append(mod)
        mod.forward = fn                                  # instance attr shadows class method

    def __exit__(self, *exc):
        for mod in self.patched:
            if "forward" in mod.__dict__:
                del mod.__dict__["forward"]
        self.patched.clear()
        return False


# =============================================================================
# model plumbing
# =============================================================================


class Recorder:
    """Forward hooks capturing block outputs (residual stream after block l)."""

    def __init__(self, blocks, idxs):
        self.blocks = blocks
        self.idxs = sorted(set(int(i) for i in idxs))
        self.acts = {}
        self.handles = []

    def __enter__(self):
        for i in self.idxs:
            self.handles.append(self.blocks[i].register_forward_hook(self._mk(i)))
        return self

    def _mk(self, i):
        def hook(_mod, _inp, out):
            self.acts[i] = out[0] if isinstance(out, (tuple, list)) else out
        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def load_model(model_id: str, attn_implementation: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(torch_dtype="auto", device_map="auto")
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    hf = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    hf.eval()
    base = hf
    for _ in range(3):
        if hasattr(base, "layers"):
            break
        if hasattr(base, "model"):
            base = base.model
        else:
            raise SystemExit("could not locate the decoder block list (expected model.model.layers)")
    blocks = base.layers
    qc = getattr(hf.config, "quantization_config", None)
    print(f"[model] {model_id}: n_layers={len(blocks)} d_model={hf.config.hidden_size} "
          f"dtype={next(hf.parameters()).dtype} "
          f"quantization={'none' if qc is None else type(qc).__name__} "
          f"device={next(hf.parameters()).device}")
    return hf, tok, base, blocks


def load_prompts(n: int, offset: int = 0, min_chars: int = 0,
                 dataset_id: str = DEFAULT_DATASET) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    texts, kept = [], 0
    for rec in ds:
        t = rec["text"]
        if len(t) < min_chars:
            continue
        if kept >= offset:
            texts.append(t)
        kept += 1
        if len(texts) == n:
            break
    if len(texts) < n:
        raise SystemExit(f"only {len(texts)} prompts matched (n={n}, min_chars={min_chars})")
    return texts


def encode(tok, prompt: str, max_seq_len: int, device):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
    return ids.to(device)


def valid_positions(seq_len: int, skip_first: int) -> torch.Tensor:
    """Positions skip_first .. seq_len-2 inclusive (first skip_first and the last excluded)."""
    if seq_len < skip_first + 2:
        raise ValueError(f"seq_len={seq_len} too short for skip_first={skip_first}")
    return torch.arange(skip_first, seq_len - 1)


def make_probes(m: int, m_shared: int, d: int, seed: int, prompt_idx: int):
    g_shared = torch.Generator(device="cpu").manual_seed(seed)
    shared = torch.randn(m_shared, d, generator=g_shared)
    g_local = torch.Generator(device="cpu").manual_seed(seed * 100003 + prompt_idx + 1)
    fresh = torch.randn(m - m_shared, d, generator=g_local) if m > m_shared else torch.zeros(0, d)
    return torch.cat([shared, fresh], dim=0), shared


# =============================================================================
# harvest
# =============================================================================


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def arm_pass(arm, hf, base, blocks, source_layers, target_layer, ids, valid,
             probes, B, m_shared, stash_dtype=torch.bfloat16):
    """One arm on one prompt: forward + ceil(m/B) backwards.

    Returns (stash, shared_mean): stash[l] is a CPU [m, n_valid, d] tensor of
    per-probe per-position gradients M_p^T r; shared_mean[l] is the
    position-mean gradient for each shared probe, fp32 CPU.
    """
    m, d = probes.shape
    K = math.ceil(m / B)
    nv = int(valid.numel())
    stash = {l: torch.empty(m, nv, d, dtype=stash_dtype) for l in source_layers}
    shared_mean = {l: torch.zeros(m_shared, d) for l in source_layers}

    ctx = relp_rules(hf) if arm == "r" else contextlib.nullcontext()
    with Recorder(blocks, [*source_layers, target_layer]) as rec, torch.enable_grad(), ctx:
        base(input_ids=ids.expand(B, -1), use_cache=False)
        tact = rec.acts[target_layer]
        sacts = [rec.acts[l] for l in source_layers]
        vdev = valid.to(tact.device)
        cot = torch.zeros_like(tact)
        for k in range(K):
            p0, p1 = k * B, min((k + 1) * B, m)
            nb = p1 - p0
            cot.zero_()
            r = probes[p0:p1].to(device=tact.device, dtype=tact.dtype)
            cot[:nb, vdev, :] = r[:, None, :]
            grads = torch.autograd.grad(
                outputs=tact, inputs=sacts, grad_outputs=cot, retain_graph=(k < K - 1)
            )
            for l, g in zip(source_layers, grads, strict=True):
                gp = g[:nb, vdev, :]                     # [nb, nv, d] = M_p^T r_b
                if not torch.isfinite(gp).all():
                    raise SystemExit(f"non-finite gradient (arm={arm}, layer {l})")
                stash[l][p0:p1] = gp.to("cpu", stash_dtype)
                for b in range(nb):
                    gi = p0 + b
                    if gi < m_shared:
                        shared_mean[l][gi] = gp[b].float().mean(dim=0).cpu()
            del grads
    return stash, shared_mean


def fold_prompt(state, source_layers, stash_j, stash_r, cross, dev, acc_dtype):
    """Add this prompt's (1/(m*nv)) * flat^T flat contributions to the running sums."""
    any_stash = stash_j if stash_j is not None else stash_r
    m, nv, d = next(iter(any_stash.values())).shape
    scale = 1.0 / (m * nv)
    for l in source_layers:
        fj = fr = None
        if stash_j is not None:
            fj = stash_j[l].reshape(m * nv, d).to(dev, torch.float32)
            state["S"]["j"][l] += (fj.T @ fj).mul_(scale).to("cpu", acc_dtype)
        if stash_r is not None:
            fr = stash_r[l].reshape(m * nv, d).to(dev, torch.float32)
            state["S"]["r"][l] += (fr.T @ fr).mul_(scale).to("cpu", acc_dtype)
        if cross and fj is not None and fr is not None:
            state["C"][l] += (fr.T @ fj).mul_(scale).to("cpu", acc_dtype)
        del fj, fr
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def consistency_line(shared_mean, lenses, shared_probes, check_layers, arms):
    msgs = []
    r0 = shared_probes[0]
    for l in check_layers:
        for arm in arms:
            if l not in shared_mean[arm]:
                continue
            ours = shared_mean[arm][l][0]
            ref = lenses[arm].jac[l].T @ r0
            c = float(ours @ ref / (ours.norm() * ref.norm() + 1e-12))
            msgs.append(f"{ARM_NAMES[arm]}L{l:02d} cos={c:+.3f}")
    return " ".join(msgs)


def cmd_harvest(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "second_moment_ckpt.pt"
    arms = list(args.arms)
    cross = ("j" in arms and "r" in arms and not args.no_cross)
    acc_dtype = getattr(torch, args.acc_dtype)

    print("[1/4] Loading lenses (conventions + consistency reference)...")
    lenses = resolve_lens_pair(args, args.arms)
    L0 = lenses[arms[0]]
    d_model = L0.d_model
    skip_first = args.skip_first if args.skip_first is not None else (L0.skip_first if L0.skip_first is not None else 4)
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(L0.max_seq_len or 128)
    dataset_id = args.dataset or L0.dataset_id or DEFAULT_DATASET
    n_prompts = args.n_prompts if args.n_prompts is not None else (L0.n_prompts or 25)

    print("[2/4] Loading model...")
    hf, tok, base, blocks = load_model(args.model, args.attn_implementation)
    if hf.config.hidden_size != d_model:
        raise SystemExit(f"model hidden_size={hf.config.hidden_size} != lens d_model={d_model}")
    n_layers = len(blocks)
    target_layer = args.target_layer if args.target_layer is not None else \
        (L0.target_layer if L0.target_layer is not None else n_layers - 2)
    src_sets = [set(lens.jac) for lens in lenses.values()]
    source_layers = sorted(set.intersection(*src_sets))
    source_layers = [l for l in source_layers if l < target_layer]
    if args.layers:
        keep = {int(x) for x in args.layers}
        source_layers = [l for l in source_layers if l in keep]
    if not source_layers:
        raise SystemExit("no source layers to harvest")
    check_layers = [l for l in (args.check_layers or []) if l in source_layers] or source_layers[:1]
    print(f"[recipe] target_layer={target_layer} skip_first={skip_first} max_seq_len={max_seq_len} "
          f"dataset={dataset_id} n_prompts={n_prompts} arms={'+'.join(ARM_NAMES[a] for a in arms)} "
          f"cross={cross} sources={source_layers[0]}..{source_layers[-1]} ({len(source_layers)})")

    print("[3/4] Loading prompts...")
    prompts = load_prompts(n_prompts, args.prompt_offset, args.min_chars, dataset_id)
    print(f"[prompts] {len(prompts)} from {dataset_id} (offset {args.prompt_offset}, min_chars {args.min_chars})")

    m = int(args.probes)
    B = int(args.probe_batch)
    m_shared = min(int(args.shared_probes), m)
    _, shared_probes = make_probes(m, m_shared, d_model, args.seed, 0)
    dev = next(hf.parameters()).device
    fold_dev = dev if dev.type == "cuda" else torch.device("cpu")

    config = {
        "model": args.model, "target_layer": int(target_layer), "source_layers": source_layers,
        "max_seq_len": int(max_seq_len), "skip_first": int(skip_first),
        "probes": m, "probe_batch": B, "shared_probes": m_shared, "seed": int(args.seed),
        "prompt_offset": int(args.prompt_offset), "n_prompts": int(n_prompts),
        "min_chars": int(args.min_chars), "dataset": dataset_id, "d_model": int(d_model),
        "arms": args.arms, "cross": bool(cross),
        "lens_path_j": lenses["j"].path if "j" in lenses else None,
        "lens_path_r": lenses["r"].path if "r" in lenses else None,
    }

    def fresh_state():
        return {
            "S": {a: {l: torch.zeros(d_model, d_model, dtype=acc_dtype) for l in source_layers} for a in arms},
            "C": ({l: torch.zeros(d_model, d_model, dtype=acc_dtype) for l in source_layers} if cross else {}),
            "shared_means": {a: {l: [] for l in source_layers} for a in arms},
            "n_done": 0, "next_idx": 0, "n_valid": [], "seq_lens": [], "config": config,
        }

    state = None
    if args.resume and ckpt_path.exists() and not args.smoke:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise SystemExit(f"checkpoint config differs from current args; delete {ckpt_path} or match args")
        for a in arms:
            state["S"][a] = {l: S.to(acc_dtype) for l, S in state["S"][a].items()}
        state["C"] = {l: Cm.to(acc_dtype) for l, Cm in state["C"].items()}
        print(f"[resume] {state['n_done']} prompts done, resuming at index {state['next_idx']}")
    if state is None:
        state = fresh_state()

    # ---- smoke extras ---------------------------------------------------------
    if args.smoke:
        smoke_checks(hf, base, blocks, tok, prompts[0], source_layers, target_layer,
                     max_seq_len, skip_first, dev)

    print("[4/4] Harvesting probe gradients (both arms share probes and prompts)...")
    t0 = time.time()
    n_todo = 1 if args.smoke else len(prompts)
    m_eff = B if args.smoke else m

    for idx in range(state["next_idx"], n_todo):
        prompt = prompts[idx]
        ids = encode(tok, prompt, max_seq_len, dev)
        seq_len = int(ids.shape[1])
        try:
            valid = valid_positions(seq_len, skip_first)
        except ValueError as exc:
            print(f"[skip] prompt {idx}: {exc}")
            state["next_idx"] = idx + 1
            continue
        nv = int(valid.numel())
        probes, _ = make_probes(m_eff, min(m_shared, m_eff), d_model, args.seed, idx)

        t_prompt = time.time()
        stash, shared = {}, {}
        for arm in arms:
            stash[arm], shared[arm] = arm_pass(
                arm, hf, base, blocks, source_layers, target_layer, ids, valid,
                probes, B, min(m_shared, m_eff))
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        fold_prompt(state, source_layers, stash.get("j"), stash.get("r"), cross, fold_dev, acc_dtype)
        for arm in arms:
            for l in source_layers:
                state["shared_means"][arm][l].append(shared[arm][l])
        del stash
        state["n_valid"].append(nv)
        state["seq_lens"].append(seq_len)
        state["n_done"] += 1
        state["next_idx"] = idx + 1
        dt = time.time() - t_prompt

        line = consistency_line(shared, lenses, shared_probes, check_layers, arms)
        print(f"[check] {line} (single prompt vs {L0.n_prompts or '?'}-prompt lens; "
              f"expect well above 0, below 1)")
        mem = f" peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if dev.type == "cuda" else ""
        print(f"[harvest] prompt {idx + 1}/{n_todo} seq={seq_len} valid={nv} probes={m_eff} "
              f"backwards={len(arms) * math.ceil(m_eff / B)} {dt:.1f}s{mem} "
              f"elapsed={time.time() - t0:.0f}s")

        if args.smoke:
            per_bwd = dt / (len(arms) * math.ceil(m_eff / B))
            full = len(prompts) * (len(arms) * math.ceil(m / B) * per_bwd + 15)
            print(f"\n[smoke] one prompt succeeded. Estimated full run at --probes {m}: "
                  f"~{full / 3600:.1f} h ({per_bwd:.1f}s per backward pass)")
            return
        if (idx + 1) % args.checkpoint_every == 0 or idx + 1 == n_todo:
            save = dict(state)
            save["S"] = {a: {l: S.float() for l, S in state["S"][a].items()} for a in arms}
            save["C"] = {l: Cm.float() for l, Cm in state["C"].items()}
            _atomic_save(save, ckpt_path)
            print(f"[checkpoint] {ckpt_path} ({state['n_done']} prompts)")

    # ---- finalize -------------------------------------------------------------
    n = state["n_done"]
    final = {
        "S": {a: {l: (state["S"][a][l] / n).float() for l in source_layers} for a in arms},
        "C": {l: (state["C"][l] / n).float() for l in state["C"]},
        "shared_means": {a: {l: torch.stack(state["shared_means"][a][l]) for l in source_layers} for a in arms},
        "shared_probes": shared_probes,
        "n_prompts": n, "n_valid": state["n_valid"], "seq_lens": state["seq_lens"],
        "config": config,
    }
    _atomic_save(final, out / "second_moment.pt")
    if args.npz:
        npz = {"shared_probes": shared_probes.numpy(), "n_prompts": np.asarray(n),
               "config_json": np.asarray(json.dumps(config))}
        for a in arms:
            for l in source_layers:
                npz[f"S{a.upper()}_L{l:02d}"] = final["S"][a][l].numpy()
                npz[f"shared_means_{a}_L{l:02d}"] = final["shared_means"][a][l].numpy()
        for l in final["C"]:
            npz[f"C_L{l:02d}"] = final["C"][l].numpy()
        np.savez_compressed(out / "second_moment.npz", **npz)
    print(f"\n[done] {n} prompts -> {out / 'second_moment.pt'}")
    print(f"Next: python {Path(__file__).name} analyze --out {out} --k 64")


def smoke_checks(hf, base, blocks, tok, prompt, source_layers, target_layer,
                 max_seq_len, skip_first, dev):
    """Forward equality of the RelP patch, and cotangent linearity of its backward."""
    print("[smoke] RelP forward-equality and backward-linearity checks...")
    ids = encode(tok, prompt, max_seq_len, dev)
    probe_layers = [source_layers[0], source_layers[len(source_layers) // 2], target_layer]

    with torch.no_grad(), Recorder(blocks, probe_layers) as rec:
        base(input_ids=ids, use_cache=False)
        ref = {l: rec.acts[l].float().cpu() for l in probe_layers}
    with torch.no_grad(), relp_rules(hf, verbose=True), Recorder(blocks, probe_layers) as rec:
        base(input_ids=ids, use_cache=False)
        alt = {l: rec.acts[l].float().cpu() for l in probe_layers}
    for l in probe_layers:
        diff = (ref[l] - alt[l]).abs().max().item()
        rel = diff / (ref[l].abs().max().item() + 1e-12)
        print(f"[smoke] forward equality L{l:02d}: max|vanilla-relp|={diff:.3e} (rel {rel:.2e}) "
              f"<- should be ~0 (kernel-rounding level)")

    valid = valid_positions(int(ids.shape[1]), skip_first)
    lin_layers = [source_layers[0], source_layers[-1]]
    with Recorder(blocks, [*lin_layers, target_layer]) as rec, torch.enable_grad(), relp_rules(hf):
        base(input_ids=ids, use_cache=False)
        tact = rec.acts[target_layer]
        sacts = [rec.acts[l] for l in lin_layers]
        vdev = valid.to(tact.device)
        d = tact.shape[-1]
        g = torch.Generator(device="cpu").manual_seed(1234)
        r1 = torch.randn(d, generator=g).to(device=tact.device, dtype=tact.dtype)
        r2 = torch.randn(d, generator=g).to(device=tact.device, dtype=tact.dtype)
        outs = []
        for r in (r1, r2, r1 + r2):
            cot = torch.zeros_like(tact)
            cot[:, vdev, :] = r
            outs.append(torch.autograd.grad(tact, sacts, grad_outputs=cot, retain_graph=True))
        for i, l in enumerate(lin_layers):
            g12 = outs[2][i].float()
            gsum = (outs[0][i] + outs[1][i]).float()
            rel = ((g12 - gsum).norm() / (g12.norm() + 1e-12)).item()
            cos = (g12.flatten() @ gsum.flatten() / (g12.norm() * gsum.norm() + 1e-30)).item()
            print(f"[smoke] backward linearity L{l:02d}: rel_err={rel:.2e} cos={cos:.6f} "
                  f"<- rel_err small and cos~1 means the patched backward is a fixed linear map")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# fit (local refit of one arm, exact one-hot rows)
# =============================================================================


def cmd_fit(args: argparse.Namespace) -> None:
    arm = args.arm
    print(f"[fit] local {ARM_NAMES[arm]}-lens refit")
    lenses = resolve_lens_pair(args, arm)          # for the recipe only
    L0 = lenses[arm]
    d_model = L0.d_model
    skip_first = args.skip_first if args.skip_first is not None else (L0.skip_first if L0.skip_first is not None else 4)
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(L0.max_seq_len or 128)
    dataset_id = args.dataset or L0.dataset_id or DEFAULT_DATASET
    n_prompts = args.n_prompts if args.n_prompts is not None else (L0.n_prompts or 25)

    hf, tok, base, blocks = load_model(args.model, args.attn_implementation)
    if hf.config.hidden_size != d_model:
        raise SystemExit(f"model hidden_size={hf.config.hidden_size} != lens d_model={d_model}")
    target_layer = args.target_layer if args.target_layer is not None else \
        (L0.target_layer if L0.target_layer is not None else len(blocks) - 2)
    source_layers = sorted(l for l in L0.jac if l < target_layer)
    dev = next(hf.parameters()).device
    prompts = load_prompts(n_prompts, args.prompt_offset, args.min_chars, dataset_id)
    B = int(args.dim_batch)
    K = math.ceil(d_model / B)
    print(f"[fit] {len(prompts)} prompts x {K} backward passes (dim_batch={B}), "
          f"target={target_layer} skip_first={skip_first} sources={len(source_layers)}")

    ckpt_path = Path(args.checkpoint or (str(Path(args.out).with_suffix("")) + "_ckpt.pt"))
    config = {"model": args.model, "arm": arm, "target_layer": int(target_layer),
              "skip_first": int(skip_first), "max_seq_len": int(max_seq_len),
              "dataset": dataset_id, "n_prompts": int(n_prompts),
              "prompt_offset": int(args.prompt_offset), "min_chars": int(args.min_chars),
              "dim_batch": B, "d_model": int(d_model), "source_layers": source_layers}
    state = None
    if not args.no_resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise SystemExit(f"checkpoint config differs; delete {ckpt_path} or match args")
        state["rows"] = {l: v.double() for l, v in state["rows"].items()}
        print(f"[resume] {state['n_done']} prompts done")
    if state is None:
        state = {"rows": {l: torch.zeros(d_model, d_model, dtype=torch.float64) for l in source_layers},
                 "n_done": 0, "next_idx": 0, "config": config}

    t0 = time.time()
    for idx in range(state["next_idx"], len(prompts)):
        ids = encode(tok, prompts[idx], max_seq_len, dev)
        seq_len = int(ids.shape[1])
        try:
            valid = valid_positions(seq_len, skip_first)
        except ValueError as exc:
            print(f"[skip] prompt {idx}: {exc}")
            state["next_idx"] = idx + 1
            continue
        rows_prompt = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}
        t_prompt = time.time()
        ctx = relp_rules(hf) if arm == "r" else contextlib.nullcontext()
        with Recorder(blocks, [*source_layers, target_layer]) as rec, torch.enable_grad(), ctx:
            base(input_ids=ids.expand(B, -1), use_cache=False)
            tact = rec.acts[target_layer]
            sacts = [rec.acts[l] for l in source_layers]
            vdev = valid.to(tact.device)
            cot = torch.zeros_like(tact)
            for k in range(K):
                d0, d1 = k * B, min((k + 1) * B, d_model)
                nb = d1 - d0
                cot.zero_()
                for b in range(nb):
                    cot[b, vdev, d0 + b] = 1.0
                grads = torch.autograd.grad(tact, sacts, grad_outputs=cot,
                                            retain_graph=(k < K - 1))
                for l, g in zip(source_layers, grads, strict=True):
                    gp = g[:nb, vdev, :].float().mean(dim=1)      # [nb, d] rows d0..d1-1 of M
                    if not torch.isfinite(gp).all():
                        raise SystemExit(f"non-finite gradient at layer {l}")
                    rows_prompt[l][d0:d1] = gp.cpu()
                del grads
        for l in source_layers:
            state["rows"][l] += rows_prompt[l].double()
        state["n_done"] += 1
        state["next_idx"] = idx + 1
        print(f"[fit] prompt {idx + 1}/{len(prompts)} seq={seq_len} {time.time() - t_prompt:.0f}s "
              f"elapsed={(time.time() - t0) / 60:.1f}m")
        if (idx + 1) % args.checkpoint_every == 0 or idx + 1 == len(prompts):
            save = dict(state)
            save["rows"] = {l: v.float() for l, v in state["rows"].items()}
            _atomic_save(save, ckpt_path)

    n = state["n_done"]
    J = torch.stack([(state["rows"][l] / n).float() for l in source_layers]
                    + [torch.eye(d_model)])
    blob = {"J": J, "source_layers": [*source_layers, int(target_layer)],
            "n_prompts": n, "d_model": int(d_model),
            "provenance": {**config, "fitted_locally": True,
                           "config_json": json.dumps(config)}}
    torch.save(blob, args.out)
    print(f"[fit] saved {args.out} (n_prompts={n}); pass it to analyze as --lens-path-{arm}")


# =============================================================================
# analyze
# =============================================================================


def haar_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((k, k)))
    d = np.sign(np.diagonal(r))
    d[d == 0] = 1.0
    return q * d


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared projection of A's (orthonormal) columns into span(B)."""
    P = B.T @ A
    return float((P ** 2).sum(axis=0).mean())


def load_harvest(out: Path):
    p = out / "second_moment.pt"
    if p.exists():
        blob = torch.load(p, map_location="cpu", weights_only=False)
        return blob, False
    ck = out / "second_moment_ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        n = state["n_done"]
        print(f"[analyze] final file missing; using checkpoint at {n} prompts")
        blob = {
            "S": {a: {l: (S / n).float() for l, S in d.items()} for a, d in state["S"].items()},
            "C": {l: (Cm / n).float() for l, Cm in state["C"].items()},
            "shared_means": {a: {l: torch.stack(v) for l, v in d.items() if v}
                             for a, d in state["shared_means"].items()},
            "shared_probes": None, "n_prompts": n, "config": state["config"],
        }
        m_sh = state["config"]["shared_probes"]
        _, blob["shared_probes"] = make_probes(m_sh, m_sh, state["config"]["d_model"],
                                               state["config"]["seed"], 0)
        return blob, True
    raise SystemExit(f"no harvest found under {out}")


def cmd_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    blob, _ = load_harvest(out)
    cfg = blob["config"]
    arms = list(cfg["arms"])
    rng = np.random.default_rng(args.seed)
    k = int(args.k)
    edt = np.float32 if args.eig_dtype == "float32" else np.float64

    # lens per arm: CLI path > harvest-recorded path
    for arm in arms:
        if getattr(args, f"lens_path_{arm}", None) is None and cfg.get(f"lens_path_{arm}"):
            setattr(args, f"lens_path_{arm}", cfg[f"lens_path_{arm}"])
    lenses = resolve_lens_pair(args, "".join(arms))

    layers = sorted(set(blob["S"][arms[0]]).intersection(*[set(blob["S"][a]) for a in arms]))
    layers = [l for l in layers if all(l in lenses[a].jac for a in arms)]
    if args.layers:
        keep = {int(x) for x in args.layers}
        layers = [l for l in layers if l in keep]
    if not layers:
        raise SystemExit("no layers shared by harvest and lenses")
    have_C = bool(blob.get("C")) and len(arms) == 2
    shared_probes = np.asarray(blob["shared_probes"].numpy())
    n_prompts = int(blob["n_prompts"])

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"Two-arm second-moment analysis: {cfg['model']}  prompts={n_prompts}  "
         f"probes/prompt={cfg['probes']}  arms={'+'.join(ARM_NAMES[a] for a in arms)}  "
         f"cross={have_C}  layers={layers[0]}..{layers[-1]} ({len(layers)})")
    emit("=" * 78)

    # ---- 1. consistency per arm ---------------------------------------------------
    emit()
    emit("Consistency (mean over prompts of probe gradient vs lens Mbar^T r, per shared probe).")
    emit("cos~1 and ratio~1 mean the lens describes this machine's backward for that arm;")
    emit("only then is that arm's G = S - Mbar^T Mbar meaningful.")
    emit(f"{'layer':>5}" + "".join(f" {ARM_NAMES[a] + '_cos':>7} {ARM_NAMES[a] + '_ratio':>8}" for a in arms))
    consistency = {a: {} for a in arms}
    for l in layers:
        row = f"{l:>5}"
        for a in arms:
            ours = blob["shared_means"][a][l].double().numpy().mean(axis=0)   # [m_shared, d]
            ref = shared_probes.astype(np.float64) @ lenses[a].jac[l].double().numpy()
            cs, rt = [], []
            for o, r_ in zip(ours, ref):
                no, nr = np.linalg.norm(o), np.linalg.norm(r_)
                cs.append(float(o @ r_ / (no * nr + 1e-12)))
                rt.append(no / (nr + 1e-12))
            consistency[a][l] = float(np.mean(cs))
            row += f" {np.mean(cs):>7.3f} {np.mean(rt):>8.3f}"
        emit(row)

    # ---- 2-5. one heavy pass per layer (keeps only that layer's matrices in RAM) --
    banks = {a: {"S": {}, "G": {}} for a in arms}
    dec_rows, own_rows, cross_rows = [], [], []
    bank_lines, bank_csv = [], []
    bank_dir = Path(args.bank) if args.bank else None
    t0 = time.time()
    for li, l in enumerate(layers):
        per_arm = {}
        for a in arms:
            S = blob["S"][a][l].double().numpy()
            S = 0.5 * (S + S.T)
            M = lenses[a].jac[l].double().numpy()
            MM = M.T @ M
            G = S - MM
            G = 0.5 * (G + G.T)
            wS, VS = np.linalg.eigh(S.astype(edt))
            wG, VG = np.linalg.eigh(G.astype(edt))
            wS, VS = wS[::-1].astype(np.float64), VS[:, ::-1]
            wG, VG = wG[::-1].astype(np.float64), VG[:, ::-1]
            _, _, Vt = np.linalg.svd(M.astype(edt), full_matrices=False)
            Vb = Vt[:k].T
            total, invariant = float(np.trace(S)), float(np.trace(MM))
            neg = float(-wG[wG < 0].sum())
            pos = float(wG[wG > 0].sum())
            gated = 1.0 - invariant / max(total, 1e-30)
            dec_rows.append({"layer": l, "arm": ARM_NAMES[a], "total_gain": total,
                             "invariant_gain": invariant, "gated_fraction": gated,
                             "neg_mass": neg / max(pos, 1e-30), "top_eig_S": float(wS[0]),
                             "top_eig_G": float(wG[0]),
                             "consistency_cos": consistency[a][l]})
            own_rows.append({"layer": l, "arm": ARM_NAMES[a],
                             "in_S": subspace_overlap(Vb, VS[:, :k]),
                             "in_G": subspace_overlap(Vb, VG[:, :k])})
            banks[a]["S"][l] = (VS[:, :k].astype(np.float32),
                                np.sqrt(np.clip(wS[:k], 0, None)).astype(np.float32))
            banks[a]["G"][l] = (VG[:, :k].astype(np.float32),
                                np.sqrt(np.clip(wG[:k], 0, None)).astype(np.float32))

            if bank_dir is not None and (bank_dir / f"L{l:02d}.npz").exists():
                z = np.load(bank_dir / f"L{l:02d}.npz")
                V = np.asarray(z["V"], dtype=np.float64)
                if V.shape[0] < V.shape[1]:
                    V = V.T
                V = V[:, :k]
                V = V / np.linalg.norm(V, axis=0, keepdims=True)

                def ratio(F):
                    tot = np.einsum("ij,jk,ki->i", F.T, S, F)
                    inv = np.sum((M @ F) ** 2, axis=0)
                    return (tot - inv) / np.maximum(inv, 1e-30)

                r_bank = ratio(V)
                r_rot = ratio(V @ haar_orthogonal(V.shape[1], rng))
                Rnd = rng.standard_normal((S.shape[0], V.shape[1]))
                Rnd /= np.linalg.norm(Rnd, axis=0, keepdims=True)
                r_rand = ratio(Rnd)
                bank_lines.append(
                    f"  L{l:02d} {ARM_NAMES[a]}: bank med={np.median(r_bank):.3f} "
                    f"(min {r_bank.min():.3f}, max {r_bank.max():.3f}) | "
                    f"rot med={np.median(r_rot):.3f} | rand med={np.median(r_rand):.3f}, "
                    f"95th={np.percentile(r_rand, 95):.3f}")
                for jdx in range(V.shape[1]):
                    bank_csv.append({"layer": l, "arm": ARM_NAMES[a], "axis_rank_1based": jdx + 1,
                                     "gatedness_ratio": float(r_bank[jdx])})

            per_arm[a] = {"M": M, "G": G, "VGk": VG[:, :k], "Vbk": Vb,
                          "trS": float(np.trace(S)), "trG": float(np.trace(G)),
                          "wG0": float(max(wG[0], 0.0))}
            del S, MM, G, VS, VG, Vt

        if len(arms) == 2:
            pj, pr = per_arm["j"], per_arm["r"]
            J_, R_ = pj["M"], pr["M"]
            mc = float((J_ * R_).sum() / (np.linalg.norm(J_) * np.linalg.norm(R_) + 1e-30))
            rec = {"layer": l, "mean_cos": mc,
                   "ov_JR": subspace_overlap(pj["Vbk"], pr["Vbk"]),
                   "R_in_GJ": subspace_overlap(pr["Vbk"], pj["VGk"]),
                   "J_in_GR": subspace_overlap(pj["Vbk"], pr["VGk"]),
                   "ov_GJ_GR": subspace_overlap(pj["VGk"], pr["VGk"])}
            if have_C and l in blob["C"]:
                C = blob["C"][l].double().numpy()
                Ccov = C - R_.T @ J_                        # cross-covariance of (J f, R f)
                rec["align_total"] = float(np.trace(C) / (np.sqrt(pj["trS"] * pr["trS"]) + 1e-30))
                rec["align_gated"] = float(np.trace(Ccov) /
                                           (np.sqrt(max(pj["trG"], 1e-30) * max(pr["trG"], 1e-30))))
                corrs = []
                tol_j = 1e-6 * max(pj["wG0"], 1e-30)
                tol_r = 1e-6 * max(pr["wG0"], 1e-30)
                for jdir in range(min(args.corr_dirs, pj["VGk"].shape[1])):
                    f = pj["VGk"][:, jdir].astype(np.float64)
                    vj = max(float(f @ pj["G"] @ f), 0.0)
                    vr = max(float(f @ pr["G"] @ f), 0.0)
                    if vj < tol_j:
                        continue                    # no J variance here: not a gated direction
                    if vr < tol_r:
                        corrs.append(0.0)           # J fluctuates, R does not: R discards it
                        continue
                    num = float(f @ Ccov @ f)
                    corrs.append(float(np.clip(num / math.sqrt(vj * vr), -1.0, 1.0)))
                corrs = np.array(corrs) if corrs else np.array([np.nan])
                rec.update({"gated_corr_median": float(np.median(corrs)),
                            "gated_corr_min": float(corrs.min()),
                            "gated_corr_max": float(corrs.max())})
                del C, Ccov
            cross_rows.append(rec)
        del per_arm
        print(f"[analyze] layer {l} done ({li + 1}/{len(layers)}, {time.time() - t0:.0f}s)")

    # ---- print section 2: per-arm decomposition -----------------------------------
    emit()
    emit("Per-arm decomposition: total = trace(S), invariant = trace(Mbar^T Mbar),")
    emit("gated_frac = 1 - invariant/total (context-dependent share of transport for that arm),")
    emit("neg_mass = |negative eigenvalue mass of G| / trace(G+), an estimator-mismatch symptom.")
    emit("Absolute totals are NOT comparable across arms (RelP rescales gradients); fractions are.")
    emit(f"{'layer':>5} {'arm':>4} {'total':>12} {'invariant':>12} {'gated_frac':>11} "
         f"{'neg_mass':>9} {'top eig S':>11} {'top eig G':>11}")
    for r in dec_rows:
        emit(f"{r['layer']:>5} {r['arm']:>4} {r['total_gain']:>12.4g} {r['invariant_gain']:>12.4g} "
             f"{r['gated_fraction']:>11.3f} {r['neg_mass']:>9.3f} {r['top_eig_S']:>11.4g} "
             f"{r['top_eig_G']:>11.4g}")

    # ---- print section 3: own-mean placement ---------------------------------------
    emit()
    emit(f"Top-{k} placement of each arm's own mean map: Mbar's top right singular vectors inside")
    emit("S's top eigenspace (should be high) and inside G's (low; G finding new directions).")
    emit(f"{'layer':>5}" + "".join(f" {ARM_NAMES[a] + ' in S':>8} {ARM_NAMES[a] + ' in G':>8}" for a in arms))
    for l in layers:
        row = f"{l:>5}"
        for a in arms:
            r = next(x for x in own_rows if x["layer"] == l and x["arm"] == ARM_NAMES[a])
            row += f" {r['in_S']:>8.3f} {r['in_G']:>8.3f}"
        emit(row)

    # ---- print section 4: cross-arm -------------------------------------------------
    if cross_rows:
        emit()
        emit("Cross-arm geometry (all scale-free; both maps read the same source space).")
        emit("  mean_cos    Frobenius cosine of Jbar vs Rbar (do the mean maps agree at all)")
        emit(f"  ov(J,R)     top-{k} right-singular subspace overlap of the two mean maps")
        emit("  R_in_GJ     Rbar's top directions inside G_J's top eigenspace: does the relevance")
        emit("              mean recover channels that were pure context-variance for the gradient")
        emit("  ov(GJ,GR)   do the two arms agree on WHICH directions are gated")
        hdr = f"{'layer':>5} {'mean_cos':>9} {'ov(J,R)':>8} {'R_in_GJ':>8} {'J_in_GR':>8} {'ov(GJ,GR)':>9}"
        if have_C:
            hdr += f" {'align_tot':>9} {'align_gated':>11} {'gcorr_med':>9} {'gcorr_rng':>15}"
        emit(hdr)
        for r in cross_rows:
            row = (f"{r['layer']:>5} {r['mean_cos']:>9.3f} {r['ov_JR']:>8.3f} {r['R_in_GJ']:>8.3f} "
                   f"{r['J_in_GR']:>8.3f} {r['ov_GJ_GR']:>9.3f}")
            if "align_total" in r:
                row += (f" {r['align_total']:>9.3f} {r['align_gated']:>11.3f} "
                        f"{r['gated_corr_median']:>9.3f} "
                        f"[{r['gated_corr_min']:>5.2f},{r['gated_corr_max']:>5.2f}]")
            emit(row)
        if have_C:
            emit()
            emit("gcorr = per-direction correlation of the two arms' context fluctuations along G_J's")
            emit(f"top {args.corr_dirs} eigendirections. Low gcorr where gated_frac_J >> gated_frac_R means")
            emit("the LRP rules discarded exactly that variance (J's noise), not shared gated signal.")

    # ---- print section 5: bank gatedness --------------------------------------------
    if bank_lines:
        emit()
        emit("Gatedness ratio (f^T S f - |Mbar f|^2) / |Mbar f|^2 for bank axes vs rotated/random nulls.")
        for s in bank_lines:
            emit(s)
        if bank_csv:
            import csv
            with (out / "bank_gatedness.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(bank_csv[0].keys()))
                w.writeheader()
                w.writerows(bank_csv)
            emit(f"  -> per-axis values in {out / 'bank_gatedness.csv'}")

    # ---- 6. export banks -------------------------------------------------------------
    for a in arms:
        for kind in ("S", "G"):
            bdir = out / f"{kind}_bank_{ARM_NAMES[a]}"
            bdir.mkdir(exist_ok=True)
            for l, (V, s) in banks[a][kind].items():
                np.savez_compressed(bdir / f"L{l:02d}.npz", V=V, S=s)
    emit()
    emit(f"Exported top-{k} banks: "
         + ", ".join(f"{kind}_bank_{ARM_NAMES[a]}" for a in arms for kind in ("S", "G"))
         + "  (scanner LXX.npz format; S values are sqrt(eigenvalue), i.e. RMS gain).")

    import csv
    with (out / "per_layer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dec_rows[0].keys()))
        w.writeheader()
        w.writerows(dec_rows)
    if cross_rows:
        keys = max((list(r.keys()) for r in cross_rows), key=len)
        with (out / "cross_layer.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(cross_rows)
    (out / "second_moment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {out / 'second_moment_report.txt'}, per_layer.csv"
          + (", cross_layer.csv" if cross_rows else ""))


# =============================================================================
# compare (local refit vs downloaded lens, one arm)
# =============================================================================


def cmd_compare(args: argparse.Namespace) -> None:
    local = load_lens(args.local)
    lenses = resolve_lens_pair(args, args.arm)
    down = lenses[args.arm]
    layers = sorted(set(local.jac) & set(down.jac))
    if args.layers:
        keep = {int(x) for x in args.layers}
        layers = [l for l in layers if l in keep]
    print(f"[compare] arm={ARM_NAMES[args.arm]} local={args.local} vs download={down.path}")
    print(f"{'layer':>5} {'frob_cos':>9} {'top8':>7} {'top32':>7} {'top64':>7}")
    svds = {}
    for l in layers:
        A = local.jac[l].double().numpy()
        Bm = down.jac[l].double().numpy()
        fc = float((A * Bm).sum() / (np.linalg.norm(A) * np.linalg.norm(Bm) + 1e-30))
        _, sa, Va = np.linalg.svd(A, full_matrices=False)
        _, sb, Vb = np.linalg.svd(Bm, full_matrices=False)
        svds[l] = (Va, sa, Vb, sb)
        ov = [subspace_overlap(Va[:kk].T, Vb[:kk].T) for kk in (8, 32, 64)]
        print(f"{l:>5} {fc:>9.3f} {ov[0]:>7.3f} {ov[1]:>7.3f} {ov[2]:>7.3f}")
    if args.axes:
        import re
        print(f"\n{'axis':>10} {'best local':>11} {'|cos|':>6} {'sigma_dl':>9} {'sigma_local':>12}")
        for name in args.axes:
            m = re.fullmatch(r"L(\d+)_SV(\d+)", name.strip())
            if not m:
                print(f"{name:>10}  bad axis name")
                continue
            l, r = int(m.group(1)), int(m.group(2))
            if l not in svds:
                print(f"{name:>10}  layer not in both lenses")
                continue
            Va, sa, Vb, sb = svds[l]
            v = Vb[r - 1]
            c = np.abs(Va @ v)
            j = int(np.argmax(c))
            print(f"{name:>10} {'SV%02d' % (j + 1):>11} {c[j]:>6.3f} {sb[r - 1]:>9.1f} {sa[j]:>12.1f}")


# =============================================================================
# CLI
# =============================================================================


def add_lens_args(p):
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-file-j", default=DEFAULT_LENS_FILE_J)
    p.add_argument("--lens-file-r", default=DEFAULT_LENS_FILE_R)
    p.add_argument("--lens-path-j", default=None, help="local J-lens .pt (skips hub download)")
    p.add_argument("--lens-path-r", default=None, help="local R-lens .pt (skips hub download)")


def add_recipe_args(p):
    p.add_argument("--n-prompts", type=int, default=None, help="default: lens provenance (25)")
    p.add_argument("--prompt-offset", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=0)
    p.add_argument("--dataset", default=None, help="default: lens provenance (NeelNanda/pile-10k)")
    p.add_argument("--max-seq-len", type=int, default=None, help="default: lens provenance, else 128")
    p.add_argument("--skip-first", type=int, default=None, help="default: lens provenance (4)")
    p.add_argument("--target-layer", type=int, default=None, help="default: lens provenance (n_layers-2)")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--seed", type=int, default=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="GPU: accumulate S_J, S_R, C via shared random probes")
    h.add_argument("--out", required=True)
    add_lens_args(h)
    add_recipe_args(h)
    h.add_argument("--arms", default="jr", choices=["jr", "j", "r"])
    h.add_argument("--probes", type=int, default=64)
    h.add_argument("--probe-batch", type=int, default=2)
    h.add_argument("--shared-probes", type=int, default=8)
    h.add_argument("--layers", nargs="*", default=None)
    h.add_argument("--check-layers", nargs="*", type=int, default=[4, 15, 25])
    h.add_argument("--no-cross", action="store_true", help="skip the cross moment C")
    h.add_argument("--acc-dtype", default="float64", choices=["float64", "float32"],
                   help="running accumulator dtype (float32 halves host RAM)")
    h.add_argument("--checkpoint-every", type=int, default=5)
    h.add_argument("--no-resume", dest="resume", action="store_false")
    h.add_argument("--npz", action="store_true", help="also export second_moment.npz")
    h.add_argument("--smoke", action="store_true", help="one prompt + patch/linearity checks, then exit")
    h.set_defaults(func=cmd_harvest)

    a = sub.add_parser("analyze", help="CPU: per-arm decomposition + cross-arm comparison")
    a.add_argument("--out", required=True)
    add_lens_args(a)
    a.add_argument("--k", type=int, default=64)
    a.add_argument("--bank", default=None, help="existing directions dir for gatedness ratios")
    a.add_argument("--layers", nargs="*", default=None)
    a.add_argument("--corr-dirs", type=int, default=8,
                   help="G_J eigendirections scored by the cross correlation")
    a.add_argument("--eig-dtype", default="float32", choices=["float32", "float64"],
                   help="dtype for eigh/svd; float32 is ~4x faster at d=4096, ample for banks")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("fit", help="GPU: local refit of one arm's lens (exact one-hot rows)")
    f.add_argument("--arm", required=True, choices=["j", "r"])
    f.add_argument("--out", required=True)
    add_lens_args(f)
    add_recipe_args(f)
    f.add_argument("--dim-batch", type=int, default=2)
    f.add_argument("--checkpoint", default=None)
    f.add_argument("--checkpoint-every", type=int, default=1)
    f.add_argument("--no-resume", action="store_true")
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("compare", help="CPU: local refit vs downloaded lens for one arm")
    c.add_argument("--arm", required=True, choices=["j", "r"])
    c.add_argument("--local", required=True)
    add_lens_args(c)
    c.add_argument("--axes", nargs="*", default=[], help="e.g. L07_SV39")
    c.add_argument("--layers", nargs="*", default=None)
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
