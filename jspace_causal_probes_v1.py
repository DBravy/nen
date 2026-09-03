#!/usr/bin/env python3
"""Causal probes for the "copied out" question, built on jspace_ablate_subject_v3.

Three subcommands, sharing the v3 model/lens/scoring machinery (imported, not
copied, so every number is produced by the same code paths as the batch runs):

  restore     Causal tracing inside an ablation. Ablate the subject positions
              (default: name_lens at layers 6-8), then in each variant restore
              the FULL clean residual at exactly one (position, layer) point,
              and record how much of the span log-prob comes back.

  substitute  Identity substitution. At the window layers, remove each subject
              position's own-token (variant-spanned) component and inject the
              donor's identity direction at matched norm. Grids over alpha
              (how much self is removed) and beta (how much donor is added).
              Scores BOTH the subject's own clean span and the donor's clean
              span, teacher-forced on the subject's prompt.

  contrast    Post-window carrier search. Estimate an identity direction or
              subspace from clean name-position states across subjects
              (difference-of-means, or shared PCA subspace), delete THAT at a
              later band, and log its cosine to the lens name direction.

Examples
  python jspace_causal_probes_v1.py restore --out runs/restore_mandela \
      --subjects "Nelson Mandela" --ablate-cond name_lens --ablate-band 6-8 \
      --name-variants

  python jspace_causal_probes_v1.py substitute --out runs/sub_mandela_mozart \
      --subjects "Nelson Mandela" --donors "Wolfgang Amadeus Mozart" \
      --band 6-8 --alpha-grid 1.0 --beta-grid 0,0.5,1.0 --name-variants

  python jspace_causal_probes_v1.py contrast --out runs/contrast_mandela \
      --subjects "Nelson Mandela" --band 9-17 --method diff --also-name-lens

Every subcommand writes a config.json with the full argument record (including
--name-variants, which the earlier batches did not persist).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch

# ------------------------------------------------------------- base import ---

def load_base(path: str):
    """Import jspace_ablate_subject_v3 from an explicit path."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--base-script not found: {p}")
    spec = importlib.util.spec_from_file_location("jspace_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jspace_base"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- shared bits ---

def add_common(p: argparse.ArgumentParser):
    p.add_argument("--out", required=True)
    p.add_argument("--base-script", default=str(Path(__file__).parent / "jspace_ablate_subject_v3.py"))
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Target subjects (default: v3 DEFAULT_SUBJECTS).")
    p.add_argument("--relations", nargs="*", default=None,
                   help="Relation ids to run (default: all 14).")
    p.add_argument("--span-len", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--name-source", choices=["self", "last"], default="self")
    p.add_argument("--name-variants", action="store_true")
    p.add_argument("--no-norm-weight", action="store_true")


def pick_relations(base, wanted):
    rels = base.DEFAULT_RELATIONS
    if wanted:
        keep = set(wanted)
        rels = [r for r in rels if r[0] in keep]
        missing = keep - {r[0] for r in rels}
        if missing:
            raise SystemExit(f"unknown relations: {sorted(missing)}")
    return rels


def single_token_ids(tok, text):
    try:
        ids_ = tok(text, add_special_tokens=False).input_ids
    except TypeError:
        ids_ = tok(text).input_ids
    return [ids_[0]] if len(ids_) == 1 else []


def name_ids_by_pos(tok, ids, spos, reader, name_source, name_variants):
    """Identical recipe to v3 main(): per position, own token (or last token)
    plus optional single-token case/space/possessive variants."""
    out = {}
    last_id = int(ids[0, spos[-1]])
    for p in list(spos) + [reader]:
        base_id = last_id if name_source == "last" else int(ids[0, p])
        idset = {base_id}
        if name_variants:
            stem = tok.decode(base_id).strip()
            for v in {stem, " " + stem, stem + "'s", " " + stem + "'s"}:
                idset.update(single_token_ids(tok, v))
        out[p] = sorted(idset)
    return out


def lens_dirs(base_mod, model, lens, layer, token_ids, use_norm_weight, device):
    """[d, k] lens directions for token ids at a layer (v3 convention: rows of
    (gamma*U) @ J, un-normalised; QR downstream orthonormalises)."""
    J = lens.jacobians[layer].to(device=device, dtype=torch.float32)
    U = model._lm_head.weight[list(token_ids)].float().to(device)
    gamma = getattr(model._final_norm, "weight", None)
    if gamma is not None and use_norm_weight:
        U = U * gamma.detach().float().to(device)
    return (U @ J).T


class PromptPack:
    """Clean-run cache for one (subject, relation)."""

    def __init__(self, base, model, lens, tok, subject, rel_name, template, span_len, max_seq_len):
        self.subject, self.rel_name = subject, rel_name
        self.prompt = template.format(S=subject)
        self.ids = model.encode(self.prompt, max_length=max_seq_len)
        self.spos = base.subject_token_positions(tok, self.prompt, subject, self.ids)
        self.prompt_len = self.ids.shape[1]
        self.reader = self.prompt_len - 1
        self.clean_logits, self.clean_prompt_acts, _ = base.run_pass(model, lens, self.ids)
        self.span, self.full_ids = base.greedy_span(model, lens, self.ids, span_len)
        self.span_strs = [tok.decode(t) for t in self.span]
        self.cidx = base.first_content_index(self.span_strs)
        self.answer_token = self.span[self.cidx]
        self.full_clean_logits, self.full_clean_acts, _ = base.run_pass(model, lens, self.full_ids)
        self.span_lp_clean = base.score_span(self.full_clean_logits, self.prompt_len, self.span)


def metric_row(base, tok, pack, logits, span_lp):
    """The v3 metric block, factored."""
    lp_clean_first = torch.log_softmax(pack.clean_logits[pack.reader], dim=-1)
    lp_cond_first = torch.log_softmax(logits[pack.reader], dim=-1)
    kl = float((lp_clean_first.exp() * (lp_clean_first - lp_cond_first)).sum())
    content_pos = pack.prompt_len - 1 + pack.cidx
    lp_content = torch.log_softmax(logits[content_pos], dim=-1)
    rank = int((lp_content > lp_content[pack.answer_token]).sum()) + 1
    top1 = int(torch.argmax(lp_content))
    return {
        "span_logp_clean": round(sum(pack.span_lp_clean), 4),
        "span_logp_cond": round(sum(span_lp), 4),
        "delta_span_logp": round(sum(span_lp) - sum(pack.span_lp_clean), 4),
        "span_lp_per_token": "|".join(f"{x:.3f}" for x in span_lp),
        "content_logp_cond": round(span_lp[pack.cidx], 4),
        "delta_content_logp": round(span_lp[pack.cidx] - pack.span_lp_clean[pack.cidx], 4),
        "rank_content_under_cond": rank,
        "content_top1_changed": int(top1 != pack.answer_token),
        "content_top1_cond": tok.decode(top1),
        "kl_first_token": round(kl, 4),
    }


def write_csv(path, rows):
    if not rows:
        print(f"[warn] no rows for {path}")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[out] {path}  ({len(rows)} rows)")


def dump_config(out_dir, args):
    cfg = {k: v for k, v in vars(args).items()}
    json.dump(cfg, (Path(out_dir) / "config.json").open("w"), indent=1, default=str)


# ------------------------------------------------------- extra hook classes --

class RestorePatch:
    """Overwrite output[0, pos] of one layer with a cached clean vector.
    Registered AFTER (i.e. entered inside) the ablation editor, so at a shared
    (layer, position) the restoration wins."""

    def __init__(self, model, layer, pos, clean_vec):
        self.model, self.layer, self.pos = model, layer, pos
        self.clean_vec = clean_vec
        self._h = None

    def _hook(self, module, inputs, output):
        t = output if torch.is_tensor(output) else output[0]
        t = t.clone()
        t[0, self.pos] = self.clean_vec.to(t.device, t.dtype)
        return t if torch.is_tensor(output) else (t, *output[1:])

    def __enter__(self):
        self._h = self.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()


def make_substitute_editor(base):
    class SubstituteEditor(base.ResidualEditor):
        """h <- h - alpha * P_self(h) + beta * ||P_self(h)|| * d_donor.
        d_donor is the equal-weight unit combination of the donor identity
        subspace's orthonormal basis at that layer."""

        def __init__(self, *a, donor_ids_by_pos=None, alpha=1.0, beta=1.0, **kw):
            super().__init__(*a, **kw)
            self.donor_ids_by_pos = donor_ids_by_pos or {}
            self.alpha, self.beta = alpha, beta

        def _make_hook(self, layer):
            def hook(module, inputs, output):
                t = output if torch.is_tensor(output) else output[0]
                t = t.clone()
                for pos in self.positions:
                    h = t[0, pos]
                    self_ids = self.name_ids_by_pos.get(pos, [])
                    if not self_ids:
                        continue
                    hf = h.float()
                    Vs = self._vectors_for_tokens(layer, h, self_ids)
                    Qs, _ = torch.linalg.qr(Vs.float())
                    removed = Qs @ (Qs.T @ hf)
                    new = hf - self.alpha * removed
                    donor_ids = self.donor_ids_by_pos.get(pos, [])
                    if donor_ids and self.beta != 0.0:
                        Vd = self._vectors_for_tokens(layer, h, donor_ids)
                        Qd, _ = torch.linalg.qr(Vd.float())
                        d = Qd.sum(dim=1)
                        d = d / (d.norm() + 1e-8)
                        new = new + self.beta * removed.norm() * d
                    t[0, pos] = new.to(h.dtype)
                return t if torch.is_tensor(output) else (t, *output[1:])
            return hook

    return SubstituteEditor


def make_contrast_editor(base):
    class ContrastEditor(base.ResidualEditor):
        """Project off a fixed, precomputed orthonormal basis per layer."""

        def __init__(self, *a, q_by_layer=None, **kw):
            super().__init__(*a, **kw)
            self.q_by_layer = q_by_layer or {}

        def _make_hook(self, layer):
            Q = self.q_by_layer.get(layer)

            def hook(module, inputs, output):
                if Q is None:
                    return output
                t = output if torch.is_tensor(output) else output[0]
                t = t.clone()
                Qd = Q.to(t.device)
                for pos in self.positions:
                    hf = t[0, pos].float()
                    t[0, pos] = (hf - Qd @ (Qd.T @ hf)).to(t.dtype)
                return t if torch.is_tensor(output) else (t, *output[1:])
            return hook

    return ContrastEditor


# ------------------------------------------------------------------ restore --

def cmd_restore(args):
    base = load_base(args.base_script)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir, args)
    subjects = args.subjects or base.DEFAULT_SUBJECTS
    relations = pick_relations(base, args.relations)
    abl_layers = base.parse_band(args.ablate_band)

    print("[1/3] loading lens..."); lens = base.load_lens(args.lens_repo, args.lens_file)
    print("[2/3] loading model..."); model = base.load_model(args.model, args.attn_implementation)
    tok = model.tokenizer
    restore_layers = base.parse_band(args.restore_layers) if args.restore_layers else list(lens.source_layers)

    corpus_mean = None
    if args.ablate_cond == "mean":
        print("  corpus mean over selected prompts...")
        acc = {l: None for l in lens.source_layers}; n = 0
        for s in subjects:
            for rel_name, template in relations:
                ids = model.encode(template.format(S=s), max_length=args.max_seq_len)
                _, acts, _ = base.run_pass(model, lens, ids)
                for l in lens.source_layers:
                    a = acts[l][1:].float()
                    acc[l] = a.sum(0) if acc[l] is None else acc[l] + a.sum(0)
                n += ids.shape[1] - 1
        corpus_mean = {l: (acc[l] / n).cpu() for l in lens.source_layers}

    rows = []
    print("[3/3] sweeping...")
    for s in subjects:
        for rel_name, template in relations:
            pack = PromptPack(base, model, lens, tok, s, rel_name, template, args.span_len, args.max_seq_len)
            nids = name_ids_by_pos(tok, pack.ids, pack.spos, pack.reader, args.name_source, args.name_variants)
            excl = {p: set(torch.topk(pack.clean_logits[p], args.exclude_clean_topn).indices.tolist())
                    for p in range(pack.prompt_len)}
            abl_positions = pack.spos if args.ablate_positions == "subject_all" else [pack.spos[-1]]

            def editor():
                return base.ResidualEditor(
                    model, lens, layers=abl_layers, positions=abl_positions, k=args.k,
                    mode=args.ablate_cond, exclude_by_pos=excl,
                    use_norm_weight=not args.no_norm_weight, rng=None,
                    corpus_mean=corpus_mean, answer_token=pack.answer_token,
                    name_ids_by_pos=nids)

            # baseline: ablated, no restoration
            abl_logits, _, _ = base.run_pass(model, lens, pack.full_ids, editor())
            span_lp_abl = base.score_span(abl_logits, pack.prompt_len, pack.span)
            gap = sum(pack.span_lp_clean) - sum(span_lp_abl)

            def emit(rpos, rlayer, span_lp, logits):
                rec = (sum(span_lp) - sum(span_lp_abl)) / gap if abs(gap) > args.min_gap else None
                row = {
                    "subject": s, "relation": rel_name, "ablate_cond": args.ablate_cond,
                    "ablate_band": args.ablate_band, "ablate_positions": args.ablate_positions,
                    "restore_pos": rpos,
                    "restore_pos_token": tok.decode(pack.full_ids[0, rpos]) if rpos >= 0 else "",
                    "restore_pos_is_subject": int(rpos in pack.spos) if rpos >= 0 else "",
                    "restore_layer": rlayer,
                    "span_logp_ablated": round(sum(span_lp_abl), 4),
                    "recovery_frac": round(rec, 4) if rec is not None else "",
                }
                row.update(metric_row(base, tok, pack, logits, span_lp))
                rows.append(row)

            emit(-1, -1, span_lp_abl, abl_logits)

            if args.restore_positions == "subject":
                rposs = list(pack.spos)
            elif args.restore_positions == "prompt":
                rposs = list(range(pack.prompt_len))
            else:  # all: prompt + span positions
                rposs = list(range(pack.full_ids.shape[1]))
            n_var = len(rposs) * len(restore_layers)
            print(f"  {s} / {rel_name}: gap {gap:+.2f}, {n_var} restoration variants")
            for rlayer in restore_layers:
                for rpos in rposs:
                    with editor():
                        with RestorePatch(model, rlayer, rpos, pack.full_clean_acts[rlayer][rpos]):
                            logits, _, _ = base.run_pass(model, lens, pack.full_ids)
                    emit(rpos, rlayer, base.score_span(logits, pack.prompt_len, pack.span), logits)

            best = sorted((r for r in rows if r["subject"] == s and r["relation"] == rel_name
                           and r["restore_pos"] >= 0 and r["recovery_frac"] != ""),
                          key=lambda r: -r["recovery_frac"])[:3]
            for b in best:
                print(f"    rescue: pos {b['restore_pos']} ({b['restore_pos_token']!r}) "
                      f"L{b['restore_layer']} -> {b['recovery_frac']:.2f}")
    write_csv(out_dir / "restore.csv", rows)


# --------------------------------------------------------------- substitute --

def cmd_substitute(args):
    base = load_base(args.base_script)
    SubstituteEditor = make_substitute_editor(base)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir, args)
    subjects = args.subjects or base.DEFAULT_SUBJECTS
    donors = args.donors
    relations = pick_relations(base, args.relations)
    layers = base.parse_band(args.band)
    alphas = [float(x) for x in args.alpha_grid.split(",")]
    betas = [float(x) for x in args.beta_grid.split(",")]

    print("[1/3] loading lens..."); lens = base.load_lens(args.lens_repo, args.lens_file)
    print("[2/3] loading model..."); model = base.load_model(args.model, args.attn_implementation)
    tok = model.tokenizer

    # donor caches: clean greedy spans and per-relation name token ids
    donor_pack = {}
    for d in donors:
        for rel_name, template in relations:
            donor_pack[(d, rel_name)] = PromptPack(base, model, lens, tok, d, rel_name,
                                                   template, args.span_len, args.max_seq_len)

    rows = []
    print("[3/3] running...")
    for s in subjects:
        for rel_name, template in relations:
            pack = PromptPack(base, model, lens, tok, s, rel_name, template, args.span_len, args.max_seq_len)
            self_ids = name_ids_by_pos(tok, pack.ids, pack.spos, pack.reader,
                                       args.name_source, args.name_variants)
            positions = pack.spos if args.positions == "subject_all" else [pack.spos[-1]]
            for d in donors:
                if d == s:
                    continue
                dpack = donor_pack[(d, rel_name)]
                # donor span teacher-forced on the SELF prompt: baseline under clean run
                donor_full = torch.cat([pack.ids, torch.tensor([dpack.span], device=pack.ids.device)], dim=1)
                dlog_clean, _, _ = base.run_pass(model, lens, donor_full)
                donor_lp_clean = base.score_span(dlog_clean, pack.prompt_len, dpack.span)

                # donor identity ids mapped onto self positions
                d_ids_map = {}
                d_nids = name_ids_by_pos(tok, dpack.ids, dpack.spos, dpack.reader,
                                         args.name_source, args.name_variants)
                for i, p in enumerate(positions):
                    if args.donor_map == "surname":
                        d_ids_map[p] = d_nids[dpack.spos[-1]]
                    else:  # positional, clamped to donor length
                        src = dpack.spos[min(i, len(dpack.spos) - 1)]
                        d_ids_map[p] = d_nids[src]

                for alpha in alphas:
                    for beta in betas:
                        ed = SubstituteEditor(
                            model, lens, layers=layers, positions=positions, k=args.k,
                            mode="substitute", exclude_by_pos={},
                            use_norm_weight=not args.no_norm_weight,
                            name_ids_by_pos=self_ids,
                            donor_ids_by_pos=d_ids_map, alpha=alpha, beta=beta)
                        logits, _, _ = base.run_pass(model, lens, pack.full_ids, ed)
                        span_lp = base.score_span(logits, pack.prompt_len, pack.span)
                        ed2 = SubstituteEditor(
                            model, lens, layers=layers, positions=positions, k=args.k,
                            mode="substitute", exclude_by_pos={},
                            use_norm_weight=not args.no_norm_weight,
                            name_ids_by_pos=self_ids,
                            donor_ids_by_pos=d_ids_map, alpha=alpha, beta=beta)
                        dlogits, _, _ = base.run_pass(model, lens, donor_full, ed2)
                        donor_lp = base.score_span(dlogits, pack.prompt_len, dpack.span)
                        row = {
                            "subject": s, "donor": d, "relation": rel_name, "band": args.band,
                            "positions": args.positions, "donor_map": args.donor_map,
                            "alpha": alpha, "beta": beta,
                            "self_span": "".join(pack.span_strs),
                            "donor_span": "".join(dpack.span_strs),
                            "donor_span_logp_cleanrun": round(sum(donor_lp_clean), 4),
                            "donor_span_logp_sub": round(sum(donor_lp), 4),
                            "delta_donor_span": round(sum(donor_lp) - sum(donor_lp_clean), 4),
                            "reader_top5_sub": " / ".join(
                                t for t, _ in base.model_topn(logits[pack.reader], 5, tok)),
                        }
                        row.update(metric_row(base, tok, pack, logits, span_lp))
                        rows.append(row)
                        print(f"  {s} / {rel_name} <- {d} a={alpha} b={beta}: "
                              f"self {row['delta_span_logp']:+.2f}, donor-span {row['delta_donor_span']:+.2f}, "
                              f"top {row['reader_top5_sub'].split(' / ')[0]!r}")
    write_csv(out_dir / "substitute.csv", rows)


# ----------------------------------------------------------------- contrast --

def cmd_contrast(args):
    base = load_base(args.base_script)
    ContrastEditor = make_contrast_editor(base)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir, args)
    targets = args.subjects or base.DEFAULT_SUBJECTS
    pool = base.DEFAULT_SUBJECTS  # estimation always uses the full roster
    relations = pick_relations(base, args.relations)
    layers = base.parse_band(args.band)
    ctx_template = dict(base.DEFAULT_RELATIONS)[args.context_relation]

    print("[1/4] loading lens..."); lens = base.load_lens(args.lens_repo, args.lens_file)
    print("[2/4] loading model..."); model = base.load_model(args.model, args.attn_implementation)
    tok = model.tokenizer

    # ---- harvest clean name-position states in the shared context ----
    print(f"[3/4] harvesting states ({args.context_relation} context)...")
    states = {}  # subject -> layer -> [n_pos, d] float cpu
    for s in pool:
        prompt = ctx_template.format(S=s)
        ids = model.encode(prompt, max_length=args.max_seq_len)
        spos = base.subject_token_positions(tok, prompt, s, ids)
        _, acts, _ = base.run_pass(model, lens, ids)
        keep = spos if args.contrast_positions == "all" else [spos[-1]]
        states[s] = {l: acts[l][keep].float().cpu() for l in layers}

    geometry = {}
    rows = []
    print("[4/4] ablating...")
    for s in targets:
        # ---- directions per layer ----
        q_by_layer, geo = {}, {}
        for l in layers:
            tgt = states[s][l].mean(dim=0)
            if args.method == "diff":
                rest = torch.cat([states[o][l] for o in pool if o != s]).mean(dim=0)
                v = tgt - rest
                Q = (v / (v.norm() + 1e-8)).unsqueeze(1)
            else:  # shared identity subspace: PCs of per-subject means
                M = torch.stack([states[o][l].mean(dim=0) for o in pool])
                M = M - M.mean(dim=0, keepdim=True)
                _, _, Vh = torch.linalg.svd(M, full_matrices=False)
                Q = Vh[:args.pcs].T.contiguous()
            q_by_layer[l] = Q

            # geometry: cosine to the lens name direction(s) of the target
            ctx_ids = model.encode(ctx_template.format(S=s), max_length=args.max_seq_len)
            ctx_spos = base.subject_token_positions(tok, ctx_template.format(S=s), s, ctx_ids)
            nids = name_ids_by_pos(tok, ctx_ids, ctx_spos, ctx_ids.shape[1] - 1,
                                   args.name_source, args.name_variants)[ctx_spos[-1]]
            D = lens_dirs(base, model, lens, l, nids, not args.no_norm_weight, Q.device)
            D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
            cos = (Q.T.to(D.dtype) @ D).abs()
            bulk = torch.cat([states[o][l] for o in pool]).mean(dim=0)
            bulk = bulk / (bulk.norm() + 1e-8)
            geo[str(l)] = {
                "max_cos_to_lens_name_dir": round(float(cos.max()), 4),
                "mean_cos_to_lens_name_dir": round(float(cos.mean()), 4),
                "cos_to_bulk_mean_state": round(float((Q.T @ bulk).abs().max()), 4),
                "subspace_dims": int(Q.shape[1]),
            }
        geometry[s] = geo

        # ---- behavior ----
        for rel_name, template in relations:
            pack = PromptPack(base, model, lens, tok, s, rel_name, template, args.span_len, args.max_seq_len)
            positions = pack.spos if args.apply_positions == "subject_all" else [pack.spos[-1]]
            ed = ContrastEditor(model, lens, layers=layers, positions=positions, k=args.k,
                                mode="contrast", exclude_by_pos={},
                                use_norm_weight=not args.no_norm_weight,
                                q_by_layer=q_by_layer)
            logits, _, _ = base.run_pass(model, lens, pack.full_ids, ed)
            span_lp = base.score_span(logits, pack.prompt_len, pack.span)
            row = {"subject": s, "relation": rel_name, "condition": f"contrast_{args.method}",
                   "band": args.band, "apply_positions": args.apply_positions,
                   "contrast_positions": args.contrast_positions,
                   "context_relation": args.context_relation, "pcs": args.pcs if args.method == "pca" else 1}
            row.update(metric_row(base, tok, pack, logits, span_lp))
            rows.append(row)
            print(f"  {s} / {rel_name} contrast_{args.method}: {row['delta_span_logp']:+.2f}")

            if args.also_name_lens:
                nids = name_ids_by_pos(tok, pack.ids, pack.spos, pack.reader,
                                       args.name_source, args.name_variants)
                ed2 = base.ResidualEditor(model, lens, layers=layers, positions=positions,
                                          k=args.k, mode="name_lens", exclude_by_pos={},
                                          use_norm_weight=not args.no_norm_weight,
                                          name_ids_by_pos=nids)
                logits2, _, _ = base.run_pass(model, lens, pack.full_ids, ed2)
                span_lp2 = base.score_span(logits2, pack.prompt_len, pack.span)
                row2 = dict(row); row2["condition"] = "name_lens"
                row2.update(metric_row(base, tok, pack, logits2, span_lp2))
                rows.append(row2)

    write_csv(out_dir / "contrast.csv", rows)
    json.dump(geometry, (out_dir / "contrast_geometry.json").open("w"), indent=1)
    print(f"[out] {out_dir}/contrast_geometry.json")


# --------------------------------------------------------------------- cli ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("restore", help="single-point clean-state restoration sweep inside an ablation")
    add_common(r)
    r.add_argument("--ablate-cond", default="name_lens",
                   choices=["name_lens", "jspace", "jspace_word", "mean",
                            "jspace_keep_name", "jspace_restore_rest"])
    r.add_argument("--ablate-band", default="6-8")
    r.add_argument("--ablate-positions", choices=["subject_all", "subject_last"], default="subject_all")
    r.add_argument("--restore-layers", default=None,
                   help="Band spec like '0-22'; default: all lens layers.")
    r.add_argument("--restore-positions", choices=["subject", "prompt", "all"], default="prompt",
                   help="'prompt' sweeps every prompt token incl. BOS (negative control); 'all' adds span positions.")
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--exclude-clean-topn", type=int, default=10)
    r.add_argument("--min-gap", type=float, default=0.25,
                   help="Minimum clean-ablated gap (nats) for recovery_frac to be reported.")
    r.set_defaults(fn=cmd_restore)

    s = sub.add_parser("substitute", help="remove self identity component, inject donor's at matched norm")
    add_common(s)
    s.add_argument("--donors", nargs="+", required=True)
    s.add_argument("--band", default="6-8")
    s.add_argument("--positions", choices=["subject_all", "subject_last"], default="subject_all")
    s.add_argument("--donor-map", choices=["surname", "positional"], default="surname",
                   help="surname: donor's last-token direction at every edited position; "
                        "positional: donor token at the same relative position (clamped).")
    s.add_argument("--alpha-grid", default="1.0", help="comma list; fraction of self component removed")
    s.add_argument("--beta-grid", default="0,1.0", help="comma list; donor injection scale (x removed norm)")
    s.add_argument("--k", type=int, default=10)
    s.set_defaults(fn=cmd_substitute)

    c = sub.add_parser("contrast", help="delete an estimated identity direction/subspace at a later band")
    add_common(c)
    c.add_argument("--band", default="9-17")
    c.add_argument("--method", choices=["diff", "pca"], default="diff")
    c.add_argument("--pcs", type=int, default=3, help="subspace dims for --method pca")
    c.add_argument("--contrast-positions", choices=["last", "all"], default="last",
                   help="which clean name-position states form each subject's class")
    c.add_argument("--apply-positions", choices=["subject_last", "subject_all"], default="subject_last")
    c.add_argument("--context-relation", default="birth_city",
                   help="template whose forward pass supplies the harvested states "
                        "(valid for subject-initial templates; the 3 mid-sentence ones differ)")
    c.add_argument("--also-name-lens", action="store_true",
                   help="also run the plain name_lens condition in-harness for comparison")
    c.add_argument("--k", type=int, default=10)
    c.set_defaults(fn=cmd_contrast)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
