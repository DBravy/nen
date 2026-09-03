#!/usr/bin/env python3
"""
jspace_ablate_subject_v3.py — v2 plus identity-restoration conditions:
  jspace_keep_name    full J-space ablation, but the component of the removed
                      content lying along the name direction(s) is added back,
                      so each ablated token keeps its identity coordinate.
  jspace_restore_rest everything the ablation removed is added back EXCEPT the
                      name-aligned component: only identity is removed, and
                      only as much of it as the full ablation would have taken.
  name_lens           project out the name direction(s) from the clean stream,
                      nothing else (the from-clean version of restore_rest).
Name direction per ablated position: the J-lens vector of its own token
(--name-source self, default) or of the last subject token (--name-source
last); --name-variants uses the span of case/space/possessive variants.

Original v2 header follows.


Second pass at "what does J-space ablation at the subject do to factual
completions", with the fixes from the first run:

  1. Answer SPANS, not first tokens. The model's own greedy continuation
     (--span-len tokens, default 4) is generated on the clean run, then scored
     (teacher-forced) under every condition. Relations whose first token is a
     space/article are now informative. We also report the first "content"
     token of the span separately.
  2. Whole-subject-span ablation. --positions accepts subject_last (the paper's
     enrichment site), subject_all (every token of the subject string), and
     last (the reader position). Closes the "the reader was reading a different
     subject token" loophole.
  3. Two J-space variants: 'jspace' (paper-faithful top-k after exclusion) and
     'jspace_word' (same, but only word-like lens tokens are eligible, so the
     intervention is not half punctuation/code priors).
  4. Positive controls: 'mean' replaces the whole residual at the ablated
     positions with the corpus-mean residual for that layer (estimated from a
     prepass over all prompts); this should break everything that depends on
     the subject. 'answer_lens' projects out only the lens vector of the answer
     token itself (the first content token of the span) — the direct
     "speech direction vs memory" test.
  5. Matched random-direction control kept ('random', --n-random replicates).
  6. Attention logging (--log-attention; needs --attn-implementation eager):
     at every layer, attention from the reader position to each subject token,
     to BOS, and the argmax key position, mean and max over heads. Shows which
     position the reader actually reads at the retrieval layers.
  7. True ablated outputs are recorded (model top-k at the reader under every
     condition), not the L22-lens proxy.

Outputs in --out: results.csv (one row per prompt x position-mode x band x
condition x replicate), summary.txt, removed_lens_tokens.json,
lens_readouts.json / .txt, attention.json (if enabled), corpus_mean.pt.

Usage:
    python jspace_ablate_subject_v2.py --out jspace_v2 \
        --positions subject_last subject_all \
        --bands 9-13 9-17 6-20 --k 10 \
        --log-attention --attn-implementation eager
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_SUBJECTS = [
    "Miles Davis", "Marie Curie", "Albert Einstein", "Serena Williams",
    "Ludwig van Beethoven", "Barack Obama", "Isaac Newton", "Charles Darwin",
    "Abraham Lincoln", "Wolfgang Amadeus Mozart", "Nelson Mandela", "Winston Churchill",
]

DEFAULT_RELATIONS = [
    ("birth_city", "{S} was born in the city of"),
    ("birth_year", "{S} was born in the year"),
    ("birth_country", "{S} was born in the country of"),
    ("death_year", "{S} died in the year"),
    ("occupation", "{S} worked as a"),
    ("profession_of", "The profession of {S} was"),
    ("instrument", "{S} played the"),
    ("famous_for", "{S} is famous for"),
    ("language", "{S} spoke the language of"),
    ("spouse", "{S} was married to"),
    ("education", "{S} studied at the University of"),
    ("citizenship", "{S} was a citizen of"),
    ("copy_first_name", "The first name of {S} is"),
    ("copy_last_name", "The last name of {S} is"),
]

GENERIC = {"", "the", "a", "an", "his", "her", "of", "in", "?", '"', "what", "to", "and", "was", "is", "____", "______"}
CONDITIONS = ["jspace", "jspace_word", "random", "mean", "answer_lens",
              "jspace_keep_name", "jspace_restore_rest", "name_lens"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="J-space ablation at the subject (v2).")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--attn-implementation", default=None,
                   help="Passed to from_pretrained; use 'eager' for --log-attention.")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--relations-json", default=None)
    p.add_argument("--positions", nargs="+", default=["subject_last", "subject_all"],
                   choices=["subject_last", "subject_all", "subject_first", "subject_inner", "last"])
    p.add_argument("--bands", nargs="+", default=["9-13", "9-17", "6-20"])
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude-clean-topn", type=int, default=10)
    p.add_argument("--span-len", type=int, default=4)
    p.add_argument("--no-norm-weight", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=2)
    p.add_argument("--readout-topn", type=int, default=15)
    p.add_argument("--log-attention", action="store_true")
    p.add_argument("--name-source", choices=["self", "last"], default="self",
                   help="Identity direction per ablated position: its own token ('self') or the last subject token everywhere ('last').")
    p.add_argument("--name-variants", action="store_true",
                   help="Use the span of case/space/possessive single-token variants of the name, not just the one token.")
    p.add_argument("--max-seq-len", type=int, default=64)
    return p.parse_args()


def parse_band(spec: str) -> list[int]:
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def wordlike(t: str) -> bool:
    s = t.strip()
    return len(s) >= 2 and s.replace("'", "").replace("-", "").isalpha()


# ------------------------------------------------------------------ model ----


def load_model(model_name: str, attn_implementation: str | None):
    import transformers
    import jlens

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    kw = {"device_map": "cuda"}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", **kw)
    except TypeError:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", **kw)
    return jlens.from_hf(hf, tok)


def load_lens(repo: str, filename: str):
    import jlens

    return jlens.JacobianLens.from_pretrained(repo, filename=filename)


def subject_token_positions(tokenizer, prompt: str, subject: str, input_ids: torch.Tensor) -> list[int]:
    """All token indices covering the subject string (ascending)."""
    start = prompt.find(subject)
    if start < 0:
        raise ValueError(f"subject {subject!r} not in prompt {prompt!r}")
    end = start + len(subject)
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=input_ids.shape[1])
        if len(enc["input_ids"]) == input_ids.shape[1]:
            cov = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start]
            if cov:
                return cov
    except Exception:
        pass
    n_before = len(tokenizer(prompt[:start].rstrip()).input_ids) if start > 0 else 1
    n_through = len(tokenizer(prompt[:end]).input_ids)
    return list(range(max(n_before, 1), n_through))


# ------------------------------------------------------------ ablation hooks --


class ResidualEditor:
    """Edits the residual at given positions at given layers, mid-forward."""

    def __init__(self, model, lens, *, layers, positions, k, mode, exclude_by_pos, use_norm_weight,
                 rng=None, corpus_mean=None, answer_token=None, name_ids_by_pos=None):
        self.model, self.lens, self.layers, self.positions = model, lens, layers, list(positions)
        self.k, self.mode, self.exclude_by_pos = k, mode, exclude_by_pos
        self.rng, self.corpus_mean, self.answer_token = rng, corpus_mean, answer_token
        self.name_ids_by_pos = name_ids_by_pos or {}
        self.removed: dict[int, dict[int, list[int]]] = {}
        self._handles = []
        self._W_U = model._lm_head.weight
        gamma = getattr(model._final_norm, "weight", None)
        self._gamma = gamma.detach().float() if (gamma is not None and use_norm_weight) else None
        self._J_cache: dict[int, torch.Tensor] = {}

    def _J(self, layer, device):
        if layer not in self._J_cache:
            self._J_cache[layer] = self.lens.jacobians[layer].to(device=device, dtype=torch.float32)
        return self._J_cache[layer]

    def _vectors_for_tokens(self, layer, h, token_ids):
        J = self._J(layer, h.device)
        U = self._W_U[list(token_ids)].float().to(h.device)
        if self._gamma is not None:
            U = U * self._gamma.to(h.device)
        return (U @ J).T  # [d, k]

    def _top_lens_tokens(self, layer, h, pos):
        J = self._J(layer, h.device)
        lens_logits = self.model.unembed((h.float() @ J.T).unsqueeze(0))[0].float()
        order = torch.argsort(lens_logits, descending=True)
        chosen = []
        excl = self.exclude_by_pos.get(pos, set())
        tok = self.model.tokenizer
        for tid in order.tolist():
            if tid in excl:
                continue
            if self.mode == "jspace_word" and not wordlike(tok.decode(tid)):
                continue
            chosen.append(tid)
            if len(chosen) == self.k:
                break
        return chosen

    @staticmethod
    def _project_off(h, V):
        Q, _ = torch.linalg.qr(V.float())
        hf = h.float()
        return (hf - Q @ (Q.T @ hf)).to(h.dtype)

    def _make_hook(self, layer):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            tensor = tensor.clone()
            for pos in self.positions:
                h = tensor[0, pos]
                if self.mode in ("jspace", "jspace_word"):
                    chosen = self._top_lens_tokens(layer, h, pos)
                    self.removed.setdefault(layer, {})[pos] = chosen
                    if chosen:
                        tensor[0, pos] = self._project_off(h, self._vectors_for_tokens(layer, h, chosen))
                elif self.mode == "random":
                    V = torch.randn(h.shape[-1], self.k, generator=self.rng, dtype=torch.float32).to(h.device)
                    tensor[0, pos] = self._project_off(h, V)
                elif self.mode == "mean":
                    tensor[0, pos] = self.corpus_mean[layer].to(h.device, h.dtype)
                elif self.mode == "answer_lens":
                    tensor[0, pos] = self._project_off(h, self._vectors_for_tokens(layer, h, [self.answer_token]))
                elif self.mode == "name_lens":
                    ids = self.name_ids_by_pos.get(pos, [])
                    if ids:
                        tensor[0, pos] = self._project_off(h, self._vectors_for_tokens(layer, h, ids))
                elif self.mode in ("jspace_keep_name", "jspace_restore_rest"):
                    chosen = self._top_lens_tokens(layer, h, pos)
                    self.removed.setdefault(layer, {})[pos] = chosen
                    ids = self.name_ids_by_pos.get(pos, [])
                    if not chosen:
                        continue
                    V = self._vectors_for_tokens(layer, h, chosen)
                    Q, _ = torch.linalg.qr(V.float())
                    hf = h.float()
                    delta = Q @ (Q.T @ hf)  # exactly what full jspace would remove
                    if ids:
                        Vn = self._vectors_for_tokens(layer, h, ids)
                        Qn, _ = torch.linalg.qr(Vn.float())
                        delta_name = Qn @ (Qn.T @ delta)  # the name-aligned part of it
                    else:
                        delta_name = torch.zeros_like(hf)
                    if self.mode == "jspace_keep_name":
                        tensor[0, pos] = (hf - delta + delta_name).to(h.dtype)
                    else:  # remove only the name-aligned part of the would-be ablation
                        tensor[0, pos] = (hf - delta_name).to(h.dtype)
            return tensor if torch.is_tensor(output) else (tensor, *output[1:])
        return hook

    def __enter__(self):
        for layer in self.layers:
            self._handles.append(self.model.layers[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for hnd in self._handles:
            hnd.remove()
        self._handles = []


# ---------------------------------------------------------------- passes -----


@torch.no_grad()
def run_pass(model, lens, input_ids, editor=None, want_attention=False):
    """Returns (logits [seq, vocab] float cpu, acts {layer: [seq, d]}, attentions or None)."""
    from jlens.hooks import ActivationRecorder

    final_layer = model.n_layers - 1
    record_at = sorted(set(lens.source_layers) | {final_layer})

    def _go():
        with ActivationRecorder(model.layers, at=record_at) as rec:
            attn = None
            if want_attention:
                try:
                    out = model._text_module(input_ids=input_ids, use_cache=False, output_attentions=True)
                    attn = getattr(out, "attentions", None)
                except Exception as exc:  # unsupported attn implementation etc.
                    print(f"[warn] attention logging failed ({exc}); continuing without", file=sys.stderr)
                    model.forward(input_ids)
            else:
                model.forward(input_ids)
            return {i: rec.activations[i].detach()[0] for i in record_at}, attn

    if editor is not None:
        with editor:
            acts, attn = _go()
    else:
        acts, attn = _go()
    logits = model.unembed(acts[final_layer]).float().cpu()
    return logits, acts, attn


def summarize_attention(attn, reader_pos, subject_positions):
    """Per layer: mean/max-over-heads attention from reader_pos to subject span, to BOS, argmax key."""
    if attn is None or any(a is None for a in attn):
        return None
    out = {}
    for layer, a in enumerate(attn):
        w = a[0, :, reader_pos, :].float().cpu()  # [heads, keys]
        subj = w[:, subject_positions].sum(dim=1)
        out[str(layer)] = {
            "to_subject_mean": round(float(subj.mean()), 4),
            "to_subject_max_head": round(float(subj.max()), 4),
            "to_bos_mean": round(float(w[:, 0].mean()), 4),
            "argmax_key_of_mean": int(torch.argmax(w.mean(dim=0))),
            "per_subject_pos_mean": [round(float(w[:, p].mean()), 4) for p in subject_positions],
        }
    return out


@torch.no_grad()
def lens_topn(model, lens, h, layer, n, tok):
    lp = torch.log_softmax(model.unembed(lens.transport(h.float().unsqueeze(0), layer))[0].float(), dim=-1)
    vals, idx = torch.topk(lp, n)
    return [[tok.decode(int(i)), round(float(v), 3)] for v, i in zip(vals, idx)]


def model_topn(logits_row, n, tok):
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    vals, idx = torch.topk(lp, n)
    return [[tok.decode(int(i)), round(float(v), 3)] for v, i in zip(vals, idx)]


def greedy_span(model, lens, input_ids, span_len):
    ids = input_ids
    span = []
    for _ in range(span_len):
        logits, _, _ = run_pass(model, lens, ids)
        t = int(torch.argmax(logits[-1]))
        span.append(t)
        ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
    return span, ids


def score_span(logits, prompt_len, span):
    """Teacher-forced log-probs of span tokens; logits are for prompt+span."""
    lps = []
    for i, t in enumerate(span):
        lp = torch.log_softmax(logits[prompt_len - 1 + i], dim=-1)
        lps.append(float(lp[t]))
    return lps


def first_content_index(span_strs):
    for i, s in enumerate(span_strs):
        st = s.strip()
        if st and st not in GENERIC and (st.isalnum() or any(ch.isalpha() for ch in st)):
            return i
    return 0


# ------------------------------------------------------------------ main -----


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or DEFAULT_SUBJECTS
    relations = DEFAULT_RELATIONS if not args.relations_json else [tuple(x) for x in json.load(open(args.relations_json))]
    bands = {spec: parse_band(spec) for spec in args.bands}

    print("[1/4] loading lens..."); lens = load_lens(args.lens_repo, args.lens_file)
    print("[2/4] loading model..."); model = load_model(args.model, args.attn_implementation)
    tok = model.tokenizer
    for spec, layers in bands.items():
        bad = [l for l in layers if l not in lens.source_layers]
        if bad:
            raise SystemExit(f"band {spec} has layers not in lens: {bad}")
    print(f"  {model} | lens layers {lens.source_layers[0]}..{lens.source_layers[-1]}")

    # ---- prepass: prompts, subject positions, corpus-mean residuals ----
    print("[3/4] prepass (subject positions, corpus mean)...")
    prompts = []
    mean_acc = {l: None for l in lens.source_layers}; mean_n = 0
    for s in subjects:
        for rel_name, template in relations:
            prompt = template.format(S=s)
            ids = model.encode(prompt, max_length=args.max_seq_len)
            spos = subject_token_positions(tok, prompt, s, ids)
            prompts.append((s, rel_name, prompt, ids, spos))
            _, acts, _ = run_pass(model, lens, ids)
            for l in lens.source_layers:
                a = acts[l][1:].float()  # drop BOS
                mean_acc[l] = a.sum(0) if mean_acc[l] is None else mean_acc[l] + a.sum(0)
            mean_n += ids.shape[1] - 1
    corpus_mean = {l: (mean_acc[l] / mean_n).cpu() for l in lens.source_layers}
    torch.save(corpus_mean, out_dir / "corpus_mean.pt")
    print(f"  corpus mean over {mean_n} tokens; subject spans e.g. {[tok.decode(prompts[0][3][0, p]) for p in prompts[0][4]]}")

    rows, removed_log, readouts, attn_log = [], {}, {}, {}
    print("[4/4] running...")
    for s, rel_name, prompt, ids, spos in prompts:
        prompt_len = ids.shape[1]
        reader = prompt_len - 1
        clean_logits, clean_acts, clean_attn = run_pass(model, lens, ids, want_attention=args.log_attention)
        span, full_ids = greedy_span(model, lens, ids, args.span_len)
        span_strs = [tok.decode(t) for t in span]
        cidx = first_content_index(span_strs)
        answer_token = span[cidx]
        full_clean_logits, _, _ = run_pass(model, lens, full_ids)
        span_lp_clean = score_span(full_clean_logits, prompt_len, span)

        readouts.setdefault(s, {"relations": {}})
        rel_read = {
            "prompt": prompt, "subject_positions": spos, "subject_tokens": [tok.decode(ids[0, p]) for p in spos],
            "greedy_span": span_strs, "content_index": cidx, "span_logp_clean": [round(x, 3) for x in span_lp_clean],
            "model_top_clean_reader": model_topn(clean_logits[reader], args.readout_topn, tok),
            "reader_lens_clean": {str(l): lens_topn(model, lens, clean_acts[l][reader], l, args.readout_topn, tok) for l in lens.source_layers},
            "reader_lens_jspace": {}, "model_top_cond_reader": {},
        }
        if "subject_last_clean" not in readouts[s]:
            readouts[s]["subject_last_clean"] = {str(l): lens_topn(model, lens, clean_acts[l][spos[-1]], l, args.readout_topn, tok) for l in lens.source_layers}
            readouts[s]["subject_last_token"] = tok.decode(ids[0, spos[-1]])
        if args.log_attention:
            attn_log.setdefault(s, {}).setdefault(rel_name, {})["clean"] = summarize_attention(clean_attn, reader, spos)

        # identity directions per position
        def _single_token_ids(text):
            try:
                ids_ = tok(text, add_special_tokens=False).input_ids
            except TypeError:
                ids_ = tok(text).input_ids
            return [ids_[0]] if len(ids_) == 1 else []
        name_ids_by_pos = {}
        last_id = int(ids[0, spos[-1]])
        for p in spos + [reader]:
            base = last_id if args.name_source == "last" else int(ids[0, p])
            idset = {base}
            if args.name_variants:
                stem = tok.decode(base).strip()
                for v in {stem, " " + stem, stem + "'s", " " + stem + "'s"}:
                    idset.update(_single_token_ids(v))
            name_ids_by_pos[p] = sorted(idset)

        # exclusion sets per position (each position's own clean top-N next tokens)
        excl_all = {p: set(torch.topk(clean_logits[p], args.exclude_clean_topn).indices.tolist()) for p in range(prompt_len)}

        for pmode in args.positions:
            positions = {"subject_last": [spos[-1]], "subject_all": spos, "subject_first": [spos[0]],
                         "subject_inner": spos[:-1] or [spos[0]], "last": [reader]}[pmode]
            for spec, layers in bands.items():
                for cond in args.conditions:
                    reps = range(args.n_random) if cond == "random" else [0]
                    for rep in reps:
                        rng = torch.Generator().manual_seed(args.random_seed * 1000 + rep) if cond == "random" else None
                        editor = ResidualEditor(model, lens, layers=layers, positions=positions, k=args.k, mode=cond,
                                                exclude_by_pos=excl_all, use_norm_weight=not args.no_norm_weight,
                                                rng=rng, corpus_mean=corpus_mean, answer_token=answer_token,
                                                name_ids_by_pos=name_ids_by_pos)
                        want_attn = args.log_attention and cond == "jspace"
                        logits, acts, attn = run_pass(model, lens, full_ids, editor, want_attention=want_attn)
                        span_lp = score_span(logits, prompt_len, span)
                        lp_clean_first = torch.log_softmax(clean_logits[reader], dim=-1)
                        lp_cond_first = torch.log_softmax(logits[reader], dim=-1)
                        kl = float((lp_clean_first.exp() * (lp_clean_first - lp_cond_first)).sum())
                        content_pos = prompt_len - 1 + cidx
                        lp_content = torch.log_softmax(logits[content_pos], dim=-1)
                        rank_content = int((lp_content > lp_content[answer_token]).sum()) + 1
                        top1_content = int(torch.argmax(lp_content))
                        row = {
                            "subject": s, "relation": rel_name, "prompt": prompt, "position_mode": pmode,
                            "ablated_positions": " ".join(map(str, positions)), "band": spec, "condition": cond, "replicate": rep,
                            "greedy_span": "".join(span_strs), "content_token": span_strs[cidx], "content_index": cidx,
                            "span_logp_clean": round(sum(span_lp_clean), 4), "span_logp_cond": round(sum(span_lp), 4),
                            "delta_span_logp": round(sum(span_lp) - sum(span_lp_clean), 4),
                            "content_logp_clean": round(span_lp_clean[cidx], 4), "content_logp_cond": round(span_lp[cidx], 4),
                            "delta_content_logp": round(span_lp[cidx] - span_lp_clean[cidx], 4),
                            "rank_content_under_cond": rank_content, "content_top1_changed": int(top1_content != answer_token),
                            "content_top1_cond": tok.decode(top1_content), "kl_first_token": round(kl, 4),
                        }
                        rows.append(row)
                        key = f"{pmode}|{spec}"
                        rel_read["model_top_cond_reader"].setdefault(cond, {})[key] = model_topn(logits[reader], 8, tok)
                        if cond in ("jspace", "jspace_word", "jspace_keep_name", "jspace_restore_rest"):
                            removed_log.setdefault(s, {}).setdefault(pmode, {}).setdefault(spec, {})[cond] = {
                                str(l): {str(p): [tok.decode(t) for t in toks] for p, toks in per_pos.items()}
                                for l, per_pos in editor.removed.items()}
                        if cond == "jspace":
                            rel_read["reader_lens_jspace"][key] = {str(l): lens_topn(model, lens, acts[l][reader], l, args.readout_topn, tok) for l in lens.source_layers}
                            if args.log_attention and attn is not None:
                                attn_log[s][rel_name][f"jspace|{key}"] = summarize_attention(attn, reader, spos)
        readouts[s]["relations"][rel_name] = rel_read

        js = [r for r in rows if r["subject"] == s and r["relation"] == rel_name and r["condition"] == "jspace" and r["position_mode"] == args.positions[0]]
        line = " | ".join(f"{r['band']}:{r['delta_span_logp']:+.2f}" for r in js)
        print(f"  {s:<24}{rel_name:<16} span={''.join(span_strs)!r:<24} content={span_strs[cidx]!r:<12} jspace[{args.positions[0]}] {line}")

    # ---- outputs ----
    keys = list(rows[0].keys())
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(removed_log, (out_dir / "removed_lens_tokens.json").open("w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(readouts, (out_dir / "lens_readouts.json").open("w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if args.log_attention:
        json.dump(attn_log, (out_dir / "attention.json").open("w", encoding="utf-8"), indent=1)
    write_readouts_txt(out_dir / "lens_readouts.txt", readouts, args.positions, list(bands), lens.source_layers)
    write_summary(out_dir / "summary.txt", rows, subjects, relations, args, removed_log, readouts)
    print(f"\n[out] {out_dir}/results.csv, summary.txt, removed_lens_tokens.json, lens_readouts.json/.txt" + (", attention.json" if args.log_attention else ""))


# --------------------------------------------------------------- reports -----


def _fmt(entries, n=12):
    return "  ".join(repr(t) for t, _ in entries[:n])


def write_readouts_txt(path, readouts, positions, bands, layers):
    lines = []
    for s, R in readouts.items():
        lines += ["=" * 100, f"SUBJECT: {s}   (last subject token {R.get('subject_last_token')!r})", "=" * 100,
                  "\n-- J-lens at the LAST SUBJECT token, clean --"]
        lines += [f"  L{l:02d}: {_fmt(R['subject_last_clean'][str(l)])}" for l in layers]
        for rel, X in R["relations"].items():
            lines += ["\n" + "-" * 100, f"RELATION {rel}: {X['prompt']!r}",
                      f"  subject tokens: {X['subject_tokens']} | greedy span: {X['greedy_span']} (content idx {X['content_index']})",
                      f"  model next-token at reader, clean: {_fmt(X['model_top_clean_reader'], 8)}"]
            for cond, d in X["model_top_cond_reader"].items():
                for key, lst in d.items():
                    lines.append(f"  model next-token at reader, {cond} [{key}]: {_fmt(lst, 6)}")
            lines.append("  -- J-lens at the READER, clean --")
            lines += [f"    L{l:02d}: {_fmt(X['reader_lens_clean'][str(l)])}" for l in layers]
            for key, d in X["reader_lens_jspace"].items():
                lines.append(f"  -- J-lens at the READER, jspace [{key}] --")
                lines += [f"    L{l:02d}: {_fmt(d[str(l)])}" for l in layers]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path, rows, subjects, relations, args, removed_log, readouts):
    L = []
    L.append("Mean delta log-prob of the model's own greedy answer SPAN under each condition (more negative = more broken).")
    L.append("Conditions: jspace = paper-faithful top-k J-lens ablation; jspace_word = word-like lens tokens only;")
    L.append("random = matched random directions; mean = corpus-mean replacement of the whole residual (positive control);")
    L.append("answer_lens = project out only the answer token's own lens vector.")
    conds = args.conditions
    for pmode in args.positions:
        for spec in args.bands:
            L.append(f"\n== positions {pmode} | band {spec} ==")
            w = {c: max(13, len(c) + 2) for c in conds}
            L.append(f"{'relation':<18}" + "".join(f"{c:>{w[c]}}" for c in conds) + f"{'content flips(js)':>20}")
            for rel_name, _ in relations:
                cells = []
                for c in conds:
                    v = [r["delta_span_logp"] for r in rows if r["relation"] == rel_name and r["position_mode"] == pmode and r["band"] == spec and r["condition"] == c]
                    cells.append(f"{np.mean(v):>{w[c]}.2f}" if v else f"{'nan':>{w[c]}}")
                js = [r for r in rows if r["relation"] == rel_name and r["position_mode"] == pmode and r["band"] == spec and r["condition"] == "jspace"]
                L.append(f"{rel_name:<18}" + "".join(cells) + f"{sum(r['content_top1_changed'] for r in js):>15}/{len(js)}")
            L.append("per subject:")
            for s in subjects:
                cells = []
                for c in conds:
                    v = [r["delta_span_logp"] for r in rows if r["subject"] == s and r["position_mode"] == pmode and r["band"] == spec and r["condition"] == c]
                    cells.append(f"{np.mean(v):>{w[c]}.2f}" if v else f"{'nan':>{w[c]}}")
                L.append(f"  {s:<24}" + "".join(cells))
    # identity-removed table
    L.append("\nWas the subject's own (last) name token among the ablated lens tokens? (jspace, subject_last, per band: layers)")
    for s in subjects:
        name = readouts[s]["subject_last_token"].strip()
        parts = []
        for spec in args.bands:
            d = removed_log.get(s, {}).get("subject_last", {}).get(spec, {}).get("jspace", {})
            hit = [f"L{int(l)}" for l, per_pos in d.items() if any(any(t.strip() == name for t in toks) for toks in per_pos.values())]
            parts.append(f"{spec}: {','.join(hit) if hit else '—'}")
        L.append(f"  {s:<24} token {name!r:<14} " + " | ".join(parts))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
