#!/usr/bin/env python3
"""
jspace_ablate_subject.py

Exploratory: what does J-space ablation at the SUBJECT token do to factual
completions, across many relations and several subjects?

For each subject S and relation template R(S), e.g. "Miles Davis was born in
the city of", we run:
  clean            : ordinary forward pass
  jspace[band]     : at the subject's last token, for each layer in the band,
                     find the k most strongly activated J-lens vectors (top-k
                     lens logits at that position), drop any whose token is in
                     the position's own clean top-N next-token predictions
                     (the paper's exclusion rule), and project the residual
                     off the span of those lens vectors. The forward pass then
                     continues on the modified stream, layer by layer.
  random[band]     : same, but project off k random orthonormal directions
                     per layer (fixed seed) — the matched-count control.

We measure, at the LAST position (where the fact is predicted): the clean
top-1 token and probability, the ablated top-1, the change in log-prob of
the clean top-1, its rank under ablation, and KL(clean || ablated).

Because every prompt starts with the subject, the subject token's residual —
and therefore the ablation applied to it — is IDENTICAL across relations for
a given subject. So this is one intervention per subject per band, read by a
dozen different downstream readers. The J-lens tokens removed at each layer
(what the lens thinks the subject position "is about to say") are saved per
subject; they are the qualitative half of the result.

J-lens vector for token t at layer l (paper: rows of W_U J_l), with the final
RMSNorm's elementwise weight folded in by default:  v_t = J_l^T (gamma * u_t).
Use --no-norm-weight for the bare rows of W_U J_l.

Usage:
    python jspace_ablate_subject.py --out jspace_subject_run \
        --bands 9-13 9-17 6-20 --k 10

Model loading mirrors the jlens README (AutoModelForCausalLM + jlens.from_hf);
if your scanner loads gpt-oss differently, replace load_model() with those
lines.
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
    "Miles Davis",
    "Marie Curie",
    "Albert Einstein",
    "Serena Williams",
    "Ludwig van Beethoven",
    "Barack Obama",
]

# (relation_name, template). {S} = subject. Includes two COPY relations
# (first/last name) as a contrast class against RECALL relations.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="J-space ablation at the subject token across factual relations.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--subjects", nargs="*", default=None, help="Override subject list.")
    p.add_argument("--relations-json", default=None,
                   help="JSON file: list of [name, template] pairs to override defaults.")
    p.add_argument("--bands", nargs="+", default=["9-13", "9-17", "6-20"],
                   help="Layer bands (inclusive) for ablation, e.g. 9-17.")
    p.add_argument("--k", type=int, default=10, help="Lens vectors ablated per layer.")
    p.add_argument("--exclude-clean-topn", type=int, default=10,
                   help="Skip lens tokens in the position's own clean top-N outputs.")
    p.add_argument("--position", choices=["subject", "last"], default="subject",
                   help="Ablate at the subject's last token, or at the final token.")
    p.add_argument("--no-norm-weight", action="store_true",
                   help="Do not fold the final norm's weight into lens vectors.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=2, help="Random-control replicates.")
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--readout-topn", type=int, default=15,
                   help="How many top J-lens tokens to record per layer at each position.")
    return p.parse_args()


def parse_band(spec: str) -> list[int]:
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


# ------------------------------------------------------------------ model ----


def load_model(model_name: str):
    """Mirror of the jlens README; swap in your scanner's loading lines if needed."""
    import transformers
    import jlens

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="cuda"
        )
    except TypeError:  # older transformers
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="cuda"
        )
    model = jlens.from_hf(hf, tok)
    return model


def load_lens(repo: str, filename: str):
    import jlens

    return jlens.JacobianLens.from_pretrained(repo, filename=filename)


def subject_last_token_index(tokenizer, prompt: str, subject: str, input_ids: torch.Tensor) -> int:
    """Index (into input_ids[0]) of the last token covering the subject string."""
    start = prompt.find(subject)
    if start < 0:
        raise ValueError(f"subject {subject!r} not found in prompt {prompt!r}")
    end = start + len(subject)
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=input_ids.shape[1])
        offsets = enc["offset_mapping"]
        if len(enc["input_ids"]) == input_ids.shape[1]:
            covering = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
            if covering:
                return max(covering)
    except Exception:
        pass
    # Fallback: prefix tokenization (BPE prefix-consistency).
    prefix_ids = tokenizer(prompt[:end]).input_ids
    return len(prefix_ids) - 1


# ------------------------------------------------------------ ablation hooks --


class ResidualEditor:
    """Forward hooks that edit the residual at one position at chosen layers.

    mode='jspace': project off the span of the top-k activated lens vectors
    (recomputed at each layer from the current, possibly already-edited
    stream). mode='random': project off k fixed random orthonormal directions
    per layer. Records which lens tokens were removed at each layer.
    """

    def __init__(
        self,
        model,
        lens,
        *,
        layers: list[int],
        position: int,
        k: int,
        mode: str,
        exclude_token_ids: set[int],
        use_norm_weight: bool,
        rng: torch.Generator | None = None,
    ) -> None:
        self.model = model
        self.lens = lens
        self.layers = layers
        self.position = position
        self.k = k
        self.mode = mode
        self.exclude = exclude_token_ids
        self.use_norm_weight = use_norm_weight
        self.rng = rng
        self.removed: dict[int, list[int]] = {}
        self._handles: list = []
        self._W_U = model._lm_head.weight  # [vocab, d]
        gamma = getattr(model._final_norm, "weight", None)
        self._gamma = gamma.detach().float() if (gamma is not None and use_norm_weight) else None
        self._J_cache: dict[int, torch.Tensor] = {}

    def _J(self, layer: int, device) -> torch.Tensor:
        if layer not in self._J_cache:
            self._J_cache[layer] = self.lens.jacobians[layer].to(device=device, dtype=torch.float32)
        return self._J_cache[layer]

    def _lens_vectors(self, layer: int, h: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Top-k activated lens vectors at this residual (after exclusions).

        Returns V [d, k'] (columns are lens vectors in layer-l space) and the
        token ids they correspond to.
        """
        J = self._J(layer, h.device)
        transported = h.float() @ J.T  # J h, [d]
        lens_logits = self.model.unembed(transported.unsqueeze(0))[0].float()  # [vocab]
        order = torch.argsort(lens_logits, descending=True)
        chosen: list[int] = []
        for tid in order.tolist():
            if tid in self.exclude:
                continue
            chosen.append(tid)
            if len(chosen) == self.k:
                break
        U = self._W_U[chosen].float().to(h.device)  # [k, d] unembedding rows
        if self._gamma is not None:
            U = U * self._gamma.to(h.device)
        V = (U @ J).T  # v_t = J^T (gamma * u_t)  -> [d, k]
        return V, chosen

    def _random_vectors(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        d = h.shape[-1]
        g = torch.randn(d, self.k, generator=self.rng, dtype=torch.float32)
        return g.to(h.device)

    def _project_off(self, h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(V.float())  # orthonormal basis of span(V)
        hf = h.float()
        return (hf - Q @ (Q.T @ hf)).to(h.dtype)

    def _make_hook(self, layer: int):
        def hook(module: nn.Module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            h = tensor[0, self.position]
            if self.mode == "jspace":
                V, chosen = self._lens_vectors(layer, h)
                self.removed[layer] = chosen
            else:
                V = self._random_vectors(layer, h)
            new_h = self._project_off(h, V)
            tensor = tensor.clone()
            tensor[0, self.position] = new_h
            if torch.is_tensor(output):
                return tensor
            return (tensor, *output[1:])

        return hook

    def __enter__(self):
        for layer in self.layers:
            self._handles.append(self.model.layers[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for hnd in self._handles:
            hnd.remove()
        self._handles = []


# ---------------------------------------------------------------- running ----


@torch.no_grad()
def run_pass(model, lens, input_ids: torch.Tensor, editor: ResidualEditor | None = None):
    """One forward pass. Returns (logits [seq, vocab] float cpu, acts {layer: [seq, d]})
    with acts recorded at every lens layer plus the final block. Editor hooks are
    registered BEFORE the recorder's so recorded tensors are post-edit."""
    from jlens.hooks import ActivationRecorder

    final_layer = model.n_layers - 1
    record_at = sorted(set(lens.source_layers) | {final_layer})

    def _go():
        with ActivationRecorder(model.layers, at=record_at) as rec:
            model.forward(input_ids)
            return {i: rec.activations[i].detach()[0] for i in record_at}

    if editor is not None:
        with editor:
            acts = _go()
    else:
        acts = _go()
    logits = model.unembed(acts[final_layer]).float().cpu()  # [seq, vocab]
    return logits, acts


@torch.no_grad()
def lens_topn(model, lens, h: torch.Tensor, layer: int, n: int, tok) -> list[list]:
    """Top-n J-lens readout at residual h (layer-l space): [[token_str, lens_logprob], ...]."""
    transported = lens.transport(h.float().unsqueeze(0), layer)
    lp = torch.log_softmax(model.unembed(transported)[0].float(), dim=-1)
    vals, idx = torch.topk(lp, n)
    return [[tok.decode(int(i)), round(float(v), 3)] for v, i in zip(vals, idx)]


def model_topn(logits_row: torch.Tensor, n: int, tok) -> list[list]:
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    vals, idx = torch.topk(lp, n)
    return [[tok.decode(int(i)), round(float(v), 3)] for v, i in zip(vals, idx)]


def summarize(clean_logits_last: torch.Tensor, abl_logits_last: torch.Tensor) -> dict:
    lp_clean = torch.log_softmax(clean_logits_last, dim=-1)
    lp_abl = torch.log_softmax(abl_logits_last, dim=-1)
    top_clean = int(torch.argmax(lp_clean))
    top_abl = int(torch.argmax(lp_abl))
    kl = float((lp_clean.exp() * (lp_clean - lp_abl)).sum())
    rank_under_abl = int((lp_abl > lp_abl[top_clean]).sum()) + 1
    return {
        "clean_top1": top_clean,
        "clean_top1_logp": float(lp_clean[top_clean]),
        "abl_top1": top_abl,
        "abl_top1_logp": float(lp_abl[top_abl]),
        "clean_top1_logp_under_abl": float(lp_abl[top_clean]),
        "delta_logp_clean_top1": float(lp_abl[top_clean] - lp_clean[top_clean]),
        "rank_of_clean_top1_under_abl": rank_under_abl,
        "kl_clean_vs_abl": kl,
        "top1_changed": int(top_clean != top_abl),
    }


def _fmt(entries: list[list], n: int = 12) -> str:
    return "  ".join(f"{t!r}" for t, _ in entries[:n])


def write_readouts_txt(path: Path, readouts: dict, bands: list[str], layers: list[int]) -> None:
    """Human-readable J-lens readouts: subject position (clean, then post-edit under
    each ablation band) and last position (clean, then under each band), per layer."""
    lines: list[str] = []
    for subject, R in readouts.items():
        lines.append("=" * 100)
        lines.append(f"SUBJECT: {subject}   (subject token read/ablated: {R.get('subject_token')!r})")
        lines.append("=" * 100)
        lines.append("\n-- J-lens at the SUBJECT token, clean (top tokens per layer) --")
        for l in layers:
            lines.append(f"  L{l:02d}: {_fmt(R['subject_position_clean'][str(l)])}")
        first = next(iter(R["relations"].values()), None)
        if first:
            for spec in bands:
                if spec in first["subject_position_jspace"]:
                    lines.append(f"\n-- J-lens at the SUBJECT token, after J-space ablation there, band {spec} (post-edit; identical across relations) --")
                    for l in layers:
                        lines.append(f"  L{l:02d}: {_fmt(first['subject_position_jspace'][spec][str(l)])}")
        for rel, X in R["relations"].items():
            lines.append("\n" + "-" * 100)
            lines.append(f"RELATION {rel}: {X['prompt']!r}")
            lines.append(f"  model next-token, clean, at last position:    {_fmt(X['model_top_clean_last'], 8)}")
            lines.append(f"  model next-token, clean, at subject position: {_fmt(X['model_top_clean_subject'], 8)}")
            lines.append("  -- J-lens at the LAST position, clean --")
            for l in layers:
                lines.append(f"    L{l:02d}: {_fmt(X['last_position_clean'][str(l)])}")
            for spec in bands:
                if spec in X["last_position_jspace"]:
                    lines.append(f"  -- J-lens at the LAST position, after J-space ablation at the subject token, band {spec} --")
                    for l in layers:
                        lines.append(f"    L{l:02d}: {_fmt(X['last_position_jspace'][spec][str(l)])}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or DEFAULT_SUBJECTS
    relations = DEFAULT_RELATIONS
    if args.relations_json:
        relations = [tuple(x) for x in json.load(open(args.relations_json))]
    bands = {spec: parse_band(spec) for spec in args.bands}

    print("[1/3] loading lens...")
    lens = load_lens(args.lens_repo, args.lens_file)
    print("[2/3] loading model...")
    model = load_model(args.model)
    tok = model.tokenizer
    for spec, layers in bands.items():
        bad = [l for l in layers if l not in lens.source_layers]
        if bad:
            raise SystemExit(f"band {spec} includes layers not in the lens: {bad}")
    print(f"  {model} | lens layers {lens.source_layers[0]}..{lens.source_layers[-1]}")

    rows: list[dict] = []
    removed_log: dict[str, dict] = {}
    readouts: dict[str, dict] = {}
    print("[3/3] running...")

    for subject in subjects:
        removed_log[subject] = {}
        readouts[subject] = {"relations": {}}
        for rel_name, template in relations:
            prompt = template.format(S=subject)
            input_ids = model.encode(prompt, max_length=args.max_seq_len)
            seq_len = input_ids.shape[1]
            pos = subject_last_token_index(tok, prompt, subject, input_ids) if args.position == "subject" else seq_len - 1
            pos_token = tok.decode(input_ids[0, pos])

            clean, clean_acts = run_pass(model, lens, input_ids)
            clean_last = clean[-1]
            last = seq_len - 1
            rel_read = {
                "prompt": prompt, "ablate_position": pos, "ablate_token": pos_token,
                "model_top_clean_last": model_topn(clean[last], args.readout_topn, tok),
                "model_top_clean_subject": model_topn(clean[pos], args.readout_topn, tok),
                "last_position_clean": {str(l): lens_topn(model, lens, clean_acts[l][last], l, args.readout_topn, tok)
                                        for l in lens.source_layers},
                "last_position_jspace": {},
                "subject_position_jspace": {},
            }
            if "subject_position_clean" not in readouts[subject]:
                readouts[subject]["subject_position_clean"] = {
                    str(l): lens_topn(model, lens, clean_acts[l][pos], l, args.readout_topn, tok)
                    for l in lens.source_layers}
                readouts[subject]["subject_token"] = pos_token
            # Exclusion set: the ablated position's own clean top-N next tokens.
            exclude = set(torch.topk(clean[pos], args.exclude_clean_topn).indices.tolist())

            base = {
                "subject": subject, "relation": rel_name, "prompt": prompt,
                "ablate_position": pos, "ablate_token": pos_token, "seq_len": seq_len,
            }
            for spec, layers in bands.items():
                # J-space ablation
                editor = ResidualEditor(model, lens, layers=layers, position=pos, k=args.k,
                                        mode="jspace", exclude_token_ids=exclude,
                                        use_norm_weight=not args.no_norm_weight)
                abl, abl_acts = run_pass(model, lens, input_ids, editor)
                stats = summarize(clean_last, abl[-1])
                rel_read["last_position_jspace"][spec] = {
                    str(l): lens_topn(model, lens, abl_acts[l][last], l, args.readout_topn, tok)
                    for l in lens.source_layers}
                rel_read["subject_position_jspace"][spec] = {
                    str(l): lens_topn(model, lens, abl_acts[l][pos], l, args.readout_topn, tok)
                    for l in lens.source_layers}
                rows.append({**base, "band": spec, "condition": "jspace", "replicate": 0, **stats,
                             "clean_top1_str": tok.decode(stats["clean_top1"]),
                             "abl_top1_str": tok.decode(stats["abl_top1"])})
                if spec not in removed_log[subject]:
                    removed_log[subject][spec] = {
                        str(l): [tok.decode(t) for t in editor.removed.get(l, [])] for l in layers
                    }
                # Random controls
                for rep in range(args.n_random):
                    gen = torch.Generator().manual_seed(args.random_seed * 1000 + rep)
                    editor_r = ResidualEditor(model, lens, layers=layers, position=pos, k=args.k,
                                              mode="random", exclude_token_ids=exclude,
                                              use_norm_weight=not args.no_norm_weight, rng=gen)
                    abl_r, _ = run_pass(model, lens, input_ids, editor_r)
                    stats_r = summarize(clean_last, abl_r[-1])
                    rows.append({**base, "band": spec, "condition": "random", "replicate": rep, **stats_r,
                                 "clean_top1_str": tok.decode(stats_r["clean_top1"]),
                                 "abl_top1_str": tok.decode(stats_r["abl_top1"])})

            readouts[subject]["relations"][rel_name] = rel_read
            r0 = [r for r in rows if r["subject"] == subject and r["relation"] == rel_name and r["condition"] == "jspace"]
            line = " | ".join(f"{r['band']}: {r['delta_logp_clean_top1']:+.2f} -> {r['abl_top1_str']!r}" for r in r0)
            print(f"  {subject:<22} {rel_name:<16} clean={r0[0]['clean_top1_str']!r} p={math.exp(r0[0]['clean_top1_logp']):.2f} | {line}")

    # ---- write outputs ----
    csv_path = out_dir / "results.csv"
    keys: list[str] = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "removed_lens_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(removed_log, f, indent=1, ensure_ascii=False)

    with (out_dir / "lens_readouts.json").open("w", encoding="utf-8") as f:
        json.dump(readouts, f, indent=1, ensure_ascii=False)
    write_readouts_txt(out_dir / "lens_readouts.txt", readouts, list(bands), lens.source_layers)

    # ---- summary: per relation, mean delta under jspace vs random, per band ----
    lines = []
    lines.append("Mean delta log-prob of the clean top-1 answer (last position), by relation and band.")
    lines.append("More negative = more broken. Random = matched-count random-direction control.")
    for spec in bands:
        lines.append(f"\n== band {spec} ==")
        lines.append(f"{'relation':<18}{'jspace':>10}{'random':>10}{'jspace-rand':>13}{'top1 changed (js)':>19}")
        for rel_name, _ in relations:
            js = [r for r in rows if r["relation"] == rel_name and r["band"] == spec and r["condition"] == "jspace"]
            rd = [r for r in rows if r["relation"] == rel_name and r["band"] == spec and r["condition"] == "random"]
            mj = float(np.mean([r["delta_logp_clean_top1"] for r in js])) if js else math.nan
            mr = float(np.mean([r["delta_logp_clean_top1"] for r in rd])) if rd else math.nan
            ch = sum(r["top1_changed"] for r in js)
            lines.append(f"{rel_name:<18}{mj:>10.2f}{mr:>10.2f}{mj - mr:>13.2f}{ch:>10}/{len(js)}")
        lines.append("\nPer subject (mean over relations):")
        for subject in subjects:
            js = [r for r in rows if r["subject"] == subject and r["band"] == spec and r["condition"] == "jspace"]
            rd = [r for r in rows if r["subject"] == subject and r["band"] == spec and r["condition"] == "random"]
            lines.append(f"  {subject:<22} jspace {np.mean([r['delta_logp_clean_top1'] for r in js]):+.2f}   "
                         f"random {np.mean([r['delta_logp_clean_top1'] for r in rd]):+.2f}")
    lines.append("\nJ-lens tokens removed at the subject token (first band, first subject shown; full log in JSON):")
    first_subject = subjects[0]
    first_band = list(bands)[0]
    for layer, toks in removed_log[first_subject][first_band].items():
        lines.append(f"  L{int(layer):02d}: {toks}")
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\n[out] {csv_path}\n[out] {out_dir / 'removed_lens_tokens.json'}\n[out] {out_dir / 'summary.txt'}"
          f"\n[out] {out_dir / 'lens_readouts.json'}\n[out] {out_dir / 'lens_readouts.txt'}")


if __name__ == "__main__":
    main()
