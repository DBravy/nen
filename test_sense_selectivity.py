#!/usr/bin/env python3
"""
test_sense_selectivity.py

Does a J-lens SVD axis separate *senses* of a word that the token-indexed
J-lens row (the paper's v_t = row t of W_U J_l) structurally cannot?

Three subcommands, run in order:

  harvest  (GPU)  Stream a held-out corpus shard, find every occurrence of the
                  target word (all tokenizer variants: " may", " May", "May",
                  "may"), run each occurrence in a BOS-prefixed window that
                  mirrors the scanner's conventions, and save the residual
                  stream at the occurrence (and optional later offsets) for the
                  requested layers, plus a background sample of non-target
                  positions. Also computes, for each saved vector, the
                  token-indexed baselines: the linear J-lens row score
                  <J_l^T u_tok, h> and the full normalized lens logit
                  W_U[tok] . norm(J_l h). Everything needing the model happens
                  here; later steps are CPU-only.

  label    (CPU)  Assign a sense to each occurrence. Always runs a rule-based
                  heuristic labeler (deontic / epistemic / month / optative /
                  other / uncertain). With --anthropic, batches contexts to
                  the Claude API for a much better labeling. Accepts external
                  labels via --labels-in.

  analyze  (CPU)  For each axis and offset, compare:
                    * the SVD axis  (unit right-singular vector)
                    * the J-lens row for the occurrence's own token
                    * the J-lens row for the canonical " may" token
                    * the full normalized lens logit for the own token
                    * a cross-validated contrast-mean direction (linear ceiling)
                    * random directions (null)
                    * all k axes of the same layer's bank (where does this axis
                      rank as a sense separator?)
                  on token-vs-background AUC and sense-vs-sense AUCs
                  (deontic vs epistemic is the primary contrast), with
                  document-cluster bootstrap CIs and Cohen's d.

Example (the "may" axis, SV39 at layers 7 and 8):

    python test_sense_selectivity.py harvest \
        --axes L07_SV39 L08_SV39 --word may \
        --directions-dir scan_svd_r0/directions \
        --seed 1 --n-docs 6000 --target-occurrences 1500 \
        --offsets 0 1 2 --out sense_may

    python test_sense_selectivity.py label --out sense_may --anthropic

    python test_sense_selectivity.py analyze --out sense_may \
        --directions-dir scan_svd_r0/directions --plots

Important caveat baked into the design: gpt-oss is causal, so the residual
stream at the "may" token only sees LEFT context. The sense label uses full
context (that's the ground truth of the utterance), so separation at offset 0
is bounded by how predictable the sense is from what precedes the word.
Offsets 1 and 2 show whether separation sharpens once the continuation has
been read.
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
from typing import Any, Iterable

import numpy as np

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LENS_REPO = "solarkyle/jspace-lenses"
DEFAULT_LENS_FILE = "gpt-oss-20b/lens.pt"
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_CONFIG = "sample-10BT"

SENSES = ["deontic", "epistemic", "month", "optative", "other", "uncertain"]


# =============================================================================
# Shared helpers
# =============================================================================


def parse_axis(name: str) -> tuple[int, int]:
    """'L07_SV39' -> (7, 39) with 1-based rank."""
    m = re.fullmatch(r"L(\d+)_SV(\d+)", name.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad axis name {name!r}; expected like L07_SV39")
    return int(m.group(1)), int(m.group(2))


def axis_name(layer: int, rank: int) -> str:
    return f"L{layer:02d}_SV{rank:02d}"


def load_bank_V(directions_dir: Path, layer: int) -> np.ndarray:
    z = np.load(directions_dir / f"L{layer:02d}.npz")
    V = np.asarray(z["V"], dtype=np.float32)
    if V.ndim != 2:
        raise ValueError("V must be 2D")
    if V.shape[0] < V.shape[1]:
        V = V.T
    return V  # [d_model, k]


def find_occurrences(ids: list[int], target_ids: set[int]) -> list[int]:
    return [i for i, t in enumerate(ids) if t in target_ids]


def build_window(
    ids: list[int], i: int, left: int, right: int, bos_id: int | None
) -> tuple[list[int], int]:
    """Window around occurrence i with BOS prefix; returns (window_ids, pos_in_window)."""
    start = max(0, i - left)
    end = min(len(ids), i + right + 1)
    window = ids[start:end]
    pos = i - start
    if bos_id is not None:
        window = [int(bos_id)] + [int(x) for x in window]
        pos += 1
    return window, pos


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


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return math.nan
    sp = math.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else math.nan


def cluster_bootstrap_auc(
    scores: np.ndarray, is_pos: np.ndarray, is_neg: np.ndarray, docs: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI for AUC by resampling documents with replacement."""
    uniq = np.unique(docs)
    if len(uniq) < 3 or n_boot <= 0:
        return math.nan, math.nan
    by_doc = {d: np.where(docs == d)[0] for d in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_doc[d] for d in pick])
        p = scores[idx][is_pos[idx]]
        n = scores[idx][is_neg[idx]]
        if len(p) and len(n):
            vals.append(auc(p, n))
    if not vals:
        return math.nan, math.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =============================================================================
# harvest
# =============================================================================


def cmd_harvest(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    axes = [parse_axis(a) for a in args.axes]
    layers = sorted({l for l, _ in axes})
    offsets = sorted(set(int(o) for o in args.offsets))
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # ---- lens: keep only the needed Jacobians on CPU ------------------------
    print("[1/4] Loading J-Lens...")
    import jlens

    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    d_model = int(lens.d_model)
    for l in layers:
        if l not in lens.source_layers:
            raise SystemExit(f"layer {l} not among fitted layers {lens.source_layers}")
    J_cpu = {l: lens.jacobians[l].detach().to("cpu", torch.float32) for l in layers}
    del lens

    # ---- axis vectors from the bank ----------------------------------------
    axis_vecs: dict[str, np.ndarray] = {}
    for l, r in axes:
        V = load_bank_V(Path(args.directions_dir), l)
        if r < 1 or r > V.shape[1]:
            raise SystemExit(f"rank {r} out of range for layer {l} (k={V.shape[1]})")
        v = V[:, r - 1].astype(np.float32)
        axis_vecs[axis_name(l, r)] = v / max(np.linalg.norm(v), 1e-12)

    # ---- model ---------------------------------------------------------------
    print("[2/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model = jlens.from_hf(hf_model, tokenizer)
    if model.d_model != d_model:
        raise SystemExit(f"model d_model={model.d_model} != lens d_model={d_model}")

    W = hf_model.get_output_embeddings().weight  # [vocab, d]
    # Final norm parameters (RMSNorm) for the exact lens logit.
    norm_mod = getattr(getattr(hf_model, "model", None), "norm", None)
    norm_w = None
    norm_eps = 1e-5
    if norm_mod is not None and hasattr(norm_mod, "weight"):
        norm_w = norm_mod.weight.detach().to("cpu", torch.float32)
        for attr in ("variance_epsilon", "eps", "epsilon"):
            if hasattr(norm_mod, attr):
                norm_eps = float(getattr(norm_mod, attr))
                break
    else:
        print("[warn] final norm module not found; lens logit uses plain RMS normalization", file=sys.stderr)

    def final_norm(x: "torch.Tensor") -> "torch.Tensor":
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps)
        return y * norm_w if norm_w is not None else y

    # ---- target token variants ----------------------------------------------
    word = args.word
    variants = [f" {word}", f" {word.capitalize()}", word.capitalize(), word]
    if args.extra_variants:
        variants += args.extra_variants
    variant_ids: dict[str, int] = {}
    for s in variants:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            variant_ids[s] = int(enc[0])
        else:
            print(f"[variants] skipping {s!r}: tokenizes to {len(enc)} tokens", file=sys.stderr)
    if not variant_ids:
        raise SystemExit("no single-token variants of the target word")
    print(f"[variants] {variant_ids}")
    id_to_variant = {v: k for k, v in variant_ids.items()}
    target_ids = set(variant_ids.values())
    main_variant = f" {word}" if f" {word}" in variant_ids else next(iter(variant_ids))

    # Token-indexed rows: v_row[layer][variant] = J_l^T u_tok  (unit-normalized copy kept too)
    u_rows = {s: W[tid].detach().to("cpu", torch.float32) for s, tid in variant_ids.items()}
    row_vecs: dict[int, dict[str, np.ndarray]] = {}
    for l in layers:
        row_vecs[l] = {}
        for s, u in u_rows.items():
            v = (J_cpu[l].T @ u).numpy().astype(np.float32)
            row_vecs[l][s] = v

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    bos_id = tokenizer.bos_token_id

    # ---- corpus ----------------------------------------------------------------
    print("[3/4] Streaming corpus...")
    if args.input_jsonl:
        def corpus() -> Iterable[dict[str, Any]]:
            with open(args.input_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        yield {"text": obj} if isinstance(obj, str) else obj
        ds: Iterable[dict[str, Any]] = corpus()
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True)
        if args.shuffle_buffer > 0:
            ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    # ---- storage -----------------------------------------------------------------
    occ_records: list[dict[str, Any]] = []
    hid: dict[tuple[int, int], list[np.ndarray]] = {(l, o): [] for l in layers for o in offsets}
    hid_valid: dict[int, list[bool]] = {o: [] for o in offsets}
    lens_logit: dict[tuple[int, int, str], list[float]] = {(l, o, s): [] for l in layers for o in offsets for s in variant_ids}
    bg_hid: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    bg_lens_logit: dict[tuple[int, str], list[float]] = {(l, s): [] for l in layers for s in variant_ids}
    bg_doc: list[int] = []

    def lens_logits_for(l: int, h: "torch.Tensor") -> dict[str, float]:
        z = final_norm(J_cpu[l] @ h)
        return {s: float(u @ z) for s, u in u_rows.items()}

    n_docs = 0
    n_occ = 0
    n_forward = 0
    t0 = time.time()
    print("[4/4] Harvesting...")
    with torch.inference_mode():
        for doc_index, item in enumerate(ds):
            if n_docs >= args.n_docs or n_occ >= args.target_occurrences:
                break
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            n_docs += 1
            if args.char_cap > 0 and len(text) > args.char_cap:
                start = rng.randint(0, len(text) - args.char_cap)
                text = text[start : start + args.char_cap]
            ids = tokenizer.encode(text, add_special_tokens=False)
            occ = find_occurrences(ids, target_ids)
            if not occ:
                continue
            if len(occ) > args.max_per_doc:
                occ = sorted(rng.sample(occ, args.max_per_doc))

            for i in occ:
                if n_occ >= args.target_occurrences:
                    break
                window, pos = build_window(ids, i, args.left_ctx, args.right_ctx, bos_id)
                input_ids = torch.tensor([window], dtype=torch.long).to(model.input_device)
                with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
                    model.forward(input_ids)
                    n_forward += 1
                    seq_len = len(window)
                    # background positions: non-special, not the target, not adjacent offsets
                    excluded = {pos + o for o in offsets}
                    cand = [p for p in range(1, seq_len) if window[p] not in special_ids and p not in excluded]
                    bg_positions = rng.sample(cand, min(args.background_per_window, len(cand))) if cand else []
                    for l in layers:
                        H = recorder.activations[l][0].detach()  # [seq, d]
                        for o in offsets:
                            p = pos + o
                            ok = p < seq_len
                            if l == layers[0]:
                                hid_valid[o].append(ok)
                            if ok:
                                h = H[p].to("cpu", torch.float32)
                                hid[(l, o)].append(h.numpy().astype(np.float16))
                                ll = lens_logits_for(l, h)
                            else:
                                hid[(l, o)].append(np.full(d_model, np.nan, dtype=np.float16))
                                ll = {s: math.nan for s in variant_ids}
                            for s in variant_ids:
                                lens_logit[(l, o, s)].append(ll[s])
                        for p in bg_positions:
                            h = H[p].to("cpu", torch.float32)
                            bg_hid[l].append(h.numpy().astype(np.float16))
                            ll = lens_logits_for(l, h)
                            for s in variant_ids:
                                bg_lens_logit[(l, s)].append(ll[s])
                    for _ in bg_positions:
                        bg_doc.append(doc_index)

                left_txt = tokenizer.decode(window[max(1, pos - args.context_tokens) : pos], skip_special_tokens=True)
                tok_txt = tokenizer.decode([window[pos]], skip_special_tokens=True)
                right_txt = tokenizer.decode(window[pos + 1 : pos + 1 + args.context_tokens], skip_special_tokens=True)
                occ_records.append(
                    {
                        "id": n_occ,
                        "document_index": doc_index,
                        "token_id": int(window[pos]),
                        "variant": id_to_variant[int(window[pos])],
                        "pos_in_window": pos,
                        "window_len": len(window),
                        "left": left_txt,
                        "token": tok_txt,
                        "right": right_txt,
                        "context_marked": f"{left_txt}⟦{tok_txt}⟧{right_txt}",
                        "url": item.get("url") if isinstance(item.get("url"), str) else None,
                    }
                )
                n_occ += 1

            if n_docs % 200 == 0:
                el = time.time() - t0
                print(f"[harvest] docs={n_docs} occurrences={n_occ} forwards={n_forward} elapsed={el:.0f}s")

    if n_occ == 0:
        raise SystemExit("no occurrences harvested")

    # ---- write --------------------------------------------------------------------
    with (out / "occurrences.jsonl").open("w", encoding="utf-8") as f:
        for r in occ_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arrays: dict[str, np.ndarray] = {}
    for (l, o), rows in hid.items():
        arrays[f"h_L{l:02d}_o{o}"] = np.stack(rows)
    for o in offsets:
        arrays[f"valid_o{o}"] = np.asarray(hid_valid[o], dtype=bool)
    for (l, o, s), vals in lens_logit.items():
        arrays[f"lenslogit_L{l:02d}_o{o}_{variant_ids[s]}"] = np.asarray(vals, dtype=np.float32)
    for l in layers:
        arrays[f"bg_h_L{l:02d}"] = np.stack(bg_hid[l]) if bg_hid[l] else np.zeros((0, d_model), np.float16)
        for s in variant_ids:
            arrays[f"bg_lenslogit_L{l:02d}_{variant_ids[s]}"] = np.asarray(bg_lens_logit[(l, s)], dtype=np.float32)
    arrays["bg_doc"] = np.asarray(bg_doc, dtype=np.int64)
    for name, v in axis_vecs.items():
        arrays[f"axis_{name}"] = v
    for l in layers:
        for s, v in row_vecs[l].items():
            arrays[f"row_L{l:02d}_{variant_ids[s]}"] = v
    np.savez_compressed(out / "hidden.npz", **arrays)

    meta = {
        "word": word,
        "variant_ids": variant_ids,
        "main_variant": main_variant,
        "axes": [axis_name(l, r) for l, r in axes],
        "layers": layers,
        "offsets": offsets,
        "d_model": d_model,
        "model": args.model,
        "lens": [args.lens_repo, args.lens_file],
        "seed": args.seed,
        "documents_seen": n_docs,
        "occurrences": n_occ,
        "background_vectors": len(bg_doc),
        "left_ctx": args.left_ctx,
        "right_ctx": args.right_ctx,
        "dataset": None if args.input_jsonl else [args.dataset, args.dataset_config, args.split],
        "final_norm_found": norm_w is not None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {n_occ} occurrences from {n_docs} docs, {len(bg_doc)} background vectors -> {out}")
    print("Next: python test_sense_selectivity.py label --out", out, "[--anthropic]")


# =============================================================================
# label
# =============================================================================

MONTH_LEFT = re.compile(
    r"(\b\d{1,2}(st|nd|rd|th)?|\b(of|in|on|by|from|until|till|through|between|since|during|early|late|mid|last|next|this|coming|end of|beginning of|month of)|[,;:(\-–—]|\b(january|february|march|april|june|july|august|september|october|november|december))\s*$",
    re.I,
)
MONTH_RIGHT = re.compile(r"^\s*(\d{1,2}(st|nd|rd|th)?\b|\d{4}\b|[,;:)\-–—/]|of\s+\d{4}|\d{1,2}\s*,)")
OPTATIVE_RIGHT = re.compile(r"^\s*(the|god|your|his|her|their|our|all|this|these|every|peace|he|she|they|it)\b", re.I)
REQUEST_RIGHT = re.compile(r"^\s*(i|we)\b", re.I)
DEONTIC_SUBJ_LEFT = re.compile(
    r"\b(you|users?|members?|participants?|students?|applicants?|candidates?|customers?|employees?|staff|residents?|patients?|visitors?|guests?|owners?|tenants?|licensees?|buyers?|sellers?|parties|party|the (applicant|user|tenant|licensee|customer|employee|member|student|board|court|committee|council|commission|agency|department|company|contractor|purchaser|borrower|lender|insurer|insured)|no (person|one|individual|member|user)|nobody|anyone|any (person|user|member|party|individual)|persons?|individuals?|companies|organi[sz]ations?|children|parents?|teachers?|drivers?|passengers?|players?|teams?|winners?|holders?)\s*$",
    re.I,
)
DEONTIC_RIGHT = re.compile(
    r"^\s*(not\s+|only\s+|also\s+|then\s+)?(be\s+(used|required|permitted|allowed|submitted|applied|granted|withdrawn|charged|entitled|eligible|contacted|asked|requested|reproduced|copied|distributed|shared|transferred|assigned|terminated|cancell?ed|renewed|extended|revoked|suspended|removed|denied|refused|admitted|excused|exempt(ed)?|reimbursed|refunded)|use|apply|submit|request|contact|choose|elect|opt|access|download|purchase|return|cancel|withdraw|enter|participate|register|enroll|attend|bring|take|obtain|receive|remove|proceed|leave|refuse|decline|include|exclude|share|reproduce|copy|distribute|transfer|assign|terminate|renew|extend|revoke|suspend|deny|admit|exempt|reimburse|refund|claim|file|appeal|vote|sign|log|visit|view|print|edit|delete|upload|post|publish|sell|buy|rent|park|smoke|drink|eat|fish|hunt|camp|swim)\b",
    re.I,
)
EPISTEMIC_RIGHT = re.compile(
    r"^\s*(or may not\b|not\s+)?(be\s+(due|because|a|an|the|more|less|able|unable|related|caused|associated|necessary|possible|difficult|easier|harder|better|worse|true|false|right|wrong|surprised|interested|helpful|useful|important|slightly|somewhat|partly|partially|entirely|completely|just|simply|that|what|why|how|one)|have\b|seem|appear|indicate|suggest|cause|lead|result|contain|include|vary|affect|help|reflect|explain|represent|differ|occur|exist|require|need|want|prefer|find|feel|think|look|sound|become|increase|decrease|reduce|improve|change|make|mean|prove|turn|actually|even|already|still|never|sometimes|well|also|often|no longer|not\b|never\b|already\b)\b",
    re.I,
)
EPISTEMIC_SUBJ_LEFT = re.compile(
    r"\b(it|this|that|there|which|they|these|those|he|she|some|many|others|results?|studies|symptoms|prices|values|levels|effects|changes|conditions|factors|data|figures|numbers|rates|costs|fees|times|dates|colou?rs|sizes|specifications|features|availability|shipping|delivery|the \w+)\s*$",
    re.I,
)
LEGAL_CUES = re.compile(r"\b(shall|must|permitted|prohibited|pursuant|hereby|herein|terms|policy|agreement|licen[cs]e|section|clause|subject to|provided that|at (its|his|her|their|our) (sole )?discretion)\b", re.I)
NAME_LEFT = re.compile(r"\b(theresa|brian|elizabeth|prime minister|pm|mrs\.?|ms\.?|mr\.?|dr\.?|robert|james|john|kim|jim)\s*$", re.I)


def heuristic_label(rec: dict[str, Any]) -> tuple[str, str]:
    left, tok, right = rec.get("left", ""), rec.get("token", ""), rec.get("right", "")
    cap = tok.strip()[:1].isupper()
    left_tail = left[-160:]
    if cap and (MONTH_RIGHT.search(right) or MONTH_LEFT.search(left_tail)):
        return "month", "cap+date-cue"
    if cap and NAME_LEFT.search(left_tail):
        return "other", "name"
    if cap and REQUEST_RIGHT.search(right):
        return "deontic", "May I/we request"
    if cap and OPTATIVE_RIGHT.search(right) and not left_tail.rstrip().endswith((".", "!", "?", ":", "\"", "“")):
        # Capitalized mid-sentence "May the ..." without date cue: could be optative or month; lean uncertain
        return "uncertain", "cap mid-sentence"
    if cap and OPTATIVE_RIGHT.search(right):
        return "optative", "May the/you..."
    if re.search(r"^\s*not\s+", right, re.I) and DEONTIC_SUBJ_LEFT.search(left_tail):
        return "deontic", "person-subject + may not"
    if DEONTIC_SUBJ_LEFT.search(left_tail) and DEONTIC_RIGHT.search(right):
        return "deontic", "person-subject + permission verb"
    if DEONTIC_RIGHT.search(right) and LEGAL_CUES.search(left_tail):
        return "deontic", "permission verb + legal cue"
    if EPISTEMIC_RIGHT.search(right):
        return "epistemic", "epistemic continuation"
    if EPISTEMIC_SUBJ_LEFT.search(left_tail) and not DEONTIC_RIGHT.search(right):
        return "epistemic", "non-agent subject"
    if DEONTIC_SUBJ_LEFT.search(left_tail):
        return "deontic", "person-subject only"
    return "uncertain", "no rule"


LABEL_PROMPT = """You are labeling the sense of the word "{word}" in English text. For each item, read the full context and output the sense of the marked occurrence (between ⟦ and ⟧).

Senses:
- deontic: permission, prohibition, or polite request/offer. Examples: "you may proceed", "users may not share", "applicants may submit", "May I help you?", "members may cancel at any time".
- epistemic: possibility or uncertainty about facts. Examples: "it may rain", "this may be due to", "results may vary", "he may have left".
- month: the calendar month of May.
- optative: a wish or blessing. Examples: "May the force be with you", "May you live long".
- other: a name (Theresa May), a fragment, a non-English use, or anything else.
- uncertain: genuinely ambiguous even with full context.

Return ONLY a JSON array of objects {{"id": <int>, "sense": "<one of: deontic, epistemic, month, optative, other, uncertain>"}}, one per item, no commentary.

Items:
{items}
"""


def cmd_label(args: argparse.Namespace) -> None:
    out = Path(args.out)
    recs = [json.loads(l) for l in (out / "occurrences.jsonl").open("r", encoding="utf-8") if l.strip()]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    word = meta["word"]

    labels: dict[int, dict[str, str]] = {}
    for r in recs:
        lab, cue = heuristic_label(r)
        labels[int(r["id"])] = {"heuristic": lab, "heuristic_cue": cue, "llm": "", "external": ""}

    if args.labels_in:
        n_ext = 0
        with open(args.labels_in, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    i = int(row["id"])
                except (KeyError, ValueError):
                    continue
                s = (row.get("sense") or row.get("label") or "").strip().lower()
                if i in labels and s in SENSES:
                    labels[i]["external"] = s
                    n_ext += 1
        print(f"[labels-in] merged {n_ext} external labels")

    if args.anthropic:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            raise SystemExit("pip install anthropic (and set ANTHROPIC_API_KEY)")
        client = anthropic.Anthropic()
        todo = [r for r in recs if not labels[int(r["id"])]["external"]]
        if args.limit:
            todo = todo[: args.limit]
        print(f"[anthropic] labeling {len(todo)} occurrences with {args.anthropic_model}, batch={args.batch_size}")
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            items = "\n".join(
                f'{r["id"]}: {r["context_marked"].replace(chr(10), " ")}' for r in batch
            )
            prompt = LABEL_PROMPT.format(word=word, items=items)
            got: dict[int, str] = {}
            for attempt in range(4):
                try:
                    resp = client.messages.create(
                        model=args.anthropic_model,
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(getattr(b, "text", "") for b in resp.content)
                    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                    arr = json.loads(text)
                    for obj in arr:
                        i = int(obj["id"])
                        s = str(obj["sense"]).strip().lower()
                        if s in SENSES:
                            got[i] = s
                    break
                except Exception as exc:  # network / parse
                    wait = 2.0 * (attempt + 1)
                    print(f"[anthropic] batch {start} attempt {attempt+1} failed: {exc}; retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
            for r in batch:
                i = int(r["id"])
                if i in got:
                    labels[i]["llm"] = got[i]
            if (start // args.batch_size) % 10 == 0:
                print(f"[anthropic] {min(start + args.batch_size, len(todo))}/{len(todo)}")

    # final = external > llm > heuristic
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "final", "source", "heuristic", "heuristic_cue", "llm", "external", "context_marked"])
        counts: dict[str, int] = {}
        for r in recs:
            i = int(r["id"])
            L = labels[i]
            if L["external"]:
                final, src = L["external"], "external"
            elif L["llm"]:
                final, src = L["llm"], "llm"
            else:
                final, src = L["heuristic"], "heuristic"
            counts[final] = counts.get(final, 0) + 1
            w.writerow([i, final, src, L["heuristic"], L["heuristic_cue"], L["llm"], L["external"], r["context_marked"].replace("\n", " ")])
    print("[labels] final sense counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if args.anthropic:
        agree = sum(1 for i, L in labels.items() if L["llm"] and L["llm"] == L["heuristic"])
        n = sum(1 for L in labels.values() if L["llm"])
        if n:
            print(f"[labels] heuristic/LLM agreement: {agree}/{n} = {agree/n:.2f}")
    (out / "to_label.jsonl").write_text(
        "\n".join(json.dumps({"id": int(r["id"]), "context_marked": r["context_marked"]}, ensure_ascii=False) for r in recs) + "\n",
        encoding="utf-8",
    )
    print(f"[out] {out/'labels.csv'}  (to_label.jsonl written for external labeling; merge with --labels-in)")


# =============================================================================
# analyze
# =============================================================================


def cmd_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    Z = np.load(out / "hidden.npz")
    recs = [json.loads(l) for l in (out / "occurrences.jsonl").open("r", encoding="utf-8") if l.strip()]
    lab_rows = list(csv.DictReader((out / "labels.csv").open("r", encoding="utf-8", newline="")))
    label_of = {int(r["id"]): r["final"] for r in lab_rows}
    labels = np.asarray([label_of.get(int(r["id"]), "uncertain") for r in recs])
    docs = np.asarray([int(r["document_index"]) for r in recs])
    tok_ids = np.asarray([int(r["token_id"]) for r in recs])
    variant_ids: dict[str, int] = meta["variant_ids"]
    main_tid = variant_ids[meta["main_variant"]]
    rng = np.random.default_rng(args.seed)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"Sense-selectivity report: word={meta['word']!r}  axes={meta['axes']}  offsets={meta['offsets']}")
    emit(f"occurrences={len(recs)} docs={len(np.unique(docs))} background={int(Z['bg_doc'].shape[0])}")
    cnt = {s: int((labels == s).sum()) for s in SENSES}
    emit(f"label counts: {cnt}")
    emit("=" * 78)

    contrasts = [("deontic", "epistemic"), ("deontic", "month"), ("epistemic", "month")]
    if args.min_per_class:
        contrasts = [(a, b) for a, b in contrasts if cnt[a] >= args.min_per_class and cnt[b] >= args.min_per_class]
    if not contrasts:
        emit(f"[warn] no sense pair has >= {args.min_per_class} examples each; only token-vs-background reported")

    def score_dir(H: np.ndarray, v: np.ndarray) -> np.ndarray:
        return H.astype(np.float32) @ v.astype(np.float32)

    def report_direction(
        name: str, s_occ: np.ndarray, s_bg: np.ndarray | None, valid: np.ndarray,
    ) -> dict[str, float]:
        res: dict[str, float] = {}
        if s_bg is not None and len(s_bg):
            res["tok_vs_bg"] = auc(s_occ[valid], s_bg)
        for a, b in contrasts:
            ia = valid & (labels == a)
            ib = valid & (labels == b)
            key = f"{a[:2]}-{b[:2]}"
            res[key] = auc(s_occ[ia], s_occ[ib])
            res[key + "_d"] = cohens_d(s_occ[ia], s_occ[ib])
            if args.bootstrap and name in args.ci_for:
                lo, hi = cluster_bootstrap_auc(s_occ, ia, ib, docs, args.bootstrap, rng)
                res[key + "_lo"], res[key + "_hi"] = lo, hi
        return res

    def fmt_row(name: str, res: dict[str, float]) -> str:
        parts = [f"{name:<26}"]
        parts.append(f"{res.get('tok_vs_bg', math.nan):>7.3f}")
        for a, b in contrasts:
            key = f"{a[:2]}-{b[:2]}"
            v = res.get(key, math.nan)
            ci = f" [{res[key+'_lo']:.2f},{res[key+'_hi']:.2f}]" if key + "_lo" in res else ""
            parts.append(f"{v:>6.3f}{ci:<13} d={res.get(key+'_d', math.nan):+.2f}")
        return "  ".join(parts)

    header = f"{'direction':<26}  {'tok/bg':>7}  " + "  ".join(f"{a[:2]}-vs-{b[:2]} AUC{'':<15}" for a, b in contrasts)

    for name in meta["axes"]:
        layer, rank = parse_axis(name)
        v_axis = Z[f"axis_{name}"].astype(np.float32)
        H_bg = Z[f"bg_h_L{layer:02d}"].astype(np.float32)
        rows_by_tid = {tid: Z[f"row_L{layer:02d}_{tid}"].astype(np.float32) for tid in variant_ids.values()}
        # normalize rows for comparability of cosine-type numbers (AUC is scale-free anyway)
        rows_unit = {tid: v / max(np.linalg.norm(v), 1e-12) for tid, v in rows_by_tid.items()}

        V_bank = None
        if args.directions_dir:
            try:
                V_bank = load_bank_V(Path(args.directions_dir), layer)
            except Exception as exc:
                emit(f"[warn] bank for layer {layer} unavailable ({exc}); skipping 64-axis control")

        for o in meta["offsets"]:
            H = Z[f"h_L{layer:02d}_o{o}"].astype(np.float32)
            valid = Z[f"valid_o{o}"].astype(bool) & ~np.isnan(H[:, 0])
            emit()
            emit("-" * 78)
            emit(f"{name}  offset {o}  (residual {o} token(s) after the target; n_valid={int(valid.sum())})")
            emit("-" * 78)
            emit("AUC = P(score of first class > score of second class); 0.5 = no separation. Sign of the SVD axis is")
            emit("arbitrary, so read |AUC-0.5|; the row/lens directions have a meaningful sign (higher = more 'may').")
            emit(header)

            # 1. SVD axis
            s_axis = score_dir(H, v_axis)
            s_axis_bg = score_dir(H_bg, v_axis)
            r_axis = report_direction("svd_axis", s_axis, s_axis_bg, valid)
            emit(fmt_row(f"SVD axis {name}", r_axis))

            # 2. J-lens row, own token
            s_row_own = np.array([H[i] @ rows_unit[int(t)] for i, t in enumerate(tok_ids)], dtype=np.float32)
            s_row_own_bg = score_dir(H_bg, rows_unit[main_tid])
            r_row_own = report_direction("row_own", s_row_own, s_row_own_bg, valid)
            emit(fmt_row("J-lens row (own token)", r_row_own))

            # 3. J-lens row, canonical ' may'
            s_row_main = score_dir(H, rows_unit[main_tid])
            r_row_main = report_direction("row_main", s_row_main, s_row_own_bg, valid)
            emit(fmt_row(f"J-lens row ({meta['main_variant']!r})", r_row_main))

            # 4. exact normalized lens logit, own token
            s_ll = np.array([Z[f"lenslogit_L{layer:02d}_o{o}_{int(t)}"][i] for i, t in enumerate(tok_ids)], dtype=np.float32)
            s_ll_bg = Z[f"bg_lenslogit_L{layer:02d}_{main_tid}"].astype(np.float32)
            r_ll = report_direction("lens_logit", s_ll, s_ll_bg, valid)
            emit(fmt_row("lens logit (own token)", r_ll))

            # 5. contrast-mean, cross-validated by document (linear ceiling proxy) — primary contrast only
            if contrasts:
                a, b = contrasts[0]
                ia = valid & (labels == a)
                ib = valid & (labels == b)
                if ia.sum() >= 5 and ib.sum() >= 5:
                    uniq = np.unique(docs[ia | ib])
                    rng.shuffle(uniq)
                    fold_of = {d: k % 2 for k, d in enumerate(uniq)}
                    fold = np.array([fold_of.get(d, -1) for d in docs])
                    s_cm = np.full(len(H), np.nan, dtype=np.float32)
                    cos_axis, cos_row = [], []
                    for k in (0, 1):
                        tr = (fold != k) & (fold >= 0)
                        te = (fold == k)
                        if (tr & ia).sum() < 3 or (tr & ib).sum() < 3:
                            continue
                        w = H[tr & ia].mean(0) - H[tr & ib].mean(0)
                        w = w / max(np.linalg.norm(w), 1e-12)
                        s_cm[te] = H[te] @ w
                        cos_axis.append(float(abs(w @ v_axis)))
                        cos_row.append(float(abs(w @ rows_unit[main_tid])))
                    okm = ~np.isnan(s_cm)
                    val = auc(s_cm[okm & ia], s_cm[okm & ib])
                    emit(f"{'contrast-mean (2-fold CV)':<26}  {'':>7}  {val:>6.3f}{'':<13} <- linear ceiling for {a} vs {b}")
                    if cos_axis:
                        emit(f"    |cos(contrast-mean, SVD axis)| = {np.mean(cos_axis):.3f}   |cos(contrast-mean, J-lens row)| = {np.mean(cos_row):.3f}")

            # 6. random-direction null
            if contrasts and args.n_random > 0:
                a, b = contrasts[0]
                ia = valid & (labels == a)
                ib = valid & (labels == b)
                seps = []
                for _ in range(args.n_random):
                    g = rng.standard_normal(H.shape[1]).astype(np.float32)
                    g /= np.linalg.norm(g)
                    s = H @ g
                    seps.append(abs(auc(s[ia], s[ib]) - 0.5))
                seps_arr = np.asarray(seps)
                sep_axis = abs(r_axis.get(f"{a[:2]}-{b[:2]}", 0.5) - 0.5)
                emit(f"{'random directions':<26}  |AUC-0.5| for {a} vs {b}: median={np.median(seps_arr):.3f}  95th={np.percentile(seps_arr, 95):.3f}  max={seps_arr.max():.3f}  (n={args.n_random});  SVD axis: {sep_axis:.3f}")

            # 7. rank among the layer's k bank axes
            if contrasts and V_bank is not None:
                a, b = contrasts[0]
                ia = valid & (labels == a)
                ib = valid & (labels == b)
                seps = []
                for j in range(V_bank.shape[1]):
                    vj = V_bank[:, j] / max(np.linalg.norm(V_bank[:, j]), 1e-12)
                    s = H @ vj
                    seps.append(abs(auc(s[ia], s[ib]) - 0.5))
                seps_arr = np.asarray(seps)
                order = np.argsort(-seps_arr)
                rank_pos = int(np.where(order == rank - 1)[0][0]) + 1
                best = order[0] + 1
                emit(f"{'bank axes (k=%d)' % V_bank.shape[1]:<26}  this axis ranks {rank_pos}/{V_bank.shape[1]} on {a} vs {b} separation; best = SV{best:02d} ({seps_arr[order[0]]:.3f}), median axis {np.median(seps_arr):.3f}")

            # per-sense means for the axis and the row
            emit(f"{'per-sense mean (SVD axis / row_main)':<26}")
            for s in SENSES:
                m = valid & (labels == s)
                if m.sum():
                    emit(f"    {s:<10} n={int(m.sum()):>4}  axis={s_axis[m].mean():+8.3f}±{s_axis[m].std():.3f}   row={s_row_main[m].mean():+8.3f}±{s_row_main[m].std():.3f}")
            if len(s_axis_bg):
                emit(f"    {'background':<10} n={len(s_axis_bg):>4}  axis={s_axis_bg.mean():+8.3f}±{s_axis_bg.std():.3f}   row={s_row_own_bg.mean():+8.3f}±{s_row_own_bg.std():.3f}")

            if args.plots:
                make_plot(out, name, o, s_axis, s_row_main, labels, valid, s_axis_bg, s_row_own_bg)

            # per-occurrence scores
            with (out / f"scores_{name}_o{o}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "document_index", "sense", "variant", "valid", "svd_axis", "row_own", "row_main", "lens_logit_own", "context_marked"])
                for i, r in enumerate(recs):
                    w.writerow([r["id"], r["document_index"], labels[i], r["variant"], int(valid[i]),
                                f"{s_axis[i]:.4f}", f"{s_row_own[i]:.4f}", f"{s_row_main[i]:.4f}", f"{s_ll[i]:.4f}",
                                r["context_marked"].replace("\n", " ")])

    emit()
    emit("Reading the result: the J-lens row is token-indexed, so it should separate the token from background")
    emit("strongly and the senses from each other weakly. If the SVD axis shows deontic-vs-epistemic AUC far from")
    emit("0.5, above the random-direction 95th percentile and near the contrast-mean ceiling, while the row sits")
    emit("near 0.5, the axis resolves a sense the token dictionary conflates. Offsets 1-2 show how much of the")
    emit("separation waits for the continuation to be read (causal model).")
    (out / "sense_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {out/'sense_report.txt'} and per-occurrence scores_*.csv")


def make_plot(out: Path, name: str, o: int, s_axis, s_row, labels, valid, s_axis_bg, s_row_bg) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable ({exc})", file=sys.stderr)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (title, s, sbg) in zip(axes, (("SVD axis " + name, s_axis, s_axis_bg), ("J-lens row", s_row, s_row_bg))):
        groups = [("background", sbg)] + [(sn, s[valid & (labels == sn)]) for sn in SENSES if (valid & (labels == sn)).sum()]
        data = [g for _, g in groups]
        tick = [f"{n}\n(n={len(g)})" for n, g in groups]
        try:
            ax.boxplot(data, tick_labels=tick, showfliers=False)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(data, labels=tick, showfliers=False)
        ax.set_title(f"{title}, offset {o}")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / f"plot_{name}_o{o}.png", dpi=140)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="GPU: collect activations at target-word occurrences")
    h.add_argument("--axes", nargs="+", required=True, help="e.g. L07_SV39 L08_SV39 (1-based rank)")
    h.add_argument("--word", default="may")
    h.add_argument("--extra-variants", nargs="*", default=[], help="extra surface forms to treat as the target")
    h.add_argument("--directions-dir", required=True, help="scan_svd_r0/directions")
    h.add_argument("--out", required=True)
    h.add_argument("--model", default=DEFAULT_MODEL)
    h.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    h.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    h.add_argument("--dataset", default=DEFAULT_DATASET)
    h.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    h.add_argument("--split", default="train")
    h.add_argument("--input-jsonl", default=None)
    h.add_argument("--shuffle-buffer", type=int, default=10000)
    h.add_argument("--seed", type=int, default=1, help="use a seed different from the scan corpus (0)")
    h.add_argument("--n-docs", type=int, default=6000)
    h.add_argument("--target-occurrences", type=int, default=1500)
    h.add_argument("--max-per-doc", type=int, default=3)
    h.add_argument("--char-cap", type=int, default=20000)
    h.add_argument("--left-ctx", type=int, default=190)
    h.add_argument("--right-ctx", type=int, default=64)
    h.add_argument("--offsets", nargs="+", default=["0"], help="token offsets after the target to record (0 = at it)")
    h.add_argument("--background-per-window", type=int, default=2)
    h.add_argument("--context-tokens", type=int, default=40, help="tokens of left/right text saved for labeling")
    h.set_defaults(func=cmd_harvest)

    l = sub.add_parser("label", help="CPU: assign senses (heuristic; optional Claude API; external CSV)")
    l.add_argument("--out", required=True)
    l.add_argument("--anthropic", action="store_true", help="label with the Anthropic API (needs ANTHROPIC_API_KEY)")
    l.add_argument("--anthropic-model", default="claude-sonnet-4-6", help="set to a model id your API access lists")
    l.add_argument("--batch-size", type=int, default=25)
    l.add_argument("--limit", type=int, default=0, help="label only the first N (for testing)")
    l.add_argument("--labels-in", default=None, help="CSV with columns id,sense from an external labeler")
    l.set_defaults(func=cmd_label)

    a = sub.add_parser("analyze", help="CPU: sense-separation statistics")
    a.add_argument("--out", required=True)
    a.add_argument("--directions-dir", default=None, help="for the 'rank among bank axes' control")
    a.add_argument("--bootstrap", type=int, default=1000, help="document-cluster bootstrap resamples for CIs (0=off)")
    a.add_argument("--ci-for", nargs="*", default=["svd_axis", "row_own"], help="directions that get bootstrap CIs")
    a.add_argument("--n-random", type=int, default=200)
    a.add_argument("--min-per-class", type=int, default=20)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--plots", action="store_true")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
