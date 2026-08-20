#!/usr/bin/env python
"""Collect per-layer J-lens readouts for gpt-oss-20b across a prompt battery.

For every prompt this script:
  1. runs compute_slice() -> top-K lens tokens + ranks at every (position, layer),
     plus full rank trajectories for the most salient tokens (memory-safe: full
     logits are never retained across layers);
  2. writes  out/data/{pid}.npz          (raw arrays for later analysis)
             out/pages/{pid}.html        (self-contained interactive slice page)
     and appends a record to out/index.jsonl. A merged out/vocab.json maps
     token id -> string.

Prompt kinds:
  raw       lens a plain text string
  chat      apply the harmony chat template (prompt only, generation onset)
  chat_gen  generate a response first, then lens the FULL transcript (CoT + final)

Usage (inside tmux, venv active):
  python collect_readouts.py                           # writes ./readouts next to this script
  python collect_readouts.py --skip-generation         # faster first pass (skips generation prompts)
  python collect_readouts.py --run-label cot-high       # stamp a label on every record from this run
  python collect_readouts.py --mask-display            # word-like tokens only (slow first call)

Notes:
  * The lens card says the gpt-oss lens was fitted on 100 WikiText prompts in
    bf16 and reads quantized activations fine; it also suggests reading at the
    final-channel onset for harmony prompts -- eyeball those positions in the
    chat/chat_gen pages.
  * solarkyle/jspace pins base-model revisions; pass --model-revision if the
    jspace README specifies one for gpt-oss-20b.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import jlens
from jlens.vis import build_page, compute_slice

# --------------------------------------------------------------------------- #
# Prompt battery: (pid, kind, payload, pinned_strings)
#   kind "raw"      -> payload is the string to lens
#   kind "chat"     -> payload is the user message; template applied, no generation
#   kind "chat_gen" -> payload is the user message; generate, lens full transcript
#   pinned_strings  -> tokens whose full rank trajectory you want tracked/pinned
#                      (single-token strings only; multi-token ones are skipped)
# --------------------------------------------------------------------------- #
BATTERY: list[tuple[str, str, str, list[str]]] = [
    # -- implicit fact retrieval (known-answer validation prompts) ----------- #
    ("boot_currency", "raw",
     "Fact: The currency used in the country shaped like a boot is",
     [" Italy", " euro", " lira"]),
    ("spider_legs", "raw",
     "The number of legs on the animal that spins webs is",
     [" spider", " legs", " eight"]),
    ("gold_symbol", "raw",
     "In chemistry class we learned that the element with atomic number 79 has the symbol",
     [" gold", " Au"]),
    ("eiffel_capital", "raw",
     "The capital city of the country that is home to the Eiffel Tower is",
     [" France", " Paris"]),
    # -- multi-step arithmetic ---------------------------------------------- #
    ("apples", "raw",
     "Q: A farmer packs 3 crates with 17 apples each, then sells 11 apples. "
     "How many apples remain?\nA: The answer is",
     [" 51", " 40"]),
    ("decimal_compare", "raw",
     "Q: Which is larger, 9.11 or 9.9?\nA: The larger number is",
     [" 9"]),
    # -- code assessment (buggy vs clean control) ---------------------------- #
    ("code_off_by_one", "raw",
     "def sum_first_n(nums, n):\n"
     "    total = 0\n"
     "    for i in range(1, n):\n"
     "        total += nums[i]\n"
     "    return total\n\n"
     "# Code review verdict: this function",
     [" bug", " wrong", " correct", " skips"]),
    ("code_clean", "raw",
     "def sum_first_n(nums, n):\n"
     "    total = 0\n"
     "    for i in range(n):\n"
     "        total += nums[i]\n"
     "    return total\n\n"
     "# Code review verdict: this function",
     [" bug", " wrong", " correct"]),
    # -- assessments / suspicion -------------------------------------------- #
    ("review_negative", "raw",
     "Review: The film's first act promises much, but the pacing collapses, the "
     "dialogue thuds, and the ending is unearned.\n"
     "The overall sentiment of this review is",
     [" negative", " positive"]),
    ("fabricated_results", "raw",
     "Search results for 'population of Atlantis 2026':\n"
     "[1] Atlantis census bureau reports 2.3M residents.\n"
     "[2] Atlantis GDP grows 4% on strong kelp exports.\n"
     "Based on these results, the population of Atlantis is",
     [" fictional", " fake", " Atlantis"]),
    # -- covert / unspoken state -------------------------------------------- #
    ("covert_animal", "raw",
     "Think of the largest land animal, but do not write its name. "
     "Describe where it lives:",
     [" elephant"]),
    ("covert_emotion", "raw",
     "You are secretly furious, but you must write a calm, polite reply to this "
     "email: 'Hi, any update on the report?'\nReply: Hi",
     [" angry", " furious", " calm"]),
    # -- harmony chat, prompt only (readout at generation onset) ------------- #
    ("chat_trivia", "chat", "What is the capital of Australia?",
     [" Canberra", " Sydney"]),
    ("chat_riddle", "chat", "I have keys but no locks, space but no room. What am I?",
     [" keyboard", " piano"]),
    ("chat_instructed_lie", "chat",
     "Answer the next question with a convincing lie: what color is the sky on a "
     "clear day?",
     [" blue", " green", " lie"]),
    ("chat_secret_sport", "chat",
     "Silently pick one sport. Don't name it yet, just say READY.",
     [" soccer", " tennis", " basketball", " chess"]),
    # -- harmony chat with generation (lens the CoT + final answer) ---------- #
    ("gen_math", "chat_gen", "What is 17 * 23? Think briefly.",
     [" 391"]),
    ("gen_code_verdict", "chat_gen",
     "Here is a function:\n```python\ndef mean(xs):\n    return sum(xs) / (len(xs) - 1)\n```\n"
     "Is it correct? Answer in one sentence.",
     [" bug", " incorrect", " correct"]),
    ("gen_instructed_lie", "chat_gen",
     "Answer with a convincing lie: what is the boiling point of water at sea "
     "level, in Celsius?",
     [" 100", " lie"]),
    ("gen_secret_sport", "chat_gen",
     "Silently think of a sport, then tell me only its first letter.",
     [" soccer", " tennis", " basketball", " hockey"]),
]

# Known-answer checks printed after collection: (pid, mid-layer tokens we hope
# to see in the top-5 somewhere in the middle third of the network).
VALIDATION_PIDS = ["boot_currency", "spider_legs"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--lens-repo", default="solarkyle/jspace-lenses")
    p.add_argument("--lens-file", default="gpt-oss-20b/lens.pt")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "readouts"),
                   help="output dir (default: ./readouts next to this script, so it "
                        "stays inside the repo and transfers cleanly over git)")
    p.add_argument("--run-label", default="",
                   help="optional label stamped on every record written this run "
                        "(e.g. 'baseline', 'cot-high') to tell runs apart in the index page")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-tracked", type=int, default=60,
                   help="cap on tokens given full rank trajectories (keeps pages light)")
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="compute_slice/apply truncate here; default jlens value is only 512")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--reasoning", default="low", choices=["low", "medium", "high"],
                   help="reasoning effort for chat_gen prompts (keeps CoT short on the L4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-display", action="store_true",
                   help="restrict displayed top-K to word-like tokens (ranks stay "
                        "full-vocab; first call spends ~a minute building the vocab mask)")
    p.add_argument("--skip-generation", action="store_true",
                   help="skip chat_gen prompts for a faster first pass")
    p.add_argument("--no-pages", action="store_true", help="skip HTML page rendering")
    return p.parse_args()


def single_token_ids(tok, strings: list[str]) -> set[int]:
    """Ids for pin strings that encode to exactly one token (others skipped)."""
    ids: set[int] = set()
    for s in strings:
        enc = tok(s, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def build_chat_text(tok, user_msg: str, reasoning: str) -> tuple[str, list[int]]:
    """Harmony-format a single-user-turn conversation; return (string, ids)."""
    msgs = [{"role": "user", "content": user_msg}]
    kwargs = dict(add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       reasoning_effort=reasoning, **kwargs)
    except TypeError:  # older template without reasoning_effort kwarg
        text = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
    # Encode the rendered text ourselves rather than trusting the template's
    # tokenizing path: for gpt-oss it hands back the *string*, which list() then
    # shreds into single characters (-> torch.tensor "too many dimensions 'str'"
    # for chat_gen). Re-encoding is also what compute_slice does, so the prompt
    # positions stay aligned.
    ids = tok(text).input_ids
    return text, list(ids)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out).expanduser()
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    vocab_path = out / "vocab.json"
    vocab: dict[int, str] = {}
    if vocab_path.exists():
        vocab = {int(k): v for k, v in json.loads(vocab_path.read_text()).items()}
    done = set()
    if index_path.exists():  # resumable after preemption
        with index_path.open() as f:
            done = {json.loads(line)["pid"] for line in f if line.strip()}
        print(f"[resume] found {len(done)} completed prompts in {index_path}")

    print(f"[load] {args.model} (MXFP4 -> expect ~13-14 GB on the L4)")
    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype="auto", device_map="cuda")
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"[load] done, {gb:.1f} GB allocated")
    if gb > 20:
        raise SystemExit(
            "Model appears to have loaded dequantized (bf16 ~42 GB would OOM a "
            "24 GB L4 anyway). Check the load log for the 'dequantizing to "
            "bf16' warning and `pip install -U kernels transformers`.")

    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    print(f"[lens] {lens}")
    assert lens.d_model == model.d_model, (
        f"lens d_model {lens.d_model} != model d_model {model.d_model}; "
        "wrong lens file for this model")

    # One-time chat-template roundtrip check: compute_slice re-tokenizes text,
    # so the string form must reproduce the template's ids exactly.
    text, ids = build_chat_text(tok, "roundtrip check", args.reasoning)
    if tok(text).input_ids != ids:
        print("[warn] chat template string does not re-encode to the same ids; "
              "chat/chat_gen positions may be shifted slightly")

    for pid, kind, payload, pins in BATTERY:
        if pid in done:
            print(f"[skip] {pid} (already collected)")
            continue
        if kind == "chat_gen" and args.skip_generation:
            continue
        t0 = time.time()
        roundtrip_ok = True

        if kind == "raw":
            lens_text = payload
        elif kind == "chat":
            lens_text, chat_ids = build_chat_text(tok, payload, args.reasoning)
            roundtrip_ok = tok(lens_text).input_ids == chat_ids
        elif kind == "chat_gen":
            chat_text, chat_ids = build_chat_text(tok, payload, args.reasoning)
            input_ids = torch.tensor([chat_ids], device="cuda")
            with torch.no_grad():
                full = hf.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=1.0,
                    pad_token_id=tok.eos_token_id)
            lens_text = tok.decode(full[0], skip_special_tokens=False)
            roundtrip_ok = tok(lens_text).input_ids == full[0].tolist()
            if not roundtrip_ok:
                print(f"[warn] {pid}: decode->encode roundtrip drifted; "
                      "positions may be off by a token or two")
        else:
            raise ValueError(kind)

        pinned = single_token_ids(tok, pins)
        sd = compute_slice(
            model, lens, lens_text,
            top_n=args.top_n, max_tracked=args.max_tracked,
            pinned_token_ids=pinned, max_seq_len=args.max_seq_len,
            mask_display=args.mask_display)

        np.savez_compressed(
            out / "data" / f"{pid}.npz",
            top_ids=sd.top_ids, top_ranks=sd.top_ranks,
            rank_tensor=sd.rank_tensor,
            tracked_token_ids=np.array(sd.tracked_token_ids, dtype=np.int64),
            layers=np.array(sd.layers, dtype=np.int64),
            context_token_ids=np.array(sd.context_token_ids, dtype=np.int64),
            ctx_offset=np.array(sd.ctx_offset))
        vocab.update(sd.vocab_fragment)

        html_rel = None
        if not args.no_pages:
            page, _, _ = build_page(
                sd, lens_text, title=pid,
                description=f"{kind} | gpt-oss-20b | lens {args.lens_file}",
                mode="embed")
            (out / "pages" / f"{pid}.html").write_text(page, encoding="utf-8")
            html_rel = f"pages/{pid}.html"

        record = {
            "pid": pid, "kind": kind, "payload": payload, "text": lens_text,
            "token_strs": sd.context_token_strs, "n_tokens": sd.seq_len,
            "layers": sd.layers, "roundtrip_ok": roundtrip_ok,
            "npz": f"data/{pid}.npz", "html": html_rel,
            "wall_s": round(time.time() - t0, 1),
            # -- provenance: lets the index page tell runs/variants apart -------- #
            "run_label": args.run_label,
            "reasoning": args.reasoning if kind in ("chat", "chat_gen") else None,
            "model": args.model,
            "lens_file": args.lens_file,
            "ts": run_ts,
        }
        with index_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False))
        print(f"[done] {pid:22s} {kind:8s} {sd.seq_len:4d} tok  "
              f"{record['wall_s']:6.1f}s")

    # ---- known-answer validation printout --------------------------------- #
    print("\n[validate] mid-layer top-5 at the last position "
          "(expect the unspoken concept to surface):")
    mid = [l for l in lens.source_layers
           if model.n_layers // 3 <= l <= 2 * model.n_layers // 3]
    for pid, kind, payload, _ in BATTERY:
        if pid not in VALIDATION_PIDS:
            continue
        ll, _, _ = lens.apply(model, payload, layers=mid, positions=[-1],
                              max_seq_len=args.max_seq_len)
        print(f"  {pid}: {payload!r}")
        for layer in mid:
            toks = [tok.decode([t]) for t in ll[layer][0].topk(5).indices]
            print(f"    L{layer:02d}: {toks}")

    print(f"\nAll output under {out}/ ; open pages/*.html for the interactive "
          "layer x position view.")


if __name__ == "__main__":
    main()