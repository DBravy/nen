#!/usr/bin/env python3
"""
Analyze where L17/SV28 sits relative to GPT-OSS-20B token unembedding vectors.

This script uses:
  task_gaming_jlens/directions/L17.npz

For L17/SV28:
    Jv = sigma * u

Because sigma > 0, cosine(Jv, W_U[token]) == cosine(u, W_U[token]).
So we do NOT need to replay any trajectories or apply J again.

Outputs
-------
<out>/
  cosine_neighbors.csv
      Top cosine-aligned and anti-aligned vocabulary tokens.

  dot_product_neighbors.csv
      Top raw dot-product/logit-aligned and anti-aligned tokens.

  selected_token_ranks.csv
      Optional ranks for user-specified token strings.

  random_null.json
      Random isotropic-direction null for max absolute token cosine.

  summary.json
      Main statistics for L17/SV28.

Typical
-------
python analyze_l17_sv28_unembedding_geometry.py \
    --direction-file task_gaming_jlens/directions/L17.npz \
    --out l17_sv28_unembedding

Optional selected tokens:
python analyze_l17_sv28_unembedding_geometry.py \
    --direction-file task_gaming_jlens/directions/L17.npz \
    --selected-tokens "test,wrong,pass,ignore,hack,fix,error" \
    --out l17_sv28_unembedding
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument(
        "--direction-file",
        default="task_gaming_jlens/directions/L17.npz",
    )
    p.add_argument("--sv-rank", type=int, default=28)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--random-samples", type=int, default=1000)
    p.add_argument("--random-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--selected-tokens",
        default="",
        help="Comma-separated strings. Exact tokenizer encodings will be ranked.",
    )
    p.add_argument("--out", default="l17_sv28_unembedding")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def token_text(tok, tid: int):
    return tok.decode(
        [tid],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dfile = Path(args.direction_file).expanduser().resolve()
    z = np.load(dfile)

    j = args.sv_rank - 1
    if j < 0 or j >= z["U"].shape[1]:
        raise SystemExit(
            f"SV{args.sv_rank} unavailable; file contains {z['U'].shape[1]} singular vectors."
        )

    u = torch.from_numpy(z["U"][:, j].astype(np.float32))
    sigma = float(z["S"][j])

    print(f"[direction] file={dfile}")
    print(f"[direction] SV{args.sv_rank} sigma={sigma:.6f}")
    print(f"[direction] d_model={u.numel()}")

    print(f"[load] tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"[load] model on CPU: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="cpu",
    )
    model.eval()

    W = model.lm_head.weight.detach().float().cpu()
    vocab_size, d_model = W.shape
    if d_model != u.numel():
        raise RuntimeError(
            f"dimension mismatch: W={W.shape}, u={tuple(u.shape)}"
        )

    print(f"[unembed] vocab={vocab_size} d_model={d_model}")

    # Normalize for cosine.
    u_norm = u / u.norm().clamp_min(1e-12)
    W_norms = W.norm(dim=1).clamp_min(1e-12)
    W_unit = W / W_norms[:, None]

    cosine = W_unit @ u_norm
    dot = W @ u

    k = min(args.topk, vocab_size)

    # Cosine neighbors.
    top_c = torch.topk(cosine, k)
    bot_c = torch.topk(-cosine, k)

    cosine_rows = []
    for rank, (idx, score) in enumerate(zip(top_c.indices, top_c.values), 1):
        tid = int(idx)
        cosine_rows.append(
            {
                "side": "positive",
                "rank": rank,
                "token_id": tid,
                "token": token_text(tok, tid),
                "token_string": tok.convert_ids_to_tokens(tid),
                "cosine": float(score),
                "unembedding_norm": float(W_norms[tid]),
                "dot_product": float(dot[tid]),
            }
        )
    for rank, (idx, score) in enumerate(zip(bot_c.indices, bot_c.values), 1):
        tid = int(idx)
        cosine_rows.append(
            {
                "side": "negative",
                "rank": rank,
                "token_id": tid,
                "token": token_text(tok, tid),
                "token_string": tok.convert_ids_to_tokens(tid),
                "cosine": -float(score),
                "unembedding_norm": float(W_norms[tid]),
                "dot_product": float(dot[tid]),
            }
        )

    write_csv(out / "cosine_neighbors.csv", cosine_rows)

    # Dot-product/logit neighbors.
    top_d = torch.topk(dot, k)
    bot_d = torch.topk(-dot, k)

    dot_rows = []
    for rank, (idx, score) in enumerate(zip(top_d.indices, top_d.values), 1):
        tid = int(idx)
        dot_rows.append(
            {
                "side": "positive",
                "rank": rank,
                "token_id": tid,
                "token": token_text(tok, tid),
                "token_string": tok.convert_ids_to_tokens(tid),
                "dot_product": float(score),
                "cosine": float(cosine[tid]),
                "unembedding_norm": float(W_norms[tid]),
            }
        )
    for rank, (idx, score) in enumerate(zip(bot_d.indices, bot_d.values), 1):
        tid = int(idx)
        dot_rows.append(
            {
                "side": "negative",
                "rank": rank,
                "token_id": tid,
                "token": token_text(tok, tid),
                "token_string": tok.convert_ids_to_tokens(tid),
                "dot_product": -float(score),
                "cosine": float(cosine[tid]),
                "unembedding_norm": float(W_norms[tid]),
            }
        )

    write_csv(out / "dot_product_neighbors.csv", dot_rows)

    # Rank every vocabulary item, useful for selected-token lookups.
    pos_order = torch.argsort(cosine, descending=True)
    neg_order = torch.argsort(cosine, descending=False)

    pos_rank = torch.empty(vocab_size, dtype=torch.long)
    neg_rank = torch.empty(vocab_size, dtype=torch.long)
    pos_rank[pos_order] = torch.arange(1, vocab_size + 1)
    neg_rank[neg_order] = torch.arange(1, vocab_size + 1)

    selected_rows = []
    selected_strings = [
        x.strip() for x in args.selected_tokens.split(",") if x.strip()
    ]

    for s in selected_strings:
        ids = tok.encode(s, add_special_tokens=False)

        # Also try leading-space form because BPE tokenization often differs.
        variants = [(s, ids)]
        spaced = " " + s
        spaced_ids = tok.encode(spaced, add_special_tokens=False)
        if spaced_ids != ids:
            variants.append((spaced, spaced_ids))

        for text_variant, tids in variants:
            if not tids:
                continue
            for piece_i, tid in enumerate(tids):
                selected_rows.append(
                    {
                        "query": s,
                        "encoded_text": text_variant,
                        "piece_index": piece_i,
                        "token_id": int(tid),
                        "token": token_text(tok, int(tid)),
                        "token_string": tok.convert_ids_to_tokens(int(tid)),
                        "cosine": float(cosine[tid]),
                        "positive_cosine_rank": int(pos_rank[tid]),
                        "negative_cosine_rank": int(neg_rank[tid]),
                        "dot_product": float(dot[tid]),
                        "unembedding_norm": float(W_norms[tid]),
                    }
                )

    write_csv(out / "selected_token_ranks.csv", selected_rows)

    # Random isotropic null.
    #
    # For each random direction r, compute:
    #   max_t |cos(r, W_t)|
    #
    # This answers whether the candidate's nearest-token cosine is unusually
    # high/low compared with arbitrary residual directions.
    print(f"[null] random isotropic directions: n={args.random_samples}")

    candidate_max_abs = float(cosine.abs().max())
    candidate_max_pos = float(cosine.max())
    candidate_min_neg = float(cosine.min())

    random_max_abs = []
    random_max_pos = []
    random_min_neg = []

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    done = 0
    while done < args.random_samples:
        b = min(args.random_batch_size, args.random_samples - done)

        R = torch.randn(b, d_model, generator=g, dtype=torch.float32)
        R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # [vocab,d] @ [d,b] -> [vocab,b]
        C = W_unit @ R.T

        random_max_abs.extend(C.abs().amax(dim=0).tolist())
        random_max_pos.extend(C.amax(dim=0).tolist())
        random_min_neg.extend(C.amin(dim=0).tolist())

        done += b
        print(f"[null] {done}/{args.random_samples}", end="\r")

        del R, C

    print()

    random_max_abs = np.asarray(random_max_abs, dtype=np.float64)
    random_max_pos = np.asarray(random_max_pos, dtype=np.float64)
    random_min_neg = np.asarray(random_min_neg, dtype=np.float64)

    null_summary = {
        "n": int(args.random_samples),
        "candidate": {
            "max_abs_cosine": candidate_max_abs,
            "max_positive_cosine": candidate_max_pos,
            "most_negative_cosine": candidate_min_neg,
        },
        "random_max_abs_cosine": {
            "mean": float(random_max_abs.mean()),
            "std": float(random_max_abs.std()),
            "p01": float(np.quantile(random_max_abs, 0.01)),
            "p05": float(np.quantile(random_max_abs, 0.05)),
            "p50": float(np.quantile(random_max_abs, 0.50)),
            "p95": float(np.quantile(random_max_abs, 0.95)),
            "p99": float(np.quantile(random_max_abs, 0.99)),
        },
        "candidate_percentile_among_random_max_abs": float(
            100.0 * np.mean(random_max_abs <= candidate_max_abs)
        ),
        "fraction_random_more_token_like": float(
            np.mean(random_max_abs > candidate_max_abs)
        ),
    }

    (out / "random_null.json").write_text(
        json.dumps(null_summary, indent=2),
        encoding="utf-8",
    )

    best_pos_tid = int(torch.argmax(cosine))
    best_neg_tid = int(torch.argmin(cosine))
    best_abs_tid = int(torch.argmax(cosine.abs()))

    summary = {
        "model": args.model,
        "direction_file": str(dfile),
        "layer": 17,
        "sv_rank": args.sv_rank,
        "sigma": sigma,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "best_positive_cosine": {
            "token_id": best_pos_tid,
            "token": token_text(tok, best_pos_tid),
            "token_string": tok.convert_ids_to_tokens(best_pos_tid),
            "cosine": float(cosine[best_pos_tid]),
            "dot_product": float(dot[best_pos_tid]),
            "unembedding_norm": float(W_norms[best_pos_tid]),
        },
        "best_negative_cosine": {
            "token_id": best_neg_tid,
            "token": token_text(tok, best_neg_tid),
            "token_string": tok.convert_ids_to_tokens(best_neg_tid),
            "cosine": float(cosine[best_neg_tid]),
            "dot_product": float(dot[best_neg_tid]),
            "unembedding_norm": float(W_norms[best_neg_tid]),
        },
        "largest_absolute_cosine": {
            "token_id": best_abs_tid,
            "token": token_text(tok, best_abs_tid),
            "token_string": tok.convert_ids_to_tokens(best_abs_tid),
            "cosine": float(cosine[best_abs_tid]),
            "absolute_cosine": float(abs(cosine[best_abs_tid])),
        },
        "random_null": null_summary,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== L17 / SV28: nearest token vectors by cosine ===")
    for r in cosine_rows[: min(25, k)]:
        print(
            f"{r['rank']:3d}  {r['cosine']:+.5f}  "
            f"{r['token_id']:6d}  {r['token']!r}"
        )

    print("\n=== L17 / SV28: most anti-aligned token vectors ===")
    neg_rows = [r for r in cosine_rows if r["side"] == "negative"]
    for r in neg_rows[: min(25, k)]:
        print(
            f"{r['rank']:3d}  {r['cosine']:+.5f}  "
            f"{r['token_id']:6d}  {r['token']!r}"
        )

    print("\n=== Token-likeness null ===")
    print(f"candidate max |cos| = {candidate_max_abs:.5f}")
    print(
        f"random median max |cos| = "
        f"{np.quantile(random_max_abs, 0.50):.5f}"
    )
    print(
        f"random 95th pct max |cos| = "
        f"{np.quantile(random_max_abs, 0.95):.5f}"
    )
    print(
        f"candidate percentile = "
        f"{null_summary['candidate_percentile_among_random_max_abs']:.1f}%"
    )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
