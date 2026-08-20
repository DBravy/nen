#!/usr/bin/env python
"""Freeze diverse natural-language corpora for J-lens SVD exploration.

This script DOES NOT load gpt-oss weights or compute J-lens readouts.  It only
samples and freezes text into the minimal ``index.jsonl`` format consumed by
``explore_jlens_svd_concepts_v2.py``.  The SVD explorer then replays these exact
texts through gpt-oss to recover activations.

The two default corpora deliberately answer different questions:

1. FineWeb ``sample-10BT``
   Random contiguous token windows from natural web documents.  Use the SVD
   explorer with ``--position-mode all`` to ask what J singular channels are
   occupied across arbitrary natural text.

2. WildChat-1M
   First English user messages from real conversations, rendered through the
   gpt-oss Harmony chat template with an assistant-generation prompt appended.
   No historical ChatGPT assistant response is included.  Use the SVD explorer
   with ``--position-mode last`` to ask what channels are present at generation
   onset across diverse real user requests.

Scientific-design choices
-------------------------
* Discovery and replication samples are frozen separately before looking at
  results.  Defaults are 400 + 400 examples per corpus.
* FineWeb windows begin at uniformly random token offsets; we do not always use
  document beginnings.
* FineWeb and WildChat are kept separate rather than pooled because they should
  be analyzed at different positions (all positions vs generation onset).
* The public WildChat dataset is sampled at the conversation level, using only
  the first user turn.  By default we keep English, non-redacted conversations.
* WildChat geographic/IP/header metadata is deliberately not copied to disk.
* Dataset revisions and tokenizer/model revision are resolved to commit SHAs
  when possible and written to ``meta.json`` for provenance.
* Existing frozen samples are not silently replaced.  Pass ``--overwrite`` to
  deliberately resample.

Typical usage
-------------
    python collect_diverse_corpora.py

Then analyze the discovery sets:

    python explore_jlens_svd_concepts_v2.py \
        --readouts diverse_corpora/fineweb_discovery \
        --position-mode all --out svd_fineweb_discovery

    python explore_jlens_svd_concepts_v2.py \
        --readouts diverse_corpora/wildchat_discovery \
        --position-mode last --out svd_wildchat_discovery

If a candidate survives, repeat unchanged on the held-out sets:

    python explore_jlens_svd_concepts_v2.py \
        --readouts diverse_corpora/fineweb_replication \
        --position-mode all --out svd_fineweb_replication

    python explore_jlens_svd_concepts_v2.py \
        --readouts diverse_corpora/wildchat_replication \
        --position-mode last --out svd_wildchat_replication

Dependencies
------------
    pip install -U datasets huggingface_hub transformers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing `datasets`. Install with: pip install -U datasets huggingface_hub"
    ) from exc

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing `huggingface_hub`. Install with: pip install -U huggingface_hub"
    ) from exc

try:
    import transformers
except ImportError as exc:
    raise SystemExit("Missing `transformers`. Install with: pip install -U transformers") from exc


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--out", default=str(here / "diverse_corpora"))
    p.add_argument("--corpora", default="fineweb,wildchat",
                   help="comma-separated subset of: fineweb,wildchat")
    p.add_argument("--n-discovery", type=int, default=400)
    p.add_argument("--n-replication", type=int, default=400)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--overwrite", action="store_true",
                   help="delete/recreate selected corpus split directories")
    p.add_argument("--shuffle-buffer", type=int, default=20_000,
                   help="streaming shuffle buffer; FineWeb sample-10BT is already a random sample")
    p.add_argument("--progress-every", type=int, default=50)

    # Tokenizer / chat template.  Matching the model/lens environment matters.
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--reasoning", default="low", choices=["low", "medium", "high"],
                   help="Harmony reasoning effort used only to render WildChat prompts")

    # FineWeb.
    p.add_argument("--fineweb-dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--fineweb-config", default="sample-10BT")
    p.add_argument("--fineweb-revision", default=None)
    p.add_argument("--fineweb-window-tokens", type=int, default=256)
    p.add_argument("--fineweb-min-retokenized", type=int, default=240,
                   help="reject rare decode/re-encode boundary cases shorter than this")
    p.add_argument("--fineweb-max-doc-chars", type=int, default=250_000,
                   help="skip extreme documents to avoid tokenizing huge pages in one shot; 0 disables")

    # WildChat.
    p.add_argument("--wildchat-dataset", default="allenai/WildChat-1M")
    p.add_argument("--wildchat-revision", default=None)
    p.add_argument("--wildchat-language", default="English",
                   help="conversation language to keep; use 'all' for no language filter")
    p.add_argument("--wildchat-allow-redacted", action="store_true",
                   help="by default skip conversations marked as having redacted PII")
    p.add_argument("--wildchat-min-user-tokens", type=int, default=8)
    p.add_argument("--wildchat-max-user-tokens", type=int, default=512)
    p.add_argument("--wildchat-max-rendered-tokens", type=int, default=900,
                   help="keep rendered Harmony prompt below the SVD explorer's default 1024-token cap")

    return p.parse_args()


def resolve_dataset_revision(repo_id: str, requested: str | None) -> str | None:
    """Resolve a dataset revision to a commit SHA when Hub access allows it."""
    if requested:
        try:
            return HfApi().dataset_info(repo_id, revision=requested).sha or requested
        except Exception as exc:
            print(f"[warn] could not resolve dataset revision {repo_id}@{requested}: {exc}")
            return requested
    try:
        return HfApi().dataset_info(repo_id).sha
    except Exception as exc:
        print(f"[warn] could not pin dataset revision for {repo_id}: {exc}")
        return None


def resolve_model_revision(repo_id: str, requested: str | None) -> str | None:
    """Resolve tokenizer/model revision to a commit SHA when possible."""
    if requested:
        try:
            return HfApi().model_info(repo_id, revision=requested).sha or requested
        except Exception as exc:
            print(f"[warn] could not resolve model revision {repo_id}@{requested}: {exc}")
            return requested
    try:
        return HfApi().model_info(repo_id).sha
    except Exception as exc:
        print(f"[warn] could not pin tokenizer revision for {repo_id}: {exc}")
        return None


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:20]


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).strip().casefold()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_split_dirs(root: Path, corpus: str, overwrite: bool) -> tuple[Path, Path]:
    d = root / f"{corpus}_discovery"
    r = root / f"{corpus}_replication"
    existing = [x for x in (d, r) if (x / "index.jsonl").exists()]
    if existing and not overwrite:
        names = ", ".join(str(x) for x in existing)
        raise SystemExit(
            f"Frozen sample already exists: {names}\n"
            "Refusing to silently resample. Use --overwrite if replacement is intentional."
        )
    if overwrite:
        for x in (d, r):
            if x.exists():
                shutil.rmtree(x)
    d.mkdir(parents=True, exist_ok=True)
    r.mkdir(parents=True, exist_ok=True)
    return d, r


def split_frozen(rows: list[dict[str, Any]], n_discovery: int, n_replication: int,
                 corpus: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert len(rows) == n_discovery + n_replication
    disc: list[dict[str, Any]] = []
    repl: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        x = dict(row)
        if i < n_discovery:
            split = "discovery"
            j = i
            target = disc
        else:
            split = "replication"
            j = i - n_discovery
            target = repl
        x["split"] = split
        x["pid"] = f"{corpus}_{split}_{j:04d}"
        x["run_label"] = f"{corpus}_{split}"
        target.append(x)
    return disc, repl


def build_chat_text(tok, user_msg: str, reasoning: str) -> tuple[str, bool]:
    """Mirror collect_readouts.py's single-user Harmony rendering."""
    msgs = [{"role": "user", "content": user_msg}]
    kwargs = dict(add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, reasoning_effort=reasoning, **kwargs
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, **kwargs)

    # We only need a stable string for replay.  Still record whether string
    # tokenization is deterministic on the current tokenizer.
    ids1 = tok(text).input_ids
    ids2 = tok(text).input_ids
    return text, ids1 == ids2


def base_record(*, corpus: str, fmt: str, text: str, payload: str,
                model: str, model_revision: str | None, source: dict[str, Any],
                n_tokens: int, reasoning: str | None = None) -> dict[str, Any]:
    return {
        # Fields consumed/inferred by explore_jlens_svd_concepts_v2.py.
        "pid": "__assigned_after_sampling__",
        "kind": fmt,
        "format": fmt,
        "generated": False,
        "payload": payload,
        "text": text,
        "n_tokens": int(n_tokens),
        "run_label": "",
        "reasoning": reasoning,
        "model": model,
        "model_revision": model_revision,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Extra provenance ignored by v2 but useful later.
        "corpus": corpus,
        "source": source,
    }


def collect_fineweb(args: argparse.Namespace, tok, dataset_revision: str | None,
                    model_revision: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = args.n_discovery + args.n_replication
    if target <= 0:
        return [], {}
    rng = random.Random(args.seed + 101)

    print(f"[fineweb] streaming {args.fineweb_dataset} config={args.fineweb_config} "
          f"revision={dataset_revision or 'unresolved/main'}")
    ds = load_dataset(
        args.fineweb_dataset,
        name=args.fineweb_config,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )
    ds = ds.shuffle(seed=args.seed + 102, buffer_size=args.shuffle_buffer)

    rows: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    scanned = 0
    skip_short = 0
    skip_huge = 0
    skip_roundtrip = 0

    for ex in ds:
        scanned += 1
        text = ex.get("text")
        if not isinstance(text, str) or not text.strip():
            skip_short += 1
            continue
        if args.fineweb_max_doc_chars > 0 and len(text) > args.fineweb_max_doc_chars:
            skip_huge += 1
            continue

        # FineWeb's token_count uses another tokenizer, so use it only as a
        # cheap prefilter; exact eligibility is checked with the gpt-oss tokenizer.
        coarse_count = ex.get("token_count")
        if isinstance(coarse_count, (int, float)) and coarse_count < args.fineweb_window_tokens:
            skip_short += 1
            continue

        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < args.fineweb_window_tokens:
            skip_short += 1
            continue

        max_start = len(ids) - args.fineweb_window_tokens
        start = rng.randint(0, max_start) if max_start > 0 else 0
        wanted = ids[start:start + args.fineweb_window_tokens]
        window = tok.decode(
            wanted,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        replay_ids = tok(window, add_special_tokens=False).input_ids
        if len(replay_ids) < args.fineweb_min_retokenized:
            skip_roundtrip += 1
            continue

        h = stable_text_hash(window)
        if h in seen_windows:
            continue
        seen_windows.add(h)

        source = {
            "dataset": args.fineweb_dataset,
            "config": args.fineweb_config,
            "dataset_revision": dataset_revision,
            "document_id": ex.get("id"),
            "dump": ex.get("dump"),
            "window_start_gptoss_token": int(start),
            "requested_window_tokens": int(args.fineweb_window_tokens),
            "replay_window_tokens": int(len(replay_ids)),
            "text_sha256_20": h,
        }
        rows.append(base_record(
            corpus="fineweb",
            fmt="raw",
            text=window,
            payload=window,
            model=args.model,
            model_revision=model_revision,
            source=source,
            n_tokens=len(replay_ids),
        ))

        if len(rows) % args.progress_every == 0 or len(rows) == target:
            print(f"[fineweb] kept={len(rows):4d}/{target} scanned={scanned:6d} "
                  f"short={skip_short} huge={skip_huge} boundary={skip_roundtrip}")
        if len(rows) >= target:
            break

    if len(rows) != target:
        raise SystemExit(f"FineWeb stream ended after collecting {len(rows)}/{target} eligible windows")

    stats = {
        "scanned": scanned,
        "accepted": len(rows),
        "skipped_short": skip_short,
        "skipped_huge": skip_huge,
        "skipped_decode_reencode_boundary": skip_roundtrip,
        "window_tokens_requested": args.fineweb_window_tokens,
    }
    return rows, stats


def first_user_turn(conversation: Any) -> dict[str, Any] | None:
    if not isinstance(conversation, list):
        return None
    for turn in conversation:
        if isinstance(turn, dict) and str(turn.get("role", "")).lower() == "user":
            return turn
    return None


def language_matches(value: Any, wanted: str) -> bool:
    if wanted.casefold() == "all":
        return True
    return str(value or "").strip().casefold() == wanted.strip().casefold()


def collect_wildchat(args: argparse.Namespace, tok, dataset_revision: str | None,
                     model_revision: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = args.n_discovery + args.n_replication
    if target <= 0:
        return [], {}

    print(f"[wildchat] streaming {args.wildchat_dataset} revision={dataset_revision or 'unresolved/main'}")
    ds = load_dataset(
        args.wildchat_dataset,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )
    ds = ds.shuffle(seed=args.seed + 202, buffer_size=args.shuffle_buffer)

    rows: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    scanned = 0
    skip_lang = 0
    skip_redacted = 0
    skip_empty = 0
    skip_length = 0
    skip_duplicate = 0

    for ex in ds:
        scanned += 1
        if not language_matches(ex.get("language"), args.wildchat_language):
            skip_lang += 1
            continue
        if not args.wildchat_allow_redacted and bool(ex.get("redacted", False)):
            skip_redacted += 1
            continue

        turn = first_user_turn(ex.get("conversation"))
        if turn is None:
            skip_empty += 1
            continue
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            skip_empty += 1
            continue

        # For English-only sampling, require the selected utterance to agree
        # when the per-utterance language annotation is populated.
        turn_lang = turn.get("language")
        if (args.wildchat_language.casefold() != "all" and turn_lang and
                not language_matches(turn_lang, args.wildchat_language)):
            skip_lang += 1
            continue

        norm = normalize_for_dedup(content)
        content_hash = stable_text_hash(norm)
        if content_hash in seen_content:
            skip_duplicate += 1
            continue

        user_ids = tok(content, add_special_tokens=False).input_ids
        if not (args.wildchat_min_user_tokens <= len(user_ids) <= args.wildchat_max_user_tokens):
            skip_length += 1
            continue

        rendered, roundtrip_ok = build_chat_text(tok, content, args.reasoning)
        rendered_ids = tok(rendered).input_ids
        if len(rendered_ids) > args.wildchat_max_rendered_tokens:
            skip_length += 1
            continue

        seen_content.add(content_hash)
        # Deliberately omit hashed_ip/country/state/header.  conversation_hash
        # and turn_identifier are public dataset provenance, not user profile data.
        source = {
            "dataset": args.wildchat_dataset,
            "dataset_revision": dataset_revision,
            "conversation_hash": ex.get("conversation_hash"),
            "turn_identifier": turn.get("turn_identifier"),
            "conversation_language": ex.get("language"),
            "turn_language": turn_lang,
            "content_sha256_20": content_hash,
            "user_tokens": int(len(user_ids)),
            "rendered_tokens": int(len(rendered_ids)),
            "chat_template_roundtrip_ok": bool(roundtrip_ok),
        }
        rows.append(base_record(
            corpus="wildchat",
            fmt="chat",
            text=rendered,
            payload=content,
            model=args.model,
            model_revision=model_revision,
            source=source,
            n_tokens=len(rendered_ids),
            reasoning=args.reasoning,
        ))

        if len(rows) % args.progress_every == 0 or len(rows) == target:
            print(f"[wildchat] kept={len(rows):4d}/{target} scanned={scanned:6d} "
                  f"lang={skip_lang} redact={skip_redacted} empty={skip_empty} "
                  f"len={skip_length} dup={skip_duplicate}")
        if len(rows) >= target:
            break

    if len(rows) != target:
        raise SystemExit(f"WildChat stream ended after collecting {len(rows)}/{target} eligible prompts")

    stats = {
        "scanned": scanned,
        "accepted": len(rows),
        "skipped_language": skip_lang,
        "skipped_redacted": skip_redacted,
        "skipped_empty": skip_empty,
        "skipped_length": skip_length,
        "skipped_duplicate": skip_duplicate,
        "language": args.wildchat_language,
        "allow_redacted": args.wildchat_allow_redacted,
    }
    return rows, stats


def save_corpus(root: Path, corpus: str, rows: list[dict[str, Any]], stats: dict[str, Any],
                args: argparse.Namespace, dataset_revision: str | None,
                model_revision: str | None) -> None:
    ddir, rdir = prepare_split_dirs(root, corpus, args.overwrite)
    disc, repl = split_frozen(rows, args.n_discovery, args.n_replication, corpus)

    common_meta = {
        "schema": "minimal collect_readouts index compatible with explore_jlens_svd_concepts_v2.py",
        "corpus": corpus,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "model": args.model,
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
        "n_discovery": args.n_discovery,
        "n_replication": args.n_replication,
        "sampling_stats": stats,
    }

    for split, directory, sample in (
        ("discovery", ddir, disc),
        ("replication", rdir, repl),
    ):
        write_jsonl(directory / "index.jsonl", sample)
        meta = dict(common_meta)
        meta.update({
            "split": split,
            "n_records": len(sample),
            "recommended_position_mode": "all" if corpus == "fineweb" else "last",
        })
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[write] {directory / 'index.jsonl'}  ({len(sample)} records)")


def main() -> None:
    args = parse_args()
    if args.n_discovery < 0 or args.n_replication < 0 or args.n_discovery + args.n_replication <= 0:
        raise SystemExit("n-discovery + n-replication must be > 0")
    if args.fineweb_min_retokenized > args.fineweb_window_tokens:
        raise SystemExit("--fineweb-min-retokenized cannot exceed --fineweb-window-tokens")

    requested = {x.strip().lower() for x in args.corpora.split(",") if x.strip()}
    unknown = requested - {"fineweb", "wildchat"}
    if unknown:
        raise SystemExit(f"unknown corpora: {sorted(unknown)}")

    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_revision = resolve_model_revision(args.model, args.model_revision)
    print(f"[tokenizer] {args.model}@{model_revision or args.model_revision or 'main'}")
    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=model_revision or args.model_revision
    )

    root_meta: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "model": args.model,
        "model_revision": model_revision,
        "n_discovery": args.n_discovery,
        "n_replication": args.n_replication,
        "corpora": sorted(requested),
    }

    if "fineweb" in requested:
        fw_rev = resolve_dataset_revision(args.fineweb_dataset, args.fineweb_revision)
        rows, stats = collect_fineweb(args, tok, fw_rev, model_revision)
        save_corpus(root, "fineweb", rows, stats, args, fw_rev, model_revision)
        root_meta["fineweb"] = {
            "dataset": args.fineweb_dataset,
            "config": args.fineweb_config,
            "revision": fw_rev,
            "sampling_stats": stats,
        }

    if "wildchat" in requested:
        wc_rev = resolve_dataset_revision(args.wildchat_dataset, args.wildchat_revision)
        rows, stats = collect_wildchat(args, tok, wc_rev, model_revision)
        save_corpus(root, "wildchat", rows, stats, args, wc_rev, model_revision)
        root_meta["wildchat"] = {
            "dataset": args.wildchat_dataset,
            "revision": wc_rev,
            "language": args.wildchat_language,
            "sampling_stats": stats,
        }

    (root / "collection_meta.json").write_text(
        json.dumps(root_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n[done] frozen corpora are ready. Recommended next commands:\n")
    if "fineweb" in requested:
        print("  python explore_jlens_svd_concepts_v2.py \\")
        print(f"      --readouts {root / 'fineweb_discovery'} \\")
        print("      --position-mode all --out svd_fineweb_discovery\n")
    if "wildchat" in requested:
        print("  python explore_jlens_svd_concepts_v2.py \\")
        print(f"      --readouts {root / 'wildchat_discovery'} \\")
        print("      --position-mode last --out svd_wildchat_discovery\n")
    print("Keep the replication directories untouched until after inspecting discovery results.")


if __name__ == "__main__":
    main()
