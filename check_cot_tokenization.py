#!/usr/bin/env python3
"""
Pre-flight for cot_intervention_server.py — validates that every rollout's prompt
and raw_response re-tokenize deterministically, so pause positions line up.

Only loads the tokenizer (no GPU / no 20B weights), so it runs anywhere the
tokenizer is reachable:

    python3 check_cot_tokenization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import transformers

ROOT = Path(__file__).resolve().parent
DATASETS = ["cot_unrealized_low", "cot_unrealized_medium", "predictive_cot_low", "predictive_cot_medium"]
MODEL_NAME = "openai/gpt-oss-20b"


def main():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    total = mismatches = roundtrip_fail = 0
    for ds in DATASETS:
        path = ROOT / ds / "rollouts.jsonl"
        if not path.exists():
            print(f"[skip] {ds}: no rollouts.jsonl")
            continue
        rows = [json.loads(l) for l in path.open() if l.strip()]
        ds_mismatch = ds_rt = 0
        for i, r in enumerate(rows):
            total += 1
            kwargs = dict(add_generation_prompt=True, reasoning_effort=r.get("reasoning_effort"),
                          tokenize=True, return_tensors="pt", return_dict=True)
            try:
                inp = tok.apply_chat_template(r["messages"], **kwargs)
            except TypeError as e:
                if "reasoning_effort" not in str(e):
                    raise
                kwargs.pop("reasoning_effort")
                inp = tok.apply_chat_template(r["messages"], **kwargs)
            prompt_len = inp["input_ids"].shape[1]
            expected = r.get("prompt_tokens")
            if expected is not None and prompt_len != expected:
                ds_mismatch += 1
                mismatches += 1
                if ds_mismatch <= 3:
                    print(f"  [mismatch] {ds}#{i} {r.get('prompt_id')}: rebuilt {prompt_len} vs stored {expected}")
            resp_ids = tok.encode(r["raw_response"], add_special_tokens=False)
            rt = tok.decode(resp_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if rt != r["raw_response"]:
                ds_rt += 1
                roundtrip_fail += 1
        print(f"[{ds}] {len(rows)} rollouts · prompt mismatches {ds_mismatch} · roundtrip diffs {ds_rt}")

    print(f"\nTOTAL {total} rollouts · {mismatches} prompt mismatches · {roundtrip_fail} roundtrip diffs")
    print("OK" if mismatches == 0 else "PROMPT MISMATCHES PRESENT — pause positions may be off for those rollouts")


if __name__ == "__main__":
    main()
