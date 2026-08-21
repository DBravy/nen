#!/usr/bin/env python3
"""
Replay one GPT-OSS assistant turn from collect_task_gaming_rollouts_v8.py and
capture sparse internal states at selected generated-token checkpoints.

The important design choice is that this script does NOT collect activations
while the model is generating. It teacher-forces the exact token IDs already
saved in replay_tokens.jsonl, uses a KV cache to replay the prefix in chunks,
and copies only selected positions/layers to CPU.

Captured per selected layer and checkpoint:
    resid_pre   : decoder-layer input (pre-attention residual stream)
    attn_out    : self-attention block output, after o_proj, before residual add
    mlp_out     : MoE block output, before residual add
    resid_post  : decoder-layer output, after attention + MoE residual updates
    router_logits / router_topk_* : MoE routing state computed from the exact
                                    normalized MLP input

No attention matrices are requested or stored.

Examples
--------
# Coarse scan every 64 generated tokens on assistant step 0:
python analyze_aha_rollout.py \
    --run-dir task_gaming_rollouts/rollouts/impossiblebench/binary_rules/\
              impossiblebench__binary_rules__s12345 \
    --step 0 --every 64

# Zoom in on particular generated-token positions:
python analyze_aha_rollout.py --run-dir RUN_DIR --step 1 \
    --positions 410,426,442,458,474 --chunk-size 128

# Only selected layers:
python analyze_aha_rollout.py --run-dir RUN_DIR --step 0 \
    --positions 300-500:20 --layers 0-5,10,15-23

Position convention
-------------------
By default, positions are 0-based offsets INTO generated_token_ids for the
selected assistant turn. If generated position g is selected, the saved state
is the model state AFTER consuming generated token g; causally, that is the
state used to predict token g+1.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import transformers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="One rollout run directory containing replay_tokens.jsonl")
    p.add_argument("--step", type=int, default=0,
                   help="Assistant/tool step to analyze (default: 0)")
    p.add_argument("--model", default=None,
                   help="HF model id. Default: infer from run_dir/prompt.json")
    p.add_argument("--model-revision", default=None,
                   help="Optional HF revision. Default: infer from prompt.json")
    p.add_argument("--out", default=None,
                   help="Output directory. Default: RUN_DIR/aha_analysis/step_XX")

    pos = p.add_argument_group("checkpoint selection")
    pos.add_argument(
        "--positions", default="",
        help=(
            "Explicit generated-token positions. Supports comma-separated integers, "
            "ranges a-b, and ranged strides a-b:s, e.g. 100,160-320:16,500"
        ),
    )
    pos.add_argument("--every", type=int, default=64,
                     help="If --positions is omitted, checkpoint every N generated tokens")
    pos.add_argument("--start", type=int, default=0,
                     help="First generated position for --every mode")
    pos.add_argument("--end", type=int, default=None,
                     help="Last generated position (inclusive) for --every mode")
    pos.add_argument("--include-last", action=argparse.BooleanOptionalAction, default=True,
                     help="Include the final generated token in --every mode")
    pos.add_argument("--text-context", type=int, default=24,
                     help="Tokens of left/right decoded context saved per checkpoint")

    cap = p.add_argument_group("capture")
    cap.add_argument("--layers", default="all",
                     help="all, or comma/range spec such as 0-5,10,20-23")
    cap.add_argument(
        "--capture",
        default="resid_pre,attn_out,mlp_out,resid_post,router",
        help="Comma-separated subset of resid_pre,attn_out,mlp_out,resid_post,router",
    )
    cap.add_argument("--save-dtype", default="float16",
                     choices=["float16", "bfloat16", "float32"],
                     help="CPU dtype for floating-point captures")
    cap.add_argument("--chunk-size", type=int, default=128,
                     help="Teacher-forced replay chunk size. Smaller = lower transient memory")

    load = p.add_argument_group("model loading")
    load.add_argument("--device-map", default=None,
                      help="HF device_map. Default: cuda if available, otherwise auto")
    load.add_argument("--dtype", default="auto",
                      choices=["auto", "float16", "bfloat16", "float32"],
                      help="Model load dtype; auto matches the collector")
    load.add_argument("--attn-implementation", default=None,
                      help="Optional Transformers attention implementation override")
    load.add_argument("--trust-remote-code", action="store_true")
    load.add_argument("--empty-cache-every", type=int, default=0,
                      help="Call torch.cuda.empty_cache() every N replay chunks (0 disables)")

    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_int_spec(spec: str) -> list[int]:
    """
    Parse: 1,4,7-10,20-40:5
    Ranges are inclusive.
    """
    out: set[int] = set()
    if not spec.strip():
        return []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" not in item:
            out.add(int(item))
            continue
        stride = 1
        core = item
        if ":" in item:
            core, stride_s = item.rsplit(":", 1)
            stride = int(stride_s)
            if stride <= 0:
                raise ValueError(f"Stride must be positive in {item!r}")
        a_s, b_s = core.split("-", 1)
        a, b = int(a_s), int(b_s)
        if b < a:
            raise ValueError(f"Range end < start in {item!r}")
        out.update(range(a, b + 1, stride))
    return sorted(out)


def load_replay_record(run_dir: Path, step: int) -> dict[str, Any]:
    path = run_dir / "replay_tokens.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec.get("step", -1)) == step:
            found = rec
            break
    if found is None:
        available = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    available.append(json.loads(line).get("step"))
                except Exception:
                    pass
        raise KeyError(f"Step {step} not found in {path}; available={available}")
    return found


def infer_model_info(run_dir: Path, args: argparse.Namespace) -> tuple[str, str | None]:
    prompt_path = run_dir / "prompt.json"
    meta: dict[str, Any] = read_json(prompt_path) if prompt_path.exists() else {}
    model_id = args.model or meta.get("model") or "openai/gpt-oss-20b"
    revision = args.model_revision if args.model_revision is not None else meta.get("model_revision")
    return model_id, revision


def torch_dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def tensor_to_cpu(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = t.detach()
    if dtype is not None and t.is_floating_point():
        t = t.to(dtype=dtype)
    return t.to(device="cpu", non_blocking=False).contiguous()


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Could not find tensor in hook output of type {type(x)}")


def get_base_model(causal_lm):
    # GPT-OSS CausalLM uses .model. Keep a fallback for future wrappers.
    if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "layers"):
        return causal_lm.model
    base = getattr(causal_lm, "base_model", None)
    if base is not None and hasattr(base, "layers"):
        return base
    raise AttributeError("Could not locate decoder stack (.model.layers or .base_model.layers)")


def layer_input_tensor(args: tuple[Any, ...]) -> torch.Tensor:
    if not args:
        raise RuntimeError("Expected decoder/MLP hidden_states as positional arg")
    return first_tensor(args[0])


def select_rows_3d(t: torch.Tensor, local_positions: list[int]) -> torch.Tensor:
    if t.ndim != 3 or t.shape[0] != 1:
        raise RuntimeError(f"Expected [1, seq, dim] tensor, got {tuple(t.shape)}")
    if not local_positions:
        return t[:, :0, :].reshape(0, t.shape[-1])
    idx = torch.tensor(local_positions, dtype=torch.long, device=t.device)
    return t[0].index_select(0, idx)


def build_checkpoint_positions(args: argparse.Namespace, generated_len: int) -> list[int]:
    if generated_len <= 0:
        raise ValueError("Selected replay step has no generated tokens")

    if args.positions.strip():
        positions = parse_int_spec(args.positions)
    else:
        if args.every <= 0:
            raise ValueError("--every must be > 0")
        start = max(0, args.start)
        end = generated_len - 1 if args.end is None else min(args.end, generated_len - 1)
        if end < start:
            raise ValueError(f"Checkpoint range is empty: start={start}, end={end}")
        positions = list(range(start, end + 1, args.every))
        if args.include_last and (generated_len - 1) >= start:
            positions.append(generated_len - 1)

    positions = sorted(set(positions))
    bad = [p for p in positions if p < 0 or p >= generated_len]
    if bad:
        raise ValueError(
            f"Generated positions out of range [0,{generated_len-1}]: {bad[:20]}"
        )
    if not positions:
        raise ValueError("No checkpoint positions selected")
    return positions


def decode_context(tok, full_ids: list[int], absolute_pos: int, radius: int) -> dict[str, Any]:
    lo = max(0, absolute_pos - radius)
    hi = min(len(full_ids), absolute_pos + radius + 1)
    token_id = int(full_ids[absolute_pos])
    try:
        token_piece = tok.convert_ids_to_tokens(token_id)
    except Exception:
        token_piece = None
    return {
        "absolute_position": absolute_pos,
        "token_id": token_id,
        "token_piece": token_piece,
        "token_decoded": tok.decode([token_id], skip_special_tokens=False),
        "context_start_absolute": lo,
        "context_end_absolute_exclusive": hi,
        "context_text": tok.decode(full_ids[lo:hi], skip_special_tokens=False),
        "prefix_tail_text": tok.decode(full_ids[lo:absolute_pos + 1], skip_special_tokens=False),
    }


def write_token_map(tok, generated_ids: list[int], input_count: int, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for g, token_id in enumerate(generated_ids):
            try:
                piece = tok.convert_ids_to_tokens(int(token_id))
            except Exception:
                piece = None
            rec = {
                "generated_position": g,
                "absolute_position": input_count + g,
                "token_id": int(token_id),
                "token_piece": piece,
                "decoded": tok.decode([int(token_id)], skip_special_tokens=False),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    replay = load_replay_record(run_dir, args.step)
    input_ids_saved = [int(x) for x in replay["input_token_ids"]]
    generated_ids = [int(x) for x in replay["generated_token_ids"]]
    full_ids = [int(x) for x in replay["full_token_ids"]]
    input_count = int(replay.get("input_token_count", len(input_ids_saved)))

    if full_ids != input_ids_saved + generated_ids:
        raise ValueError(
            "Replay integrity check failed: full_token_ids != input_token_ids + generated_token_ids"
        )
    if input_count != len(input_ids_saved):
        raise ValueError(
            f"Replay integrity check failed: input_token_count={input_count} "
            f"but len(input_token_ids)={len(input_ids_saved)}"
        )

    generated_positions = build_checkpoint_positions(args, len(generated_ids))
    absolute_positions = [input_count + p for p in generated_positions]
    selected_abs = set(absolute_positions)
    max_abs = max(absolute_positions)

    model_id, revision = infer_model_info(run_dir, args)
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else run_dir / "aha_analysis" / f"step_{args.step:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[replay] run={run_dir}")
    print(f"[replay] step={args.step} prompt_tokens={input_count} generated_tokens={len(generated_ids)}")
    print(f"[replay] checkpoints={len(generated_positions)} generated range="
          f"{generated_positions[0]}..{generated_positions[-1]}")
    print(f"[load] {model_id} revision={revision!r}")

    tok = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=args.trust_remote_code,
    )

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    else:
        load_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "auto"
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    base = get_base_model(model)
    layers = base.layers
    n_layers = len(layers)

    if args.layers.strip().lower() == "all":
        selected_layers = list(range(n_layers))
    else:
        selected_layers = parse_int_spec(args.layers)
        bad_layers = [x for x in selected_layers if x < 0 or x >= n_layers]
        if bad_layers:
            raise ValueError(f"Layer indices out of range [0,{n_layers-1}]: {bad_layers}")

    valid_capture = {"resid_pre", "attn_out", "mlp_out", "resid_post", "router"}
    capture_kinds = {x.strip() for x in args.capture.split(",") if x.strip()}
    unknown = capture_kinds - valid_capture
    if unknown:
        raise ValueError(f"Unknown --capture values: {sorted(unknown)}")

    save_dtype = torch_dtype_from_name(args.save_dtype)
    if save_dtype == "auto":
        raise AssertionError("save dtype may not be auto")

    # Nested payload: captures[generated_position][layer][kind] -> CPU tensor.
    captures: dict[int, dict[int, dict[str, torch.Tensor]]] = {
        p: {layer_idx: {} for layer_idx in selected_layers}
        for p in generated_positions
    }

    # Hooks consult these mutable per-chunk values. This avoids registering/removing
    # dozens of hooks on every chunk.
    hook_state: dict[str, Any] = {
        "chunk_start": 0,
        "local_positions": [],
        "absolute_positions": [],
    }

    def current_generated_positions() -> list[int]:
        return [a - input_count for a in hook_state["absolute_positions"]]

    def save_rows(layer_idx: int, kind: str, tensor3: torch.Tensor) -> None:
        local = hook_state["local_positions"]
        if not local:
            return
        rows = select_rows_3d(tensor3, local)
        rows_cpu = tensor_to_cpu(rows, save_dtype)
        for i, g in enumerate(current_generated_positions()):
            captures[g][layer_idx][kind] = rows_cpu[i].clone()

    handles = []

    for layer_idx in selected_layers:
        layer = layers[layer_idx]

        if "resid_pre" in capture_kinds:
            def make_layer_pre(li: int):
                def hook(_module, hargs):
                    save_rows(li, "resid_pre", layer_input_tensor(hargs))
                return hook
            handles.append(layer.register_forward_pre_hook(make_layer_pre(layer_idx)))

        if "attn_out" in capture_kinds:
            def make_attn_post(li: int):
                def hook(_module, _hargs, output):
                    save_rows(li, "attn_out", first_tensor(output))
                return hook
            handles.append(layer.self_attn.register_forward_hook(make_attn_post(layer_idx)))

        # Capture router state from the normalized input that is ACTUALLY passed to
        # the MoE. This remains robust even if a kernelized MLP bypasses the Python
        # router module's forward hook.
        if "router" in capture_kinds:
            def make_mlp_pre(li: int, layer_ref):
                def hook(_module, hargs):
                    local = hook_state["local_positions"]
                    if not local:
                        return
                    h = layer_input_tensor(hargs)
                    rows = select_rows_3d(h, local)
                    router = getattr(layer_ref.mlp, "router", None)
                    if router is None or not hasattr(router, "weight"):
                        raise RuntimeError(f"Layer {li}: could not access mlp.router weight")
                    logits = F.linear(rows, router.weight, getattr(router, "bias", None))
                    top_k = int(getattr(router, "top_k", getattr(model.config, "num_experts_per_tok", 4)))
                    top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                    top_w = torch.softmax(top_vals, dim=-1)

                    logits_cpu = tensor_to_cpu(logits, save_dtype)
                    vals_cpu = tensor_to_cpu(top_vals, save_dtype)
                    weights_cpu = tensor_to_cpu(top_w, save_dtype)
                    idx_cpu = tensor_to_cpu(top_idx, None).to(torch.int16)
                    for i, g in enumerate(current_generated_positions()):
                        slot = captures[g][li]
                        slot["router_logits"] = logits_cpu[i].clone()
                        slot["router_topk_logits"] = vals_cpu[i].clone()
                        slot["router_topk_weights"] = weights_cpu[i].clone()
                        slot["router_topk_indices"] = idx_cpu[i].clone()
                return hook
            handles.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(layer_idx, layer)))

        if "mlp_out" in capture_kinds:
            def make_mlp_post(li: int):
                def hook(_module, _hargs, output):
                    save_rows(li, "mlp_out", first_tensor(output))
                return hook
            handles.append(layer.mlp.register_forward_hook(make_mlp_post(layer_idx)))

        if "resid_post" in capture_kinds:
            def make_layer_post(li: int):
                def hook(_module, _hargs, output):
                    save_rows(li, "resid_post", first_tensor(output))
                return hook
            handles.append(layer.register_forward_hook(make_layer_post(layer_idx)))

    # Token metadata is written before replay so an OOM/interrupt still leaves a
    # useful map for selecting narrower positions next time.
    write_token_map(tok, generated_ids, input_count, out_dir / "token_map.jsonl")
    (out_dir / "generated_text.txt").write_text(
        tok.decode(generated_ids, skip_special_tokens=False), encoding="utf-8"
    )

    checkpoint_meta = []
    for g, a in zip(generated_positions, absolute_positions):
        rec = decode_context(tok, full_ids, a, args.text_context)
        rec["generated_position"] = g
        checkpoint_meta.append(rec)
    with (out_dir / "checkpoints.jsonl").open("w", encoding="utf-8") as f:
        for rec in checkpoint_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Replay only through the latest requested checkpoint; there is no scientific
    # reason to process the remainder of the assistant turn for these activations.
    replay_ids = full_ids[:max_abs + 1]
    embed_device = base.embed_tokens.weight.device
    if str(embed_device) == "meta":
        raise RuntimeError("Embedding layer is still on meta device; model did not materialize correctly")

    if torch.cuda.is_available():
        print(f"[load] CUDA allocated={torch.cuda.memory_allocated()/1e9:.2f} GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"[model] layers={n_layers} selected={selected_layers}")
    print(f"[model] hidden={getattr(model.config, 'hidden_size', None)} "
          f"experts={getattr(model.config, 'num_local_experts', None)} "
          f"top_k={getattr(model.config, 'num_experts_per_tok', None)}")
    print(f"[replay] processing {len(replay_ids)} tokens through absolute position {max_abs}")

    t0 = time.time()
    past_key_values = None
    chunks = math.ceil(len(replay_ids) / args.chunk_size)

    try:
        with torch.inference_mode():
            for chunk_i, start in enumerate(range(0, len(replay_ids), args.chunk_size)):
                end = min(len(replay_ids), start + args.chunk_size)
                ids = torch.tensor(
                    [replay_ids[start:end]], dtype=torch.long, device=embed_device
                )

                abs_here = sorted(a for a in selected_abs if start <= a < end)
                local_here = [a - start for a in abs_here]
                hook_state["chunk_start"] = start
                hook_state["absolute_positions"] = abs_here
                hook_state["local_positions"] = local_here

                outputs = base(
                    input_ids=ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                # Do not retain last_hidden_state; all desired vectors have already
                # been copied to CPU by hooks.
                del outputs, ids

                if args.empty_cache_every and torch.cuda.is_available():
                    if (chunk_i + 1) % args.empty_cache_every == 0:
                        torch.cuda.empty_cache()

                mark = " *capture*" if abs_here else ""
                print(
                    f"[replay] chunk {chunk_i+1:03d}/{chunks:03d} "
                    f"abs={start}:{end-1}{mark}"
                )

    finally:
        for h in handles:
            h.remove()

    # Validate that every requested tensor family actually fired. This catches
    # architectural / kernel changes instead of silently producing partial data.
    missing: list[dict[str, Any]] = []
    expected_keys: set[str] = set()
    for kind in capture_kinds:
        if kind == "router":
            expected_keys.update({
                "router_logits", "router_topk_logits",
                "router_topk_weights", "router_topk_indices",
            })
        else:
            expected_keys.add(kind)

    for g in generated_positions:
        for li in selected_layers:
            got = set(captures[g][li])
            miss = sorted(expected_keys - got)
            if miss:
                missing.append({"generated_position": g, "layer": li, "missing": miss})

    if missing:
        preview = missing[:10]
        raise RuntimeError(
            "Some requested hooks did not produce captures. "
            f"First missing entries: {preview}"
        )

    elapsed = time.time() - t0

    payload = {
        "format_version": 1,
        "model": model_id,
        "model_revision": revision,
        "transformers_version": transformers.__version__,
        "run_dir": str(run_dir),
        "step": args.step,
        "position_space": "generated_token_ids_0_based",
        "state_semantics": (
            "Capture at generated position g is after consuming token g and is "
            "therefore part of the state predicting token g+1."
        ),
        "input_token_count": input_count,
        "generated_token_count": len(generated_ids),
        "generated_positions": generated_positions,
        "absolute_positions": absolute_positions,
        "selected_layers": selected_layers,
        "capture_kinds": sorted(capture_kinds),
        "save_dtype": args.save_dtype,
        "chunk_size": args.chunk_size,
        "captures": captures,
    }
    torch.save(payload, out_dir / "activations.pt")

    metadata = {
        k: v for k, v in payload.items() if k != "captures"
    }
    metadata.update({
        "elapsed_s": round(elapsed, 3),
        "files": {
            "activations": "activations.pt",
            "checkpoints": "checkpoints.jsonl",
            "token_map": "token_map.jsonl",
            "generated_text": "generated_text.txt",
        },
        "capture_definitions": {
            "resid_pre": "input to decoder layer, before input_layernorm/self-attention",
            "attn_out": "self-attention output after o_proj, before residual addition",
            "mlp_out": "MoE output before residual addition",
            "resid_post": "decoder-layer output after attention and MoE residual additions",
            "router_logits": "linear router logits from normalized MLP input",
            "router_topk_logits": "top-k router logits",
            "router_topk_weights": "softmax over top-k router logits (actual mixture weights)",
            "router_topk_indices": "selected expert indices",
        },
    })
    write_json(metadata, out_dir / "metadata.json")

    print(f"[done] elapsed={elapsed:.1f}s")
    print(f"[done] {out_dir / 'activations.pt'}")
    print(f"[done] {out_dir / 'checkpoints.jsonl'}")
    print(f"[done] {out_dir / 'token_map.jsonl'}")
    if torch.cuda.is_available():
        print(f"[done] CUDA allocated={torch.cuda.memory_allocated()/1e9:.2f} GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

    del past_key_values, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
