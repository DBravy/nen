#!/usr/bin/env python3
"""
Interactive CoT SV-intervention server.

Pick a rollout, pause at a token, amplify/suppress a set of J-lens singular
vectors (SVs) *at that one token*, and re-generate the continuation. Returns the
steered continuations alongside an unsteered control and the rollout's original
continuation.

Model runs here (CUDA VM); the UI is served as static files for a laptop browser.

    python3 cot_intervention_server.py --host 0.0.0.0 --port 8123

Design notes
------------
* Rollouts store `messages` + `raw_response` (Harmony text) but no token ids, so
  we re-tokenize deterministically, mirroring the collector's prompt build
  (apply_chat_template with add_generation_prompt=True, reasoning_effort=...).
* Steering scope is "pause token only": the residual edit fires exactly on the
  last position of the prefill forward; every decode step is untouched. The nudge
  still propagates through the KV cache that later tokens attend to.
* Edit math matches causal_l17_sv28_interventions.py:
      coeff = h . v ;  new = multiplier*coeff + bias ;  add (new-coeff)*v
  where v is a unit-norm column of the per-layer SVD V matrix.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import mimetypes
import threading
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "cot_intervention_dashboard"

# dataset key -> (rollout dir, direction basis). low/medium share the same V per
# basis, so the direction basis is what actually selects a directions/ folder.
DATASETS = {
    "cot_unrealized_low": {"dir": "cot_unrealized_low", "basis": "unrealized"},
    "cot_unrealized_medium": {"dir": "cot_unrealized_medium", "basis": "unrealized"},
    "predictive_cot_low": {"dir": "predictive_cot_low", "basis": "predictive"},
    "predictive_cot_medium": {"dir": "predictive_cot_medium", "basis": "predictive"},
}
BASIS_DIRS = {
    "unrealized": "cot_unrealized_low",
    "predictive": "predictive_cot_low",
}
MODEL_NAME = "openai/gpt-oss-20b"


# --------------------------------------------------------------------------- #
# Edit primitives (mirrors causal_l17_sv28_interventions.py)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TargetEdit:
    key: str
    layer: int
    sv_rank: int  # 1-based
    v: torch.Tensor  # unit-norm (d_model,)
    multiplier: float = 1.0
    bias: float = 0.0


def find_decoder_layers(model):
    """Locate the GPT-OSS decoder ModuleList across Transformers layouts."""
    direct = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("layers", getattr(model, "layers", None)),
    ]
    for name, value in direct:
        if isinstance(value, torch.nn.ModuleList) and len(value) > 22:
            print(f"[model] decoder layers: {name} ({len(value)})")
            return value
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 22:
            print(f"[model] decoder layers: {name} ({len(module)}) [discovered]")
            return module
    raise RuntimeError("Could not locate a decoder-layer ModuleList on the model.")


class PauseTokenEditController:
    """
    Apply additive SV edits to the residual stream at exactly one position: the
    last token of the prefill forward (= the pause token). Decode steps (seq_len
    == 1) are never edited, so only the pause token's hidden state is nudged.
    """

    def __init__(self, model, decoder_layers, edits: list[TargetEdit]):
        self.model = model
        self.decoder_layers = decoder_layers
        self.edits_by_layer = defaultdict(list)
        for e in edits:
            self.edits_by_layer[e.layer].append(e)
        self.handles = []
        self.enabled = False
        # per-edit coefficient audit for telemetry
        self.audit = {e.key: {"n": 0, "sum_pre": 0.0, "sum_post": 0.0} for e in edits}

    def _layer_hook(self, layer: int):
        def hook(module, inputs, output):
            if not self.enabled or not self.edits_by_layer.get(layer):
                return None
            tensor = output if torch.is_tensor(output) else output[0]
            seq_len = tensor.shape[-2]
            if seq_len <= 1:
                # decode step: pause token already cached, do not edit.
                return None
            idx = torch.as_tensor([seq_len - 1], device=tensor.device, dtype=torch.long)
            x = tensor.index_select(-2, idx).float()  # [batch, 1, d]
            total_delta = torch.zeros_like(x)
            for edit in self.edits_by_layer[layer]:
                v = edit.v.to(device=x.device, dtype=torch.float32)
                coeff = torch.einsum("bsd,d->bs", x, v)
                new_coeff = edit.multiplier * coeff + edit.bias
                total_delta.add_((new_coeff - coeff).unsqueeze(-1) * v)
                a = self.audit[edit.key]
                a["n"] += int(coeff.numel())
                a["sum_pre"] += float(coeff.sum().item())
                a["sum_post"] += float(new_coeff.sum().item())
            selected = tensor.index_select(-2, idx)
            selected.add_(total_delta.to(dtype=tensor.dtype))
            tensor.index_copy_(-2, idx, selected)
            return None

        return hook

    def __enter__(self):
        for layer in sorted(self.edits_by_layer):
            self.handles.append(
                self.decoder_layers[layer].register_forward_hook(self._layer_hook(layer))
            )
        self.enabled = True
        return self

    def __exit__(self, *exc):
        self.enabled = False
        for h in self.handles:
            h.remove()
        self.handles = []

    def telemetry(self):
        out = {}
        for key, a in self.audit.items():
            n = max(1, a["n"])
            out[key] = {
                "n_forward_vector_instances": a["n"],
                "mean_pre": a["sum_pre"] / n,
                "mean_post": a["sum_post"] / n,
            }
        return out


# --------------------------------------------------------------------------- #
# Data + model store
# --------------------------------------------------------------------------- #
def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Store:
    def __init__(self, device: str):
        print(f"[store] loading tokenizer + model {MODEL_NAME} ...")
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            device_map=device if device != "cpu" else None,
        )
        self.model.eval()
        self.decoder_layers = find_decoder_layers(self.model)

        self.rollouts = {}  # dataset key -> list[rollout]
        self.summaries = {}  # dataset key -> list[summary dict]
        for key, spec in DATASETS.items():
            path = ROOT / spec["dir"] / "rollouts.jsonl"
            if not path.exists():
                print(f"[store] WARNING: missing {path}, skipping {key}")
                continue
            rows = jsonl(path)
            self.rollouts[key] = rows
            self.summaries[key] = [
                {
                    "index": i,
                    "prompt_id": r.get("prompt_id"),
                    "category": (r.get("source") or {}).get("category"),
                    "difficulty": (r.get("source") or {}).get("difficulty"),
                    "reasoning_effort": r.get("reasoning_effort"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "generated_tokens": r.get("generated_tokens"),
                }
                for i, r in enumerate(rows)
            ]
            print(f"[store] {key}: {len(rows)} rollouts")

        # Cache of tokenized prefixes: (dataset, index) -> {prompt_ids, response_ids}
        self._tok_cache = {}
        self._tok_lock = threading.Lock()

        # Direction cache: (basis, layer) -> np.load result; sv std lookup per basis.
        self._dir_cache = {}
        self._sv_std = {}  # basis -> {(layer, sv_index_0): std_activation}
        self._load_sv_std()

        # Serialize generation (single GPU, in-place residual edits).
        self.gen_lock = threading.Lock()

    # -- directions -------------------------------------------------------- #
    def _directions_dir(self, basis: str) -> Path:
        return ROOT / BASIS_DIRS[basis] / "directions"

    def direction_vector(self, basis: str, layer: int, sv_rank: int) -> torch.Tensor:
        cache_key = (basis, layer)
        z = self._dir_cache.get(cache_key)
        if z is None:
            p = self._directions_dir(basis) / f"L{layer:02d}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing directions file {p}")
            z = np.load(p)
            self._dir_cache[cache_key] = z
        V = z["V"]
        j = sv_rank - 1
        if j < 0 or j >= V.shape[1]:
            raise ValueError(f"SV{sv_rank} out of range (layer {layer} has {V.shape[1]} SVs)")
        v = torch.from_numpy(V[:, j].astype(np.float32))
        return v / v.norm().clamp_min(1e-8)

    def _load_sv_std(self):
        for basis, dir_name in BASIS_DIRS.items():
            path = ROOT / dir_name / "sv_rankings.csv"
            table = {}
            if path.exists():
                with path.open(encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            layer = int(row["layer"])
                            sv0 = int(row["sv_index_0"])
                            std = float(row.get("std_activation") or 0.0)
                        except (KeyError, ValueError):
                            continue
                        table[(layer, sv0)] = std
            self._sv_std[basis] = table

    def sv_std(self, basis: str, layer: int, sv_rank: int) -> float:
        return self._sv_std.get(basis, {}).get((layer, sv_rank - 1), 0.0)

    # -- tokenization ------------------------------------------------------ #
    def _tokenized(self, dataset: str, index: int):
        cache_key = (dataset, index)
        with self._tok_lock:
            cached = self._tok_cache.get(cache_key)
        if cached is not None:
            return cached

        rollout = self.rollouts[dataset][index]
        kwargs = dict(
            add_generation_prompt=True,
            reasoning_effort=rollout.get("reasoning_effort"),
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        try:
            inputs = self.tok.apply_chat_template(rollout["messages"], **kwargs)
        except TypeError as e:
            if "reasoning_effort" not in str(e):
                raise
            kwargs.pop("reasoning_effort", None)
            inputs = self.tok.apply_chat_template(rollout["messages"], **kwargs)

        prompt_ids = [int(x) for x in inputs["input_ids"][0].tolist()]
        response_ids = self.tok.encode(rollout["raw_response"], add_special_tokens=False)

        expected = rollout.get("prompt_tokens")
        prompt_mismatch = expected is not None and len(prompt_ids) != expected
        result = {
            "prompt_ids": prompt_ids,
            "response_ids": [int(x) for x in response_ids],
            "prompt_mismatch": prompt_mismatch,
            "prompt_len": len(prompt_ids),
            "expected_prompt_tokens": expected,
        }
        with self._tok_lock:
            self._tok_cache[cache_key] = result
        return result

    def rollout_view(self, dataset: str, index: int):
        rollout = self.rollouts[dataset][index]
        t = self._tokenized(dataset, index)
        response_ids = t["response_ids"]

        # Per-token decoded string + channel, by scanning Harmony markers.
        channel_id = self.tok.convert_tokens_to_ids("<|channel|>")
        message_id = self.tok.convert_tokens_to_ids("<|message|>")
        end_ids = {
            x
            for x in (
                self.tok.convert_tokens_to_ids("<|end|>"),
                self.tok.convert_tokens_to_ids("<|call|>"),
                self.tok.convert_tokens_to_ids("<|return|>"),
                self.tok.convert_tokens_to_ids("<|start|>"),
                self.tok.eos_token_id,
            )
            if isinstance(x, int) and x >= 0
        }
        tokens = []
        channel = None
        in_header = False
        header_ids = []
        for i, tid in enumerate(response_ids):
            text = self.tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            cur_channel = channel
            if tid in end_ids:
                channel, in_header, header_ids = None, False, []
                cur_channel = "marker"
            elif tid == channel_id:
                channel, in_header, header_ids = None, True, []
                cur_channel = "marker"
            elif in_header:
                if tid == message_id:
                    hdr = self.tok.decode(header_ids, skip_special_tokens=False).lower()
                    channel = next((c for c in ("analysis", "commentary", "final") if c in hdr), None)
                    in_header, header_ids = False, []
                    cur_channel = "marker"
                else:
                    header_ids.append(tid)
                    cur_channel = "marker"
            tokens.append({"index": i, "text": text, "token_id": tid, "channel": cur_channel})

        return {
            "dataset": dataset,
            "index": index,
            "prompt_id": rollout.get("prompt_id"),
            "prompt_text": self._prompt_text(rollout),
            "tokens": tokens,
            "final": rollout.get("final") or "",
            "prompt_mismatch": t["prompt_mismatch"],
            "prompt_len": t["prompt_len"],
            "expected_prompt_tokens": t["expected_prompt_tokens"],
        }

    @staticmethod
    def _prompt_text(rollout):
        parts = []
        for m in rollout.get("messages", []):
            parts.append(f"[{m.get('role')}]\n{m.get('content')}")
        return "\n\n".join(parts)

    # -- generation -------------------------------------------------------- #
    def _generate(self, prefix_ids, *, n_samples, temperature, top_p, seed, max_new_tokens):
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        do_sample = temperature > 0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=n_samples,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        prefix_len = input_ids.shape[1]
        texts = []
        for row in out:
            cont = row[prefix_len:]
            texts.append(self.tok.decode(cont, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        return texts

    def intervene(self, body):
        dataset = body["dataset"]
        index = int(body["rollout_index"])
        pause_index = int(body["pause_index"])
        basis = body.get("basis", "unrealized")
        n_samples = int(body.get("n_samples", 3))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.95))
        seed = int(body.get("seed", 0))
        max_new_tokens = int(body.get("max_new_tokens", 256))

        t = self._tokenized(dataset, index)
        prompt_ids, response_ids = t["prompt_ids"], t["response_ids"]
        if pause_index < 0 or pause_index >= len(response_ids):
            raise ValueError(f"pause_index {pause_index} out of range (0..{len(response_ids)-1})")
        prefix_ids = prompt_ids + response_ids[: pause_index + 1]
        original_continuation = self.tok.decode(
            response_ids[pause_index + 1 :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        edits = []
        for e in body.get("edits", []):
            layer = int(e["layer"])
            sv_rank = int(e["sv_rank"])
            v = self.direction_vector(basis, layer, sv_rank)
            # bias may be given directly, or in SD units of the SV's activation.
            if e.get("bias_sd") is not None:
                std = self.sv_std(basis, layer, sv_rank)
                bias = float(e["bias_sd"]) * (std if std else 1.0)
            else:
                bias = float(e.get("bias", 0.0))
            edits.append(
                TargetEdit(
                    key=f"{basis}:L{layer:02d}_SV{sv_rank}",
                    layer=layer,
                    sv_rank=sv_rank,
                    v=v,
                    multiplier=float(e.get("multiplier", 1.0)),
                    bias=bias,
                )
            )

        with self.gen_lock:
            baseline = self._generate(
                prefix_ids,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens,
            )
            telemetry = {}
            if edits:
                controller = PauseTokenEditController(self.model, self.decoder_layers, edits)
                with controller:
                    intervened = self._generate(
                        prefix_ids,
                        n_samples=n_samples,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        max_new_tokens=max_new_tokens,
                    )
                telemetry = controller.telemetry()
            else:
                # No edits: intervened == baseline (same seed). Regenerate for parity.
                intervened = self._generate(
                    prefix_ids,
                    n_samples=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                )

        return {
            "pause_token": self.tok.decode([response_ids[pause_index]], skip_special_tokens=False),
            "prefix_len": len(prefix_ids),
            "original_continuation": original_continuation,
            "baseline": baseline,
            "intervened": intervened,
            "telemetry": telemetry,
            "edits": [
                {"layer": e.layer, "sv_rank": e.sv_rank, "multiplier": e.multiplier, "bias": e.bias, "key": e.key}
                for e in edits
            ],
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/datasets":
                return self.send_json(
                    {"datasets": [{"key": k, "rollouts": self.store.summaries.get(k, [])}
                                  for k in DATASETS if k in self.store.summaries]}
                )
            if u.path == "/api/rollout":
                return self.send_json(
                    self.store.rollout_view(q["dataset"][0], int(q["index"][0]))
                )
            if u.path == "/api/sv_meta":
                std = self.store.sv_std(q["basis"][0], int(q["layer"][0]), int(q["sv"][0]))
                return self.send_json({"std": std})
            path = "index.html" if u.path == "/" else u.path.lstrip("/")
            target = (STATIC / path).resolve()
            if STATIC.resolve() != target and STATIC.resolve() not in target.parents:
                raise FileNotFoundError
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
            self.send_json({"error": str(e)}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/api/intervene":
                return self.send_json(self.store.intervene(body))
            raise FileNotFoundError(u.path)
        except (KeyError, ValueError, IndexError) as e:
            self.send_json({"error": str(e)}, 400)
        except FileNotFoundError as e:
            self.send_json({"error": str(e)}, 404)
        except Exception as e:  # generation failures shouldn't kill the server
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    Handler.store = Store(args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose = args.verbose
    url = f"http://{args.host}:{server.server_port}"
    print(f"CoT SV-Intervention server: {url}")
    if not args.no_open and args.host in ("127.0.0.1", "localhost"):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
