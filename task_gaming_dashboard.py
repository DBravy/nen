#!/usr/bin/env python3
"""Local, dependency-free browser for task_gaming_candidate_data."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "task_gaming_candidate_data"
STATIC = ROOT / "task_gaming_dashboard"


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Store:
    def __init__(self, data: Path):
        self.data = data
        self.meta = json.loads((data / "meta.json").read_text())
        self.stats = json.loads((data / "activation_stats.json").read_text())
        self.directions = json.loads((data / "direction_readouts.json").read_text())
        self.trials = jsonl(data / "trials.jsonl")
        self.turns = jsonl(data / "turns.jsonl")
        self.trial_by_run = {x["run_id"]: x for x in self.trials}
        self.turn_by_key = {(x["run_id"], x["step"]): x for x in self.turns}
        self.activations = defaultdict(list)
        with (data / "activations.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = {
                    "row_id": int(row["row_id"]), "token": row["token"],
                    "token_string": row["token_string"], "channel": row["channel"],
                    "token_index_in_turn": int(row["token_index_in_turn"]),
                    "full_position": int(row["full_position"]),
                }
                for c in self.meta["candidates"]:
                    n = c["name"]
                    item[n] = float(row[f"{n}_activation"])
                    item[n + "_scale"] = float(row[f"{n}_contribution_scale"])
                self.activations[(row["run_id"], int(row["step"]))].append(item)
        self.moments = self._build_moments()
        self.token_path = data / "token_data.jsonl"
        self.offsets = [0]
        with self.token_path.open("rb") as f:
            while f.readline():
                self.offsets.append(f.tell())

    def _build_moments(self):
        """Reviewed, turn-level landmarks for finding gaming in the CoT."""
        moments = []
        for tr in self.trials:
            # These categories explicitly contain no intentional gaming moment.
            if tr["intent_class"] in ("no_gaming", "accidental_proxy_failure"):
                continue
            run = tr["run_id"]
            decision = tr.get("decision_step")
            if decision is not None:
                if tr["intent_class"] == "considered_rejected":
                    kind, summary = "rejection", "A shortcut is considered here and then rejected."
                elif tr["intent_class"] == "attempted_shortcut_failed":
                    kind, summary = "failed", "The model chooses a shortcut, but the attempt does not succeed."
                else:
                    kind, summary = "decision", "The reasoning commits to the gaming strategy."
                moments.append({"run_id": run, "step": decision, "kind": kind, "summary": summary})
                if tr.get("gaming_action"):
                    later = [x["step"] for x in self.turns if x["run_id"] == run and x["step"] > decision]
                    if later:
                        moments.append({"run_id": run, "step": min(later), "kind": "action",
                                        "summary": "The selected gaming strategy is executed."})
        for i, m in enumerate(moments, 1):
            m["id"] = i
            tr = self.trial_by_run[m["run_id"]]
            turn = self.turn_by_key.get((m["run_id"], m["step"]), {})
            text = turn.get("generated_text", "")
            for marker in ("<|channel|>", "<|message|>", "<|end|>", "<|start|>"):
                text = text.replace(marker, " ")
            text = " ".join(text.split())
            m.update(environment=tr["environment"], condition=tr["condition"],
                     intent_class=tr["intent_class"], label_notes=tr["label_notes"],
                     excerpt=(text[:280] + "…") if len(text) > 280 else text)
            tokens = self.activations.get((m["run_id"], m["step"]), [])
            cues = ("cheat", "hack", "hardcod", "special", "satisf", "test", "expect",
                    "ignore", "suppress", "skip", "truncat", "clear", "replace", "rewrite",
                    "bypass", "glob", "mypy", "dummy", "trivial", "wrong")
            hits = [j for j, tok in enumerate(tokens)
                    if any(cue in tok["token"].lower() for cue in cues)]
            marked = set()
            for hit in hits:
                marked.update(range(max(0, hit - 5), min(len(tokens), hit + 7)))
            # Action turns are often mostly a tool call; keep them visible even if
            # tokenizer splitting prevents a cue match.
            if not marked and m["kind"] == "action":
                marked.update(range(len(tokens)))
            m["highlight_row_ids"] = [tokens[j]["row_id"] for j in sorted(marked)]
        return moments

    def token(self, row_id: int):
        if row_id < 1 or row_id >= len(self.offsets):
            raise KeyError(row_id)
        with self.token_path.open("rb") as f:
            f.seek(self.offsets[row_id - 1])
            return json.loads(f.readline())

    def bootstrap(self):
        fields = ("run_id", "environment", "condition", "seed", "reasoning",
                  "intent_class", "usable", "gaming_considered", "gaming_attempted",
                  "gaming_action", "accidental_shortcut", "baseline_step", "decision_step",
                  "label_confidence", "label_notes")
        return {
            "meta": self.meta, "stats": self.stats,
            "trials": [{k: x.get(k) for k in fields} for x in self.trials],
            "turns": self.turns, "moments": self.moments,
        }


class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/bootstrap":
                return self.send_json(self.store.bootstrap())
            if u.path == "/api/directions":
                return self.send_json(self.store.directions)
            if u.path == "/api/turn":
                key = (q["run_id"][0], int(q["step"][0]))
                return self.send_json({"turn": self.store.turn_by_key[key],
                                       "tokens": self.store.activations[key]})
            if u.path == "/api/token":
                return self.send_json(self.store.token(int(q["row_id"][0])))
            path = "index.html" if u.path == "/" else u.path.lstrip("/")
            target = (STATIC / path).resolve()
            if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                raise FileNotFoundError
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
            self.send_json({"error": str(e)}, 404)

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    Handler.store = Store(args.data)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose = args.verbose
    url = f"http://{args.host}:{server.server_port}"
    print(f"Task Gaming Lens Explorer: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
