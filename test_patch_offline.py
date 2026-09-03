#!/usr/bin/env python3
"""Offline structural test for patch_second_moment.py (no real torch/model).

Installs a numpy-backed 'torch' shim and a fake 'workspace_second_moment',
then runs cmd_select and cmd_analyze end-to-end on synthetic data. Catches
key/shape/logic errors in the CPU paths; the GPU patch loop is not covered.
"""
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- torch shim
class T:
    def __init__(self, a):
        self.a = np.asarray(a)

    # dtype/device no-ops
    def float(self): return T(self.a.astype(np.float64))
    def double(self): return T(self.a.astype(np.float64))
    def half(self): return T(self.a.astype(np.float32))
    def cpu(self): return self
    def to(self, *_a, **_k): return self
    def clone(self): return T(self.a.copy())
    def numpy(self): return self.a
    def item(self): return self.a.item()
    def numel(self): return self.a.size
    def tolist(self): return self.a.tolist()
    @property
    def shape(self): return self.a.shape
    @property
    def T_(self): return T(self.a.T)
    T = property(lambda self: T(self.a.T))
    def norm(self): return float(np.linalg.norm(self.a))
    def mean(self, axis=0): return T(self.a.mean(axis=axis))
    def sum(self): return T(self.a.sum())
    def argmax(self): return T(np.argmax(self.a))
    def __matmul__(self, o): return T(self.a @ (o.a if isinstance(o, T) else o))
    def __rmatmul__(self, o): return T((o.a if isinstance(o, T) else o) @ self.a)
    def __add__(self, o): return T(self.a + (o.a if isinstance(o, T) else o))
    def __radd__(self, o): return self.__add__(o)
    def __sub__(self, o): return T(self.a - (o.a if isinstance(o, T) else o))
    def __rsub__(self, o): return T((o.a if isinstance(o, T) else o) - self.a)
    def __mul__(self, o): return T(self.a * (o.a if isinstance(o, T) else o))
    def __rmul__(self, o): return self.__mul__(o)
    def __truediv__(self, o): return T(self.a / (o.a if isinstance(o, T) else o))
    def __neg__(self): return T(-self.a)
    def __ge__(self, o): return T(self.a >= (o.a if isinstance(o, T) else o))
    def __getitem__(self, i):
        r = self.a[i.a if isinstance(i, T) else i]
        return T(r) if isinstance(r, np.ndarray) else r
    def __int__(self): return int(self.a)
    def __float__(self): return float(self.a)
    def __len__(self): return len(self.a)


_saved = {}

torch = types.ModuleType("torch")
torch.float32 = "float32"
torch.int32 = "int32"
torch.Tensor = T
torch.tensor = lambda x, dtype=None: T(np.asarray(x))
torch.stack = lambda xs, dim=0: T(np.stack([x.a for x in xs], axis=dim))
torch.from_numpy = lambda a: T(a)
torch.zeros = lambda *s, **k: T(np.zeros(s))
torch.set_grad_enabled = lambda *_: None


def _t_save(obj, path):
    key = str(path)
    if key.endswith(".tmp"):
        key = key[:-4]
    _saved[key] = obj
    Path(path).write_bytes(pickle.dumps("placeholder"))


def _t_load(path, **_k):
    return _saved[str(path)]


torch.save = _t_save
torch.load = _t_load
sys.modules["torch"] = torch

# ------------------------------------------------- fake workspace_second_moment
wsm = types.ModuleType("workspace_second_moment")
D = 24


class FakeLens:
    def __init__(self, jac, path):
        self.jac = jac
        self.path = path


rng = np.random.default_rng(0)


def _psd(scale):
    A = rng.standard_normal((D, D))
    return scale * (A @ A.T) / D


M_J = rng.standard_normal((D, D))
M_R = rng.standard_normal((D, D)) * 0.5
S_J = M_J.T @ M_J + _psd(30.0)
S_R = M_R.T @ M_R + _psd(5.0)

_blob = {
    "S": {"j": {1: T(S_J), 2: T(S_J * 0.5)}, "r": {1: T(S_R), 2: T(S_R * 0.5)}},
    "config": {"arms": "jr", "d_model": D, "model": "fake/model",
               "target_layer": 3, "skip_first": 1, "max_seq_len": 16,
               "dataset": "fake/ds", "prompt_offset": 0, "n_prompts": 5,
               "lens_path_j": "lj.pt", "lens_path_r": "lr.pt"},
}

wsm.load_harvest = lambda p: (_blob, False)
wsm.resolve_lens_pair = lambda args, arms: {
    "j": FakeLens({1: T(M_J), 2: T(M_J * 0.7)}, "lj.pt"),
    "r": FakeLens({1: T(M_R), 2: T(M_R * 0.7)}, "lr.pt"),
}
wsm.add_lens_args = lambda p: None
sys.modules["workspace_second_moment"] = wsm

import patch_second_moment as psm  # noqa: E402  (after shims)


# ------------------------------------------------------------------- select
class NS:
    pass


args = NS()
args.harvest = "fakeharvest"
args.out = "/tmp/pt_test/directions.pt"
args.layers = None
args.k_top = 3
args.k_orth = 2
args.n_pairs = 2
args.n_rand = 2
args.pair_energy_tol = 3.0
args.pair_gamma_ratio = 1.5
args.min_residual = 0.3
args.r_sets = "auto"
args.seed = 0
args.lens_path_j = None
args.lens_path_r = None

Path("/tmp/pt_test").mkdir(exist_ok=True)
psm.cmd_select(args)
dirs = _saved[args.out]
assert set(dirs["layers"]) <= {1, 2} and dirs["layers"], "select produced no layers"
for l, rec in dirs["layers"].items():
    for name, s in rec["sets"].items():
        n = s["V"].shape[1]
        assert len(s["names"]) == n == len(s["gamma_J"]) == len(s["fSf_J"])
        assert "gamma_R" in s, "r-arm metadata missing"
print(f"[test] select OK: layers={sorted(dirs['layers'])} "
      f"sets={sorted(dirs['layers'][sorted(dirs['layers'])[0]]['sets'])}")

# ------------------------------------------------------------------ analyze
SK, SKH = 16, 8
layers = [1]
dirs_l = [("topG", 0), ("topG", 1), ("topJbar", 2), ("rand", 3)]
templates = {i: rng.standard_normal(SK) for i in range(4)}
gain = {0: lambda c: 1.0 + 0.8 * c, 1: lambda c: 0.5 + 0.6 * c,
        2: lambda c: 1.0, 3: lambda c: 0.2}
alphas = [0.01, 0.1]
rows = {k: [] for k in ("set", "prompt", "layer", "pos", "dir",
                        "alpha", "sign", "eps", "dlog_norm", "dh_norm",
                        "dh_tail_norm", "kl", "flip")}
sk_log, sk_h = [], []
meta = []
for (s, d_) in dirs_l:
    meta.append({"layer": 1, "set": s, "dir": d_, "name": f"L01:{s}:{d_:02d}",
                 "gamma_J": 5.0 if s == "topG" else 0.5, "fSf_J": 10.0,
                 "inv_J": 2.0, "gamma_R": 1.0, "fSf_R": 8.0})
for c in range(4):
    eps = 1.0
    for a in alphas:
        for (s, d_) in dirs_l:
            g = gain[d_](c)
            base = a * eps * g * templates[d_]
            for sign in (1, -1):
                v = sign * base + 0.001 * rng.standard_normal(SK)
                rows["set"].append(s); rows["prompt"].append(c)
                rows["layer"].append(1); rows["pos"].append(7)
                rows["dir"].append(d_); rows["alpha"].append(a)
                rows["sign"].append(sign); rows["eps"].append(a * eps)
                rows["dlog_norm"].append(float(np.linalg.norm(v)))
                rows["dh_norm"].append(float(np.linalg.norm(v)) * 0.5)
                rows["dh_tail_norm"].append(0.1); rows["kl"].append(0.01)
                rows["flip"].append(-1)
                sk_log.append(v.astype(np.float32))
                sk_h.append((0.5 * g * a * eps * np.ones(SKH)).astype(np.float32))
    # a couple of zero-row noise records
    rows["set"].append("-"); rows["prompt"].append(c); rows["layer"].append(1)
    rows["pos"].append(7); rows["dir"].append(-1); rows["alpha"].append(0.0)
    rows["sign"].append(0); rows["eps"].append(0.0)
    rows["dlog_norm"].append(0.001); rows["dh_norm"].append(0.001)
    rows["dh_tail_norm"].append(0.0); rows["kl"].append(0.0); rows["flip"].append(-1)
    sk_log.append((0.0005 * rng.standard_normal(SK)).astype(np.float32))
    sk_h.append(np.zeros(SKH, dtype=np.float32))

patches = {
    "config": {"model": "fake/model", "n_prompts": 4, "prompt_offset": 5,
               "layers": layers, "alphas": alphas, "directions": "d.pt",
               "directions_hash": "abc", "sets": ["topG", "topJbar", "rand"]},
    "prompt_meta": [], "direction_meta": meta,
    "set": rows["set"],
}
for k in ("alpha", "sign", "eps", "dlog_norm", "dh_norm", "dh_tail_norm", "kl"):
    patches[k] = T(np.array(rows[k], dtype=np.float64))
for k in ("prompt", "layer", "pos", "dir", "flip"):
    patches[k] = T(np.array(rows[k], dtype=np.int64))
patches["sk_log"] = T(np.stack(sk_log))
patches["sk_h"] = T(np.stack(sk_h))
_saved["/tmp/pt_test/patches.pt"] = patches

a2 = NS()
a2.out = "/tmp/pt_test"
psm.cmd_analyze(a2)

import csv
with open("/tmp/pt_test/per_direction.csv") as f:
    pd_rows = list(csv.DictReader(f))
assert len(pd_rows) == 4
topg = [r for r in pd_rows if r["set"] == "topG"]
assert all(float(r["template_stability"]) > 0.9 for r in topg), "template stability broken"
assert all(abs(float(r["corr_pred"])) > 0.9 for r in topg), "corr_pred broken"
with open("/tmp/pt_test/decoding.csv") as f:
    dec = list(csv.DictReader(f))
hi_a = [r for r in dec if float(r["alpha"]) == 0.1]
assert hi_a and all(float(r["within_acc"]) > 0.9 for r in hi_a), \
    f"within-context decoding should be near-perfect on clean synthetic data: {hi_a}"
report = Path("/tmp/pt_test/patch_report.txt").read_text()
assert "Response curves" in report and "Decodability" in report
print("[test] analyze OK: per_direction stats and decoding behave on synthetic data")
print("[test] ALL OFFLINE CHECKS PASSED")
