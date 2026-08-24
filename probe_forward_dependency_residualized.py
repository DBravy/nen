#!/usr/bin/env python3
"""
probe_forward_dependency_residualized.py

Second-generation probe for a candidate J-Lens right-singular-vector concept.

Working hypothesis:
    FORWARD SEMANTIC DEPENDENCY / PROSPECTIVE RELATIONAL COMPLETION

The previous probe found:
  * all 8 lexical matched pairs favored the hypothesized use at both layers;
  * generic the/of/to controls were weaker;
  * because/if were fairly strong;
  * namely/for example were unexpectedly weak;
  * activation was essentially uncorrelated with entropy/surprisal;
  * the activation ranking was highly preserved across the two layers.

This script is designed to clarify what remains:

1. LEXICAL BASELINE
   Several words (who, where, spot, feel...) activate substantially even in
   non-hypothesized uses. We therefore collect MANY uses of each lexical item
   in both roles and estimate the functional effect after controlling for token
   identity.

2. POSITION / CONTEXT LENGTH
   We explicitly record target-token position and sequence length and include
   them as nuisance covariates.

3. HIDDEN-STATE NORM
   Raw projection can be inflated by residual-stream norm. Cosine projection is
   the primary outcome, and norm is included in the raw-activation regression.

4. NOVEL GENERALIZATION
   We test operators not present in the original top-activation list:
       which, whose, whether, why, how, what, means, called, named
   If these generalize, that is stronger evidence for an abstract functional
   category rather than a cluster of memorized lexical prototypes.

5. BOUNDARIES
   We separately test:
       - generic grammatical dependence: the/of/to/a/can/will
       - clausal relations: because/if/when/although/unless
       - discourse-only payload markers: namely/for example/specifically
       - punctuation: elaborative colon vs time/ratio, semicolon, comma

Outputs:
  <out_dir>/items.csv
  <out_dir>/results.json
  <out_dir>/summary.md

SVD indices are ZERO-BASED.

Example:
    python probe_forward_dependency_residualized.py \
        --model openai/gpt-oss-20b \
        --directions-dir unrealized_words_fineweb/directions \
        --layers 7,8 \
        --sv-indices 12,12 \
        --out-dir forward_dependency_probe_v2

If sign differs:
    --signs 1,-1

Optional:
    --custom-cases extra_cases.jsonl
Each JSONL case can contain:
    id, lexical_key, family, tier, role, target, occurrence, text

role:
    positive | negative | boundary | control | natural

tier:
    core | novel | clausal | discourse_only | generic | punctuation | natural
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------

CASES: List[Dict[str, Any]] = []


def add(
    id: str,
    lexical_key: str,
    family: str,
    tier: str,
    role: str,
    target: str,
    text: str,
    occurrence: int = 0,
):
    CASES.append(
        dict(
            id=id,
            lexical_key=lexical_key,
            family=family,
            tier=tier,
            role=role,
            target=target,
            text=text,
            occurrence=occurrence,
        )
    )


# =========================
# CORE: observed lexical families
# =========================

# COLON: elaboration/specification vs non-discourse punctuation.
add("colon_pos_1","colon","colon","core","positive",":",
    "There is one reason the experiment failed: the sensor was unplugged.")
add("colon_pos_2","colon","colon","core","positive",":",
    "The central problem is simple: nobody checked the calibration.")
add("colon_pos_3","colon","colon","core","positive",":",
    "She gave us one warning: the road would close before sunset.")
add("colon_pos_4","colon","colon","core","positive",":",
    "The report reached a surprising conclusion: the smaller model was more reliable.")
add("colon_neg_1","colon","colon","core","negative",":",
    "The meeting starts at 3:30 tomorrow afternoon.")
add("colon_neg_2","colon","colon","core","negative",":",
    "The final score was 4:2 after extra time.")
add("colon_neg_3","colon","colon","core","negative",":",
    "The ratio of red to blue pieces is 3:5.")
add("colon_neg_4","colon","colon","core","negative",":",
    "The timestamp reads 12:45:08.", occurrence=0)

# WHO: relative clause vs metalinguistic / quoted lexical object.
for i, (p, n) in enumerate([
    ("I spoke with Maya, who had already read the report.",
     "The editor underlined the word who in red."),
    ("The engineer who designed the device later explained the flaw.",
     "They printed who on a separate line."),
    ("We hired Elena, who had worked on the original system.",
     "The teacher asked the class to spell who."),
    ("I met a researcher who studies language models.",
     "The search box already contained the word who."),
],1):
    add(f"who_pos_{i}","who","relative","core","positive","who",p)
    add(f"who_neg_{i}","who","relative","core","negative","who",n)

# WHERE
for i, (p, n) in enumerate([
    ("We returned to the cabin, where the supplies had been stored.",
     "The editor underlined the word where in red."),
    ("This is the point where the two explanations begin to differ.",
     "They printed where on the first line."),
    ("They reached the valley, where the river turns north.",
     "The teacher asked the class to spell where."),
    ("I remember the room where we first ran the experiment.",
     "The search query consisted only of the word where."),
],1):
    add(f"where_pos_{i}","where","relative","core","positive","where",p)
    add(f"where_neg_{i}","where","relative","core","negative","where",n)

# THAT: complementizer vs demonstrative determiner.
for i, (p, n) in enumerate([
    ("I realized that the second measurement was wrong.",
     "I bought that book at the airport yesterday."),
    ("The results suggest that the effect is much smaller than expected.",
     "We moved that chair into the hallway."),
    ("She noticed that the final column was missing.",
     "Please hand me that folder on the desk."),
    ("They concluded that the error came from preprocessing.",
     "I remember that building across the street."),
],1):
    add(f"that_pos_{i}","that","complementizer","core","positive","that",p)
    add(f"that_neg_{i}","that","complementizer","core","negative","that",n)

# FACT: "in fact" discourse transition vs ordinary noun.
for i, (p, n) in enumerate([
    ("In fact, the second model performed substantially better.",
     "The fact surprised everyone in the room."),
    ("In fact, nobody had checked whether the file existed.",
     "That fact remained controversial for years."),
    ("In fact, the effect grew after the intervention.",
     "The fact was mentioned briefly in the appendix."),
    ("In fact, both experiments produced the same pattern.",
     "A single fact changed the entire investigation."),
],1):
    add(f"fact_pos_{i}","fact","in_fact","core","positive","fact",p)
    add(f"fact_neg_{i}","fact","in_fact","core","negative","fact",n)

# POINT: temporal/discourse frame vs concrete/geometric noun.
for i, (p, n) in enumerate([
    ("At one point, I thought the entire analysis had failed.",
     "Mark each point on the graph with a small circle."),
    ("At one point the crowd became completely silent.",
     "The red point is slightly above the horizontal line."),
    ("At one point, the two trajectories briefly overlapped.",
     "Move the point two units to the left."),
    ("At one point we considered discarding the whole dataset.",
     "Each point represents a separate observation."),
],1):
    add(f"point_pos_{i}","point","temporal_frame","core","positive","point",p)
    add(f"point_neg_{i}","point","temporal_frame","core","negative","point",n)

# FEEL: predicative complement opening vs noun.
for i, (p, n) in enumerate([
    ("The lighting makes the room feel much larger than it is.",
     "I like the feel of the paper in this notebook."),
    ("A few details make the interface feel surprisingly polished.",
     "The soft feel was the main reason she chose the fabric."),
    ("The music makes the scene feel strangely empty.",
     "The fabric has a smooth feel against the skin."),
    ("Small delays can make the conversation feel awkward.",
     "She preferred the heavier feel of the older keyboard."),
],1):
    add(f"feel_pos_{i}","feel","predicative","core","positive","feel",p)
    add(f"feel_neg_{i}","feel","predicative","core","negative","feel",n)

# SPOT: transitive predicate opening an object vs noun.
for i, (p, n) in enumerate([
    ("If you spot a problem, report it before restarting the machine.",
     "We found a quiet spot beside the river."),
    ("Experts can spot subtle errors that beginners usually miss.",
     "A dark spot appeared near the center of the image."),
    ("You can spot the difference once the curves are overlaid.",
     "She saved a parking spot near the entrance."),
    ("I immediately spot the mismatch when the rows are aligned.",
     "The dog slept in its usual spot by the window."),
],1):
    add(f"spot_pos_{i}","spot","transitive","core","positive","spot",p)
    add(f"spot_neg_{i}","spot","transitive","core","negative","spot",n)


# =========================
# NOVEL: predicted by the abstract hypothesis
# =========================

# WHICH: relative clause. Metalinguistic negatives.
for i,(p,n) in enumerate([
    ("The device, which had failed twice before, was replaced.",
     "The editor circled the word which in the paragraph."),
    ("The first result, which surprised us, disappeared after filtering.",
     "They printed which on its own line."),
    ("We used the older method, which requires fewer assumptions.",
     "The teacher asked the class to spell which."),
],1):
    add(f"which_pos_{i}","which","relative","novel","positive","which",p)
    add(f"which_neg_{i}","which","relative","novel","negative","which",n)

# WHOSE: relational modifier.
for i,(p,n) in enumerate([
    ("The scientist whose paper we discussed joined the call.",
     "The editor highlighted the word whose in yellow."),
    ("We interviewed a programmer whose code had caused the failure.",
     "They printed whose beneath the title."),
    ("I met the author whose book inspired the project.",
     "The teacher asked the class to spell whose."),
],1):
    add(f"whose_pos_{i}","whose","relative","novel","positive","whose",p)
    add(f"whose_neg_{i}","whose","relative","novel","negative","whose",n)

# WHETHER: proposition/question content.
for i,(p,n) in enumerate([
    ("We tested whether the effect survived randomization.",
     "The editor underlined the word whether in the sentence."),
    ("She asked whether the backup file still existed.",
     "They printed whether on a flash card."),
    ("The main question is whether the direction generalizes.",
     "The teacher asked the class to spell whether."),
],1):
    add(f"whether_pos_{i}","whether","embedded_question","novel","positive","whether",p)
    add(f"whether_neg_{i}","whether","embedded_question","novel","negative","whether",n)

# WHY
for i,(p,n) in enumerate([
    ("I finally understood why the two runs disagreed.",
     "The editor underlined the word why in the sentence."),
    ("She explained why the experiment had to be repeated.",
     "They printed why in large letters."),
    ("Nobody knew why the signal vanished after layer eight.",
     "The teacher asked the class to spell why."),
],1):
    add(f"why_pos_{i}","why","embedded_question","novel","positive","why",p)
    add(f"why_neg_{i}","why","embedded_question","novel","negative","why",n)

# HOW
for i,(p,n) in enumerate([
    ("The report explains how the estimate was calculated.",
     "The editor underlined the word how in the sentence."),
    ("We learned how the failure propagated through the system.",
     "They printed how on a separate card."),
    ("She showed us how the two vectors were aligned.",
     "The teacher asked the class to spell how."),
],1):
    add(f"how_pos_{i}","how","embedded_question","novel","positive","how",p)
    add(f"how_neg_{i}","how","embedded_question","novel","negative","how",n)

# WHAT
for i,(p,n) in enumerate([
    ("Tell me what happened after the model loaded.",
     "The editor underlined the word what in the sentence."),
    ("We need to determine what the direction is actually tracking.",
     "They printed what on a separate card."),
    ("Nobody could explain what caused the sudden change.",
     "The teacher asked the class to spell what."),
],1):
    add(f"what_pos_{i}","what","embedded_question","novel","positive","what",p)
    add(f"what_neg_{i}","what","embedded_question","novel","negative","what",n)

# MEANS: copular-ish semantic relation vs resources/sense.
for i,(p,n) in enumerate([
    ("This means the second hypothesis is probably wrong.",
     "The lab has the means to repeat the experiment."),
    ("A positive score means the feature is present.",
     "They lacked the means to purchase new equipment."),
    ("That result means we should inspect the earlier layers.",
     "She achieved her goal by every legal means available."),
],1):
    add(f"means_pos_{i}","means","semantic_relation","novel","positive","means",p)
    add(f"means_neg_{i}","means","semantic_relation","novel","negative","means",n)

# CALLED: naming/classification complement vs ordinary phone-call verb.
for i,(p,n) in enumerate([
    ("The method uses a technique called contrastive activation addition.",
     "She called the office before lunch."),
    ("They discovered a phenomenon called phase locking.",
     "He called his brother after the meeting."),
    ("We tested a metric called effective rank.",
     "The manager called everyone yesterday morning."),
],1):
    add(f"called_pos_{i}","called","naming","novel","positive","called",p)
    add(f"called_neg_{i}","called","naming","novel","negative","called",n)

# NAMED: naming relation vs past-tense appointment/mention.
for i,(p,n) in enumerate([
    ("The paper introduces a benchmark named LongBench.",
     "The committee named her chair of the group."),
    ("We used a dataset named FineWeb for the scan.",
     "The article named three possible suspects."),
    ("They built a tool named VectorScope for the analysis.",
     "The report named every contributor in the appendix."),
],1):
    add(f"named_pos_{i}","named","naming","novel","positive","named",p)
    add(f"named_neg_{i}","named","naming","novel","negative","named",n)


# =========================
# CLAUSAL BOUNDARIES: nearby hypothesis
# =========================

for i,text in enumerate([
    "The experiment stopped because the temperature rose too quickly.",
    "We reran the analysis because the first result looked suspicious.",
    "The model failed because the cache was corrupted.",
],1):
    add(f"because_{i}","because","clausal_relation","clausal","boundary","because",text)

for i,text in enumerate([
    "The result changes if the final token is removed.",
    "The effect disappears if the direction is randomized.",
    "The script retries if the first load attempt fails.",
],1):
    add(f"if_{i}","if","clausal_relation","clausal","boundary","if",text)

for i,text in enumerate([
    "The instability appears when the singular values become degenerate.",
    "The score rises when the relevant token is encountered.",
    "The output changes when the context is shortened.",
],1):
    add(f"when_{i}","when","clausal_relation","clausal","boundary","when",text)

for i,text in enumerate([
    "The result survived although the sample size was small.",
    "The model answered correctly although the prompt was ambiguous.",
    "The direction persisted although the representation rotated.",
],1):
    add(f"although_{i}","although","clausal_relation","clausal","boundary","although",text)

for i,text in enumerate([
    "The script will continue unless the checkpoint is missing.",
    "The effect should remain unless the layer is ablated.",
    "We cannot compare them unless the signs are aligned.",
],1):
    add(f"unless_{i}","unless","clausal_relation","clausal","boundary","unless",text)


# =========================
# DISCOURSE-ONLY PAYLOAD MARKERS
# These were predicted by the old "payload imminent" hypothesis but some were weak.
# =========================

for i,text in enumerate([
    "Only one explanation remained plausible, namely a calibration error.",
    "Two variables changed, namely temperature and pressure.",
    "The failure had one immediate cause, namely a missing file.",
],1):
    add(f"namely_{i}","namely","discourse_payload","discourse_only","boundary","namely",text)

for i,text in enumerate([
    "For example, the same pattern appears when the input is reversed.",
    "For example, a colon can introduce an explanation.",
    "For example, this direction activates strongly on relative clauses.",
],1):
    add(f"example_{i}","example","discourse_payload","discourse_only","boundary","example",text)

for i,text in enumerate([
    "More specifically, the effect begins around layer seven.",
    "Specifically, we compared the first sixty-four singular vectors.",
    "The problem is specifically the mismatch between the two files.",
],1):
    add(f"specifically_{i}","specifically","discourse_payload","discourse_only","boundary","specifically",text)

for i,text in enumerate([
    "The key observation is the following: the direction persists across layers.",
    "Consider the following: every matched pair moved in the same direction.",
    "The following result was reported in the appendix.",
],1):
    add(f"following_{i}","following","discourse_payload","discourse_only","boundary","following",text)


# =========================
# GENERIC GRAMMATICAL DEPENDENCY CONTROLS
# =========================

generic_controls = [
    ("the","She quietly opened the wooden box."),
    ("the","They carefully measured the final distance."),
    ("of","He placed a glass of water on the desk."),
    ("of","We compared the output of both systems."),
    ("to","They decided to postpone the meeting."),
    ("to","She tried to reproduce the result."),
    ("a","He carried a heavy suitcase upstairs."),
    ("a","They observed a small change in accuracy."),
    ("can","A careful reader can notice the difference."),
    ("can","The program can load several checkpoints."),
    ("will","The script will save the output automatically."),
    ("will","The model will generate another token."),
]
for i,(target,text) in enumerate(generic_controls,1):
    add(f"generic_{target}_{i}",target,"generic_dependency","generic","control",target,text)


# =========================
# PUNCTUATION BOUNDARIES
# =========================

add("semicolon_1","semicolon","punctuation","punctuation","boundary",";",
    "The first model failed; the second completed the task.")
add("semicolon_2","semicolon","punctuation","punctuation","boundary",";",
    "The signal disappeared; we restarted the recording.")
add("comma_1","comma","punctuation","punctuation","control",",",
    "After the meeting, we returned to the lab.")
add("comma_2","comma","punctuation","punctuation","control",",",
    "Slowly, the noise began to decrease.")
add("dash_1","dash","punctuation","punctuation","boundary","—",
    "There was one remaining possibility—the cache had not been cleared.")
add("dash_2","dash","punctuation","punctuation","boundary","—",
    "The conclusion was unavoidable—the two vectors represented the same mode.")


# =========================
# NATURAL REPLICATION EXAMPLES
# =========================

add("natural_colon_medicine","colon","natural","natural","natural",":",
    "Big news in the world of medicine surfaced this week: The National Institutes of Health promised a whopping $10.1 million to fund the scientific study of ailments.")
add("natural_fact_skoda","fact","natural","natural","natural","fact",
    "The survey was the biggest car owner survey in the UK. In fact, Skoda took three of the top four spots.")
add("natural_point_chicken","point","natural","natural","natural","point",
    "At one point I was wrestling with a whole chicken trying to separate it into the various parts.")
add("natural_who_crystal","who","natural","natural","natural","who",
    "I got a call from Crystal, who is the daughter of the owner Ainsley. She told me that the restaurant was in trouble.")
add("natural_spot_bugs","spot","natural","natural","natural","spot",
    "There is a feedback section in the options in case you spot any more bugs and want to notify the company.")
# IMPORTANT: target the FIRST "that", fixing the bug in the prior probe.
add("natural_that_model","that","natural","natural","natural","that",
    "It may be that your internal model of the world is flawed, and that will deplete your body budget.",
    occurrence=0)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--directions-dir", required=True)
    p.add_argument("--direction-file", action="append", default=[],
                   help="Optional explicit LAYER=/path/file.pt, repeatable.")
    p.add_argument("--layers", required=True)
    p.add_argument("--sv-indices", required=True,
                   help="ZERO-BASED SV index for each layer.")
    p.add_argument("--signs", default=None)
    p.add_argument("--hook-position", choices=["pre", "post"], default="post")
    p.add_argument("--out-dir", default="forward_dependency_probe_v2")
    p.add_argument("--custom-cases", default=None)
    p.add_argument("--trace-left", type=int, default=3)
    p.add_argument("--trace-right", type=int, default=5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--attn-implementation", default=None)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Direction loading
# -----------------------------------------------------------------------------

def parse_explicit(items):
    out = {}
    for x in items:
        if "=" not in x:
            raise ValueError("--direction-file must be LAYER=PATH")
        k, v = x.split("=", 1)
        out[int(k)] = Path(v)
    return out


def discover_file(directory: Path, layer: int) -> Path:
    files = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pt",".pth",".npy",".npz",".bin"}
    ]
    pats = [
        re.compile(rf"(^|[^0-9])layer[_-]?0*{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^A-Za-z0-9])L0*{layer}([^0-9]|$)", re.I),
    ]
    scored = []
    for p in files:
        score = 0
        for j, pat in enumerate(pats):
            if pat.search(p.stem):
                score = 100 - 10*j
                break
        if score:
            if "direction" in p.stem.lower(): score += 5
            if "sv" in p.stem.lower(): score += 2
            scored.append((score, len(str(p)), p))
    if not scored:
        if len(files) == 1:
            return files[0]
        raise FileNotFoundError(
            f"Could not discover direction file for layer {layer} in {directory}. "
            f"Use --direction-file {layer}=PATH"
        )
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def load_obj(path: Path):
    if path.suffix.lower() in {".pt",".pth",".bin"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix.lower() == ".npz":
        return dict(np.load(path, allow_pickle=True))
    raise ValueError(path)


PREFERRED_KEYS = [
    "Vh", "V", "right_singular_vectors", "directions",
    "singular_vectors", "vectors", "svs", "components",
]


def arrays(obj):
    out = []
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, dict):
        for k in PREFERRED_KEYS:
            if k in obj:
                out.extend(arrays(obj[k]))
        for k, v in obj.items():
            if k not in PREFERRED_KEYS:
                out.extend(arrays(v))
    elif isinstance(obj, (list,tuple)):
        for x in obj[:100]:
            out.extend(arrays(x))
    return out


def extract_vec(obj, sv: int, d_model: int, layer: int):
    if isinstance(obj, dict):
        for k in [layer, str(layer), f"layer_{layer}", f"L{layer}", f"L{layer:02d}"]:
            if k in obj:
                try:
                    return extract_vec(obj[k], sv, d_model, layer)
                except Exception:
                    pass

    for a in arrays(obj):
        t = torch.as_tensor(a)
        if t.ndim == 1 and t.numel() == d_model:
            return t.float()
        if t.ndim == 2:
            if t.shape[1] == d_model and sv < t.shape[0]:
                return t[sv].float()
            if t.shape[0] == d_model and sv < t.shape[1]:
                return t[:,sv].float()
        if t.ndim == 3 and layer < t.shape[0]:
            x = t[layer]
            if x.ndim == 2:
                if x.shape[1] == d_model and sv < x.shape[0]:
                    return x[sv].float()
                if x.shape[0] == d_model and sv < x.shape[1]:
                    return x[:,sv].float()

    raise ValueError(f"Could not extract layer={layer}, sv={sv}, d={d_model}")


# -----------------------------------------------------------------------------
# Model / token helpers
# -----------------------------------------------------------------------------

def get_blocks(model):
    for fn in [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]:
        try:
            z = fn(model)
            if len(z):
                return z
        except Exception:
            pass
    raise RuntimeError("Could not locate decoder blocks.")


def occurrences(text: str, target: str):
    out = []
    start = 0
    while True:
        i = text.find(target, start)
        if i < 0:
            break
        out.append((i, i+len(target)))
        start = i + max(1, len(target))
    return out


def locate_target(tokenizer, text: str, target: str, occurrence: int, max_length: int):
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    occs = occurrences(text, target)
    if not occs:
        occs = occurrences(text.lower(), target.lower())
    if not occs:
        raise ValueError(f"Target {target!r} not found in: {text}")

    which = occurrence if occurrence >= 0 else len(occs)+occurrence
    if which < 0 or which >= len(occs):
        raise ValueError(
            f"occurrence={occurrence} invalid for {target!r}; found {len(occs)}"
        )
    c0, c1 = occs[which]
    hits = [
        i for i,(a,b) in enumerate(offsets)
        if not (a == b == 0) and max(a,c0) < min(b,c1)
    ]
    if not hits:
        raise ValueError(f"No token overlap for target {target!r}")
    # Last subtoken of the matched string.
    return enc, hits[-1], hits


def decode_tok(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def finite(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def avg(xs):
    xs = finite(xs)
    return float(np.mean(xs)) if xs else None


def sd(xs):
    xs = finite(xs)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def corr(a,b):
    pairs = [(x,y) for x,y in zip(a,b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs], float)
    y = np.asarray([p[1] for p in pairs], float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    return (x-mu)/s


def hc3_ols(y, X, names):
    """
    OLS with HC3 robust standard errors.
    Returns coefficients, SEs, t stats, n, rank, R^2.
    Uses pseudo-inverse so dummy-heavy fixed-effects designs are safe.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]

    beta = np.linalg.pinv(X) @ y
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)

    Hdiag = np.sum((X @ XtX_inv) * X, axis=1)
    denom = np.clip(1.0 - Hdiag, 1e-6, None)
    u = resid / denom
    meat = X.T @ ((u*u)[:,None] * X)
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se>0)

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y-y.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else None

    return {
        "n": int(len(y)),
        "rank": int(np.linalg.matrix_rank(X)),
        "r2": r2,
        "coefficients": {
            name: {
                "beta": float(beta[i]),
                "se_hc3": float(se[i]),
                "t_hc3": float(t[i]) if np.isfinite(t[i]) else None,
            }
            for i,name in enumerate(names)
        }
    }


def fixed_effect_role_regression(rows, outcome: str, include_norm: bool):
    """
    Core+novel rows only; positive/negative roles only.

    outcome ~ role_positive + lexical fixed effects
              + target_position_z + seq_len_z
              + entropy_z + surprisal_z [+ hidden_norm_z]

    The role coefficient is the key estimate: does the functional role matter
    after subtracting token identity and nuisance variables?
    """
    rr = [
        r for r in rows
        if r["tier"] in {"core","novel"}
        and r["role"] in {"positive","negative"}
        and r[outcome] is not None
    ]
    if not rr:
        return None

    lexical = sorted({r["lexical_key"] for r in rr})
    baseline = lexical[0]

    pos = np.asarray([r["target_token_index"] for r in rr], float)
    seqlen = np.asarray([r["sequence_length"] for r in rr], float)
    entropy = np.asarray([r["next_token_entropy_nats"] for r in rr], float)
    surprisal = np.asarray([r["actual_next_surprisal_nats"] for r in rr], float)
    hnorm = np.asarray([r["hidden_norm"] for r in rr], float)

    columns = [np.ones(len(rr)), np.asarray([r["role"]=="positive" for r in rr], float)]
    names = ["intercept", "role_positive"]

    for lex in lexical[1:]:
        columns.append(np.asarray([r["lexical_key"]==lex for r in rr], float))
        names.append(f"lexical_FE[{lex}]")

    columns += [zscore(pos), zscore(seqlen), zscore(entropy), zscore(surprisal)]
    names += ["target_position_z","sequence_length_z","entropy_z","surprisal_z"]

    if include_norm:
        columns.append(zscore(hnorm))
        names.append("hidden_norm_z")

    X = np.column_stack(columns)
    y = np.asarray([r[outcome] for r in rr], float)
    result = hc3_ols(y, X, names)
    result["lexical_baseline"] = baseline
    result["outcome"] = outcome
    return result


def per_lexical_effects(rows):
    out = {}
    for lex in sorted({r["lexical_key"] for r in rows}):
        rr = [
            r for r in rows
            if r["lexical_key"] == lex
            and r["tier"] in {"core","novel"}
            and r["role"] in {"positive","negative"}
        ]
        pos = [r for r in rr if r["role"]=="positive"]
        neg = [r for r in rr if r["role"]=="negative"]
        if not pos or not neg:
            continue
        out[lex] = {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "positive_mean_raw": avg([r["activation_raw"] for r in pos]),
            "negative_mean_raw": avg([r["activation_raw"] for r in neg]),
            "raw_difference": avg([r["activation_raw"] for r in pos]) - avg([r["activation_raw"] for r in neg]),
            "positive_mean_cos": avg([r["activation_cos"] for r in pos]),
            "negative_mean_cos": avg([r["activation_cos"] for r in neg]),
            "cos_difference": avg([r["activation_cos"] for r in pos]) - avg([r["activation_cos"] for r in neg]),
        }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    svs = [int(x.strip()) for x in args.sv_indices.split(",") if x.strip()]
    if len(layers) != len(svs):
        raise ValueError("--layers and --sv-indices must have same length.")

    signs = [1]*len(layers)
    if args.signs:
        signs = [int(x.strip()) for x in args.signs.split(",") if x.strip()]
        if len(signs) != len(layers) or any(x not in (-1,1) for x in signs):
            raise ValueError("--signs must supply +1/-1 for each layer.")

    cases = list(CASES)
    if args.custom_cases:
        with open(args.custom_cases,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    c.setdefault("occurrence",0)
                    c.setdefault("lexical_key",c["target"])
                    c.setdefault("family","custom")
                    c.setdefault("tier","novel")
                    c.setdefault("role","boundary")
                    cases.append(c)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cases] {len(cases)}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    kwargs = dict(
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    print(f"[model] loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    blocks = get_blocks(model)
    d_model = int(model.config.hidden_size)

    explicit = parse_explicit(args.direction_file)
    dirs = {}
    direction_meta = {}
    for layer, sv, sign in zip(layers,svs,signs):
        path = explicit.get(layer)
        if path is None:
            path = discover_file(Path(args.directions_dir), layer)
        v = extract_vec(load_obj(path), sv, d_model, layer).reshape(-1)
        v = float(sign) * v / v.norm().clamp_min(1e-12)
        dirs[layer] = v.cpu()
        direction_meta[str(layer)] = {
            "sv_index_zero_based": sv,
            "sign": sign,
            "file": str(path),
        }
        print(f"[direction] L{layer} SV{sv} sign={sign:+d} <- {path}")

    captured = {}
    handles = []
    for layer in layers:
        if args.hook_position == "pre":
            def make_hook(idx):
                def hook(module, inputs):
                    captured[idx] = inputs[0][0].detach().float().cpu()
                return hook
            handles.append(blocks[layer].register_forward_pre_hook(make_hook(layer)))
        else:
            def make_hook(idx):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output,(tuple,list)) else output
                    captured[idx] = h[0].detach().float().cpu()
                return hook
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))

    input_device = model.get_input_embeddings().weight.device
    rows = []
    traces = []

    with torch.inference_mode():
        for ci, case in enumerate(cases,1):
            try:
                enc, target_idx, target_subtokens = locate_target(
                    tokenizer, case["text"], case["target"],
                    int(case.get("occurrence",0)), args.max_length
                )
            except Exception as e:
                print(f"[skip] {case['id']}: {e}")
                continue

            input_ids = enc["input_ids"].to(input_device)
            mask = enc.get("attention_mask")
            if mask is not None:
                mask = mask.to(input_device)

            captured.clear()
            output = model(
                input_ids=input_ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0].detach().float().cpu()
            ids = input_ids[0].detach().cpu()
            seq_len = int(ids.numel())

            lp = torch.log_softmax(logits[target_idx], dim=-1)
            prob = lp.exp()
            entropy = float(-(prob*lp).sum())
            top_prob = float(prob.max())
            next_id = int(ids[target_idx+1]) if target_idx+1 < seq_len else None
            surprisal = float(-lp[next_id]) if next_id is not None else None
            next_token = decode_tok(tokenizer,next_id) if next_id is not None else None

            for layer, sv in zip(layers,svs):
                H = captured[layer]
                v = dirs[layer]
                raw = H @ v
                hnorm = H.norm(dim=-1)
                cos = raw / hnorm.clamp_min(1e-12)

                row = {
                    **case,
                    "layer": layer,
                    "sv_index_zero_based": sv,
                    "target_token_index": int(target_idx),
                    "target_position_fraction": float(target_idx/max(1,seq_len-1)),
                    "sequence_length": seq_len,
                    "target_token_decoded": decode_tok(tokenizer,int(ids[target_idx])),
                    "activation_raw": float(raw[target_idx]),
                    "activation_cos": float(cos[target_idx]),
                    "hidden_norm": float(hnorm[target_idx]),
                    "activation_prev1": float(raw[target_idx-1]) if target_idx>0 else None,
                    "activation_next1": float(raw[target_idx+1]) if target_idx+1<seq_len else None,
                    "activation_next2": float(raw[target_idx+2]) if target_idx+2<seq_len else None,
                    "next_token_entropy_nats": entropy,
                    "next_token_top_probability": top_prob,
                    "actual_next_surprisal_nats": surprisal,
                    "actual_next_token": next_token,
                }
                rows.append(row)

                lo = max(0,target_idx-args.trace_left)
                hi = min(seq_len,target_idx+args.trace_right+1)
                traces.append({
                    "case_id": case["id"],
                    "layer": layer,
                    "trace": [
                        {
                            "rel": j-target_idx,
                            "token": decode_tok(tokenizer,int(ids[j])),
                            "raw": float(raw[j]),
                            "cos": float(cos[j]),
                            "hidden_norm": float(hnorm[j]),
                        }
                        for j in range(lo,hi)
                    ],
                })

            print(
                f"[{ci:03d}/{len(cases)}] {case['id']:<24} "
                f"{case['tier']:<14} {case['role']:<8} "
                f"idx={target_idx}"
            )

            del output, logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # -------------------------
    # Analysis
    # -------------------------
    summary = {
        "model": args.model,
        "hypothesis": "forward semantic dependency / prospective relational completion",
        "n_cases_requested": len(cases),
        "n_rows": len(rows),
        "layers": {},
        "cross_layer": {},
        "directions": direction_meta,
    }

    for layer in layers:
        rr = [r for r in rows if r["layer"]==layer]
        tiers = {}
        for tier in sorted({r["tier"] for r in rr}):
            tr = [r for r in rr if r["tier"]==tier]
            tiers[tier] = {}
            for role in sorted({r["role"] for r in tr}):
                sr = [r for r in tr if r["role"]==role]
                tiers[tier][role] = {
                    "n": len(sr),
                    "mean_raw": avg([r["activation_raw"] for r in sr]),
                    "mean_cos": avg([r["activation_cos"] for r in sr]),
                    "mean_hidden_norm": avg([r["hidden_norm"] for r in sr]),
                    "mean_entropy": avg([r["next_token_entropy_nats"] for r in sr]),
                }

        lex = per_lexical_effects(rr)
        core_lex = {
            k:v for k,v in lex.items()
            if any(r["lexical_key"]==k and r["tier"]=="core" for r in rr)
        }
        novel_lex = {
            k:v for k,v in lex.items()
            if any(r["lexical_key"]==k and r["tier"]=="novel" for r in rr)
        }

        reg_cos = fixed_effect_role_regression(rr,"activation_cos",include_norm=False)
        reg_raw = fixed_effect_role_regression(rr,"activation_raw",include_norm=True)

        all_core_novel = [
            r for r in rr
            if r["tier"] in {"core","novel"} and r["role"] in {"positive","negative"}
        ]

        summary["layers"][str(layer)] = {
            "tier_role_means": tiers,
            "core_lexical_effects": core_lex,
            "novel_lexical_effects": novel_lex,
            "fixed_effect_regression_cos": reg_cos,
            "fixed_effect_regression_raw": reg_raw,
            "activation_entropy_r_all": corr(
                [r["activation_cos"] for r in rr],
                [r["next_token_entropy_nats"] for r in rr],
            ),
            "activation_surprisal_r_all": corr(
                [r["activation_cos"] for r in rr],
                [r["actual_next_surprisal_nats"] for r in rr],
            ),
            "positive_minus_negative_win_fraction_by_lexical_key": (
                float(np.mean([v["cos_difference"]>0 for v in lex.values()]))
                if lex else None
            ),
            "n_core_novel_rows": len(all_core_novel),
        }

    if len(layers) >= 2:
        for i in range(len(layers)):
            for j in range(i+1,len(layers)):
                a,b = layers[i],layers[j]
                da = {r["id"]:r for r in rows if r["layer"]==a}
                db = {r["id"]:r for r in rows if r["layer"]==b}
                ks = sorted(set(da)&set(db))
                summary["cross_layer"][f"{a}_vs_{b}"] = {
                    "n_common": len(ks),
                    "raw_r": corr(
                        [da[k]["activation_raw"] for k in ks],
                        [db[k]["activation_raw"] for k in ks],
                    ),
                    "cos_r": corr(
                        [da[k]["activation_cos"] for k in ks],
                        [db[k]["activation_cos"] for k in ks],
                    ),
                    "core_novel_cos_r": corr(
                        [da[k]["activation_cos"] for k in ks if da[k]["tier"] in {"core","novel"}],
                        [db[k]["activation_cos"] for k in ks if da[k]["tier"] in {"core","novel"}],
                    ),
                }

    # -------------------------
    # Write
    # -------------------------
    json_path = out_dir/"results.json"
    with json_path.open("w",encoding="utf-8") as f:
        json.dump(
            {"summary":summary,"items":rows,"traces":traces},
            f, indent=2, ensure_ascii=False
        )

    csv_path = out_dir/"items.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    md_path = out_dir/"summary.md"
    with md_path.open("w",encoding="utf-8") as f:
        f.write("# Forward Semantic Dependency Probe v2\n\n")
        f.write("Primary hypothesis: **forward semantic dependency / prospective relational completion**.\n\n")
        f.write("The key statistic is the `role_positive` coefficient in the cosine fixed-effects regression. ")
        f.write("It estimates the functional-role effect after controlling for lexical identity, token position, ")
        f.write("sequence length, entropy, and surprisal.\n\n")

        for layer in layers:
            s = summary["layers"][str(layer)]
            f.write(f"## Layer {layer} / SV{svs[layers.index(layer)]} (zero-based)\n\n")

            reg = s["fixed_effect_regression_cos"]
            if reg:
                c = reg["coefficients"]["role_positive"]
                f.write("### Residualized functional-role effect (cosine)\n\n")
                f.write(f"- beta: `{c['beta']}`\n")
                f.write(f"- HC3 SE: `{c['se_hc3']}`\n")
                f.write(f"- HC3 t: `{c['t_hc3']}`\n")
                f.write(f"- regression R^2: `{reg['r2']}`\n\n")

            f.write("### Core lexical effects\n\n")
            f.write("| lexical item | positive cos | negative cos | difference |\n")
            f.write("|---|---:|---:|---:|\n")
            for k,v in s["core_lexical_effects"].items():
                f.write(
                    f"| {k} | {v['positive_mean_cos']:.5f} | "
                    f"{v['negative_mean_cos']:.5f} | {v['cos_difference']:+.5f} |\n"
                )
            f.write("\n### Novel generalization effects\n\n")
            f.write("| lexical item | positive cos | negative cos | difference |\n")
            f.write("|---|---:|---:|---:|\n")
            for k,v in s["novel_lexical_effects"].items():
                f.write(
                    f"| {k} | {v['positive_mean_cos']:.5f} | "
                    f"{v['negative_mean_cos']:.5f} | {v['cos_difference']:+.5f} |\n"
                )

            f.write("\n### Boundary tiers\n\n")
            for tier in ["clausal","discourse_only","generic","punctuation","natural"]:
                if tier in s["tier_role_means"]:
                    f.write(f"- **{tier}**: `{s['tier_role_means'][tier]}`\n")
            f.write("\n")

        if summary["cross_layer"]:
            f.write("## Cross-layer consistency\n\n")
            for k,v in summary["cross_layer"].items():
                f.write(f"- **{k}**: raw r={v['raw_r']}, cosine r={v['cos_r']}, core+novel cosine r={v['core_novel_cos_r']}\n")

        f.write("\n## Interpretation guide\n\n")
        f.write("- Strong positive `role_positive` after fixed effects => the role is not reducible to token identity/position/predictive uncertainty.\n")
        f.write("- Novel lexical items with positive within-token effects => abstraction generalizes beyond the words that originally revealed the SV.\n")
        f.write("- Clausal > discourse-only would support relational completion over generic 'payload incoming'.\n")
        f.write("- Generic controls equally high would argue for a broader grammatical-dependency interpretation.\n")
        f.write("- Natural examples should reproduce, but are not used as the clean causal contrast.\n")

    print("\n[done]")
    print(json_path)
    print(csv_path)
    print(md_path)
    print("\nFeed back results.json.")


if __name__ == "__main__":
    main()
