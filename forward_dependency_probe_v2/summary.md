# Forward Semantic Dependency Probe v2

Primary hypothesis: **forward semantic dependency / prospective relational completion**.

The key statistic is the `role_positive` coefficient in the cosine fixed-effects regression. It estimates the functional-role effect after controlling for lexical identity, token position, sequence length, entropy, and surprisal.

## Layer 7 / SV12 (zero-based)

### Residualized functional-role effect (cosine)

- beta: `0.013213933957784125`
- HC3 SE: `0.002086263788772801`
- HC3 t: `6.3337790881933165`
- regression R^2: `0.844247447328239`

### Core lexical effects

| lexical item | positive cos | negative cos | difference |
|---|---:|---:|---:|
| colon | 0.08490 | 0.05597 | +0.02892 |
| fact | 0.08502 | 0.07196 | +0.01307 |
| feel | 0.08360 | 0.07037 | +0.01323 |
| point | 0.08365 | 0.06876 | +0.01489 |
| spot | 0.08738 | 0.07954 | +0.00783 |
| that | 0.08503 | 0.06852 | +0.01651 |
| where | 0.08672 | 0.08025 | +0.00647 |
| who | 0.09147 | 0.09057 | +0.00090 |

### Novel generalization effects

| lexical item | positive cos | negative cos | difference |
|---|---:|---:|---:|
| called | 0.05982 | 0.05259 | +0.00724 |
| how | 0.05456 | 0.04162 | +0.01294 |
| means | 0.05645 | 0.02583 | +0.03062 |
| named | 0.03321 | 0.02979 | +0.00342 |
| what | 0.09444 | 0.07818 | +0.01626 |
| whether | 0.05870 | 0.05213 | +0.00658 |
| which | 0.07393 | 0.07825 | -0.00432 |
| whose | 0.04845 | 0.03548 | +0.01297 |
| why | 0.05969 | 0.04404 | +0.01565 |

### Boundary tiers

- **clausal**: `{'boundary': {'n': 15, 'mean_raw': 39.1079709370931, 'mean_cos': 0.05354307318727176, 'mean_hidden_norm': 727.9027994791667, 'mean_entropy': 3.445000743865967}}`
- **discourse_only**: `{'boundary': {'n': 12, 'mean_raw': 24.49512768785159, 'mean_cos': 0.033165585099292606, 'mean_hidden_norm': 751.2979075113932, 'mean_entropy': 2.2976887126763663}}`
- **generic**: `{'control': {'n': 12, 'mean_raw': 37.48099088668823, 'mean_cos': 0.05240716319531202, 'mean_hidden_norm': 719.0074310302734, 'mean_entropy': 4.369558572769165}}`
- **punctuation**: `{'boundary': {'n': 4, 'mean_raw': 42.65255928039551, 'mean_cos': 0.06084360275417566, 'mean_hidden_norm': 699.72021484375, 'mean_entropy': 6.395753383636475}, 'control': {'n': 2, 'mean_raw': 39.455970764160156, 'mean_cos': 0.053576743230223656, 'mean_hidden_norm': 733.8774108886719, 'mean_entropy': 5.020391225814819}}`
- **natural**: `{'natural': {'n': 6, 'mean_raw': 67.89259465535481, 'mean_cos': 0.09388792266448338, 'mean_hidden_norm': 724.1615498860677, 'mean_entropy': 2.773435433705648}}`

## Layer 8 / SV12 (zero-based)

### Residualized functional-role effect (cosine)

- beta: `0.01602030192017167`
- HC3 SE: `0.0022907649166078355`
- HC3 t: `6.993429052464506`
- regression R^2: `0.7582502929432637`

### Core lexical effects

| lexical item | positive cos | negative cos | difference |
|---|---:|---:|---:|
| colon | 0.07183 | 0.03306 | +0.03877 |
| fact | 0.08213 | 0.07508 | +0.00705 |
| feel | 0.08414 | 0.06983 | +0.01431 |
| point | 0.08359 | 0.06877 | +0.01482 |
| spot | 0.08686 | 0.07180 | +0.01506 |
| that | 0.07275 | 0.05265 | +0.02010 |
| where | 0.08877 | 0.07911 | +0.00966 |
| who | 0.08724 | 0.07450 | +0.01275 |

### Novel generalization effects

| lexical item | positive cos | negative cos | difference |
|---|---:|---:|---:|
| called | 0.06513 | 0.05501 | +0.01012 |
| how | 0.06364 | 0.04462 | +0.01901 |
| means | 0.06015 | 0.03603 | +0.02412 |
| named | 0.04963 | 0.04415 | +0.00548 |
| what | 0.08903 | 0.06634 | +0.02269 |
| whether | 0.06707 | 0.05535 | +0.01172 |
| which | 0.07239 | 0.07449 | -0.00210 |
| whose | 0.05926 | 0.04355 | +0.01571 |
| why | 0.06243 | 0.03622 | +0.02621 |

### Boundary tiers

- **clausal**: `{'boundary': {'n': 15, 'mean_raw': 50.887977345784506, 'mean_cos': 0.058395543073614435, 'mean_hidden_norm': 872.9450113932292, 'mean_entropy': 3.445000743865967}}`
- **discourse_only**: `{'boundary': {'n': 12, 'mean_raw': 36.78959679603577, 'mean_cos': 0.04198039183393121, 'mean_hidden_norm': 887.8504130045573, 'mean_entropy': 2.2976887126763663}}`
- **generic**: `{'control': {'n': 12, 'mean_raw': 47.994285583496094, 'mean_cos': 0.054640463863809906, 'mean_hidden_norm': 886.3412577311198, 'mean_entropy': 4.369558572769165}}`
- **punctuation**: `{'boundary': {'n': 4, 'mean_raw': 57.835537910461426, 'mean_cos': 0.06882528774440289, 'mean_hidden_norm': 840.3242950439453, 'mean_entropy': 6.395753383636475}, 'control': {'n': 2, 'mean_raw': 51.010589599609375, 'mean_cos': 0.05562855489552021, 'mean_hidden_norm': 912.7646179199219, 'mean_entropy': 5.020391225814819}}`
- **natural**: `{'natural': {'n': 6, 'mean_raw': 76.62381935119629, 'mean_cos': 0.08714414263765018, 'mean_hidden_norm': 877.6413269042969, 'mean_entropy': 2.773435433705648}}`

## Cross-layer consistency

- **7_vs_8**: raw r=0.930363495798167, cosine r=0.9119949520959351, core+novel cosine r=0.8770403558994472

## Interpretation guide

- Strong positive `role_positive` after fixed effects => the role is not reducible to token identity/position/predictive uncertainty.
- Novel lexical items with positive within-token effects => abstraction generalizes beyond the words that originally revealed the SV.
- Clausal > discourse-only would support relational completion over generic 'payload incoming'.
- Generic controls equally high would argue for a broader grammatical-dependency interpretation.
- Natural examples should reproduce, but are not used as the clean causal contrast.
