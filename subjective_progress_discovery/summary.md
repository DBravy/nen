# Subjective-progress discovery summary

- Events: **58** total; **45** direct-analysis eligible across **19** rollouts.
- Objective statuses among eligible events: correct=24, incorrect=17, ambiguous=4.
- Model representation: 23 layers, hidden size 2880; scanned SVs/layer=32.
- Direct transition: mean t=0:3 minus mean t=-5:-1, then event minus matched controls.
- Direct validation: leave-one-rollout-out; rollout is the independent unit.

## Top direct residual-space layers

| rank | layer | CV cosine | std effect | Holm p | rollout consistency | correct | incorrect | status-dir cosine | nearest SV | |cos| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.1208 | 1.742 | 0.0023 | 1.000 | 0.1624 | 0.1047 | 0.082 | 2 | 0.015 |
| 2 | 21 | 0.1233 | 1.683 | 0.0023 | 1.000 | 0.1610 | 0.1085 | 0.092 | 28 | 0.026 |
| 3 | 19 | 0.1159 | 1.613 | 0.0023 | 1.000 | 0.1548 | 0.1022 | 0.060 | 30 | 0.012 |
| 4 | 22 | 0.1174 | 1.578 | 0.0023 | 1.000 | 0.1489 | 0.1060 | 0.091 | 32 | 0.013 |
| 5 | 18 | 0.1177 | 1.543 | 0.0023 | 1.000 | 0.1692 | 0.0865 | 0.001 | 2 | 0.011 |
| 6 | 13 | 0.1212 | 1.267 | 0.0023 | 0.842 | 0.1459 | 0.1313 | 0.047 | 28 | 0.022 |
| 7 | 17 | 0.1107 | 1.402 | 0.0023 | 0.947 | 0.1520 | 0.0872 | -0.022 | 32 | 0.011 |
| 8 | 4 | 0.1636 | 1.305 | 0.0023 | 1.000 | 0.2033 | 0.1352 | 0.207 | 15 | 0.053 |
| 9 | 10 | 0.1206 | 1.225 | 0.0023 | 0.842 | 0.1486 | 0.1282 | 0.076 | 29 | 0.036 |
| 10 | 11 | 0.1271 | 1.219 | 0.0023 | 0.842 | 0.1430 | 0.1430 | 0.118 | 15 | 0.029 |

## Top SV candidates (delta)

| rank | layer | SV | effect | std effect | Holm p | consistency | correct | incorrect | preservation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1 | -0.4476 | -0.745 | 1 | 0.842 | -0.4680 | -0.5028 | 1.000 |
| 2 | 11 | 25 | -4.1660 | -0.638 | 1 | 0.737 | -7.3859 | -4.5502 | 1.000 |
| 3 | 2 | 1 | -0.2591 | -0.592 | 1 | 0.842 | -0.2834 | -0.3337 | 1.000 |
| 4 | 17 | 32 | 6.8974 | 0.637 | 1 | 0.737 | 6.7432 | 8.4520 | 0.978 |
| 5 | 4 | 24 | 1.9932 | 0.630 | 1 | 0.842 | 1.6906 | 1.9317 | 0.848 |
| 6 | 5 | 32 | 3.6224 | 0.750 | 1 | 0.789 | 4.4519 | 2.3697 | 0.654 |
| 7 | 21 | 28 | -48.9014 | -0.852 | 1 | 0.842 | -49.9528 | -24.2382 | 0.496 |
| 8 | 8 | 27 | 6.4376 | 0.722 | 1 | 0.684 | 9.1595 | 4.6752 | 0.726 |
| 9 | 19 | 28 | -10.2469 | -0.554 | 1 | 0.789 | -13.5622 | -9.8280 | 0.959 |
| 10 | 22 | 24 | -16.3805 | -0.570 | 1 | 0.632 | -25.6290 | -18.0944 | 1.000 |

## What to inspect next

The strongest dopamine-like candidate is not simply the largest effect. Prefer a layer/direction with positive held-out CV cosine across rollouts, preservation in both correct and incorrect subjective-progress events, and good correct↔incorrect cross-status transfer. The nearest-SV column shows whether the directly learned direction is approximately one existing J-lens singular vector or a mixture outside any single scanned SV.

`progress_directions.npz` contains the full-data unit direction for every layer. Use a top-ranked direction only in a new causal intervention experiment; do not use the present campaign as the final behavioral test set.
