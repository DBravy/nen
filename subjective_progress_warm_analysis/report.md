# Subjective-progress W-arm analysis

This report is exploratory candidate discovery, not held-out confirmation or a causal result.

## Data and design

- Input annotations: 61 (58 mapped and analyzed; 3 excluded).
- Event-bearing rollouts: 20.
- Objective status among analyzed events: {'ambiguous': 5, 'correct': 32, 'incorrect': 21}.
- Window: t=-20…+20 in true generated-token coordinates, bounded to one assistant message, with missing positions retained.
- Normalization: robust_zscore within rollout using channel `analysis` and event-relevant message indices [0].
- Controls: target 5 per event (minimum 3), same rollout and message role/index, outside every event span ±20 generated tokens.
- Scalar delta: mean t=0:3 minus mean t=-5:-1.
- Peak/trough baseline: mean t=-15:-5; search interval t=-3:5.
- Inference: event minus its mean matched control, then equal-weighted over rollout means.
- Trace controls are masked to each matched event's available offsets, so plotted support is paired at every t.

## Ranked candidate arms

Ranking metric: `delta`. The score combines absolute rollout-standardized effect and rollout directional consistency, with a 0.25 floor plus a 0.75 multiplier for preservation of the oriented effect in both correct and incorrect events.

| rank | arm | score | grouped effect | bootstrap 95% CI | Holm p | rollout consistency | correct effect | incorrect effect | preservation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | L22_SV32 | 0.066 | 0.294 | [-0.014, 0.665] | 0.3793 | 0.632 | 0.011 | 0.266 | 0.038 |
| 2 | L17_SV28 | 0.027 | -0.109 | [-0.364, 0.145] | 0.7679 | 0.579 | -0.615 | 0.681 | 0.000 |
| 3 | L18_SV30 | 0.021 | 0.134 | [-0.146, 0.428] | 0.7679 | 0.421 | -0.031 | 0.344 | 0.000 |

## Control audit

- Match tiers: {'same_rollout_broad_category': 13, 'same_rollout_exact_token': 118, 'same_rollout_fine_class': 27, 'same_step_broad_category': 11, 'same_step_exact_token': 94, 'same_step_fine_class': 27}.
- Events below the target control count: 0.
- Same-step controls: 132 / 290.
- Exact-token controls: 212 / 290.
- Exact message-boundary coverage matches: 129 / 290 (median coverage distance 4.0).
- Reused control anchors: 66 / 290.

## Interpretation cautions

- All 58 annotated t=0 tokens are punctuation tokens in this dataset. Lexical matching reduces, but cannot eliminate, the sentence-boundary confound.
- Partial windows are status-dependent; each CSV row reports the exact usable event and rollout count.
- A peak statistic is selection-biased by construction, so only its matched-control contrast is interpretable.
- SVD direction signs are arbitrary. A repeatable decrease can be as interesting as an increase.
- Selection and inference use the same 20 event-bearing rollouts. Validate any selected arm on wholly unseen rollouts before steering.
- Incorrect events are a target condition, not label noise: preservation there is central to the subjective-progress hypothesis.

## Files

- `event_windows.jsonl`: raw and normalized event/control traces with tokens and locations.
- `event_measurements.csv`: paired event-level scalar measurements.
- `control_measurements.csv`: each sampled control's measurements and match metadata.
- `trace_summary.csv`: event-weighted and equal-rollout event-triggered means and SEMs.
- `grouped_statistics.csv`: bootstrap, sign-flip, status, and paired-status results.
- `arm_rankings.csv` / `.json`: transparent ranking components.
- `plot_manifest.json`: authoritative list of plots produced by this run.

Current-run plots:

- `plots/L22_SV32.svg`
- `plots/L17_SV28.svg`
- `plots/L18_SV30.svg`
