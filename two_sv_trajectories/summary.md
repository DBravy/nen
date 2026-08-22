# Focused SV trajectory analysis

**Indexing:** zero-based throughout this report. `SV_02` is column 2; `SV_09` is column 9.

The earlier scanner's human-readable candidate labels were one-based, so its `SV03` maps to this report's `SV_02`, and its `SV10` maps to this report's `SV_09`.

## Headline geometry

| trajectory | layers | mean adjacent |cos| | min adjacent |cos| | endpoint |cos| | path angle | endpoint angle | reciprocal adjacent | sign flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SV_02 | L2–L11 | 0.851 | 0.757 | 0.031 | 274.4° | 88.2° | 1.000 | 0 |
| SV_09 | L9–L15 | 0.968 | 0.890 | 0.684 | 77.2° | 46.8° | 1.000 | 2 |

## Interpretation aids

- **High adjacent + high endpoint retention** indicates a nearly fixed persistent axis.
- **High adjacent + low endpoint retention** indicates a continuous rotating trajectory: local identity survives even though the endpoint eventually becomes geometrically different.
- `cumulative_path_angle_deg` sums the unoriented adjacent angles; `endpoint_axis_displacement_deg` measures only start-to-current displacement. Their ratio helps distinguish a short/direct path from a long curved one.
- Reciprocal best-match checks are performed among the top-K SVs; they are stricter than same-rank cosine alone.
- Raw SVD sign flips are reported but are not treated as identity changes. Canonical signs are parallel-transported along each fixed-rank chain.

## Spectral context

- **SV_02**: sigma 174.720 → 40.861; relative local gap range 0.1314–0.3811; greedy identity trace leaves the starting rank on 0 layer(s) in the requested window.
- **SV_09**: sigma 23.135 → 21.296; relative local gap range 0.0141–0.0497; greedy identity trace leaves the starting rank on 0 layer(s) in the requested window.

## Lexical-neighborhood retention

- **SV_02**: mean adjacent canonical top-token Jaccard 0.230; longest-span Jaccard 0.000.
- **SV_09**: mean adjacent canonical top-token Jaccard 0.559; longest-span Jaccard 0.041.

Lexical overlap uses only the retained nearest/farthest token lists from the scanner, not the full-vocabulary cosine profile.

## Files to inspect

- `trajectory_layers.csv`: anchor retention and cumulative rotation at each layer.
- `trajectory_steps.csv`: adjacent continuity, reciprocal matching, margins, sign flips, and gaps.
- `pairwise_retention.csv`: all within-window pairwise comparisons.
- `greedy_identity_trace.csv`: whether the direction changes singular-value rank when followed by identity.
- `lexical_pairwise_retention.csv`: sign-canonicalized positive/negative token-neighborhood overlap.
- `lexical_layers.csv`: readable canonical token neighborhoods by layer.
