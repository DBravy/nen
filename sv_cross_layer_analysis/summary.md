# Cross-layer singular-vector alignment

Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
Top K directions: 64
Layer pairs analyzed: 253
Random rank-label permutations per pair: 2000

## Headline statistics

- Chance exact-rank rate for K=64: **0.0156** (1.56%).
- Mean exact-rank best-match rate across analyzed pairs: **0.130**.
- Mean same-rank |cosine| across analyzed pairs: **0.164**.
- Adjacent-layer exact-rank best-match rate: **0.465**.
- Adjacent-layer same-rank |cosine|: **0.478**.
- Adjacent-layer raw same-rank sign-flip rate: **0.379**.

The primary identity metric is absolute cosine because SVD sign is arbitrary. The raw sign-flip statistic is reported only to characterize orientation changes.

## Most rank-persistent SVs

| SV | exact-rank rate | mean same-rank |cos| | adjacent exact rate | adjacent |cos| |
|---:|---:|---:|---:|---:|
| SV00 | 0.751 | 0.346 | 1.000 | 0.847 |
| SV01 | 0.589 | 0.355 | 0.955 | 0.862 |
| SV02 | 0.344 | 0.257 | 0.909 | 0.806 |
| SV03 | 0.265 | 0.239 | 0.864 | 0.750 |
| SV06 | 0.245 | 0.236 | 0.682 | 0.635 |
| SV18 | 0.229 | 0.235 | 0.545 | 0.546 |
| SV11 | 0.209 | 0.236 | 0.682 | 0.635 |
| SV05 | 0.186 | 0.238 | 0.682 | 0.654 |
| SV04 | 0.182 | 0.212 | 0.682 | 0.646 |
| SV14 | 0.182 | 0.207 | 0.636 | 0.612 |
| SV22 | 0.178 | 0.213 | 0.545 | 0.550 |
| SV09 | 0.178 | 0.200 | 0.500 | 0.515 |
| SV10 | 0.166 | 0.206 | 0.409 | 0.495 |
| SV13 | 0.162 | 0.182 | 0.591 | 0.561 |
| SV08 | 0.146 | 0.199 | 0.500 | 0.507 |

## Spectral-gap relationship

The table below correlates the minimum relative local singular-value gap across a same-rank layer pair with alignment persistence.

| subset | alignment metric | Pearson r | Spearman rho | n |
|---|---|---:|---:|---:|
| all_pairs | same_rank_abs_cosine | 0.181 | 0.164 | 16192 |
| all_pairs | same_rank_fraction_of_best | 0.303 | 0.221 | 16192 |
| all_pairs | same_rank_is_best | 0.338 | 0.235 | 16192 |
| all_pairs | same_rank_deficit_from_best | -0.202 | -0.192 | 16192 |
| adjacent_pairs | same_rank_abs_cosine | 0.237 | 0.380 | 1408 |
| adjacent_pairs | same_rank_fraction_of_best | 0.207 | 0.343 | 1408 |
| adjacent_pairs | same_rank_is_best | 0.237 | 0.357 | 1408 |
| adjacent_pairs | same_rank_deficit_from_best | -0.182 | -0.324 | 1408 |

## Rank-label permutation tests

- Pairs with mean diagonal |cosine| above the random-rank null at p<=0.05: **152/253**.
- Pairs with exact-rank best-match rate above the random-rank null at p<=0.05: **111/253**.

See `pair_summary.csv` for pair-level permutation p-values and `sv_pair_matches.csv` for the full source-SV -> target-SV matching table.
