# Overlay subset composition at the opener grade

## Question

The all-six joint overlay stack was measured **resolvably worse** than baseline:
combined 50.37% vs baseline 53.36%, season-blocked P+ 0.00145 (**read**:
`artifacts/overlay_stack_backtest/20260819T191534Z/result.json`). This study asks
the complementary question: what is the highest composed-policy accuracy at the
Tuesday-opener grade achievable by composing SUBSETS of the project's existing
pick-flipping overlays, rather than the naive all-six stack?

Baseline (**read**: `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`,
sha256 recorded in the artifact): the active `weak_stack` model's 1,537 REG games
2020-2025, graded with the PRODUCTION probability rule
(`home_cover_probability >= 0.5`), baseline accuracy **53.3599%** on 1,503 scored
games (34 pushes retained). **Measured** this session by re-scoring that archive.

## Method

**Measured** (script: `scripts/overlay_subset_composition.py`, artifact:
`artifacts/overlay_subset_composition/20260821T174356Z/result.json`):

- All non-empty subsets of SEVEN flips: the six prospective overlays
  (`coach_fade`, `injury_value_lost_tilt`, `division_revenge_tilt`,
  `backup_qb_fade`, `surface_switch_tilt`, `spread_gap_zone_fade`) plus the
  reconstructed player-arrests back-side flip = **127 subsets**. The 63 subsets
  of the six overlays alone are the rows of the table below without `arrest`.
- Combination rule identical to `scripts/overlay_stack_backtest.py`: each flip
  sets the pick to the complement of the unflipped baseline pick (verified
  programmatically via `verify_no_direction_conflicts`); a subset flips a game
  if ANY member fires. Arrest-member conditions were computed against the
  unflipped baseline, matching how the published arrest evaluation scored it.
- Arrest reconstruction reused `scripts/player_arrests_policy_eval.py`'s own
  frozen machinery (`broad_incident_game_flags` + `apply_frozen_policy`)
  against `data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet`
  (sha256 in the artifact). It reproduces the published 25 flips exactly.
- Paired bootstrap, candidate subset vs unflipped baseline, 20,000 samples,
  seed 20260821, BOTH week-blocked (107 blocks) and season-blocked (6 blocks,
  degenerate per the runner's small-block warning) for every subset.
- The vectorized blocked bootstrap was verified EXACTLY equivalent to
  `nfl_ats.clv.week_blocked_bootstrap` on two check subsets (coach+arrest and
  all-seven, both blockings): all four checks true (**measured**, printed by
  the script run and stored under
  `equivalence_check_vs_nfl_ats_week_blocked_bootstrap`).

This is continuous evidence on already-looked-at windows (several member
overlays were screened/tuned on spans this archive re-touches). No
rotation-registry window is spent.

## Production-chain reproduction check

Published reference (**read**: `docs/player_arrests_policy_eval.md`): candidate
53.7591% vs production 53.3599%, +0.3992 pts, from applying the arrest policy
directly to the frozen opener baseline.

**Measured** this session:

| reference | candidate accuracy | note |
|---|---|---|
| arrest-only on baseline | **53.7591%** | exact reproduction of the published figure |
| coach -> arrest sequential | 54.1583% | true production order (`coach_fade_then_player_arrests_v1`); differs because coach fade is applied first |
| coach+arrest joint OR | 54.2249% | the subset-study composition |

**Inferred**: the published 53.76% figure corresponds to arrest-on-baseline;
the sequential production chain scores higher still on this archive. The
joint-OR convention used across the subset study sits between them.

## Full subset table (127 subsets, sorted by point estimate)

Point estimate is blocking-independent; week-blocked and season-blocked 95%
intervals shown separately. Season intervals come from 6 degenerate blocks —
use the estimate and P+, not the endpoints, as the calibrated read.


| members | flips | candidate % | delta pts (95% week) | P+ wk | delta pts (95% season) | P+ seas |
|---|---|---|---|---|---|---|
| coach+divrev+arrest+sgz | 419 | 55.4225 | +2.063 [-0.602, +4.765] | 0.9304 | +2.063 [-0.986, +4.430] | 0.9154 |
| coach+arrest+sgz | 298 | 55.2229 | +1.863 [-0.464, +4.257] | 0.9350 | +1.863 [-1.598, +5.153] | 0.8581 |
| coach+divrev+sgz | 402 | 55.0898 | +1.730 [-0.922, +4.396] | 0.8978 | +1.730 [-0.991, +3.609] | 0.9128 |
| divrev+arrest+sgz | 338 | 54.9568 | +1.597 [-0.853, +4.074] | 0.8975 | +1.597 [-1.164, +3.965] | 0.8710 |
| coach+divrev+arrest | 269 | 54.8902 | +1.530 [-0.539, +3.629] | 0.9201 | +1.530 [-0.065, +2.915] | 0.9681 |
| coach+sgz | 280 | 54.8237 | +1.464 [-0.737, +3.765] | 0.8940 | +1.464 [-1.728, +4.247] | 0.8433 |
| coach+divrev | 249 | 54.7572 | +1.397 [-0.596, +3.377] | 0.9090 | +1.397 [+0.066, +2.554] | 0.9760 |
| arrest+sgz | 215 | 54.7572 | +1.397 [-0.602, +3.499] | 0.9061 | +1.397 [-1.183, +3.462] | 0.8628 |
| coach+divrev+arrest+sgz+surface | 588 | 54.4245 | +1.065 [-2.135, +4.331] | 0.7297 | +1.065 [-2.128, +3.673] | 0.7615 |
| divrev+arrest | 170 | 54.4245 | +1.065 [-0.534, +2.721] | 0.8912 | +1.065 [-0.616, +2.776] | 0.8796 |
| divrev+sgz | 318 | 54.4245 | +1.065 [-1.320, +3.516] | 0.8056 | +1.065 [-1.468, +3.226] | 0.8066 |
| bkqb+coach+divrev+arrest+sgz | 545 | 54.2249 | +0.865 [-2.472, +4.255] | 0.6845 | +0.865 [-1.226, +2.301] | 0.7854 |
| coach+arrest | 125 | 54.2249 | +0.865 [-0.662, +2.397] | 0.8594 | +0.865 [-0.723, +2.724] | 0.8150 |
| coach+divrev+sgz+surface | 572 | 54.1583 | +0.798 [-2.379, +4.050] | 0.6787 | +0.798 [-2.195, +3.333] | 0.7111 |
| sgz | 194 | 54.1583 | +0.798 [-1.105, +2.811] | 0.7786 | +0.798 [-1.446, +2.447] | 0.7655 |
| bkqb+coach+arrest+sgz | 441 | 54.0918 | +0.732 [-2.310, +3.803] | 0.6658 | +0.732 [-1.731, +2.884] | 0.7301 |
| divrev | 147 | 54.0918 | +0.732 [-0.804, +2.283] | 0.8138 | +0.732 [-0.996, +2.458] | 0.7783 |
| bkqb+coach+divrev+sgz | 530 | 54.0253 | +0.665 [-2.661, +3.995] | 0.6446 | +0.665 [-1.346, +2.307] | 0.7373 |
| coach | 104 | 54.0253 | +0.665 [-0.600, +1.952] | 0.8343 | +0.665 [-0.782, +2.288] | 0.7734 |
| coach+divrev+arrest+surface | 460 | 54.0253 | +0.665 [-2.123, +3.491] | 0.6668 | +0.665 [-1.330, +2.646] | 0.7252 |
| coach+arrest+sgz+surface | 487 | 53.9587 | +0.599 [-2.368, +3.677] | 0.6384 | +0.599 [-2.917, +3.902] | 0.6274 |
| divrev+arrest+sgz+surface | 517 | 53.9587 | +0.599 [-2.442, +3.718] | 0.6391 | +0.599 [-2.221, +3.073] | 0.6710 |
| bkqb+divrev+arrest+sgz | 470 | 53.8922 | +0.532 [-2.649, +3.701] | 0.6150 | +0.532 [-1.645, +2.606] | 0.6751 |
| bkqb+coach+sgz | 425 | 53.8257 | +0.466 [-2.508, +3.488] | 0.6028 | +0.466 [-1.944, +2.615] | 0.6488 |
| coach+divrev+surface | 443 | 53.8257 | +0.466 [-2.287, +3.278] | 0.6179 | +0.466 [-1.509, +2.454] | 0.6616 |
| bkqb+coach+divrev+arrest | 416 | 53.7591 | +0.399 [-2.436, +3.269] | 0.5962 | +0.399 [-0.571, +1.497] | 0.7417 |
| bkqb+arrest+sgz | 364 | 53.7591 | +0.399 [-2.359, +3.293] | 0.5911 | +0.399 [-1.398, +1.944] | 0.6728 |
| arrest | 24 | 53.7591 | +0.399 [-0.269, +1.072] | 0.8516 | +0.399 [+0.131, +0.732] | 0.9982 |
| bkqb+coach+divrev | 399 | 53.6926 | +0.333 [-2.493, +3.152] | 0.5811 | +0.333 [-0.987, +1.592] | 0.6767 |
| coach+sgz+surface | 470 | 53.6261 | +0.266 [-2.642, +3.262] | 0.5608 | +0.266 [-3.043, +3.395] | 0.5664 |
| arrest+sgz+surface | 415 | 53.5595 | +0.200 [-2.611, +3.083] | 0.5441 | +0.200 [-2.627, +2.567] | 0.5673 |
| bkqb+divrev+arrest | 324 | 53.4930 | +0.133 [-2.424, +2.728] | 0.5258 | +0.133 [-2.092, +1.936] | 0.5639 |
| bkqb+divrev+sgz | 452 | 53.4930 | +0.133 [-2.987, +3.271] | 0.5182 | +0.133 [-1.974, +2.263] | 0.5372 |
| divrev+arrest+surface | 378 | 53.4930 | +0.133 [-2.332, +2.677] | 0.5272 | +0.133 [-1.900, +2.247] | 0.5182 |
| divrev+sgz+surface | 498 | 53.4930 | +0.133 [-2.842, +3.211] | 0.5252 | +0.133 [-2.557, +2.657] | 0.5387 |
| bkqb+sgz | 345 | 53.2934 | -0.067 [-2.785, +2.770] | 0.4625 | -0.067 [-1.664, +1.521] | 0.4300 |
| bkqb+divrev | 304 | 53.2269 | -0.133 [-2.672, +2.399] | 0.4449 | -0.133 [-2.713, +1.912] | 0.4657 |
| bkqb+coach+arrest | 291 | 53.1603 | -0.200 [-2.569, +2.130] | 0.4181 | -0.200 [-1.225, +0.994] | 0.3237 |
| coach+arrest+surface | 337 | 53.1603 | -0.200 [-2.593, +2.250] | 0.4245 | -0.200 [-2.307, +1.888] | 0.4132 |
| bkqb+coach+divrev+arrest+sgz+surface | 698 | 53.0938 | -0.266 [-3.874, +3.400] | 0.4345 | -0.266 [-2.793, +2.065] | 0.4173 |
| divrev+surface | 358 | 53.0938 | -0.266 [-2.726, +2.251] | 0.4022 | -0.266 [-2.374, +1.917] | 0.3923 |
| bkqb+coach | 273 | 53.0273 | -0.333 [-2.606, +1.915] | 0.3721 | -0.333 [-1.389, +1.209] | 0.2664 |
| divrev+injury+arrest+sgz | 781 | 53.0273 | -0.333 [-4.316, +3.746] | 0.4314 | -0.333 [-4.155, +3.360] | 0.3737 |
| sgz+surface | 395 | 53.0273 | -0.333 [-3.026, +2.467] | 0.3982 | -0.333 [-2.922, +1.909] | 0.4039 |
| coach+divrev+injury+arrest+sgz | 840 | 52.9607 | -0.399 [-4.522, +3.792] | 0.4253 | -0.399 [-4.410, +3.258] | 0.4276 |
| bkqb+coach+divrev+sgz+surface | 683 | 52.8942 | -0.466 [-4.048, +3.204] | 0.3922 | -0.466 [-3.003, +2.209] | 0.3569 |
| bkqb+arrest | 197 | 52.8942 | -0.466 [-2.460, +1.471] | 0.3032 | -0.466 [-1.697, +0.725] | 0.2169 |
| coach+surface | 319 | 52.8942 | -0.466 [-2.763, +1.863] | 0.3339 | -0.466 [-2.421, +1.639] | 0.3237 |
| bkqb+divrev+arrest+sgz+surface | 631 | 52.7611 | -0.599 [-4.068, +2.953] | 0.3598 | -0.599 [-3.026, +1.787] | 0.3196 |
| divrev+injury+arrest | 675 | 52.7611 | -0.599 [-4.301, +3.158] | 0.3745 | -0.599 [-4.099, +2.621] | 0.3720 |
| divrev+injury+sgz | 771 | 52.7611 | -0.599 [-4.560, +3.460] | 0.3822 | -0.599 [-4.230, +2.973] | 0.3510 |
| bkqb+coach+arrest+sgz+surface | 606 | 52.6946 | -0.665 [-4.069, +2.772] | 0.3395 | -0.665 [-3.425, +2.132] | 0.3234 |
| coach+divrev+injury+arrest | 746 | 52.6946 | -0.665 [-4.614, +3.291] | 0.3679 | -0.665 [-3.867, +2.330] | 0.3475 |
| coach+divrev+injury+sgz | 830 | 52.6946 | -0.665 [-4.784, +3.499] | 0.3755 | -0.665 [-4.545, +2.868] | 0.3688 |
| injury+arrest+sgz | 706 | 52.6946 | -0.665 [-4.519, +3.267] | 0.3627 | -0.665 [-4.527, +2.972] | 0.3741 |
| arrest+surface | 254 | 52.6946 | -0.665 [-2.738, +1.397] | 0.2533 | -0.665 [-2.379, +0.748] | 0.1991 |
| bkqb+coach+divrev+arrest+surface | 587 | 52.6281 | -0.732 [-4.008, +2.600] | 0.3215 | -0.732 [-2.605, +1.322] | 0.2435 |
| coach+injury+arrest+sgz | 767 | 52.6281 | -0.732 [-4.736, +3.329] | 0.3574 | -0.732 [-4.933, +3.156] | 0.3675 |
| bkqb | 176 | 52.5615 | -0.798 [-2.670, +1.051] | 0.1853 | -0.798 [-2.370, +0.671] | 0.1476 |
| divrev+injury | 664 | 52.5615 | -0.798 [-4.491, +2.933] | 0.3360 | -0.798 [-4.099, +2.285] | 0.3174 |
| bkqb+coach+divrev+surface | 571 | 52.4950 | -0.865 [-4.151, +2.446] | 0.2926 | -0.865 [-3.097, +1.276] | 0.2365 |
| coach+divrev+injury | 735 | 52.4950 | -0.865 [-4.781, +3.054] | 0.3307 | -0.865 [-3.747, +1.933] | 0.3052 |
| bkqb+coach+sgz+surface | 590 | 52.4285 | -0.931 [-4.288, +2.495] | 0.2850 | -0.931 [-3.675, +1.963] | 0.2609 |
| bkqb+arrest+sgz+surface | 538 | 52.4285 | -0.931 [-4.124, +2.357] | 0.2775 | -0.931 [-3.187, +1.318] | 0.2081 |
| injury+sgz | 696 | 52.4285 | -0.931 [-4.756, +2.990] | 0.3115 | -0.931 [-4.727, +2.532] | 0.3226 |
| bkqb+divrev+sgz+surface | 613 | 52.3619 | -0.998 [-4.452, +2.525] | 0.2791 | -0.998 [-3.482, +1.512] | 0.2311 |
| coach+injury+sgz | 757 | 52.3619 | -0.998 [-4.977, +3.020] | 0.3095 | -0.998 [-5.067, +2.761] | 0.3198 |
| bkqb+divrev+arrest+surface | 510 | 52.2954 | -1.065 [-4.158, +2.065] | 0.2397 | -1.065 [-3.423, +1.175] | 0.1811 |
| surface | 233 | 52.2289 | -1.131 [-3.063, +0.808] | 0.1168 | -1.131 [-2.743, +0.264] | 0.0528 |
| coach+divrev+injury+arrest+sgz+surface | 927 | 52.0958 | -1.264 [-5.523, +3.046] | 0.2803 | -1.264 [-5.260, +2.244] | 0.2713 |
| divrev+injury+arrest+sgz+surface | 874 | 52.0293 | -1.331 [-5.503, +2.884] | 0.2642 | -1.331 [-5.197, +2.177] | 0.2570 |
| injury+arrest | 586 | 52.0293 | -1.331 [-4.806, +2.202] | 0.2247 | -1.331 [-4.164, +1.103] | 0.1406 |
| bkqb+divrev+surface | 491 | 51.9627 | -1.397 [-4.453, +1.672] | 0.1766 | -1.397 [-3.997, +1.044] | 0.1487 |
| bkqb+sgz+surface | 519 | 51.9627 | -1.397 [-4.567, +1.841] | 0.1908 | -1.397 [-3.588, +0.884] | 0.1118 |
| coach+divrev+injury+arrest+surface | 845 | 51.9627 | -1.397 [-5.394, +2.635] | 0.2480 | -1.397 [-4.934, +1.794] | 0.2146 |
| coach+injury+arrest | 659 | 51.9627 | -1.397 [-5.106, +2.359] | 0.2268 | -1.397 [-4.390, +1.456] | 0.1748 |
| coach+divrev+injury+sgz+surface | 918 | 51.8962 | -1.464 [-5.730, +2.823] | 0.2485 | -1.464 [-5.402, +1.898] | 0.2349 |
| bkqb+coach+arrest+surface | 475 | 51.8297 | -1.530 [-4.447, +1.393] | 0.1436 | -1.530 [-3.539, +0.742] | 0.0866 |
| divrev+injury+arrest+surface | 785 | 51.8297 | -1.530 [-5.354, +2.341] | 0.2197 | -1.530 [-5.034, +1.860] | 0.2107 |
| divrev+injury+sgz+surface | 865 | 51.8297 | -1.530 [-5.688, +2.681] | 0.2361 | -1.530 [-5.284, +1.847] | 0.2112 |
| injury | 575 | 51.8297 | -1.530 [-4.953, +1.969] | 0.1897 | -1.530 [-3.912, +0.749] | 0.1071 |
| coach+divrev+injury+surface | 836 | 51.7631 | -1.597 [-5.592, +2.421] | 0.2167 | -1.597 [-4.890, +1.424] | 0.1686 |
| coach+injury | 648 | 51.7631 | -1.597 [-5.253, +2.115] | 0.1921 | -1.597 [-4.400, +1.027] | 0.1290 |
| bkqb+coach+surface | 458 | 51.6301 | -1.730 [-4.582, +1.121] | 0.1094 | -1.730 [-3.654, +0.739] | 0.0732 |
| bkqb+divrev+injury+arrest+sgz | 870 | 51.6301 | -1.730 [-6.105, +2.688] | 0.2169 | -1.730 [-5.226, +1.680] | 0.1693 |
| divrev+injury+surface | 776 | 51.6301 | -1.730 [-5.563, +2.151] | 0.1911 | -1.730 [-5.137, +1.536] | 0.1341 |
| bkqb+arrest+surface | 397 | 51.5635 | -1.796 [-4.450, +0.811] | 0.0868 | -1.796 [-3.838, +0.267] | 0.0396 |
| coach+injury+arrest+sgz+surface | 863 | 51.5635 | -1.796 [-5.937, +2.449] | 0.1954 | -1.796 [-5.810, +1.549] | 0.1744 |
| injury+arrest+sgz+surface | 809 | 51.5635 | -1.796 [-5.843, +2.411] | 0.1935 | -1.796 [-5.444, +1.227] | 0.1404 |
| bkqb+coach+divrev+injury+arrest+sgz | 924 | 51.4970 | -1.863 [-6.367, +2.628] | 0.2061 | -1.863 [-5.467, +1.304] | 0.1403 |
| bkqb+divrev+injury+sgz | 862 | 51.4970 | -1.863 [-6.221, +2.522] | 0.1993 | -1.863 [-5.441, +1.546] | 0.1450 |
| bkqb+divrev+injury+arrest | 779 | 51.4305 | -1.929 [-6.125, +2.264] | 0.1810 | -1.929 [-4.958, +1.054] | 0.1006 |
| bkqb+coach+divrev+injury+sgz | 916 | 51.3639 | -1.996 [-6.469, +2.492] | 0.1872 | -1.996 [-5.620, +1.173] | 0.1257 |
| bkqb+divrev+injury | 770 | 51.3639 | -1.996 [-6.170, +2.159] | 0.1715 | -1.996 [-4.809, +0.914] | 0.0933 |
| bkqb+injury+arrest+sgz | 804 | 51.3639 | -1.996 [-6.221, +2.326] | 0.1795 | -1.996 [-5.491, +1.108] | 0.1168 |
| coach+injury+sgz+surface | 854 | 51.3639 | -1.996 [-6.109, +2.209] | 0.1696 | -1.996 [-6.047, +1.291] | 0.1404 |
| injury+sgz+surface | 800 | 51.3639 | -1.996 [-5.994, +2.171] | 0.1677 | -1.996 [-5.574, +0.896] | 0.1158 |
| bkqb+coach+divrev+injury+arrest | 844 | 51.2309 | -2.129 [-6.494, +2.213] | 0.1638 | -2.129 [-4.860, +0.605] | 0.0644 |
| bkqb+coach+injury+arrest+sgz | 860 | 51.2309 | -2.129 [-6.527, +2.307] | 0.1683 | -2.129 [-5.939, +1.249] | 0.1209 |
| bkqb+injury+sgz | 796 | 51.2309 | -2.129 [-6.348, +2.153] | 0.1626 | -2.129 [-5.749, +0.949] | 0.0974 |
| bkqb+coach+divrev+injury | 835 | 51.1643 | -2.196 [-6.512, +2.109] | 0.1561 | -2.196 [-4.993, +0.470] | 0.0532 |
| bkqb+surface | 377 | 51.1643 | -2.196 [-4.799, +0.335] | 0.0432 | -2.196 [-4.382, -0.066] | 0.0214 |
| bkqb+coach+injury+sgz | 852 | 51.0978 | -2.262 [-6.614, +2.143] | 0.1531 | -2.262 [-6.133, +1.118] | 0.1067 |
| coach+injury+arrest+surface | 768 | 51.0978 | -2.262 [-6.089, +1.614] | 0.1213 | -2.262 [-5.229, +0.443] | 0.0484 |
| injury+arrest+surface | 707 | 51.0313 | -2.329 [-5.995, +1.404] | 0.1064 | -2.329 [-4.931, +0.000] | 0.0243 |
| coach+injury+surface | 759 | 50.8982 | -2.462 [-6.250, +1.395] | 0.1008 | -2.462 [-5.487, +0.132] | 0.0284 |
| injury+surface | 698 | 50.8317 | -2.528 [-6.155, +1.198] | 0.0873 | -2.528 [-5.026, -0.269] | 0.0086 |
| bkqb+injury+arrest | 700 | 50.6986 | -2.661 [-6.618, +1.324] | 0.0907 | -2.661 [-4.772, -0.625] | 0.0018 |
| bkqb+injury | 691 | 50.6321 | -2.728 [-6.640, +1.198] | 0.0830 | -2.728 [-4.778, -0.750] | 0.0016 |
| bkqb+divrev+injury+arrest+sgz+surface | 950 | 50.5655 | -2.794 [-7.254, +1.668] | 0.1071 | -2.794 [-6.059, +0.000] | 0.0236 |
| bkqb+coach+divrev+injury+arrest+sgz+surface | 999 | 50.4990 | -2.861 [-7.368, +1.671] | 0.1055 | -2.861 [-6.312, -0.198] | 0.0073 |
| bkqb+coach+injury+arrest | 767 | 50.4990 | -2.861 [-6.972, +1.266] | 0.0831 | -2.861 [-5.294, -0.652] | 0.0012 |
| bkqb+coach+injury | 758 | 50.4325 | -2.927 [-7.019, +1.186] | 0.0757 | -2.927 [-5.423, -0.829] | 0.0000 |
| bkqb+divrev+injury+sgz+surface | 942 | 50.4325 | -2.927 [-7.368, +1.527] | 0.0961 | -2.927 [-6.258, -0.133] | 0.0141 |
| bkqb+coach+divrev+injury+sgz+surface | 991 | 50.3659 | -2.994 [-7.498, +1.539] | 0.0943 | -2.994 [-6.370, -0.396] | 0.0014 |
| bkqb+divrev+injury+arrest+surface | 875 | 50.3659 | -2.994 [-7.258, +1.238] | 0.0822 | -2.994 [-5.906, -0.132] | 0.0157 |
| bkqb+coach+divrev+injury+arrest+surface | 930 | 50.2994 | -3.061 [-7.444, +1.274] | 0.0819 | -3.061 [-5.956, -0.384] | 0.0095 |
| bkqb+divrev+injury+surface | 867 | 50.2329 | -3.127 [-7.388, +1.095] | 0.0722 | -3.127 [-5.973, -0.330] | 0.0127 |
| bkqb+injury+arrest+sgz+surface | 889 | 50.2329 | -3.127 [-7.454, +1.265] | 0.0781 | -3.127 [-6.262, -0.891] | 0.0000 |
| bkqb+coach+divrev+injury+surface | 922 | 50.1663 | -3.194 [-7.569, +1.138] | 0.0727 | -3.194 [-6.048, -0.528] | 0.0044 |
| bkqb+coach+injury+arrest+sgz+surface | 939 | 50.0998 | -3.260 [-7.667, +1.210] | 0.0717 | -3.260 [-6.908, -0.732] | 0.0004 |
| bkqb+injury+sgz+surface | 881 | 50.0998 | -3.260 [-7.569, +1.130] | 0.0698 | -3.260 [-6.456, -1.044] | 0.0000 |
| bkqb+coach+injury+sgz+surface | 931 | 49.9667 | -3.393 [-7.789, +1.065] | 0.0636 | -3.393 [-7.101, -0.904] | 0.0003 |
| bkqb+injury+arrest+surface | 802 | 49.6341 | -3.726 [-7.769, +0.334] | 0.0345 | -3.726 [-5.786, -1.974] | 0.0000 |
| bkqb+coach+injury+arrest+surface | 858 | 49.5010 | -3.859 [-8.013, +0.267] | 0.0326 | -3.859 [-6.391, -1.923] | 0.0000 |
| bkqb+injury+surface | 794 | 49.5010 | -3.859 [-7.864, +0.197] | 0.0289 | -3.859 [-5.935, -2.177] | 0.0000 |
| bkqb+coach+injury+surface | 850 | 49.3679 | -3.992 [-8.133, +0.133] | 0.0278 | -3.992 [-6.585, -2.178] | 0.0000 |

## POST-HOC ATTRIBUTION: greedy forward selection

**POST-HOC ATTRIBUTION — mining on already-looked-at data. It proposes, it does
not conclude.** Greedy forward selection maximizing the full-sample point
estimate at each step (**measured**, artifact `greedy_forward_selection`):

| step | added | members so far | point estimate (pts) |
|---|---|---|---|
| 1 | spread_gap_zone | sgz | +0.798 |
| 2 | coach_fade | coach+sgz | +1.464 |
| 3 | player_arrests | coach+arrest+sgz | +1.863 |
| 4 | division_revenge | coach+divrev+arrest+sgz | +2.063 |
| 5 | surface_switch | +surface | +1.065 (worse) |
| 6 | backup_qb_fade | +bkqb | -0.266 (worse) |
| 7 | injury_value_lost | +injury | -2.861 (worse) |

The greedy path terminates at the same four-member subset the exhaustive
enumeration ranks first (`coach+divrev+arrest+sgz`, +2.063 pts), and it
reproduces the known leave-one-out marginals: injury_value_lost and
backup_qb_fade are the destructive members, surface_switch mildly negative
(**read**: marginals in
`artifacts/overlay_stack_backtest/20260819T191534Z/result.json`). Because this
ordering was chosen after looking at the same outcomes, no step of it is
confirmatory.

## What the EV rule would play prospectively

The pool is forced picks: a card must be submitted either way, so the decision
is expected value at the opener grade, full stop. Under that rule the card to
play next is the **four-member composition coach_fade + division_revenge +
player_arrests + spread_gap_zone**: candidate accuracy **55.4225%** vs baseline
53.3599%, delta **+2.0625 pts**, week-blocked 95% [-0.602, +4.765], P+ 0.9304;
season-blocked P+ 0.9154 (**measured**, artifact `subsets[0]`).

**Inferred, and binding on how this number may be quoted**: because that subset
was selected as the maximum over 127 correlated candidates scored on the very
archive that produced the estimate, its +2.06 pts is an UPPER BOUND inflated by
selection-on-same-data, not a prospective expectation. The honest prospective
read is the predeclared, non-cherry-picked identities recorded below (the
production-chain extensions), whose point estimates (+0.86 to +1.86 pts) and
P+ values (0.859-0.935 week-blocked) still favour playing a composed subset
over the unflipped baseline under an EV rule. Per AGENTS.md, an interval
crossing zero is not grounds for declining the play; probability_positive is
the continuous read and every one of the playable candidates sits well above
0.5 on at least one blocking.

## Registry records

All four predeclared identities were declared before results were examined and
recorded via `nfl-ats weak-signals record` (league nfl, effect-units
accuracy_points, seasons 2020-2025, source =
`artifacts/overlay_subset_composition/20260821T174356Z/result.json`,
classification `unresolved_below_power`; all four commands exited 0;
**measured** this session). Week-blocked primary:

| signal | delta pts | 95% week | P+ wk |
|---|---|---|---|
| overlay_subset_production_chain_coach_arrest | +0.8649 | [-0.6618, +2.3968] | 0.8594 |
| overlay_subset_production_plus_division_revenge | +1.5303 | [-0.5387, +3.6291] | 0.9201 |
| overlay_subset_production_plus_spread_gap_zone | +1.8629 | [-0.4636, +4.2568] | 0.9350 |
| overlay_subset_all_seven_joint | -2.8609 | [-7.3678, +1.6712] | 0.1055 |

None is terminal: no refuted mechanism (the three playable candidates have
positive signs with high P+) and no positive-control bound was established.
The all-seven joint result extends the all-six stack finding — adding arrest
to the full stack makes it worse still (50.4990%) — but its week-blocked
interval does not sit wholly below zero, so it remains category 3, unresolved.

## Reproducibility

```powershell
.\.tools\uv.exe run python scripts/overlay_subset_composition.py
```

Artifact: `artifacts/overlay_subset_composition/20260821T174356Z/result.json`
(source-artifact, player-feature-table and incidents-table sha256 hashes,
seeds, n_games/seasons, per-subset intervals, equivalence checks, greedy steps
all inside).

### 2026-08-21 holdout de-biasing

How much of the +2.06 pts survives honest de-biasing? Three predeclared reads
(stated verbatim in `scripts/overlay_selection_holdout.py`'s docstring BEFORE
any of its outputs existed), all attribution on already-scored archive data -
no rotation-registry window is spent. Script:
`scripts/overlay_selection_holdout.py`; artifact
`artifacts/overlay_selection_holdout/20260821T195512Z/result.json`
(**measured** this session, machinery reused unchanged from
`scripts/overlay_subset_composition.py`, 20k-sample bootstraps, seed 20260821).

**Read 1 - split-half holdout.** Selection on seasons {2020, 2021, 2022}
(704 scored games), evaluation on {2023, 2024, 2025} (799 games). The winner
on the selection half alone is a THREE-member subset, coach_fade +
division_revenge + spread_gap_zone (**measured**: it does not include arrest,
which was worth only +0.28 pts there). It scores **+2.6989 pts** on its own
selection half and **+0.8761 pts** on the holdout: week-blocked 95%
[-2.8751, +4.7859], P+ 0.6599; season-blocked P+ 0.7341 (3 blocks, degenerate
per the runner's warning - read the estimate and P+, not the endpoints).

**Read 2 - reverse split.** Selection on {2023, 2024, 2025}, evaluation on
{2020, 2021, 2022}. Winner: coach_fade + player_arrests + spread_gap_zone
(+1.6270 pts on selection). Holdout: **+2.1307 pts**, week-blocked 95%
[-0.8499, +5.1136], P+ 0.9123; season-blocked P+ 1.0000 (again 3 degenerate
blocks). One direction holds up fully, the other shrinks ~3x; that asymmetry
is itself the noise floor this design exists to expose.

**Read 3 - rank stability across all 127 subsets.** Spearman rank correlation
between halves **rho = 0.7207**; OLS slope of holdout delta on selection-half
delta (the shrinkage factor) **0.6356** - on average roughly 36% of any
selection-half advantage is luck. The full-slate global max
(coach+divrev+arrest+sgz) ranks 3-of-127 out-of-sample in the forward
evaluation and 2-of-127 in the reverse - genuinely near the top in BOTH
halves, not an artifact of one cut (**all measured**, artifact fields
`rank_stability`, `global_max_subset_out_of_sample`).

Reference anchors inside each half (**measured**, artifact
`references_per_half`): naive all-seven is strongly negative in both halves
(-2.27 / -3.38 pts); arrest-only is mildly positive (+0.28 / +0.50); the true
production chain (coach -> arrest) scores +0.85 / +0.75 over each half's own
baseline (53.55% / 53.19%).

Both holdout identities were recorded via `nfl-ats weak-signals record`
(league nfl, effect-units accuracy_points, seasons matching each evaluation
half, source = the holdout artifact, classification `unresolved_below_power`,
week-blocked primary, all values machine-read from the artifact; both
commands exited 0; **measured** this session):

| signal | delta pts | 95% week | P+ wk | n games | weeks |
|---|---|---|---|---|---|
| overlay_subset_holdout_2023_2025_frozen | +0.8761 | [-2.8751, +4.7859] | 0.6599 | 799 | 54 |
| overlay_subset_holdout_2020_2022_reverse | +2.1307 | [-0.8499, +5.1136] | 0.9123 | 704 | 53 |

Neither is terminal under the AGENTS.md taxonomy: no refuted mechanism (both
directions positive; the composition family shows real split-half structure,
rho 0.72) and no positive-control bound.

**Verdict.** The honest read is that roughly a THIRD to a HALF of the
headline +2.06 pts is selection inflation: the shrinkage slope says expect
~64% of any selection advantage to survive (~+1.3 pts), while the forward
holdout of the actual frozen subset came in at +0.88 pts (P+ 0.66) and the
reverse at +2.13 pts (P+ 0.91). The four-member global-max subset itself
stays top-3 out-of-sample in both directions, so the FAMILY is real even
where the headline number is inflated. Under the EV rule the implication for
registering a 2026 prospective challenger stands: expected value at the
opener still favours playing a composed subset over the unflipped baseline -
every holdout estimate is positive and the reverse direction is strongly so -
but the card should carry a de-inflated expectation near **+1 pt, not
+2.06**, with the four-member subset (whose members appear in every winning
selection) as the natural registration candidate. Per AGENTS.md, P+ 0.66 in
the weaker direction is a reason to size expectations honestly, never grounds
to decline the play.
