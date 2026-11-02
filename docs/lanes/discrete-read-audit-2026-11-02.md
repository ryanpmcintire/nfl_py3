# discrete-read-audit-2026-11-02

## Goal
Find the mechanism by which the served discrete margin read (mass_preserving_lattice.band_read, policy discrete_conditional_non_push_v1) moves picks off the smooth read, and whether it is a defect.

## State
Scratch (agentA/ under session scratchpad): recon.py, patch.py, cal.py, tab2.py, patch_hist.parquet, patch_2026.parquet.
- KC_DEN W8 reproduced with production DiscretePushReader.for_week (pool 1466 games to 2026-10-26, opener lines): line -1.5, point = line + residual 0.81 + probability-rule shift 0.43 = -0.26 (mean target; no home-side offset, it was 0). cover .480 (feed .478), band 2.5, band_games 389, atoms 64, theta +0.010, push 0. Same pool at point = line gives .443; the band's raw P(margin > -1.5) is .453. Sign conventions consistent (nflverse, home covers if margin > line).
- Mechanism (design, not a sign bug): tilted_atoms matches the MEAN of the pooled margins to point, but the point is built with the residual MEDIAN shift (residual_location, gaussian_median). Band margins are not recentred to the anchor line (band of lines -4..1 has mean line -2.0, mean margin -1.76, median margin -3.0, mass piled at -3), so the tilted lattice has mean at point but median well below, and a slightly-above-line mean yields cover under .5. Smooth rule is median-centred.
- 2020-25 opener (n=1503, pushes dropped): smooth 820-683, stored discrete 797-706, disagreements 239 with discrete 108-131 (exact p<=0.077, probability discrete better 0.07 by Beta). 2026 disagreements 13-11; pooled 121-142. Brier 2020-25 smooth .25162, discrete .25191; 2026 smooth .24854, discrete .24289.
- Discrete home-pick rate .385 vs smooth .433 vs home cover rate .496; shade is on home favourites (mean discrete-smooth -0.016, home-fav picks 38.8% vs 47.5%; home dog -0.002).
- Disagreements by |line| (n, discrete wins): <=1.5 15,7; 1.5-3 59,27; 3-4.5 33,13; 4.5-6.5 44,17; 6.5-9 56,32; 9-14 27,10; >14 5,2. Key number at cover boundary (|L| within .5 of 3,7,10,14): 135 games 61 wins, mean diff -0.017; not at key 104 games 47 wins, -0.020. 7 buckets plus 2 key flags plus 2 fav flags = looks counted; no bucket stands out, the shade is not located at key numbers.
- Patch scratch variants (history point = L + residual_at_open - 0.4, the shift that reproduces stored discrete unbiased, mean abs err .013; 2026 point exact from feed, reproduces stored 56-47):
  A mean tilt (production): 2020-25 787-716 (recon), 2026 56-47; disagrees with smooth 329 / 24.
  B median-mid tilt: 794-709, disagrees 252; 2026 54-49, disagrees 6 (= smooth 54-49).
  C recentre pool to anchor (same fractional part, shift margins), mean tilt: 777-726; 2026 55-48.
  D recentre + median: 788-715; 2026 55-48.
  Smooth 820-683 / 54-49. No variant beats smooth on the opener grade; median tilt matches 2026 smooth but not 2020-25.
- Recorded: weak-signals discrete_vs_smooth_side_2020_25, unresolved_below_power, effect -1.53 accuracy pts, SE 1.03, [-3.55, 0.49], probability_positive 0.069.

## Tried
Mean vs median target (B), pool recentring (C, D). None recovers smooth in 2020-25.

## Next
Proposed patch (not applied, pinned files): in band_read/tilted_atoms replace mean-matching by median-matching (mid-CDF at point = .5, brentq on theta in [-1,1], same bracket), consistent with residual_location. Estimated: 2020-25 794-709 vs served 797-706 (no gain), 2026 54-49 vs 56-47 (13-11 -> 3-3 on 6 disagreements). The gain is mechanical consistency, not accuracy; under the multimodal rule the discrete read stays served unless a smooth read beats it on the opener grade through the played card (smooth +1.5 pts, probability .93, 2020-25). Decision belongs to the orchestrator.
Remaining unexplained: history discrete shade toward away on home favourites persists under median tilt and recentring; point shift proxy for history is approximate (-0.4 vs +0.43 in 2026).

## Open
- Per-week residual median shifts for 2020-25 are not stored; reconstruction used a constant fitted -0.4.
- Whether the 2026 +0.43 median shift (opposite sign to the history proxy) is itself the Week 8 driver.
