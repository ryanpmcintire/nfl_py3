# mod25c-noise-mechanisms

## Goal
Cut excess within-game noise by fitting ball security, clock burn, OT and timeout behaviour on real plays. Re-choose strength from its own targets, then re-check the shape held out. Parent docs/lanes/mod25-synthetic-data.md. Rule: no parameter touches mass at 3/7/10/14/17.

## State
Code `scripts/mod25c_noise.py` (imports mod25_mechanisms; commands measure, grid, gate; engine untouched, no src edits). Outputs `artifacts/mod25c/` (measure_m1..m3, grid_c, gate_*, real_metrics.json, policies2 npz, run_all.sh). Variants: base, b (= mod25b fine2), c = b + tov + clk + tout (+ c_ot, ablations c_tov c_clk c_tout). Fit on 2009-2017 only.
Mechanisms added (all fit to their own behaviour): P(turnover | down, dist, field, score, time, play type) and P(timeout used | state) as HGB grids driving class-restricted resampling; clock burn as state-transfer of seconds per snap (real cell mean by play class x time zone x score band, applied as delta to the drawn play); OT-only neighbour pool (variant ot).
**Measured divergences (neutral 12k games, b vs real eval 2018-25 / train 2009-17):** overall turnover per run/pass play .0226 vs .0199/.0229 (matches; the orchestrator's .119 vs .061 does not reproduce as a general rate; tied q4 le 120s: base .045 vs .019/.033, b .033). Clock: tied/trailing last 2 min snap 14.3 vs 12.6/12.4 (b); c_clk 12.2 (fixed); q4 le300 tied 21.4 vs 23.3/24.3, c_clk 25.2. Timeouts per game 7.2 vs 7.7/7.4: late q4 le300 1.5 vs 1.24, q2 final 2 min 1.9 vs 2.65/2.14 (c tout: 1.76, not closed). OT: tie given OT .17 vs .068/.049; drive score .35 vs .43/.40; FG per drive .23 vs .26/.30; drive length 144 s vs 162/174 (c_clk 161).
**Variance anatomy (measured):** sim neutral margin variance 225 vs real 205 (eval)/221 (train); variance of score diff at 2700/1800/900/300 s left matches real train within 5 percent at every time. So no late-game noise source; the sim's neutral variance already equals real TOTAL variance, leaving no room for strength. Inferred: uniform play-level overdispersion (pooled plays), not ball security or clock.
c_clk lengthens OT drives, so OT tie rose to .26-.27 while OT scoring per drive stays low (.29-.30 vs .40-.43): OT scoring (4th-down/FG share in OT) is the unresolved defect.
Strength grid (declared before run: scale 1.5/2/2.5/3, yard_gain 2, 6 worlds x 8 seasons, loss = sum of squared z of autocorr + 3 R2 buckets vs real 2011-2017, z by real-target half-width): losses 66.1, 48.9, 15.4, 3.8; extension 3.5 and 4.0 (declared after the edge result) gave 8.3 and 12.8, so 3.0 is the minimum (SD 16.2).
**Held-out gate (scale 3, yard gain 2, 8 worlds x 8 seasons, 6 post-burn; real 2018-2025) gate_g2.txt:** mass3 real .144 base .080 b .103 c .102; mass7 .085 / .063 .072 .073; SD 14.31 / 16.89 16.95 16.41; share le3 .247 / .168 .174 .182; autocorr .156 / .215 .213 .154 (c passes); R2 w10-18 .150 / .120 .123 .101; nonstrength var 177.5 / 251.5 254.9 246.6.

## Tried
OT-only pool: no gain on OT scoring (drive score .29 vs .30). Timeout grid initial bug (coarse time axis) fixed with finer axis.

## Next
OT: fit 4th-down/FG decision per OT drive state and kickoff start. Play-level overdispersion: compare per-play yard SD sim vs real conditional on team.

## Open
Noise excess not localized to any of the fitted mechanisms; margin SD stays above gate when strength is set from its own targets.
