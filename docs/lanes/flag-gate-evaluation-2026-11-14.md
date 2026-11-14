# flag-gate-evaluation-2026-11-14

## Goal
Measure the in-season flag membership gate (count a flag once its record since discovery is winning) as a mechanism: null behaviour, power, what it did to 2026 picks, and the admissible alternatives with their effect on the Week 10 card. Measurement only: nothing served changed, no rule proposed, nothing chosen.

## State (2026-11-14, done; measured unless labelled)
- Rule (read): pick_probability_fit.py:216-266 grades each candidate flag at the Splash opener on games after its discovery date; :354-358 counts it when wins > losses, strictly. No minimum games (1-0 counts; 0-0 and level records do not). Re-evaluated at every fit (daily plus extra runs), so a flag leaves the moment its record is level. :359 sums the counted flags unweighted; one fitted weight multiplies the sum (pick_probability.py:225-231). The weight moves with the set: all seven .260, protection only .319, division and protection .234 (refit of the current set reproduces the served coefficients to four decimals).
- Counted set (read, fit metadata): all seven through 20261025T161243Z; protection only from 20261025T181629Z (protection 18-8, division 0-0); division and protection from 20261103T173458Z (division 3-0) to the active 20261114T184614Z.
- Flags, record since discovery (reproduces the fit metadata on all seven; exact two-sided p):

| flag | found | record | by week | counted |
|---|---|---|---|---|
| protection | 08-25 | 24-9, p .014 | 3-1, 2-0, 3-3, 2-2, 6-1, 2-1, -, 3-0, 3-1 | every fit since 10-25 |
| division | 08-19 | 3-1, p .625 | W8 3-0, W9 0-1; 3 unplayed W10, 30 later | since 11-03 |
| coach | 08-18 | 20-33, p .098 | 2-4, 6-4, 1-7, 1-3, 4-4, 2-3, 3-4, 1-4 | pre-gate only |
| bye | 09-02 | 4-10, p .180 | W6 0-2, W7 2-2, W8 1-3, W9 1-3; 16 unplayed | pre-gate only |
| cold visitor | 08-20 | 1-3 | W6 0-1, W7 0-2, W9 1-0; 2 unplayed W10 | pre-gate only |
| arrests / tank zone | 08-20 / 09-02 | 0-1 / 0-0 | W1 / fires W14-18 | pre-gate only |

- Null, a flag with no information (seed 20261114, 20,000 draws, real fire dates). Counting one flag at .234 changes the side on .377 of its games (57 served Week 7-10 probabilities; .414 on 2020-25).

| flag (fires known) | counted W4 / W7 / W11 / W18 | changes a season | counted then dropped | picks moved |
|---|---|---|---|---|
| coach (53) | .42 / .50 / .50 / .50 | 1.41 | .38 | 8.3 |
| protection (37) | .39 / .43 / .51 / .51 | 1.48 | .40 | 5.8 |
| division (37) | 0 / 0 / .50 / .42 | 1.85 | .53 | 5.7 |
| bye (30) | 0 / .25 / .40 / .43 | 1.40 | .41 | 3.9 |

  Seven such flags together: 1.3 counted at Week 4, 2.8 at Week 11; at least one counted .98; 8.5 membership changes (5-95%: 4-14); 25.6 picks moved (8-45).
- Power, probability counted by Week 11 at a true 52 / 54 / 56%: coach .61 / .72 / .81, protection .59 / .68 / .75, division .55 / .59 / .63, bye .46 / .53 / .59. Only coach at 56% reaches .8 (Week 9). Dropped at least once after being counted: .37-.52, .32-.50, .28-.48. Exact fires needed for .8: 491, 133, 63; at division's 37 a season that is 13.3, 3.6, 1.7 seasons.
- 2026 replay (flip audit games.csv of 20:40Z, Weeks 7-10; served side reproduced 57 of 57; gate in force at the deadline on 48 games, 35 settled; other terms held as served). Decisive games first: gated side v every flag at the served weight differs on 6 (GB_DET, LA_LV, ATL_TB, CLE_PIT, NE_MIA, LAC_BAL unplayed), gated side 1-4, p .375, P+ .11; v every flag at the fixed 2020-25 weight .260 differs on 5, 1-3, p .625, P+ .19; v no flags differs on 2 (JAX_BAL, PIT_CIN unplayed), 1-0, p 1.0. Totals: gated 13-22, all flags 16-19, fixed weight 15-20, none 12-23. Seven looks, one family.
- 2020-25 out-of-season level by set, in-sample to the flags' discovery: all seven 863-640, log loss .68277; division and protection 829-674, .68670; protection only 833-670, .68719; none 825-678, .68841.
- Week 10 unplayed sides that depend on the set: PIT at CIN (CIN 55.2% with division counted; PIT by .026 logit without it); LAC at BAL (LAC 54.1%; BAL if every flag is counted, cold-visitor flag). MIN at GB, LA at ARI, JAX at TEN carry flags and hold.
- Reading (inferred): at this sample size membership is close to a coin for any flag below about 56%; the choice is walk-forward in time, so no scored-game constant reaches src, but the statistic is the sign of a record as short as three games, which the promotion rule says is not sufficient to serve. Protection is the only flag whose record is far from even.
- Registry: 3 cells, family flag_gate_evaluation_2026_11_14, accuracy_points, unresolved_below_power; pool ran once, exit 0, 4,625 signals, sign test p .067. Simulated cells unrecorded.

## Tried
- First run treated every Week 7-10 game as gated (the counted list is in all fit metadata); the gate era is now the fits that carry post-discovery records. scripts/flag_gate_evaluation.py -> artifacts/flag_gate_evaluation/ (summary.json, replay_games.csv, flag_games_2026.csv, record_batch.json); two runs, exit 0; ruff check and format clean.

## Next (alternatives, none chosen; Week 10 effect)
1. Keep the gate: CIN and LAC stay.
2. Every flag always counted at the 2020-25 weight, no in-season choice, in-sample to discovery: LAC at BAL becomes BAL.
3. No flags: PIT at CIN becomes PIT.
4. Counted set frozen for the season at a declared date (protection only): PIT.
5. A pre-declared evidence bar set out of season: only protection is near any such bar: PIT.
6. Each flag its own fitted, shrunk weight (scripts/per_flag_weights_eval.py, not read): not measured.

## Open
- Held games are read at their Tuesday record and revised games at their last revision; Week 10 probabilities are as of 20:40Z.
- The gate was designed after coach had lost (pick-chain-audit-2026.md:26-27), so its 54-23 forward test shares weeks with its design (reported).

## Root decision (2026-11-14 16:22 ET)
- Week 10 is served as it stands. No pinned edit the night before the games: the only side that changes without the division flag is PIT at CIN, both sides sit near even, and on 35 settled gated games no arm is distinguishable (gated 13-22, all flags 16-19, fixed weight 15-20, none 12-23).
- The star does not rest on the flag: a no-record run of the Sunday refresh (measured this afternoon) names SEA at LV, 62.4 percent, as the re-nomination candidate; the star moves off PIT at CIN at the 10:00 Sunday pass unless the probabilities change.
- The gate itself does not meet the repository's rules: membership is chosen in season on records as short as three games with no null, and under the null it counts a no-information flag half the time and moves about 26 picks a season (AGENTS: every parameter chosen out of season; a short record is not a reason to serve; no rule flips a pick without a mechanism).
- Tuesday 2026-11-17 before the noon lock: freeze the counted set at protection only, the set the 2026-10-25 audit declared, remove the automatic re-admission, and review membership only at a declared date with this lane's null and power tables. A unit prepares the patch and a replay first; it is a pinned edit, so the validation cohort is re-enrolled. Until then CIN -3.5 (55.2 percent) is a pick that exists because of a flag admitted at 3-0, now 3-1.
