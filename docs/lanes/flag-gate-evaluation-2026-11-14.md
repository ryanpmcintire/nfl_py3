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

## Frozen set patch (2026-11-14 evening; prepared, NOT applied; measured unless labelled)
- Patch: `tests/scratch/flag_set_freeze/flag_set_freeze.patch`, built by `make_patch.py` there from byte copies (`a/` original, `b/` patched); `git apply --check` exit 0, ruff check and format clean on `b/`; 22 insertions, 7 deletions, two files.
  - `pick_probability.py:92-104` adds `DECLARED_COUNTED_MEMBERS` (protection), `DECLARED_COUNTED_FLAG_COLUMNS` (still drops owner-held members) and `COUNTED_FLAG_DECLARATION` (set as of 2026-10-25, source fit 20261025T181629Z and the audit lane, frozen from 2026-11-17 by this lane, review at a declared date only); `:1110-1112` rewrites the Model-page sentence, which would otherwise keep stating the winning-record rule; `__all__` gains the two names.
  - `pick_probability_fit.py:356` `counted = list(DECLARED_COUNTED_FLAG_COLUMNS)` replaces the wins > losses filter (today :354-358); `:383` writes `counted_flag_declaration` into the fit metadata. `post_discovery_flag_records` and its metadata key are untouched, so the weekly report keeps every flag's record; the records no longer decide membership. The one flag-sum weight is still refitted leave-one-season-out.
- Declared set (read, pick-chain-audit-2026.md:32-41, :52-58, :71-72): the 10-25 audit shipped the gate and recorded that day's output, "counted: protection", with protection the only qualifier in every forward-chained week; it never worded protection only as a fixed list, and its Next says the Tuesday refit recomputes the gate. The patch therefore names the 10-25 fit as the set's source and this lane as the freeze. Protection was picked on its Week 1-6 record (18-8), so those weeks are in-sample to the choice. No review date is in source: the orchestrator declares it here.
- Tests (grep of `tests/` outside scratch): no durable test pins the gate; zero matches for `post_discovery`, `fit_pick_probability`, `build_fit_population`, `rule_gate_sentence` or the sentence text. `tests/test_cli.py:703, :740, :754` only seed a fixture artifact with `COUNTED_FLAG_COLUMNS`, the unchanged candidate list, and state no rationale. No test edit, no second patch.
- Evidence: `tests/scratch/flag_set_freeze/evidence.py` (patched modules loaded from `b/`, nothing written under `artifacts/`, one run, exit 0) -> `evidence.json`, `evidence_log.txt`, `replay_2026.csv`, `record_batch.json` beside it.
- Refit on the frozen set, same population, no artifact: flag weight .3191 (today .2336 on division and protection; that arm reproduces the active coefficients to five decimals), model .3091, move .2213, hook .8053. Fold flag weights 2020-25: .348, .305, .344, .192, .327, .390.
- 2020-25 out of season, served construction (stated probability at the half-point pool line plus hook, fold coefficients; 1,322 rows, 661 games; today's arm reproduces 754-568, .67833):

| counted set | two-stage | log loss | fit harness | game level |
|---|---|---|---|---|
| division and protection (today) | 754-568 | .67833 | 755-567 | 829-674, .68670 |
| protection only (frozen) | 723-599 | .68199 | 729-593 | 833-670, .68719 |

  Decisive rows first: 97 rows (75 games) change side, frozen side 33-64, exact p .002, P+ .001; by season 10-7, 7-9, 1-22, 6-7, 7-7, 2-12. Frozen minus today: accuracy -2.35 points [-4.01, -0.68], P+ .003; log-loss gain -.0037 [-.0077, +.0004], P+ .037 (game bootstrap, 10,000 draws, seed 20261114). Every flag was found on these seasons, so this is the division flag's fit to its own discovery data, not a measured forward cost (inferred); the only clean record is 2026, division 3-1.
- 2026 replay, Weeks 2-9, 120 settled games (Week 1's 16 predate the fitted probability; Week 10 has no result in the snapshot). Side and probability from the last publication before each deadline, other terms held as served, frozen weight .3191:

| arm | record | settled sides differing from frozen | frozen side on those |
|---|---|---|---|
| served card | 61-59 | 19 (W2 4, W3 4, W4 3, W5 1, W6 3, W7 4; 17 carry the coach flag) | 15-4, p .019, P+ .994 |
| gate run all season | 72-48 | 0 | none |
| frozen set | 72-48 | | |

  All 19 fall before the 10-25 declaration, so they are in-sample to the choice of protection (the audit's post-hoc coach result); since the declaration no settled side differs. The all-season gate counts protection only through Week 8 and adds division from Week 9 (fit metadata where the gate was live, results known before the day of each fit elsewhere; that emulation gives the metadata's set on 36 of 42 gated fits, the six misses being 11-02 fits that had not loaded Sunday's results). Against the earlier audit replay (Tuesday or last-revision probability) the served side matches 57 of 57 and the frozen side 56 of 57: IND at MIN Week 7 flips only on the later probability; without it 15-3. Three looks, one family.
- Week 10 unplayed: PIT at CIN is the only side that changes (CIN 55.2 percent served, PIT 50.7 percent frozen). MIN at GB (MIN 51.1 to 56.9) and LA at ARI (LA 57.1 to 51.3) move and hold.
- Registry: 3 cells, family flag_set_freeze_2026_11_14, unresolved_below_power (accuracy_points, log_loss_improvement); record exit 0; pool ran once, exit 0, 4,632 signals, sign test p .061. The 2020-25 accuracy interval is wholly negative but in-sample to discovery, so no closing ground was claimed; the orchestrator verifies that call.
- Open before Tuesday: declare the membership review date here; settle the enrol-or-fit order (step 4 note); the frozen set gives up 2.35 points of fitted-history level that belongs to the division flag, and nothing measured after 10-25 separates the arms.
- Runbook, Tuesday before noon ET (commands from the repo root, `.\.tools\uv.exe run --no-sync` prefix):
  0. Before applying: `git apply --check tests\scratch\flag_set_freeze\flag_set_freeze.patch` exits 0 (if a file moved, rerun `make_patch.py`); `pytest tests -q` green on the unpatched tree; rerun `evidence.py` and confirm the tables below reproduce, Week 10 results added.
  1. `git apply tests\scratch\flag_set_freeze\flag_set_freeze.patch`.
  2. `pytest tests -q`. A failing test that pins an owner or safety decision is not broken: stop, `git apply -R` the patch, quote its rationale to the owner lane; never exempt or edit it.
  3. Re-enrol validation (the pinned edit stops v9, as on 11-10): copy `registry/studies/combined_vs_raw_2026_v9.json` to `_v10.json` (study_id `combined_vs_raw_2026_w11_w18_v10`, start_week 11, exclusions and supersedes naming v9 and this change); set `DEFAULT_STUDY = "independent_validation_v10"` (independent_validation.py:30) and the `--protocol` default (cli_commands/prospective.py:519); `nfl-ats independent-validation enroll --protocol registry/studies/combined_vs_raw_2026_v10.json`; `nfl-ats independent-validation status` must read source_unchanged true.
  4. `nfl-ats fit-pick-probability`; the new metadata must show `counted_flag_columns` `["flag_protection"]`, the declaration block, every flag's record, and the flag weight in the table below. The v9 protocol freezes coefficients at enrollment (read, :22), so if `enroll` reads the active fit (independent_validation.py:98-157, not read) run this step before step 3 (inferred).
  5. `python scripts\check_served_terms.py` exits 0.
  6. `python scripts\capture_scheduler.py --run-job weekly_lock`, publish, read the Model page sentence; every PICK-CHANGE needs its NOTIFY-OK id.

## Root decision revised (2026-11-14 16:38 ET), supersedes the Tuesday plan above
- The freeze is NOT applied on Tuesday. The gate stays as served. The patch stays on file unapplied (tests/scratch/flag_set_freeze/flag_set_freeze.patch).
- Why the plan changed (measured by the patch unit): (1) its premise was wrong: the 2026-10-25 audit declared the gate and recorded its output at the time, counted: protection; it never declared a fixed list, so freezing at protection only would be a new in-season membership choice made today, the very thing the gate is criticised for; (2) through Week 9 the frozen set and the gate pick the same side in every game, and the only side that differs so far is PIT at CIN in Week 10; (3) on 2020-25 the frozen set reads -2.35 accuracy points against the current set [-4.01, -0.68], which cannot count for the division flag because that history is in-sample to its discovery, and equally gives no ground to remove it.
- What the null result means in expectation (inferred): a flag with no information moves a pick between two sides that are each about even, so the gate's churn costs variance, not accuracy; a flag with information is counted more often than not. A pinned edit, a cohort re-enrolment and a hand-picked set are not justified by that.
- Season-boundary review item (2026-11-18, hou-best-pick-check): division (53.0% 2020-25, 5-2 live) shares one weight with protection (58.2%, 24-9 live), worth +5.7 pts on Week 11 HOU, the Best Pick. Split weights .148/.332 lose held-out (Brier -0.00047, P .086) but both flags are in-sample to discovery; review per-flag weights with the null and power tables above.
- What stands from this lane: the null and power tables, the per-flag records, and the fact that a counted flag admitted on a short record is behind a given pick (this week CIN -3.5). The weekly audit reports which picks depend on the counted set. Revisit membership rules only as a declared change at a season boundary, with a pre-declared evidence bar measured against this null.
