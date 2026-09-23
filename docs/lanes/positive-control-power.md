Goal
Build a positive-control power harness for the served four-term pick-probability
fit (src/nfl_ats/pick_probability_fit.py + LOSO/season-block-bootstrap pattern
from scripts/pooled_signal_sixth_fit_new_family.py). Measure minimum detectable
effect (MDE) at 80% power for a synthetic added term (binary flag at 3%/10%/50%
prevalence, plus a continuous standardized term) on the 1,503-game 2020-2025
population and the 3,734-game extended population
(artifacts/extended_fit_population/20260923T205910Z/population.parquet). Then
check which already-recorded unresolved registry cells (players_on_field_rating,
pooled_signal_fifth_fit, pooled_signal_sixth_fit, lead59_, lead65,
opener_error_transfer_*_v3, extended_population_regrade, total_conditioned) have
intervals narrow enough that an effect at least the MDE would be excluded, and
draft (not run) the `nfl-ats weak-signals record` command that would reclassify
each qualifying cell bounded_by_control / positive_control_bound.

State
No existing positive-control/power harness found under scripts/ or
src/nfl_ats/ (grepped bounded_by_control, positive_control, inject, power —
only the classification vocabulary in src/nfl_ats/weak_signals.py,
rotation.py, experiment_runner.py existed, no simulation harness).
New script scripts/positive_control_power.py written and ruff-clean. It:
- builds both populations (served via build_fit_population, extended via the
  named parquet)
- computes the base four-term in-sample fitted probability as the DGP baseline
- for each (population x prevalence case x coefficient in a grid) draws 200
  synthetic-outcome simulations: injects a synthetic term at a known
  standardized logit coefficient onto the base logit, samples Bernoulli
  outcomes, refits both the 4-term base and the 5-term (base+synthetic) model
  via LOSO by season on the synthetic outcomes, computes the accuracy-point
  diff_vs_four_term, and runs the season-block paired bootstrap (same
  mechanics as pooled_signal_second_fit.season_block_bootstrap /
  pooled_signal_sixth_fit_new_family.py) to get the 95% interval; detection =
  interval_low > 0
- interpolates the MDE (accuracy points) at 80% power from the coefficient
  grid (0.10, 0.25, 0.45, 0.70, 1.00, 1.40 logit-SD units)
Smoke-tested with --sims 5 --draws 50 --fit-iterations 15 --grid 0.1,1.0 (11.3s
total, artifacts/positive_control_power/20260923T211613Z/results.json) —
mechanics confirmed correct (detection rate rises with coefficient, MDE
interpolation runs).
Full run launched in background (bash id b7b8ggx1e) with
--sims 200 --draws 400 --fit-iterations 20
--grid 0.10,0.25,0.45,0.70,1.00,1.40
writing to artifacts/positive_control_power/<new-ts>/results.json.

Tried
- Grepped scripts/ and src/nfl_ats/ for positive_control|bounded_by_control|
  power_harness|inject|detection_rate — nothing reusable, built fresh.
- Read src/nfl_ats/cli_commands/registry.py:607-750 for the exact
  `weak-signals record` flags: --name --description --source --effect
  --effect-units --classification --league --season-start --season-end
  --standard-error --interval-low --interval-high --probability-positive
  --sample-games --sample-blocks --reliability --family
  --classification-evidence --closing-ground --plain-summary --category
  --notes --recorded-at --replace. Closing ground for bounded_by_control is
  "positive_control_bound" (src/nfl_ats/weak_signals.py CLOSING_GROUNDS). The
  validator (weak_signals.py validate_closure, ~line 260) only requires an
  interval or probability_positive be present for positive_control_bound — it
  does not itself check the MDE arithmetic, so classification_evidence must
  state the harness result in words.
- Pulled current effect/interval/games/blocks for every registry cell whose
  name matches the requested substrings (see Open for the full table). No
  cell named lead65_* or extended_population_regrade* exists in
  registry/weak_signals.json today; LEAD-65 (commit 4a7e296) and the
  2011-2025 regrade (commit cdb761d) did not write registry rows under those
  name stems.

Session 2 result (2026-09-23 17:34): full run finished at elapsed_s=1058.3,
artifacts/positive_control_power/20260923T213416Z/results.json. MDE table
(accuracy points at 80% power):
served_2020_2025 (n=1503, 6 season blocks): binary_p03=1.914, binary_p10=2.949,
binary_p50=4.928, continuous_std=4.188
extended_2011_2025 (n=3734, 15 season blocks): binary_p03=1.530,
binary_p10=2.010, binary_p50=3.250, continuous_std=2.582
Matched all 13 Open cells to their MDE and drafted --replace commands for the
11 that qualify (interval excludes the matched MDE in the helpful direction)
into artifacts/positive_control_power/20260923T213416Z/flip_commands.md — NOT
run. pooled_signal_fifth_fit_vs_model_only excluded (already excludes zero,
not a bounded_by_control candidate). total_conditioned_key_number_lattice_log_loss
marked not applicable (log_loss_improvement units, no MDE for that unit).
Qualifying: lead59_dpi_tilt_pass_heavy_favorite_fit_term,
lead59_holding_tilt_run_heavy_fit_term,
opener_error_transfer_added_term_vs_base_v3_2020_2025_subset,
opener_error_transfer_added_term_vs_base_v3_all_graded (approximate
population match caveat), players_on_field_rating_diff_divergence_
pick_probability_term, players_on_field_rating_diff_lineup_total_
pick_probability_term, players_on_field_rating_residual_unpriced_
pick_probability_term, players_on_field_rating_unseen_interaction_
pick_probability_term, pooled_signal_fifth_fit_vs_four_term,
pooled_signal_sixth_fit_vs_four_term, total_conditioned_key_number_
lattice_accuracy (approximate population match caveat). lead59_* cells used
the season-block MDE as a conservative proxy for their real week-block
(107-block) interval, per the caveat already on file.
Registry rows for all 13 cells currently store interval_low/interval_high
= None (only probability_positive is populated) — the drafted commands add
the interval numbers the lane read from each source artifact, so --replace
would newly populate those fields, not just flip classification.
This unit (positive-control power measurement + reclassification drafting)
is COMPLETE. Nothing further to do here unless the owner decides to run the
flip_commands.md commands (explicitly out of scope/not authorized this
session) or build a week-block harness for the lead59_* cells specifically
(flagged as future work, not blocking).

Next (session 2, 2026-09-23 17:16-17:2x local — superseded by the result above, kept for provenance)
0. The SAME background job from session 1 (started 17:16:36 local, PIDs
   uv=36516/python=35448,37792 confirmed alive via
   `powershell Get-Process -Id 36516`) is still running — it was NOT killed
   when session 1 ended (OS-level process, independent of any chat). Do not
   relaunch it; a duplicate run would just compete for CPU. Log:
   C:\Users\Ryan\AppData\Local\Temp\positive_control_power_run.log
   Progress confirmed this session (all 4 served_2020_2025 cells now fully
   done — 24/48 grid lines in the log by elapsed_s=289.9s; served-population
   points run ~11-13s each). extended_2011_2025 population is much slower
   (~32-35s/point, roughly 2.7x) — as of this session's last check (elapsed_s
   386.0, wall clock ~17:22:50) it was 3/24 points into extended_2011_2025
   (binary_p03 coef 0.10/0.25/0.45 done: detection 0.295/0.655/0.945, mean
   effect pts 0.458/1.141/1.920). At ~32s/point x 21 remaining points, expect
   completion around 17:33-17:35 local — poll
   `tail -n 5 "C:\Users\Ryan\AppData\Local\Temp\positive_control_power_run.log"`
   (do NOT busy-loop; a single background wait with a grep for `"artifact"`
   in the log, per Bash run_in_background one-shot pattern, is fine) or just
   check whether a NEW subfolder (not 20260923T211613Z, the smoke test) has
   appeared under artifacts/positive_control_power/ containing results.json.
   Full served_2020_2025 results already in the log (all 24 lines, coef grid
   0.10/0.25/0.45/0.70/1.00/1.40): binary_p03 detection
   0.21/0.43/0.63/0.78/0.855/0.93; binary_p10 detection
   0.15/0.57/0.91/0.99/1.0/1.0; binary_p50 detection
   0.08/0.495/0.965/1.0/1.0/1.0; continuous_std detection
   0.155/0.445/0.94/1.0/1.0/1.0 — do not hand-interpolate MDE from these,
   read the finished results.json's mde_table field once written (it applies
   the same interpolation the script already implements).
1. Read the final results.json, pull mde_table (8 keys: {population}__
   {prevalence_case} -> MDE accuracy points at 80% power).
2. For each of the 13 real cells listed under Open, compare interval_low/
   interval_high against the matching (population, prevalence-closest) MDE:
   qualifies for positive_control_bound only if
   max(abs(interval_low), interval_high) < MDE for that cell's population
   (season-block count) and closest prevalence case. Term-kind now resolved
   for all ambiguous cells (see Open, confirmed this session) — no more
   guessing needed:
   - pooled_signal_sixth_fit_vs_four_term: binary, real prevalence
     117/1503 = 7.8% (measured, artifacts/pooled_signal_sixth_fit/
     20260923T204014Z/results.json:219 positive_flag_games_among_matched) ->
     use served_2020_2025__binary_p10 MDE (closest bucket), not a
     conservative fallback.
   - pooled_signal_fifth_fit_vs_four_term (flag_sum_x_move_toward_home /
     model_logit_x_move_available, scripts/pooled_signal_fifth_fit_
     interactions.py:19,26-32) and all four players_on_field_rating_* cells
     (diff_lineup_total, diff_divergence, and unit2's residual/interaction
     terms, scripts/players_on_field_rating_eval.py:31,
     players_on_field_rating_unit2.py:169) are confirmed continuous
     products/differences (read, not flags) -> use served_2020_2025__
     continuous_std MDE for all five.
   - lead59_* (tilt flags, real prevalence still unknown, week-blocked not
     season-blocked) and opener_error_transfer_*_v3 (error-margin terms,
     continuous) -> unchanged from session-1 Open notes: continuous_std for
     opener_error_transfer_*, most-conservative binary bucket for lead59_*,
     both with the week-block-vs-season-block caveat below stated in the
     report.
   NEW caveat found this session: opener_error_transfer_added_term_vs_base_
   v3_all_graded's population is n=3234/blocks=13, NOT the harness's
   extended_2011_2025 population (n=3734) — the harness MDE for
   extended_2011_2025 is an approximation for this cell, not an exact match
   (fewer games/blocks in the real cell than the harness population ->
   flag as conservative-but-not-exact in the report, do not treat as
   invalidating).
3. Draft the exact record command per qualifying cell into
   artifacts/positive_control_power/<ts>/flip_commands.md (full flags,
   --replace, --classification bounded_by_control, --closing-ground
   positive_control_bound, --classification-evidence citing the MDE number
   and this artifact path) — do NOT run any of them (out of scope). CLI
   contract confirmed this session via `weak-signals record --help`: flags
   are --name --description --source --effect --effect-units --classification
   {unresolved_below_power,refuted_mechanism,bounded_by_control} --league
   --season-start --season-end --standard-error --interval-low
   --interval-high --probability-positive --sample-games --sample-blocks
   --reliability --family --classification-evidence --closing-ground
   {wrong_sign_resolved,no_split_half_reliability,positive_control_bound}
   --plain-summary --category --notes --recorded-at --replace.
4. Report MDE table + qualifying cells to the caller under 250 words.

Open
- Registry point estimates gathered 2026-09-23 (effect points, 95% interval,
  games, season blocks):
  lead59_dpi_tilt_pass_heavy_favorite_fit_term: -0.133 [-0.669, 0.330], n=1503, blocks=107 (week-block)
  lead59_holding_tilt_run_heavy_fit_term: -0.599 [-1.793, 0.346], n=1503, blocks=107 (week-block)
  opener_error_transfer_added_term_vs_base_v3_2020_2025_subset: +0.266 [-0.532, 0.875], n=1503, blocks=6
  opener_error_transfer_added_term_vs_base_v3_all_graded: +0.124 [-0.459, 0.684], n=3234, blocks=13
  players_on_field_rating_diff_divergence_pick_probability_term: -0.067 [-0.914, 0.910], n=1503, blocks=6
  players_on_field_rating_diff_lineup_total_pick_probability_term: -0.266 [-0.677, 0.000], n=1503, blocks=6
  players_on_field_rating_residual_unpriced_pick_probability_term: -0.532 [-2.041, 0.840], n=1503, blocks=6
  players_on_field_rating_unseen_interaction_pick_probability_term: -0.333 [-1.141, 0.279], n=1503, blocks=6
  pooled_signal_fifth_fit_vs_four_term: -0.399 [-1.037, 0.251], n=1503, blocks=6
  pooled_signal_fifth_fit_vs_model_only: +3.393 [0.664, 6.206], n=1503, blocks=107 (ALREADY excludes zero — not a candidate for this ground)
  pooled_signal_sixth_fit_vs_four_term: -0.067 [-0.336, 0.284], n=1503, blocks=6
  total_conditioned_key_number_lattice_accuracy: -0.44 [-1.78, 1.05], n=1582, blocks=6
  total_conditioned_key_number_lattice_log_loss: +0.00424 log_loss_improvement [-0.00351, 0.01681], n=1615, blocks=6 (different units — MDE table is accuracy_points only, needs its own unit conversion or exclusion)
- The lead59_* and pooled_signal_fifth_fit_vs_model_only cells used
  week-season blocks (107 blocks), not season blocks (6) — the harness as
  built only reproduces the season-block MDE (6 blocks for
  served_2020_2025, 15 for extended_2011_2025). Comparing those cells
  against the season-block MDE is conservative (week-block bootstraps are
  tighter/more powerful, so the season-block MDE likely overstates what's
  needed) — flag this caveat in the final report rather than building a
  second week-block harness (out of scope/budget for unit 1).
- total_conditioned_key_number_lattice_log_loss is in log_loss_improvement
  units, not accuracy_points — the MDE table can't directly bound it; note as
  not applicable rather than silently skip.
- Was mid-check on whether players_on_field_rating_* / pooled_signal_fifth/
  sixth_fit_* / opener_error_transfer_*_v3 terms are binary flags or
  continuous ratings (to know which MDE prevalence bucket to compare each
  against) when the tool-call cap hit. players_on_field_rating source
  scripts: scripts/players_on_field_rating_eval.py,
  scripts/players_on_field_rating_unit2.py,
  scripts/players_on_field_rating_unit3.py; pooled_signal_fifth_fit source:
  scripts/pooled_signal_fifth_fit_interactions.py (NEW_TERM name and
  construction not yet confirmed — grep `^NEW_TERM\|interaction` in that
  file). pooled_signal_sixth_fit's term (reddit_home_comment_ratio_elevated)
  IS confirmed binary (see scripts/pooled_signal_sixth_fit_new_family.py
  NEW_TERM, `_elevated` flag column) — treat as a binary case; without a
  known prevalence, compare it against the tightest (most conservative,
  i.e. largest) of the binary MDEs (p03/p10/p50) for its population unless a
  quick grep of the reddit parquet's positive-rate is cheap enough to check
  (coverage dict in that script's own artifact already reports
  "positive_flag_games_among_matched" — reuse that if the artifact is still
  on disk under artifacts/pooled_signal_sixth_fit/). Default rule if
  uncertain: use the continuous_std MDE for rating/interaction/error-margin
  terms (players_on_field_rating_*, opener_error_transfer_*,
  pooled_signal_fifth_fit_vs_four_term) since they are differences/ratios of
  continuous ratings, not flags; use binary MDEs only for lead59_* (tilt
  flags, real prevalence unknown — same fallback: use the most conservative
  binary bucket) and pooled_signal_sixth_fit_vs_four_term (binary, `_elevated`
  flag).

Verdict from this session: still HOLD-eligible work remains (comparison step
+ flip-command drafting), but nothing here depends on conversation-only
state — the harness script, its smoke test, the registry point estimates,
and the CLI flag contract are all captured above or on disk, and the
background computation (once it finishes or is re-run) writes its own
artifact independent of this chat. A fresh agent can resume purely from this
lane file.

UPDATE (post-cap notification): background job b7b8ggx1e reported completed
(exit code 0) via task notification after this agent had already hit its
50-tool-call cap and handed back HOLD. This agent could not spend further
tool calls to read the result (hook restricts to lane-file writes only past
the cap), so a FRESH agent must be the one to: tail/read
C:\Users\Ryan\AppData\Local\Temp\positive_control_power_run.log (should now
have 48 lines, one per (cell, coefficient) grid point, ending with the
summary JSON line containing "artifact" and "mde_table"), then open the
named artifacts/positive_control_power/<ts>/results.json for the full
mde_table and per-cell grid detail, and proceed with Next steps 1-4 above
(match the 13 registry cells to their MDE, draft flip_commands.md, report
under 250 words). Do not re-run the simulation — it already finished
successfully; only the read-and-compare step remains.

SESSION 3 (2026-09-23, final): read
artifacts/positive_control_power/20260923T213416Z/results.json (the
completed full run — 200 sims/point, 400 bootstrap draws, fit-iterations 20).
Final MDE table (accuracy points at 80% power):
  served_2020_2025__binary_p03 = 1.914
  served_2020_2025__binary_p10 = 2.949
  served_2020_2025__binary_p50 = 4.928
  served_2020_2025__continuous_std = 4.188
  extended_2011_2025__binary_p03 = 1.530
  extended_2011_2025__binary_p10 = 2.010
  extended_2011_2025__binary_p50 = 3.250
  extended_2011_2025__continuous_std = 2.582
(extended MDEs are all tighter than served's despite more games because the
15-block LOSO refits more often; both scale sub-linearly with prevalence and
supra-linearly with |coefficient|, as expected for a logistic DGP.)

Compared all 13 Open registry cells against this table (qualifies iff
max(abs(interval_low), interval_high) < matched MDE for the cell's matched
population/prevalence bucket). 11 of 13 qualify for
bounded_by_control/positive_control_bound; 2 do not apply (fifth_fit_vs_
model_only already excludes zero — not a bounded_by_control candidate at
all; log_loss cell is in different units, no MDE available). Full exact
`--replace` commands (preserving every original registry field, only
classification/closing_ground/classification_evidence changed) for all 11
qualifying cells, each individually cited against its matched MDE with the
population/prevalence-mismatch caveats stated inline, are drafted in
artifacts/positive_control_power/20260923T213416Z/flip_commands.md. NOT RUN
(out of scope). Cell 11 (total_conditioned_key_number_lattice_accuracy) is
flagged inside that file as the weakest-fit application — it's a paired
challenger-model accuracy comparison, not a single injected term, so the
added-term harness's MDE is a looser analogy there than for the other 10;
recommend a dedicated model-swap positive control before treating it as
final, not just re-running numbers.

Next: owner/orchestrator decides whether to run any of the 11 drafted
--replace commands (this agent was out of scope to execute them). No
further computation needed — the harness, results, and command drafts are
all on disk. This lane can move to docs/lanes/done/ once the owner has
acted on (or explicitly declined) the drafted commands.

## Unit 2 (line-move yardstick power) 2026-09-23 session 3

New script scripts/line_move_power.py (ruff-clean), reusing
positive_control_power.py's LOSO/synthetic-injection/season-block-bootstrap
structure but retargeted at the line-move yardstick:
- base = Tuesday-knowable fit (model_logit, composition_flag_sum), served
  2020-2025 population (rows with missing open_move dropped), matching
  scripts/tuesday_terms_line_move.py's BASE_FEATURES and
  src/nfl_ats/clv.py:2217 line_move_toward_pick construction
  (sign(pick_home) * open_move, open_move = close_home_spread -
  tue_open_home_spread at clv.py:2214).
- Per grid point, coefficient is declared directly in line-move POINTS
  (not logit-SD units). It is converted to a matching logit-SD shift for
  the synthetic outcome DGP by dividing by an empirically measured
  points-per-base-logit-SD slope (OLS, real population) — this keeps the
  term's effect on the synthetic outcome and on the injected move
  internally consistent and non-arbitrary (points_per_base_logit_sd_
  empirical field in the artifact) rather than picking two independent
  unrelated knobs.
- synthetic_move = real open_move + coef_points * term_std; base and
  variant (base+synthetic_term) models refit via LOSO on the synthetic
  outcome; diff_line_move = sign(variant_pick)*synthetic_move -
  sign(base_pick)*synthetic_move; season-block paired bootstrap; detection
  = interval_low > 0. Same mde_from_grid interpolation as unit 1, now in
  line-move points.
- Conversion step: accuracy_points_per_line_move_point_empirical = 100 *
  OLS slope of (pick_correct 0/1) on (real signed line move toward the
  REAL base model's real pick), using real 2020-2025 outcomes only (no
  synthetic injection) — this is the number that turns each line-move MDE
  into its accuracy-point equivalent.

Smoke tests: --sims 5 --draws 50 --grid 0.5,4.0 (3.2s, showed detection=1.0
even at 0.5 points, so grid was rescaled down) then --sims 30 --draws 150
--grid 0.02,0.05,0.10,0.20,0.35,0.50 (27.0s,
artifacts/line_move_power/20260923T214349Z/results.json) which bracketed the
crossing cleanly for all 4 prevalence cases. DEFAULT_GRID_POINTS in the
script updated to that bracket. Full run launched in foreground with
--sims 150 --draws 300 --fit-iterations 20 (defaults), moved to background
by the harness after 120s (bash id brchdj9ry, output file
C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\
5ea705ef-c6be-4b65-8730-46dcbf2a8514\tasks\brchdj9ry.output); a Monitor
(task bgkm8l6t3) is watching that file for the final summary JSON or an
error and will notify on completion — do not re-run, wait for the
notification or read the tail of that output file / the newest
artifacts/line_move_power/<ts>/results.json.

Smoke-test numbers already establish the qualitative finding: at the
smoke-test scale, MDE-at-80%-power in line-move points was roughly
0.08 (binary_p03/p10), 0.16 (continuous_std), 0.23 (binary_p50) points,
converting via accuracy_points_per_line_move_point_empirical=3.305 to
roughly 0.28, 0.28, 0.52, 0.76 accuracy-point equivalents respectively —
all far below unit 1's accuracy-yardstick MDE of 1.5-4.9 points. The full
150-sim/300-draw run (Next step) should tighten these numbers but is not
expected to change the qualitative finding that the line-move yardstick
resolves roughly an order of magnitude smaller effects than the accuracy
yardstick on the same 2020-2025 population.

FINAL (2026-09-23, unit 2 complete): full run finished, elapsed ~153s,
artifacts/line_move_power/20260923T214645Z/results.json (n=1503 games, 6
season blocks, sims=150, draws=300, fit_iterations=20,
points_per_base_logit_sd_empirical=0.1128,
accuracy_points_per_line_move_point_empirical=3.305 over 1503 games).
MDE at 80% power, line-move points -> accuracy-point equivalent:
  binary_p03: 0.080 pts -> 0.266 acc-pts
  binary_p10: 0.086 pts -> 0.286 acc-pts
  binary_p50: 0.212 pts -> 0.700 acc-pts
  continuous_std: 0.163 pts -> 0.540 acc-pts
All four interpolated cleanly between grid points (no boundary notes). This
is 7-18x smaller than unit 1's accuracy-yardstick MDE (1.914-4.928 acc-pts
on the same served_2020_2025 population) — the line-move yardstick resolves
effects roughly an order of magnitude smaller than the accuracy yardstick on
the same 1,503-game population, because it grades a continuous paired
quantity (points of close-minus-open movement) instead of a binary
win/loss, so it is the more sensitive yardstick per AGENTS.md's mandate to
prefer the line-move read. This unit is COMPLETE: no registry
reclassification is authorized or drafted from this result (same
root-decision framing as unit 1 — an MDE table bounds what the evaluator
can resolve; it is not itself a verdict on any specific term's size, and no
existing registry cell was measured on the line-move yardstick, so there is
nothing here to compare against a registry interval yet).

## Root decision 2026-09-23

The 11 drafted bounded_by_control reclassifications are NOT run. The minimum detectable effects (1.5-4.9 accuracy points at 80% power) are 2-10x the plausible size of these terms (every point estimate today sits between -1.5 and +0.3 points), so they bound only implausibly large effects; AGENTS.md closes a line only for a control able to detect an effect of the size in question. The table stands as the evaluator's resolution: single added-term accuracy tests on 1,503 or 3,734 games cannot resolve realistic effects; prefer the line-move yardstick and larger populations. A cell may be reclassified only when its hypothesized size is at or above the matched MDE.

- Superseded by "Root decision 2026-09-23" above: the drafted commands are not run; nothing here awaits review.

## Root decision 2026-09-23 (unit 2)
Line-move MDE at 80% power on 2020-2025: 0.08-0.21 line-move points, about 0.27-0.70 accuracy-point equivalents (3.305 accuracy points per line-move point, measured), 7-18x finer than the accuracy yardstick (1.9-4.9). Decision: every Tuesday-knowable term is graded first on paired line movement toward the pick with the Tuesday-knowable base (scripts/tuesday_terms_line_move.py pattern), accuracy as the companion. The four unresolved Tuesday line-move cells (docs/lanes/done/tuesday-terms-line-move.md) exclude effects at the continuous MDE (0.163 points); they are not reclassified yet because the accuracy conversion rests on one synthetic outcome model. Next: a second control that injects the effect directly into the real move series without the conversion, then reclassify cells whose interval excludes that MDE.

## Unit 3 (second, conversion-free line-move control) 2026-09-23 session 4

New script scripts/line_move_power_direct.py (ruff-clean), imports shared
helpers (season_block_bootstrap, synth_term, standardize, mde_from_grid,
PREVALENCE_CASES, TARGET_POWER) from line_move_power.py rather than
duplicating them. Design, deliberately different from unit 2 to be
conversion-free:
- The base model (model_logit, composition_flag_sum) is LOSO-refit exactly
  ONCE against the REAL, unmodified 2020-2025 home_covered outcomes (no
  synthetic Bernoulli draw anywhere in the script -> literally satisfies "no
  synthetic cover outcomes"). Its real out-of-fold logit is deterministic and
  reused for every simulation.
- Per simulation: draw a synthetic term (binary at 3/10/50% prevalence, or
  standard-normal continuous). Its only effect is (a) a known additive
  coef_points shift added directly to the REAL open_move series
  (synthetic_move = real_open_move + coef_points * standardized_term, same
  construction as units 1/2) and (b) a matched shift to the real base logit,
  coef_logit = coef_points / logistic_scale, where logistic_scale =
  margin_sd_points_empirical * sqrt(3)/pi -- a closed-form variance match
  between a logistic distribution and a Normal(0, margin_sd_points), and
  margin_sd_points_empirical is the plain standard deviation of the real
  population's margin_vs_open column (a descriptive statistic of ONE
  variable, not a regression slope fit between two variables the way unit
  2's points_per_base_logit_sd_empirical was -- this is the conversion-free
  distinction). Measured margin_sd_points_empirical = 12.9696 points.
- The variant "pick" is a deterministic threshold (variant_logit >= 0), never
  a re-drawn outcome -- no per-sim LOSO refit of a noise regressor is needed
  or done (that would only measure noise since a term uncorrelated with real
  outcomes would fit to ~0; this design sidesteps that by applying the known
  declared shift directly to the real fitted logit instead of asking a
  second model to discover it).
- diff_line_move = sign(variant_pick)*synthetic_move - sign(base_pick)*
  synthetic_move; same season_block_bootstrap paired stat as units 1/2;
  detection = interval_low > 0; same mde_from_grid interpolation, reported
  ONLY in line-move points (no accuracy-point conversion computed or
  reported at all, per the "no accuracy conversion" instruction).

Smoke test --sims 10 --draws 60 --grid 0.01,0.05,0.2,0.5 (2.3s) confirmed
mechanics (detection rises with coefficient) and margin_sd_points_empirical=
12.97 sanity-checks close to the commonly-cited ~13.5-point NFL margin SD.
Full run --sims 150 --draws 300 --fit-iterations 20 (defaults) --grid
0.05,0.1,0.2,0.35,0.5,0.65,0.85, run in the foreground, 20.6s total:
artifacts/line_move_power_direct/20260923T215739Z/results.json (games=1503,
games_scored=1503, season_blocks=6, seasons 2020-2025 -- exact population
match to the four registry cells below).

MDE at 80% power, line-move points (all four interpolated cleanly, no
boundary notes):
  binary_p03 = 0.05337
  binary_p10 = 0.07289
  binary_p50 = 0.06286
  continuous_std = 0.06535

Comparison with unit 2's MDE (same served_2020_2025 population, same units):
  unit 2 (synthetic-outcome, OLS-derived conversion): binary_p03=0.080,
  binary_p10=0.086, binary_p50=0.212, continuous_std=0.163
  unit 3 (direct, conversion-free): binary_p03=0.053, binary_p10=0.073,
  binary_p50=0.063, continuous_std=0.065
Unit 3 is tighter (more sensitive) than unit 2 across all four cells --
consistent with unit 3 having one fewer layer of injected randomness (no
synthetic-outcome resampling noise). Both controls agree on the qualitative,
decision-relevant finding: the line-move yardstick resolves effects on the
order of 0.05-0.2 points at 80% power on 1,503 games, roughly an order of
magnitude finer than the accuracy yardstick (1.9-4.9 accuracy points, unit
1) -- this corroborates unit 2's finding with an independent mechanism, so
the root decision (prefer the line-move read) does not rest on one synthetic
outcome model alone anymore.

Registry check: the four unresolved *_tuesday_line_move cells (reddit one
excluded per instruction) are all continuous/count-type terms --
diff_divergence and diff_lineup_total are continuous lineup-rating
differences (players_on_field_rating_eval.py), cfb_transfer_logit is a
continuous transfer logit (opener_error_transfer_unit2.py), and
week_gated_protection_flag_sum is gated_flag_sum = composition_flag_sum
(FLAG_SUM_COLUMN, a multi-flag COUNT, confirmed via
src/nfl_ats/pick_probability.py:79) minus a protection flag after week 4 --
a modified count, not a single 0/1 flag -- so all four are matched against
continuous_std = 0.06535, not a binary bucket. All four have n=1503 games, 6
season blocks (exact match to this harness's population). Qualification:
max(abs(interval_low), interval_high) < 0.06535.
  cfb_transfer_logit_tuesday_line_move: interval [-0.0249, 0.0714] -> max
    0.0714 > 0.06535 -> DOES NOT QUALIFY (close call, exceeds by 0.008).
  diff_divergence_tuesday_line_move: interval [-0.0586, 0.0113] -> max
    0.0586 < 0.06535 -> QUALIFIES.
  diff_lineup_total_tuesday_line_move: interval [-0.0233, 0.0027] -> max
    0.0233 < 0.06535 -> QUALIFIES.
  week_gated_protection_flag_sum_tuesday_line_move: interval [-0.032,
    0.0045] -> max 0.032 < 0.06535 -> QUALIFIES.

Exact --replace commands for the 3 qualifying cells (every original field
preserved, only classification/closing-ground/classification-evidence
changed, `weak-signals record --help` checked this session for the current
flag set) are drafted in
artifacts/line_move_power_direct/20260923T215739Z/flip_commands.md. NOT RUN
(out of scope this session; no registry writes, no commits).

Next: owner/orchestrator decides whether to run the 3 drafted commands (and
whether cfb_transfer_logit_tuesday_line_move's near-miss, 0.0714 vs MDE
0.0654, warrants a slightly larger run to tighten the boundary rather than
leaving it unresolved_below_power). This unit is COMPLETE; nothing further
to compute here.
