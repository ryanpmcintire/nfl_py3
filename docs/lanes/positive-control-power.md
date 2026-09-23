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

## Root decision 2026-09-23

The 11 drafted bounded_by_control reclassifications are NOT run. The minimum detectable effects (1.5-4.9 accuracy points at 80% power) are 2-10x the plausible size of these terms (every point estimate today sits between -1.5 and +0.3 points), so they bound only implausibly large effects; AGENTS.md closes a line only for a control able to detect an effect of the size in question. The table stands as the evaluator's resolution: single added-term accuracy tests on 1,503 or 3,734 games cannot resolve realistic effects; prefer the line-move yardstick and larger populations. A cell may be reclassified only when its hypothesized size is at or above the matched MDE.

- Superseded by "Root decision 2026-09-23" above: the drafted commands are not run; nothing here awaits review.
