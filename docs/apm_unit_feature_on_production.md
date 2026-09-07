# PER-09 hierarchical team-unit feature: predeclaration

Frozen 2026-09-07 before building or scoring.

## Binding taxonomy (verbatim fleet brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.


## Frozen design

Inferred hypothesis: I think shrinkage toward roster-position units can recover
persistent team offense and defense strength beyond the active weak stack.
Read: docs/hierarchical_apm.md:40-51 fixes Ridge alpha=1000, team scale=11,
EPA clip=5 and unit-mean shrinkage k=500; use that algebra without tuning.
Read: docs/unit_apm_ratings.md:51-63 defines the four position units.

Columns, exactly: home_apm_off_rating, home_apm_def_rating,
away_apm_off_rating, away_apm_def_rating, apm_off_rating_diff,
apm_def_rating_diff. Differences are home minus away. Defense is positive
for EPA suppression. Team rating is its mean historical lineup sum of
shrunk player coefficients plus 11 times its team coefficient, in EPA/play.
Offense and defense shrink separately toward league position-unit means;
unmapped positions retain their flat coefficient. No target lineup is used.

As-of: require explicit UTC decision_timestamp and source completed_at.
Only plays from games with completed_at STRICTLY before decision_timestamp
enter. Missing completion times are excluded. For historical date-only games,
completion availability is next midnight America/New_York; decisions are the
preceding Tuesday midnight America/New_York. Same-time games remain excluded.
A supplied decision_timestamp overrides that date-only approximation.
Roster positions use only roster weeks represented by completed source games;
modal unit per player from that subset, lexicographic ties, never full-season
future roster positions. Seed is the preceding season's final available fit,
using ONLY completed prior-season games. Refit current-season available plays
walk-forward and blend each team-side rating with its seed at n/(n+500),
n=current-season team-side plays. With no current plays retain seed; without
seed use current fit; without either emit NaN. Prior-season-only seeding, no
multi-year carry-forward. Source years lacking participation remain missing.

Profile weak_stack_apm_unit adds only those six columns to weak_stack.
Population: regular-season games in the established 1,537-game 2020-2025
Tuesday-opener archive; train/replay the active 2ceaf63b56b7ce25 configuration
and verify its saved opener probabilities and picks exactly. No active edits.
All candidate tables live inside artifacts/apm_unit_opener_eval/<stamp>/.

Primary metric: candidate minus active model's OWN opener probability-rule
accuracy, in percentage points, paired on games covered on all four team-side
ratings, excluding pushes. Also report full archive policy using baseline
picks where features are missing, coverage, flips and per-season deltas.
20,000 week-blocked paired bootstrap draws, seed 20260817; within-week game
correlation is ZERO, never estimated or padded. Frozen-pick null: keep both
pick vectors fixed and shuffle outcomes within week, 200 draws, same seed.
No feature or hyperparameter selection; all signs reported.

Rotation family apm_unit_on_production inherits participation_offense_defense_rapm
and unit_apm_ratings. The incumbent stack is the comparator, not a new stack
selection exercise. Acknowledge 2018-2025 mining and earlier APM/EPA work:
results are discounted reused historical evidence, not independent confirmation.
Request earliest two-season window BEFORE scoring. Score only assigned windows;
record each through weak-signals record and rotation record, then assign the
next window, irrespective of sign, until all three two-season archive windows
are covered. On any assignment refusal report its exact text and stop scoring;
complete the builder. The full archive summary is permitted only after all its
windows have been assigned and scored. Never score an unassigned window.
Classification defaults unresolved_below_power; thresholds govern claims only.
No automatic promotion, played-profile change, card, or ledger writes.


## Results appended after scoring, 2026-09-07

Measured: `uv run --no-sync python -m scripts.apm_unit_opener_eval` wrote
`artifacts/apm_unit_opener_eval/20260907T144504Z/opener_summary.json`.
Measured in that artifact: candidate 52.9607% versus incumbent 53.3599%,
delta -0.399202 accuracy points, week-blocked 95% [-1.860465, +1.061712],
probability_positive=0.28295; 1,537/1,537 covered, 1,503 nonpush, 107 weeks,
116 changed picks. Inferred: I think the expected-value read favors retaining
the incumbent; this does not refute the hierarchical APM mechanism.
Measured: three sequential assigned windows were scored and recorded through
both CLIs, each unresolved_below_power/unresolved; exact commands and outputs
are in the artifact's registry_commands.log. No unassigned window was scored.
Measured: season deltas for 2020 through 2025 are -2.727273, -1.271186,
0.000000, +0.375940, -1.879699, +2.621723 points respectively; their intervals
and probability_positive are preserved in opener_summary.json.
Measured: frozen-pick null mean +0.213573, central 95% [-1.333999, +1.729874],
observed percentile 0.225 (200 outcome shuffles with picks held fixed).

Read: src/nfl_ats/clv.py:2028-2030 discloses that the inherited evaluator
swaps only spread_line to the opener; the incumbent's other features retain
close-era information. Inferred: the new APM columns satisfy their as-of rule,
but this inherited evaluator limitation precludes claiming that the entire
incumbent table is a Tuesday-vintage snapshot. Read: the frozen design above
acknowledges earlier APM/EPA work and mined-era reuse; all results carry that
discount. Measured: sources.json inventories 269,651 valid source plays from
2016-2025, including prior-season seeds; the source snapshots therefore cover
more seasons than the earlier 2019-2024 hierarchy screen described.
