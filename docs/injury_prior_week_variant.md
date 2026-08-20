# Tuesday-visible injury variant: prior-week report and prior-week absence

> **Owner-corrected 2026-08-20:** this screen's motivation (section 1 below)
> is built on the premise that our PICKS lock Tuesday noon. They do not --
> only the pool's LINE locks then; picks are editable up to each game's real
> deadline (**refined 2026-08-20: min(kickoff, Sunday 16:00 ET) -- SNF/MNF
> lock early at Sunday 4pm**) (see `docs/pool_edge_plan.md`,
> `docs/injury_news_sourcing.md` §5.1). That makes this screen's motivation
> moot, not its numbers: the two
> measured arms (`injury_value_lost_prior_week_report` -0.219 pts, P+
> 0.37875; `injury_value_lost_prior_week_absence` -0.219 pts, P+ 0.3347,
> both `unresolved_below_power`, both leaning negative/coin-flip-adjacent)
> stand exactly as recorded. What is moot is the reason this design was
> worth building in the first place: it exists to find a Tuesday-visible
> substitute for the current-week official report because that report was
> believed structurally unreadable by any pool-playable cutoff. Since the
> Saturday-cutoff channel (`injury_value_lost_narrowed`, +1.316 pts, P+
> 0.8875) is directly playable via a late-week pick refresh, no substitute
> is needed -- the original signal already clears the bar this variant was
> trying to work around. These two arms remain honest, recorded negatives on
> their own terms; they just answer a question that no longer needs asking.

Predeclared 2026-08-19, before any cover-rate or sign is computed. Written
before `scripts/injury_prior_week_variant_experiment.py` is run.

## Binding closing-grounds taxonomy (restated verbatim, per AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero." The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## 1. Context and motivation

`docs/injury_news_sourcing.md` section 5.1 measured (2026-08-19) that
`injury_value_lost_narrowed`'s recorded +1.316 pt / P+ 0.8875 edge
(`docs/injury_value_lost.md` sec 4, `decision_hours_before_kickoff=24`,
i.e. a Saturday cutoff) collapses to +0.000 pts / P+ 0.3965 under a true
Tuesday-noon-ET decision cutoff (`docs/pool_edge_plan.md` line 80: "Picks
lock Tuesday at 12"), because only 0.43% of official injury-report rows
have `date_modified` at or before their own game's Tuesday noon -- the
league's own practice-report rule means the first report of a game week
files Wednesday, structurally after the pool's lock.

One injury information set IS fully final and public by Tuesday noon of the
current week: **the previous week's own final injury report, and the
previous week's actual game participation (who did and did not play)**.
Both describe events that already happened before the current week's
Tuesday lock -- no report-filing-schedule race is possible. Hypothesis:
players ruled Out/Doubtful (or who simply did not play) last week often
remain out or diminished this week (short-term injuries, IR-adjacent moves,
recovery timelines), so a value-weighted lag-1 unavailability signal may
carry playable information the Tuesday-noon opener does not fully price,
even though the *current*-week official report cannot be used.

## 2. Two arms, predeclared construction

Both arms replace the production `injuries` table entirely with a synthetic
table keyed at the **current** (season, week, team) but built only from
**prior**-week information, then feed that synthetic table into the
unmodified `nfl_ats.players.enrich_with_player_features` exactly as
production would consume a real injuries table (same severity weights via
`nfl_ats.availability.fixed_unavailability`, same role-share and
career-value EWMA state, both of which are already strictly built from
completed games prior to the current one and are untouched by this
substitution).

**Team schedule / prior-week mapping.** For each team, its games are sorted
chronologically within a season (from the `home_team`/`away_team` long form
of the game features table) and each week's "prior week" is the team's
immediately preceding row in that sort -- the actual previous game played,
not `week - 1` arithmetic, so bye weeks are handled correctly (a team's week
3 after a week-2 bye maps to week 1, not a nonexistent week 2). **Week 1 of
each season has no prior game and is excluded by construction**: both sides
of a week-1 matchup get zero synthetic injury rows, which flows through
`enrich_with_player_features` exactly like any other game with no visible
injury data (`diff_*` metrics fall back to 0.0, per its own documented
"without both reports there is no directional injury information" rule) --
not dropped from the window, just contributes no injury-driven signal in
either arm. The predeclared report counts exactly how many team-games in the
456-game window are week 1 (no prior week) before any result is read.

**Arm A -- `prior_week_report`.** For each team, take the LATEST-filed
("Friday-final") official injury-report row per player from the team's
*prior* week (mirrors `docs/injury_news_sourcing.md` sec 5.1 experiment 2's
final-report construction). Keep only rows for players who did **not**
record any snaps (offense + defense + special teams) in that prior game --
this is the predeclared "game-specific and resolved" exclusion: a
Questionable/Probable listing followed by the player actually taking the
field is treated as resolved by kickoff and dropped; an Out/Doubtful listing
followed by zero snaps is treated as unresolved and carried forward. Carried
fields: `gsis_id`, `position`, `report_status`, `practice_status`,
`date_modified` (the original prior-week timestamp, which by construction
is always more than a full week before the current game's kickoff and
therefore trivially satisfies any cutoff at or after the current week's
Tuesday noon). Severity uses the same `fixed_unavailability` mapping
production already uses (Out=1.0, Doubtful=0.85, Questionable=0.35,
Probable=0.05, practice-status fallback otherwise) -- unchanged, just
applied to last week's designation instead of this week's.

**Arm B -- `prior_week_absence`.** For each team, take every player on that
team's active game-week roster (`status` in `{ACT, INA}`, the same set
`nfl_ats.players._active_roster_features` already treats as "on the active
roster") in the *prior* week who recorded **zero** snaps in that team's
prior game (absent from the snap-counts table entirely, or present with
offense+defense+special-teams snaps summing to zero). This is a strictly
outcome-based definition -- it does not require an official report entry at
all, so it also catches roster-decision absences (healthy scratches,
suspensions) alongside injury-driven ones; that is a deliberate, declared
difference from Arm A, not an oversight. Every carried-forward row is
assigned `report_status="Out"` (severity 1.0, matching "did not play"),
`practice_status` missing, `position` from the roster row, and
`date_modified` = the prior game's own kickoff timestamp + 1 hour (safely
after the prior game concluded, and by construction always more than a
week before the current game's Tuesday noon).

**Known scope limit, declared before running:** because both arms are built
from exactly one week of lag, a player on multi-week injured reserve drops
off both the weekly injury report and the ACT/INA roster status after their
IR move is final, so neither arm re-derives "still out because still on
IR" beyond the first lagged week where that shows up in either source. This
construction tests a one-week persistence signal, not a full-IR-duration
signal; that is a scope limit to state in the writeup, not a defect to
silently work around.

`decision_hours_before_kickoff=24` is used for both arms when calling
`enrich_with_player_features`, matching the production Saturday default --
this choice is **inert by construction** here, since every synthetic row's
`date_modified` already sits more than a week before the current game's
kickoff regardless of the hours offset chosen; it is fixed only so the
non-injury feature machinery (QB expected-EPA blending, etc.) behaves
identically to the reference arms.

## 3. Population, contrast, and statistics (identical to the reference arms)

- Same 456-game `[2020, 2021]` opener-graded window that
  `injury_value_lost_narrowed` / `injury_value_lost_gradient` /
  `injury_value_lost_tuesday_cutoff_*` all used -- a free re-read of the
  already-spent `mod07_weak_signal_stack` window (no
  `rotation.assign`/`record_look`; `[2022, 2023]` is never referenced).
- Same exact snapshots as `scripts/injury_tuesday_cutoff_experiment.py`
  (`player_snapshot=20260812T200527Z`, `pbp_snapshot=20260812T142851Z`,
  `player_value_snapshot=20260813T121050Z`), so results are directly
  comparable to the recorded +1.316 (Saturday) and 0.000 (Tuesday-official)
  arms on the identical 456 games.
- Same D-A contrast: `profile="player"` (A, baseline) vs
  `profile="player_value"` (D, candidate) ridge models
  (`ridge_alpha=10.0`, `target="market_residual"`,
  `min_train_games=500`), ridge-fit forward-chained through each window,
  scored at the opener (`opener_pick_evaluation`).
- Same statistic: week-blocked bootstrap (`week_blocked_bootstrap`,
  `block="week"`, `samples=20000`, `seed=20260819`) on
  `right_correct - left_correct` (candidate minus baseline accuracy),
  reported as `delta_points`, its 95% interval, and `probability_positive`.
- A fresh `saturday` arm (real, unmodified injuries,
  `decision_hours_before_kickoff=24`) is rebuilt in the same run for a
  reproduction check against the recorded +1.316 / P+ 0.8875, and to supply
  paired per-game data for a **channel-delta** contrast (Saturday D-A minus
  candidate D-A, paired on the same games, same bootstrap machinery) against
  each of the two new arms -- the same technique
  `scripts/injury_tuesday_cutoff_experiment.py` used to isolate the
  Tuesday-to-Saturday channel.

## 4. Registry naming (checked for collisions before running)

`registry/weak_signals.json` currently has no entry named
`injury_value_lost_prior_week_report` or `injury_value_lost_prior_week_absence`
(checked directly against the 206 existing signal names, all of which were
enumerated). Both names are free to use. Recorded reliability will cite the
already-measured value-lost trait reliability (0.87-0.93,
`injury_value_lost_gradient`), not independently re-measured for this
narrower construction (same caveat `injury_value_lost_narrowed` already
carries).

## 5. DO-NOT-POOL

Both new arms share the 456-game `[2020, 2021]` window with
`injury_value_lost_gradient`, `injury_value_lost_narrowed`,
`injury_value_lost_tuesday_cutoff_official`,
`injury_value_lost_tuesday_cutoff_pft_augmented`, and
`mod07_opener_bias_ablation` -- decompositions/re-reads of the same window,
not independent evidence. Any future pool must treat them as correlated
with that whole family (see `overlap_warnings` in
`nfl-ats weak-signals pool`).

## 6. What would make this playable (decision framing, stated in advance)

Per AGENTS.md, an interval crossing zero after this run is not grounds to
close the question -- it stays `unresolved_below_power` unless a resolved
wrong sign or a positive-control bound is cleared, neither of which this
design attempts to clear. The actionable question for the EV statement is
simply whether either arm's `probability_positive` clears 0.5 (a genuine
Tuesday-visible edge worth wiring as a challenger) and how its magnitude
compares to the 0.000 Tuesday-official floor and the +1.316 Saturday
ceiling already on record.
