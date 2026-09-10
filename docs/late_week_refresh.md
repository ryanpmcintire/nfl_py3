# Late-week pick-refresh flow (POL-11)

Owner correction, **2026-08-20**: pool picks are editable up to each game's
own kickoff. Only the **grading lines** freeze at the Tuesday noon lock. That
had not been the project's working assumption, and it unlocks a structural
edge the project had never spent: a pick can be reconsidered, mid-week, with
information the frozen Tuesday line never had a chance to price -- a Friday
injury designation, a kickoff-nearest weather forecast, or simply a fresher
run of the active model. This document is the design and operating record
for the flow that spends it: `nfl-ats refresh-picks`
(`src/nfl_ats/pick_refresh.py`).

It is deliberately self-contained. `docs/prospective_evidence.md` and
`docs/pool_edge_plan.md` are owned by other work in flight and are not edited
here; cross-links can be added once that work lands.

## The one sentence that governs everything below

**Grading is always against the frozen Tuesday line; deciding can happen any
time before a pick's own deadline.** Two different questions, two different
freeze points, and the whole design is just keeping them from ever touching
the wrong artifact.

## Cadence

```
Tuesday noon ET   publish-predictions --record-decisions   (unchanged, exactly as-is)
                    |  locks the grading lines for the week
                    v
Wednesday 6:15 PM   refresh-picks --record-decisions --note wednesday_opener
                    |  only matters in a week with a Wednesday game (2026 Week 1
                    |  opens Wednesday); otherwise finds nothing changed
                    v
Thursday (pre-TNF) refresh-picks --record-decisions --note thursday_afternoon
                    |  finalizes TNF picks -- Tuesday-to-Thursday information only
                    v
Saturday            refresh-picks --record-decisions --note saturday_pass
                    |  everything not yet locked gets one more look
                    v
Sunday morning       refresh-picks --record-decisions --note sunday_morning_final --publish-card
                    |  FINAL pass: locks the rest of the week, including SNF/MNF,
                    |  at Sunday 4:00 PM ET (see "Per-game deadline" below)
```

Four named passes, not a fixed weekly cadence bolted onto a single Tuesday-
Sunday window, because **Thursday games exist**: the week's first kickoff can
be less than 48 hours after the Tuesday lock, so "refresh once, on Sunday" is
too late for TNF and "refresh once, on Thursday" is too early for everything
else. None of the passes names a week: `--season`/`--week` default to the active
model's linked weekly forecast (`artifacts/active_ats_model.json`), which is
the week `publish-predictions` locked, so the scheduled jobs in
`scripts/capture_scheduler.py` stay correct all season without editing
(added 2026-09-07 after the first in-season Sunday passes failed on the then-
required pair; `tests/test_capture_scheduler.py` now parses every scheduled
`nfl-ats` argv against the real parser).
Each pass is the *same* command, run again; nothing about
`refresh-picks` itself is pass-specific except the free-text `--note` label
recorded alongside anything it changes (`thursday_afternoon`,
`saturday_pass`, `sunday_morning_final`, or whatever cadence a given week
actually uses -- the label is not validated against this list, it is purely
for later legibility of the ledger and the card).

Running it more often than this cadence is harmless: a pass that finds
nothing changed writes nothing (see "No-op refresh" below), so there is no
penalty to refreshing more often than the plan above, only to refreshing less
often.

## Per-game deadline, not one weekly cutoff

Every game's pick can change until **`min(that game's own kickoff, that
week's Sunday 4:00 PM ET)`** (`nfl_ats.pick_refresh.pick_deadline`,
`sunday_pick_lock`). Two owner directives, both 2026-08-20, produce this
formula:

1. A pick may change up until its own game's kickoff -- the literal "picks
   are editable to kickoff" rule.
2. Nothing may change after **Sunday 4:00 PM ET** of that week, even for
   games that kick off later (SNF, MNF) -- so Sunday-night and Monday-night
   picks lock **early**, at the same moment as the rest of the week, not at
   their own kickoff.

The two rules do not conflict; they compose into one `min()`. A Thursday
game's own kickoff is always earlier than that week's Sunday, so rule 2 never
touches it -- rule 1 alone governs TNF. A Sunday-early or Sunday-afternoon
game's kickoff is also before the 4:00 PM cap in every case this project
schedules against, so rule 1 governs those too. Only SNF and MNF are ever
bound by rule 2 instead of their own kickoff, and for those two the deadline
is **earlier** than kickoff by design.

**The Sunday anchor is computed from the week's own games, not a calendar
guess.** `sunday_pick_lock` reuses the exact Tue..Mon week-cycle anchor
`nfl_ats.odds_backfill.plan_backfill` already uses for its own decision
timestamps (mode Sunday among the week's kickoffs, so one isolated
Tuesday/Wednesday reschedule cannot shift the whole week's lock instant), at
`America/New_York` local time via `zoneinfo`, which resolves the EDT/EST
offset correctly for whichever calendar date the week actually falls on.

**Consequence for reading this channel's evidence later, stated once here so
it does not have to be rediscovered:** the four passes above give a
Thursday-night pick at most a Tuesday-to-Thursday information window, while a
Sunday or Monday pick can use everything through Sunday afternoon. The
channel's information depth is **not uniform across a week's games**, and any
analysis of refresh-driven accuracy that pools TNF and MNF picks together
without accounting for this will understate what the Sunday/Monday half of
the channel can do and overstate what the Thursday half can.

The guard is enforced in code on every call, not left to caller discipline:
`plan_refresh` computes `eligible`/`ineligible_reason`
(`"kickoff_passed"` or `"sunday_pick_lock_passed"`) per game, and
`record_plan` only ever appends a revision for a `changed` game, which by
construction is already `eligible`. `tests/test_pick_refresh.py` pins both
halves directly: a game whose own kickoff has passed is never revised even
hours before the Sunday cap, and an MNF game is refused **after** the Sunday
4:00 PM cap **even though its own kickoff is still a day away**.

## Grading vs. deciding

| | Frozen at | Source | Ever rewritten? |
|---|---|---|---|
| **Grading line** (what the pool settles against) | Tuesday lock | `nfl_ats.clv.load_paper_decisions` (`decision_home_spread`) | Never -- `record_paper_decisions` already guarantees a republished card never moves it |
| **Deciding side** (what gets submitted) | Each game's own deadline, above | latest row in the pick-revision ledger, or the Tuesday pick if none | Never in place -- only ever appended to |

`refresh-picks` **never** re-reads the current market line to score a game.
It recomputes the active model's probability with **current features**
(current injury designations if the feature table has been rebuilt since
Tuesday, current everything upstream of the spread) but always evaluates
that recompute at the **original, frozen** spread
(`nfl_ats.lines.apply_external_lines`, unchanged, reused exactly as
`pool-card-at-lines` already uses it). A game with no recorded original line
**cannot be refreshed** -- `plan_refresh` fails closed for that game
specifically (reported in `unrefreshable_game_ids`), and fails closed for the
whole week if the Tuesday card was never recorded at all (no
`--record-decisions` run happened yet: `plan_refresh` raises, naming the
missing step).

The original line is read from the **paper-decision ledger**
(`artifacts/clv_ledger/decisions.parquet`), not from a fresh read of the
linked weekly-forecast artifact. That is a deliberate choice: the ledger is
the one place in this codebase already proven to survive a same-week
republish without its anchor moving (`record_paper_decisions`: "a
republished card with a moved line never rewrites the CLV anchor"), and it is
already what the pool's own grading depends on. A direct consequence: **the
Tuesday `publish-predictions --record-decisions` run is a prerequisite for
every refresh pass that follows it**, exactly as before this feature existed.

### Model identity

`refresh-picks` reads the SAME active-model manifest
(`artifacts/active_ats_model.json`) the Tuesday card was built from --
method, feature profile, regressor, ridge alpha, probability method -- and
refuses to run if that CONFIGURATION no longer matches the one the recorded
original card's forecast (`forecast_artifact/metadata.json`) was produced
under (`plan_refresh` raises
`"...refuses to recompute picks under a different model identity..."`).
This mirrors `nfl_ats.weekly.assert_synchronized`'s spirit for the Tuesday
card: a refresh must never silently recompute this week's picks under a
model the pool's frozen line was never actually locked against.

**Why the configuration and not the `model_id` (2026-09-09).** The id hashes
the feature table's digest, so every daily player-data refresh (`lineups_*`
-> `weekly-run --refresh-player-data`) mints a new id. Until 2026-09-09 the
check compared ids, which meant the first daily refresh after the Tuesday
lock left every refresh pass for the rest of the week refusing -- measured
on the first in-season Wednesday (`--run-job refresh_wed_inactives_primetime
--dry`: recorded `9ef62ef158338785`, active `c657058903f3232b`), hours
before the Week 1 opener, with the late-week follow rule still unfired.
Recomputing under current data is this command's stated purpose; the
identity that must not drift is the configuration. A missing forecast
artifact or a genuine configuration change still fails closed.

## Overlays

The played Tuesday card now uses one frozen four-member policy. Coach fade,
division revenge, player arrests, and spread-gap zone are evaluated
independently against the raw model card; their game ids are unioned and an
affected raw pick is complemented exactly once. `refresh-picks` never reruns
those detectors. It reads the four Tuesday flags and `composed_overlay_flip`
from the paper ledger, refits the raw model at the frozen Tuesday line, then
applies that frozen union exactly once. Only after that does the observed-
movement rule get a chance to override the side.

The refresh never queries a newer arrest or schedule snapshot: a later source
revision could backfill information dated before Tuesday and retroactively
alter the decision-time information set. The paper ledger's `pick_side` is the
final Tuesday played side; `model_pick_side`, `former_policy_pick_side`, the
four member flags, and the source hashes preserve both the raw and former-
production counterfactuals.

Other overlays remain prospective attribution arms unless separately promoted.
In particular, injury value-lost, backup-QB fade, and surface-switch are not
part of this production union.

## Observed-movement pick policy (POL-11 addendum, 2026-08-20)

Unlike every overlay above, this one **is** wired into the played pick, not
left as challenger-only evidence. It is a market-based decision rule, not a
pick-level overlay, so it sits outside the "no overlay logic is touched"
statement above and is documented separately here.

**The evidence base, and what it does and does not establish.** Measured
2026-08-20 (`docs/observed_movement_channel.md`,
`scripts/observed_movement_channel.py`) against the identical frozen
`weak_stack`/ridge/alpha-10 production recipe `docs/opener_evaluation.md`
runs, paired against the production pick on the same games, week-blocked
bootstrap (20,000 samples, seed 20260819). All six entries below are
[read] from `registry/weak_signals.json` as recorded, and every one of them
is `unresolved_below_power` -- an interval crossing zero is never grounds to
reject a signal (AGENTS.md's binding rule), and none of these is being
represented as a resolved finding:

| Cell | Grading | Effect (accuracy pts) | Interval (week-blocked) | P+ | n |
|---|---|---|---|---|---|
| `observed_movement_threshold_0_5` | Tuesday to close, full-slate | +1.663 | [-1.193, +4.536] | 0.873 | 1,503 |
| `observed_movement_threshold_1_0` | Tuesday to close, full-slate | +1.863 | [-0.469, +4.267] | 0.935 | 1,503 |
| `observed_movement_threshold_0_5_sunday_am_realism` | Sunday-morning realism, 2023-2025 | +1.627 | [-2.574, +5.808] | 0.764 | 799 |
| `observed_movement_threshold_1_0_sunday_am_realism` | Sunday-morning realism, 2023-2025 | +3.254 | [+0.251, +6.266] | 0.981 | 799 |
| `observed_movement_oracle_full_slate` | Tuesday to close, oracle | +1.730 | [-1.110, +4.564] | 0.883 | 1,503 |
| `observed_movement_oracle_sunday_am_realism` | Sunday-morning realism, oracle | +2.253 | [-2.122, +6.582] | 0.832 | 799 |

An independent family reads even stronger on a different partial window:
`odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025` (Tuesday-to-
Wednesday oracle, always-flip design) reads +4.4669 points, P+ 0.9326,
n=347 -- reconciled, not double-counted, in
`docs/observed_movement_channel.md`.

**The threshold: 1.0, frozen from the predeclared 0.5/1.0 grid, not
re-tuned here.** At BOTH gradings measured -- the Tuesday-to-close
full-slate grading and the Sunday-morning-realism grading -- the >=1.0
cell reads a larger paired-delta effect AND a higher `probability_positive`
than the >=0.5 cell (1.863 pts / P+ 0.935 vs 1.663 pts / P+ 0.873 at the
first grading; 3.254 pts / P+ 0.981 vs 1.627 pts / P+ 0.764 at the second).
That is the entire justification for choosing 1.0 -- it is the stronger of
the two predeclared cells at both readings, selected from a grid fixed
before either number was seen, not tuned after the fact.

**This is an EV decision, not a claim that the channel is resolved.** Per
AGENTS.md's "A promotion bar is not a decision bar": the pool is forced
picks, so declining a change that is ~93-98% likely to help (at the
strongest, most realistic grading) is not caution, it is taking the losing
side of the bet. `probability_positive` 0.981 on the Sunday-morning-realism
cell, and 0.935 on the full-slate Tuesday-to-close cell, both comfortably
clear the standing "play forced-pick positive EV" order. Wiring this policy
is that decision, made and executed, not merely proposed -- write-ups of it
must say so plainly rather than hedge it back into "under consideration."

**Honest caveats, carried forward from the predeclaration, not smoothed
over:**
- The Sunday-morning-realism cells are built from the 2023-2025
  `intraday_hourly` archive only, whose real coverage ceiling is ~10:55 ET
  Sunday (not the nominal 13:00/16:00 cutoffs) -- see
  `docs/observed_movement_channel.md`'s addendum. The measured
  Sunday-realism numbers are therefore a **lower bound**: real
  market movement between ~11:00 ET and kickoff (same-day injury news
  especially) is not captured by this archive and is not measured here.
- The Tuesday-to-close full-slate oracle overstates reachability for
  SNF/MNF/late-Sunday games specifically, since the close prints after the
  true Sunday 16:00 ET pick deadline; the Sunday-realism cells are the
  deadline-respecting variant for that subset.
- No positive control has been run for this channel, so none of the six
  cells above can be classified `bounded_by_control`; none is
  `wrong_sign_resolved` either (no whole interval sits below zero). All six
  therefore stay `unresolved_below_power`, correctly, in the registry.

**The rule, exactly as implemented (`nfl_ats.pick_refresh.plan_refresh`,
`current_captured_home_spread`, `MOVEMENT_POLICY_THRESHOLD`).** At each
refresh pass, for each still-open game (the existing per-game deadline
guard already decides which games are even reachable): let
`delta = current_captured_line - decision_home_spread` -- the current
locally-captured home spread minus the frozen Tuesday line, home-oriented,
the identical sign convention `open_move` uses in
`scripts/observed_movement_channel.py` and throughout `nfl_ats.clv`
(`tue_open_home_spread` / `close_home_spread` are the same `home_spread_line`
column that convention already relies on). If `abs(delta) >= 1.0`: the
refreshed pick becomes the side the market moved toward (`delta > 0` picks
HOME, else AWAY -- reused verbatim from the measurement script's
`_threshold_pick`). Otherwise: the refreshed pick is the model's own
recomputed pick, exactly the prior behavior.

**Where "the current captured line" comes from -- read-only, no live fetch
from inside `refresh-picks`.** `scripts/odds_capture.ps1` (a Windows Task
Scheduler job) already runs `nfl-ats odds-ingest` several times each
morning (`docs/ops_runbook.md`: "the scheduled live odds captures land"
~06:00-09:00 ET) and writes into `data/market/raw` via
`nfl_ats.market_data.write_market_snapshot` -- the SAME store the
historical `odds-backfill` executor also writes into.
`current_captured_home_spread` reads that store with
`nfl_ats.market_data.load_quote_history` / `spread_consensus`, the
IDENTICAL "current line" adapter `nfl_ats.best_pick_nomination.week_dispersion_pool`
already uses for exactly the reason stated there ("never calls the odds
API") and the one `odds-summary` surfaces to a human. `refresh-picks`
never triggers a live capture itself -- `predict_close_for_week`
(`nfl_ats.clv`) was checked and confirmed to be equally read-only (it
raises `ClosePredictionUnavailable` rather than fetching on demand), so
this mirrors the existing production pattern rather than inventing a new
one, and avoids spending API quota or making a network call on every
`refresh-picks` invocation (which, per the cadence above, can run several
times a week and in rehearsal without `--record-decisions`).

**Fail-open, explicitly.** "Fresh" means the newest quote's
`observed_at_utc` across the whole local market store falls on the same
America/New_York calendar date as the refresh's own `now`. If the store is
empty, or its newest quote is not from today (the scheduled capture hasn't
landed yet, or is stale), `current_captured_home_spread` returns an EMPTY
mapping and the movement policy is a **no-op for every game that pass** --
every pick falls back to the model's own recompute, logged in
`refresh_summary`'s `movement_policy` block
(`current_line_fresh: false`, with a `current_line_reason`). The
model-pick refresh itself always proceeds regardless. The same fail-open
applies per game even when the store IS fresh overall, if that specific
game's line was not matched/captured this pass.

**Both arms recorded, always.** Every pick-revision ledger row now carries
four additional columns: `movement_policy` (`movement_ge_1.0` or
`model_only`), `movement_delta` (the signed point move, `null` when no
fresh line was available), `movement_pick_side` (the side the market
moved toward, computed whenever a delta exists, even on rows where the
policy did not select it), and `model_only_pick_side` (the model's own
recomputed pick, always present -- the counterfactual). `new_pick_side`
is always the PLAYED pick; `new_home_cover_probability` is always the
model's own probability estimate and is never altered by the movement
policy (only the discrete side can be overridden) -- so on a
movement-governed row, `new_pick_side` may not equal the usual
>=0.5-on-probability rule. That is the one deliberate, disclosed exception
to that invariant in this codebase, fully recoverable from these four
columns on every row.

**Tracked challenger.** `model_only_refresh_incumbent`
(`artifacts/prospective/challengers.json`) is the counterfactual arm:
"what would this week's refresh have picked with no movement override."
Its evidence is entirely reconstructable from `pick_revisions.parquet`'s
`model_only_pick_side` column against the same `decision_home_spread`
grading line every other arm uses -- see that registration for the full
evidence block and honest caveats. The settlement path exists as of
2026-09-09: `nfl-ats settle` grades `pick_revisions.parquet`'s
`previous_pick_side` and `new_pick_side` as two arms at the frozen
`decision_home_spread`, taking the latest pass recorded before each game's
own deadline.

## The pick-revision ledger

`artifacts/prospective/pick_revisions.parquet`
(`nfl_ats.pick_refresh.PICK_REVISION_COLUMNS`), append-only, one row per
**changed, eligible** game per refresh pass -- a pass that finds nothing to
change writes zero rows (see "No-op refresh").

| Column | Meaning |
|---|---|
| `revision_recorded_at_utc` | When this specific revision was written |
| `refresh_run_id` | Groups every row one pass produced |
| `season`, `week`, `game_id`, `home_team`, `away_team`, `kickoff` | Game identity |
| `decision_home_spread` | The FROZEN Tuesday line -- identical on every revision of a game, by construction |
| `original_recorded_at_utc` | The Tuesday paper-decision ledger's own `recorded_at_utc` for this game |
| `previous_pick_side` | The pick immediately before this revision -- the Tuesday post-overlay pick for a game's first revision, the prior revision's `new_pick_side` for every one after |
| `previous_home_cover_probability` | The prior revision's probability, or blank for a game's first revision (the paper-decision ledger never recorded the Tuesday probability, only the side) |
| `new_pick_side`, `new_home_cover_probability` | This revision's result -- the PLAYED pick; the probability is always the model's own estimate, never altered by the movement policy below |
| `decision_policy_id`, `decision_policy_fingerprint` | Exact production policy frozen on Tuesday |
| `coach_fade_flip`, `division_revenge_flip`, `player_arrests_flip`, `spread_gap_zone_flip` | Frozen Tuesday member flags; each member was evaluated against the raw card |
| `composed_overlay_flip` | OR of the four member flags; the refitted raw side is complemented once when true |
| `player_arrests_snapshot_id`, `player_arrests_safe_index_sha256` | Provenance copied from Tuesday's paper row; refresh never opens that snapshot or a newer one |
| `movement_policy` | `late_week_leader_median_follow_1_0` (the promoted late-week follow governed this pick -- see "Promoted late-week follow" below), `late_week_leader_median_follow_1_0_news_veto` (that follow fired and post-Tuesday injury news contradicted it, so the Tuesday pick stands -- see "Injury-news veto" below), `movement_ge_1.0` (the observed-movement policy governed this pick), or `model_only` (below both thresholds, or no market evidence -- see "Observed-movement pick policy" above) |
| `movement_delta` | The governing arm's signed move in home-oriented points: the late-week leader-median net move when the late-week arm governs, else the consensus delta when a fresh captured line exists, else the late-week net when only that arm has evidence; blank/null when neither arm does |
| `movement_pick_side` | The side the governing (or counterfactual) market arm points at, whenever `movement_delta` is not null -- the candidate side even on rows where `movement_policy` did not select it |
| `model_only_pick_side` | The recomputed production-policy pick (post frozen four-member union, pre movement-policy override) -- always present; the counterfactual the `model_only_refresh_incumbent` challenger tracks |
| `late_week_net_move` | The served late-week arm's own evidence: the MEDIAN Wednesday-to-deadline net move across the three leading books; null when the live intraday archive has no usable quotes for this game this pass |
| `late_week_pick_side` | The leading books' side at the served full-point threshold; blank when unavailable |
| `late_week_eligible_books` | How many of the three leading books contributed an increment; 0 when none did (the arm cannot fire). The equal-book arm's own twelve-book count stays on its paired challenger ledger |
| `consensus_delta` | The 1.0-point consensus arm's own evidence: `current_captured_home_spread - decision_home_spread`; null when no fresh captured line exists for this game |
| `consensus_pick_side` | The side the consensus move points at; blank when unavailable |
| `follow_news_veto` | True when the follow fired and post-Tuesday injury news pointed against it, so the Tuesday pick was kept. The un-vetoed side stays on `movement_pick_side`, which is the paired OFF arm |
| `follow_news_source` | Which reader answered: `official` (the injury report, when that season's rows carry a real timestamp), `pft_fallback` (ProFootballTalk headlines when they do not), or `none` |
| `follow_news_team` | The team the market moved TOWARD -- the one the contrary injury news is about; blank when the follow did not fire or nothing was readable |
| `model_id` | The active model this revision was computed under |
| `feature_table_sha256` | Provenance: which exact feature-table build produced this revision |
| `reason` | `"pick_refresh recompute"`, or `"pick_refresh recompute (<note>)"` when `--note` was passed |

**Anti-backdating guarantees, all enforced in code (`tests/test_pick_refresh.py`
pins every one of these):**

1. **Append-only.** Nothing already written is ever rewritten in place --
   not the Tuesday card (`publish-predictions` is untouched by this
   feature), not an earlier revision. A game revised twice in a week has two
   rows; the chain (Tuesday pick -> revision 1 -> revision 2 -> ...) is
   fully recoverable by sorting on `revision_recorded_at_utc`.
2. **Per-game kickoff guard.** A game whose own kickoff has passed is never
   revised, regardless of the Sunday cap.
3. **Week-wide Sunday 4:00 PM ET cap.** No game -- including one whose own
   kickoff is still a day away -- is ever revised after that week's Sunday
   lock instant.
4. **Recording is opt-in**, exactly like `publish-predictions
   --record-decisions`: `record_plan`/`record_refresh` default
   `record_decisions=False`. `plan_refresh` always computes and reports what
   *would* change; only passing `--record-decisions` writes anything.
5. **The rehearsal-lock-window guard is reused, unchanged.**
   `nfl_ats.clv.refuse_if_outside_recording_lock_window` -- the exact
   function that already guards `publish-predictions --record-decisions`
   and `prospective-record` -- is called against the week's ORIGINAL
   kickoffs before any revision write. A refresh invoked weeks before a
   real lock week (the 2026-08-18 incident this constant exists to prevent
   from recurring) cannot reach the ledger, independent of the per-game
   deadline check above.
6. **Model-identity check.** Covered above under "Model identity" -- a
   changed active model refuses the whole refresh, not just a silent
   recompute under the wrong model.

### Recovering both the Tuesday pick and the final pick

- `nfl_ats.pick_refresh.original_card(artifacts_root, season=, week=)` --
  the Tuesday-recorded pick and line, untouched.
- `nfl_ats.pick_refresh.final_pick_per_game(artifacts_root, season=, week=)`
  -- the FINAL pre-kickoff pick per game: the latest revision if one
  exists, else the Tuesday pick, with a `revised` flag so a later scoring
  pass can tell the two apart. Both are available for every game
  simultaneously, so a season can eventually be scored both ways (Tuesday
  vs. final) without re-deriving either from the other.

## `--publish-card`: additive, never touches the Tuesday section

`CURRENT_PREDICTIONS.md` stays exactly what `publish-predictions` wrote by
default. `refresh-picks --publish-card` is opt-in and appends (or, on a
later pass, replaces just its own) a clearly-labeled section:

```
## Late-week refresh (as of <timestamp>)

<N> picks changed since the Tuesday card (<note>), recomputed with current
data but scored at the frozen Tuesday grading line. Only games whose
deadline (their own kickoff, or that week's Sunday 4:00 PM ET if earlier)
had not yet passed were eligible. "Policy" is
`late_week_leader_median_follow_1_0` when the three leading books moved the
line at least a full point since Tuesday and the pick followed them,
`late_week_leader_median_follow_1_0_news_veto` when they moved that far but
the injury report points the other way, so Tuesday's pick stands,
`movement_ge_1.0` when the pool's own captured line
instead moved >=1.0 point and the pick followed it, or `model_only` when
neither market arm fired -- see the movement-policy sections above. This is
research output, not a wagering recommendation.

| Matchup | Previous pick | New pick | Model estimate | Policy | Market move |
|---|---|---|---|---|---|
...
```

Implementation: `nfl_ats.pick_refresh.append_refresh_to_card`, marked by
`<!-- LATE_WEEK_REFRESH:START/END -->` (a separate marker pair from
`publishing.py`'s own `<!-- CURRENT_PREDICTIONS:START/END -->` in
`README.md` -- the two never interact). Re-running the append (Saturday
after a Thursday pass, Sunday after a Saturday pass) replaces the section in
place rather than stacking duplicate blocks, so the card always shows only
the most recent pass's changes; it fails closed (raises) if no published
card exists yet at the destination -- `refresh-picks` is a second step, and
`--publish-card` cannot invent a first one. When a pass changes nothing,
the section still writes, saying so plainly ("No pick changes since the
Tuesday card"), rather than silently leaving a stale section from an earlier
pass in place.

## Exact commands

## Promoted late-week follow (MKT-15, leader median served 2026-09-09)

**What changed on 2026-09-09.** The served arm is no longer the equal-book
mean of all twelve books. It is the **median Wednesday-to-deadline net move of
the three leading books** -- Bovada, William Hill (US), MyBookie, the three
`docs/book_leadership.md` identifies as leading roughly three-fifths of their
own line moves -- at the then-frozen 0.5-point threshold. The equal-book rule
it replaced keeps recording as the paired OFF challenger.

**What changed on 2026-09-10: the gate is a FULL point, not half a point.**
`docs/follow_threshold_live_card.md` measured a predeclared 0.5 / 0.75 / 1.0 /
1.5 grid on the nine-member card that is actually played, 2023-2025, 799
opener-graded games. Against the served 0.5 arm, the 1.0 arm reads **+2.003
accuracy points, week-blocked 95% [-0.769, +4.851], `probability_positive`
0.918** (season-blocked [+0.376, +4.869]) on 140 changed picks, and **+1.252
points against not following at all** (`probability_positive` 0.846) where the
served 0.5 arm reads **-0.751** (`probability_positive` 0.332). The 1.0 arm is
the only one positive on BOTH the played card and the raw model card
(+1.252 / +1.752). Every cell is `unresolved_below_power`; an interval
containing zero is not a rejection ground (AGENTS.md), and the pool is forced
picks, so the arm with the higher expected accuracy at the deadline is served.

**The mechanism, named, because a threshold that flips the played card must
name one (AGENTS.md).** The leaders' median lives on a half-point lattice, and
the band decomposition splits cleanly by move size: inside 0.5-to-0.75 the
market side LOSES the picks it reverses -- 22 of 52 member-flip reversals
(42.3%) and 40 of 88 raw-pick reversals (45.5%) -- while at a full point or
more it WINS them (58.5% and 52.5%). The served 0.5 rule spent 288 of its 514
scored firings in the losing band; the [0.75, 1.0) band is empty on the whole
816-game archive, so raising the gate to 1.0 deletes exactly that band and
keeps every firing where the market side wins. This is a statement about how
much information a market move of a given SIZE carries, not an accuracy dip
located at a spread number. Higher is not automatically better either: the 1.5
arm is worse than 1.0 on both surfaces (+1.126 vs +2.003 against the 0.5 arm),
which is what one losing band -- rather than a monotone trend -- looks like.

**The constant is separate on purpose.** The served gate is
`sharp_book_movement_features.LEADER_FOLLOW_THRESHOLD` (1.0); the equal-book
paired challenger stays on `THRESHOLD` (0.5), the constant it was measured at,
so the two arms remain comparable game for game. The retired half-point
leader-median arm keeps recording as its own paired OFF challenger
(`late_week_leader_median_follow_0_5_off_incumbent`,
`leader_median_half_would_be_pick_side` on the follow ledger), and the policy
id on every revision row became `late_week_leader_median_follow_1_0`, so the
ledger keeps the two eras distinguishable without a migration. The rule had
never fired on a played card before the Thursday 2026-09-10 refresh, so
nothing is reversed retroactively.

## Injury-news veto on the follow (F3p, 2026-09-10)

`docs/follow_news_gate.md` measured a second, separately predeclared change
inside the follow branch. On the 816-game archive, over the games where the
leaders moved and the injury report can be read, following the market is a
literal coin flip against just keeping the Tuesday pick (189-189). Split those
fires by whether an injury filed since Tuesday noon agrees with the move and
the halves separate by 7.5 points: **following a CONFIRMED move is worth +3.18
accuracy points, following a CONTRADICTED one costs 4.35**, and the "neither"
bucket is an exact 43-43 tie.

**The served rule.** When the follow fires and post-Tuesday injury news points
AGAINST the move -- the team the market moved TOWARD is the one whose
skill-position injury situation just got worse -- the market side is discarded
and the Tuesday pick stands. Paired against following every move, that reads
**+1.126 accuracy points, week-blocked 95% [-1.242, +3.461],
`probability_positive` 0.830**, on 79 pick changes across three seasons. The
positive control (perfect foresight on the same decision set) resolves at
about +29 points, so the instrument is proven and nothing here is bounded by
it; every cell is `unresolved_below_power`.

**The reader, exactly.** `injury_signal_refresh_tilt.follow_news_for_game`.
For each skill-position player (QB/RB/WR/TE) on each team, `prior_sev` is that
player's latest designation filed at or before this week's Tuesday noon ET,
searched across the whole season (a cross-week prior -- the own-week baseline
was measured identically zero on 494 of 494 games), and `final_sev` is the
latest own-week designation filed by the pass instant. Only players with at
least one own-week row filed inside `(Tuesday noon, now]` contribute, so a
player who simply never reappears on this week's report cannot be credited
with a recovery. Severity is `Out=4, Doubtful=3, Questionable=2, Probable=1`.
The direction is defined from the MARKET, never from the pick:
`news_toward_market = news(the team the market moved against) - news(the team
it moved toward)`, with the abandoned/favored roles read off
`sign(leader_median_net_move)`. At or above `INJURY_NET_THRESHOLD` (2.0) the
news CONFIRMS the move; at or below -2.0 it CONTRADICTS it and the veto fires.
On a season whose official rows carry no usable timestamp -- measured: every
2025 and 2026 row of snapshot `20260909T223500Z` is `week_proxy` with a null
`date_modified` -- the reader falls back to the ProFootballTalk headline path
at its own bar of 1.0, which is what makes this F3p rather than F3 and what
lets it act at all on the live season. Fail-open everywhere: no snapshot, no
headline archive, an unreadable store or a zero move all read as "no news",
and no news never vetoes.

**Precedence.** The veto is a guard INSIDE the follow branch, never a step
below it: a vetoed game counts as "the follow fired", so it does not fall
through to the 1.0-point consensus arm, the heavy-handle rule or the
rookie-crew step. That is how it was measured. Its OFF arm is the un-vetoed
side, which every revision row already carries as `movement_pick_side`, and
the follow ledger records it explicitly as
`late_week_follow_no_news_veto_off_incumbent`.

**What the reader sees.** "The line moved, but the injury report points the
other way." -- the sentence on the refreshed-pick line, and one clause in the
board's late-week note.

**Two things stated plainly.** The higher-scoring arm in that battery is F3w
(+1.752, `probability_positive` 0.964), the same veto computed on an
unrestricted news delta; it is not served because its confirmation flag's
split-half reliability is -0.027 against F3's +0.035 and its "news" credits a
team whenever a player merely fails to reappear on the report -- a filing
artifact. And F2 ("news lowers the movement bar") is the one resolved result
in that battery and it is negative (-1.752, whole week-blocked interval below
zero), recorded `refuted_mechanism` on `wrong_sign_resolved`; what that refutes
is narrow -- a sub-half-point single-book move plus agreeing news is not a
follow signal -- and it closes nothing about the veto.

Measured 2026-09-09
(`artifacts/sharp_weighted_follow/20260909T233606Z/`, predeclared in
`docs/sharp_weighted_follow.md` before any arm was computed): on the frozen
2026-09-05 baseline card, 2023-2025, 799 opener-graded games, the leader-median
arm (S1) adds **+3.004 accuracy points over the baseline, week-blocked 95%
[-0.993, +6.953], `probability_positive` 0.930**, against the served
equal-book arm's (S4) +1.752 [-0.870, +4.326], P+ 0.907. Head to head on the
same games and the same blocks, **S1 - S4 = +1.252 accuracy points,
week-blocked 95% [-0.990, +3.522], `probability_positive` 0.864**;
season-blocked [0.000, +2.256], P+ 0.982. Both arms are positive in every
season. On the current model the same head-to-head reads +0.38, P+ 0.63
(reported by the lane that measured it; not re-measured here).

**Mechanism, not a threshold search** (as of the 2026-09-09 promotion; the
gate itself moved to a full point the next day, above). The threshold was
unchanged at 0.5 and no grid was searched. S1's fire set almost exactly
contains S4's (measured on
the 816-game archive frame): 325 games where both fire, agreeing on the side in
all 325, 4 where only the equal-book mean fires, and **198 where the leaders
cleared half a point but the twelve-book mean was diluted below it**. Those 198
(194 of them opener-scored) are where the difference lives -- the Tuesday card
is 47.94% right on them, and following the leaders lifts that to 52.06%. A
loosened equal-book gate matched for fire count (threshold 0.25, 514 fires
against S1's 523) does not recover it: S1 beats that gate-matched equal arm by
+1.377 accuracy points, `probability_positive` 0.961 for S1. So the gain is the
leaders' information, not simply firing more often.

Every cell is recorded `unresolved_below_power` under the
`sharp_weighted_follow` family. An interval containing zero is not a rejection
ground (AGENTS.md), and the pool is forced picks: the arm with the higher
expected accuracy at the deadline is the one served.

**The rule, exactly as served
(`nfl_ats.pick_refresh.plan_refresh`, `nfl_ats.sharp_book_movement_features.late_week_follow_frame`).**
At each refresh pass, for each still-open game: difference each book's line
series over the `[Monday, Sunday)` window and sum only the increments observed
at or after that week's Wednesday, strictly before the refresh instant, with
provider updates later than the observation refused -- read from live intraday
snapshots only. Let `net` be the **median** of those per-book net moves over
the three leading books that contributed one. If
`|net| >= LEADER_FOLLOW_THRESHOLD` (1.0 since 2026-09-10): the served
pick becomes the side the market moved toward (`net > 0` picks HOME, else
AWAY), unless the injury-news veto below discards it. If no leading book
contributed, the arm cannot fire and the Tuesday
pick stands -- it never falls through to the twelve-book mean. Otherwise the
existing logic stands unchanged (the model's own recompute, possibly
1.0-consensus-overridden). Sunday passes consume Saturday evidence; Sunday
moves are outside the rule, preserving the measured construct. Missing archives
or unusable stores are fail-open: the arm reports itself unavailable and the
pass proceeds exactly as before.

**One computation, four records.** `late_week_follow_frame` returns every arm
on every row -- `leader_median_net_move` / `leader_books` for the served arm,
`leader_median_half_would_be_pick_side` for the retired half-point gate, and
`equal_net_move` / `eligible_books` for the equal-book arm -- so the served
pick and the paired ledger can never drift apart. The
`late_week_move_follow_refresh_decisions.parquet` ledger keeps its
`late_week_move_follow_refresh_v1` challenger id, recording the EQUAL-BOOK
arm (`equal_would_be_pick_side`, `equal_movement_flip`) as one paired OFF
challenger, beside the served arm's own side, a
`served_challenger_id` of `late_week_leader_median_follow_v1`, and two more OFF
arms: `late_week_leader_median_follow_0_5_off_incumbent` (the half-point gate)
and `late_week_follow_no_news_veto_off_incumbent`
(`news_veto_would_be_pick_side` is the served side, `movement_would_be_pick_side`
the un-vetoed one). Every
pick-revision row additionally carries both market arms' evidence
(`late_week_*`, `consensus_*`), the governing `movement_policy`, and the
`model_only_pick_side` counterfactual, so a later settlement pass can score
Tuesday vs final, model-only vs played, leader vs equal, and the two market
arms against each other without re-deriving any of them.

**First live fire:** the Thursday 2026-09-10 15:00 ET refresh (the Tuesday
2026-09-08 lock is unchanged: it records the Tuesday card the rule reads). The
equal-book rule it replaces never fired on a played card, so no served pick
changes retroactively and no continuity is broken.

## Served heavy-handle follow (H1, owner order 2026-09-09)

Measured 2026-09-09 (`docs/handle_follow_on_card.md`,
`artifacts/handle_follow_on_card/20260909T232503Z/result.json`): on top of the
served nine-member card, over the 260 opener-graded games the public-betting
backfill reaches, flipping the pick to the side holding at least 70% of the
spread money adds **+0.38 accuracy points, week-blocked 95% [-3.00, +4.28],
`probability_positive` 0.5695** (season-blocked 0.6031), firing on 73 games and
changing 37 of them (18-19 becomes 19-18). Recorded `unresolved_below_power`;
the interval crossing zero is not a rejection ground (AGENTS.md), and on the
forced-pick decision rule a candidate above a coin flip on top of what is
played is played. The two narrowings measured beside it do NOT carry the
effect -- H2 (the same rule restricted to tickets <= 60%, the sharp-money
shape) is an exact dead heat at 7-7, and H3 (restricted to lines within 3
points of pick'em) is -0.77 points -- which is a diagnosis for a session
extending this lane, not a verdict on H1.

**The rule, exactly as served (`nfl_ats.pick_refresh.plan_refresh`,
`nfl_ats.public_betting_live.load_latest_public_handle`).** At each refresh
pass at or after **Saturday 12:00 ET** of that week, for each still-open game
where **neither** market rule above fired: read the latest public-betting
capture strictly at or before the pass instant that carries rows for this
season and week (`data/raw/public_betting_live/`, written by the
`public_betting_sat` and `public_betting_sun` scheduler jobs), take the side
with the larger share of the spread money, and if that share is **>= 70%** and
the pick is on the other side, the served pick becomes the money's side.
Otherwise everything stands exactly as before.

**Precedence is strictly below both market rules, on purpose.** Heavy handle
is largely the cause of the line move `LATE_WEEK_MOVE_FOLLOW_POLICY` and the
1.0-point consensus rule already read, so applying it on top of them would
count the same money twice. It may only apply where neither fired.

**Thursday and Wednesday games never see a reading.** Both captures land after
those kickoffs, and the clock gate refuses a pass before Saturday noon ET
outright rather than letting it reuse the previous week's capture. Everything
about the arm is fail-open: no store, no capture before this pass, no row for
this week, or an unreadable store all keep the pick and report a named reason
under `movement_policy.handle_follow` in the pass summary.

**One computation, two records.** Every pick-revision row keeps the pre-rule
pick and the numbers the decision was made on (`handle_pre_rule_pick_side`,
`handle_pick_side`, `handle_money_pct`, `handle_ticket_pct`), and the flipped
rows carry the reason `Followed the heavy-money side`. The paired
`handle_follow_refresh_off_incumbent` challenger
(`prospective/handle_follow_refresh_decisions.parquet`) records both arms --
served and off -- for every eligible game that carried a reading on the pass,
so the OFF arm accrues game for game instead of being reconstructed later.

## Served rookie-crew step (2026-09-09, below both market arms)

Closing-grounds taxonomy, verbatim, because this section reports intervals: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. Only a RESOLVED wrong sign (whole interval on the wrong side of
zero), zero split-half reliability, or a positive control proven able to detect
an effect that size ever closes a line of work. Everything else is
`unresolved_below_power`; report `probability_positive`, never "contains zero".

Officiating-crew assignments publish Wednesday-Thursday, after the Tuesday
lock (`docs/referee_assignments_capture.md`), so the reconciled rookie-crew
rule of `docs/rookie_crew_reconciliation.md` can only ever reach the card
through this path. It is now the third step in the chain, and it is the LAST
one consulted before the model's own side:

1. `late_week_move_follow_0_5` -- the promoted follow rule above;
2. `movement_ge_1.0` -- the 1.0-point consensus rule;
3. `rookie_crew_underdog_v1` -- this step;
4. `model_only`.

**The rule, exactly as served** (`nfl_ats.pick_refresh._rookie_crew_lookup`,
one function, one call site). Take the newest
`data/players/referee_assignments` snapshot covering the pass's (season, week)
and consider only games whose own `pick_deadline` -- `min(kickoff, Sunday 16:00
ET)` -- is strictly after that snapshot's `captured_at_utc`. A game is FLAGGED
when its head referee has at most one prior season in the archive-extended
officials table (`load_officials(include_archive=True)`, 2009-2025, so the
season floor is that population's own first season plus one, 2010). The flag is
signed by the UNDERDOG side of the frozen Tuesday line: `+1` when the home team
is the underdog, `-1` when the away team is. That signed column is attached to
the feature table -- Tuesday-opener consensus spread where the opener store
reaches, archived nflverse spread as a close proxy before 2020 -- and the
week's model is refit on profile `weak_stack_rookie_crew_underdog`. On a
FLAGGED game whose refit side differs from the model-only side, and only when
neither market arm fired, the served pick becomes the refit side. Everything
else is untouched.

**It is a model feature, not a hand-set flip.** The ridge learns the
coefficient, so the served direction is whatever the mechanism says on that
game rather than a fixed "back the dog" instruction. That is what keeps the
serve equal to the arm that was measured.

**Narrowed to flagged games, deliberately.** The predeclared arm in
`docs/rookie_crew_reconciliation.md` lets the refit move UNFLAGGED games too
(coefficient drift), and on the played nine-member card that arm is **+0.133
accuracy points, week 95% [-0.197, +0.466], `probability_positive` 0.790**, six
changed picks. The served step is the flagged-games subset, because AGENTS.md
requires a pick flip to name a mechanism and a drift-only change on an
unflagged game names none. Measured 2026-09-09 on the same 1,503 non-push
games, that subset is **+0.067 accuracy points, week 95% [-0.134, +0.328],
`probability_positive` 0.708** (season-blocked 0.718), three changed picks, on
a card baseline of 56.886%. Per era: 2020-2021 +0.219
`probability_positive` 0.820 on one changed pick, 2022-2023 0.000 on two,
2024-2025 0.000 on none -- magnitude statements per era, never absence.
Registry cell `rookie_crew_served_step_flagged_games_card_2020_2025`,
`unresolved_below_power`; nothing here is closed, and the pool is forced picks,
so a step 71% likely to be better is a step to play.

**Fails open, always.** No published assignment, a snapshot at or after every
eligible game's deadline, a week with no rookie crew, an unreadable officials
history, or a refit that will not fit each return the model-only side with the
reason named in the pass's `rookie_crew.reason`. The refit itself is skipped
entirely unless a flagged game exists, so an ordinary week pays nothing for it.

**Ledger and paired arm.** Every pick-revision row carries `rookie_crew_flag`,
`rookie_crew_referee` and `rookie_crew_pick_side` beside the governing
`movement_policy`; the OFF arm is the `model_only_pick_side` column already on
every row, registered as `rookie_crew_underdog_off_incumbent` in
`artifacts/prospective/challengers.json`. Because the ledger records only games
whose served pick CHANGED, a game where this step held the Tuesday pick against
a model-only flip writes no row -- every pass's JSON lists those game ids under
`rookie_crew.rookie_crew_reads`, and that is the known gap the registration
names.

**Week 1, 2026 (measured, `nfl-ats refresh-picks --note crew_probe`,
2026-09-10T00:08Z).** One of sixteen crews is new this season: Alex Moore, one
prior season, working BUF at HOU, where the home team is the 1.5-point
underdog, so the flag is `+1`. The refit's side is HOME, the model-only side is
HOME, so the step changes nothing and the row is recorded `model_only`. The
other fifteen crews carry three or more prior seasons.

## Exact commands

```powershell
# Tuesday noon ET -- unchanged, exactly as documented before this feature.
.\.tools\uv.exe run nfl-ats weekly-run --season 2026 --week 1 --record-decisions

# Thursday afternoon (pre-TNF): recompute, record what changed, no card edit yet.
.\.tools\uv.exe run nfl-ats refresh-picks `
    --season 2026 --week 1 --record-decisions --note thursday_afternoon

# Saturday: another look, same shape.
.\.tools\uv.exe run nfl-ats refresh-picks `
    --season 2026 --week 1 --record-decisions --note saturday_pass

# Sunday morning, BEFORE 4:00 PM ET -- the final pass; locks SNF/MNF too.
.\.tools\uv.exe run nfl-ats refresh-picks `
    --season 2026 --week 1 --record-decisions --publish-card `
    --note sunday_morning_final

# A dry look at any time, with nothing written (default -- --record-decisions
# is what makes a call "real"; omit it for a rehearsal or a status check):
.\.tools\uv.exe run nfl-ats refresh-picks --season 2026 --week 1
```

`--features` defaults to the active model's own card-path feature table
(`data/processed/game_features_player.parquet` or
`game_features_weak_stack.parquet`, matching `nfl_ats.weekly.CARD_PATH_TABLES`)
-- rebuild it first (`build-player-features` / the weak-stack build step)
if a pass is meant to see fresher injury/weather-adjacent data than Tuesday's
build; `refresh-picks` only reads an existing table, it never rebuilds one.
`--min-train-games` defaults to `nfl_ats.constants.DEFAULT_MIN_TRAIN_GAMES`
(500), matching every other production command.

**Season note:** Week 1 2026 locks Tuesday 2026-09-08. That Tuesday
`weekly-run` (or `publish-predictions`) run still needs `--record-decisions`
-- unchanged by anything in this document -- and it is the prerequisite the
week's `refresh-picks` passes above depend on: no recorded Tuesday card, no
frozen line to refresh against, no refresh.

## Ops checklist for a live week

1. Tuesday: `weekly-run --record-decisions` (or `publish-predictions
   --record-decisions` if running the steps by hand). Confirms the card is
   SYNCHRONIZED and locks the grading lines.
2. Before Thursday kickoff: rebuild the feature table if fresher data is
   worth having, then `refresh-picks --record-decisions --note
   thursday_afternoon`.
3. Saturday: same, `--note saturday_pass`.
4. Sunday morning, before 4:00 PM ET: same, plus `--publish-card`, `--note
   sunday_morning_final`. This is the last pass that can touch anything.
5. After the pool locks -- verify: `nfl-ats refresh-picks --season <S> --week
   <W>` (no `--record-decisions`) should report an empty `changed_game_ids`
   for every game, and every game should show `"kickoff_passed"` or
   `"sunday_pick_lock_passed"` once its deadline has passed.

## Deliberately deferred

- **Wiring another research overlay into the refresh's actual decision path.**
  Any future production change must update the frozen Tuesday policy identity
  and flags; refresh must never infer it from live sources mid-week.
- ~~**Settling `final_pick_per_game` against outcomes.**~~ Shipped
  2026-09-09 as `nfl-ats settle` (`nfl_ats.settlement`), which reports the
  Tuesday arm and the refreshed arm side by side for every refresh ledger,
  not only this one. It grades each arm at that row's own frozen
  `decision_home_spread` and takes the latest pass recorded before the
  game's deadline, so the week's three or four passes count once.
- **Re-running the per-overlay challenger recorders from `refresh-picks`
  itself.** Those recorders are first-write-wins per game
  (`nfl_ats.prospective_scoring`), the same anti-backdating discipline this
  ledger uses, so calling them again mid-week from `refresh-picks` would be
  a safe no-op for any game already recorded on Tuesday -- not useless, but
  not load-bearing either, and adding the wiring without a clear use for it
  is unjustified scope for this build.
