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
else. Each pass is the *same* command, run again; nothing about
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
refuses to run if the recorded original card's `model_id` no longer matches
the currently active one (`plan_refresh` raises
`"...refuses to recompute picks under a different model identity..."`).
This mirrors `nfl_ats.weekly.assert_synchronized`'s spirit for the Tuesday
card: a refresh must never silently recompute this week's picks under a
model the pool's frozen line was never actually locked against.

## Overlays

Two pick-level policies are applied to the real card today. `refresh-picks`
re-applies the year-1-coach fade first, then applies the player-arrest
back-side policy from the **frozen Tuesday flags stored in the paper ledger**.
It never queries a newer arrest snapshot: a later source refresh could add or
revise an incident dated before Tuesday, and using that newer view would
retroactively alter the decision-time information set. The paper ledger's
`pick_side` is already the final Tuesday played side; `model_pick_side` and
`pre_arrest_pick_side` preserve the raw and coach-only counterfactual arms.

**No overlay logic is touched by this feature.** Every other pick-level
overlay this project has built -- the injury value-lost tilt (the specific
channel that motivated this whole feature, `docs/injury_value_lost.md`,
+1.316 accuracy points, `probability_positive` 0.8875,
`unresolved_below_power`), division revenge, backup-QB fade, surface-switch
tilt, spread-gap-zone fade, the ECDF-mapping-incumbent and era-weighted
challengers, forecast cold-visitor tilt -- stays exactly what it already is:
challenger-tracked evidence collected by `publish-predictions
--record-decisions`, **never applied to the played pick**. That is a
deliberate, labeled scope decision for this build, not an oversight or a
missing wire-up:

- Promoting a research overlay onto the real forced pick is the same kind of
  call that promoted coach-fade (`docs/coach_fade_overlay.md`, an explicit
  owner decision, 2026-08-18) -- a one-way door for real, submitted picks,
  and not something a feature-flow implementation should do unilaterally as
  a side effect of building the flow.
- `injury_value_tilt_overlay.py`'s own module docstring is explicit that "no
  such decision has been made for this candidate" and "the task that built
  this module was explicit: dual-track it, do not touch the production
  pick." Wiring it into `refresh-picks`'s decision path would silently
  reverse that standing instruction.

What this feature *does* change, structurally: the mechanism now exists
(`resolve_overlay`'s pattern -- score at current data, apply an overlay,
compare against a baseline) to add the injury tilt (or any other pick-level
overlay) into the refresh's decision chain the same way coach-fade is
wired in today, **once that promotion is a deliberate owner decision**. That
is future work, not part of this build; see "Deliberately deferred" below.

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
evidence block, honest caveats, and the (currently absent) settlement
path.

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
| `coach_fade_flip` | Whether the coach-fade overlay flipped this specific recompute |
| `player_arrests_flip` | Whether the frozen Tuesday arrest flags flipped this recompute after coach fade |
| `player_arrests_snapshot_id`, `player_arrests_safe_index_sha256` | Provenance copied from Tuesday's paper row; refresh never opens that snapshot or a newer one |
| `movement_policy` | `movement_ge_1.0` (the observed-movement policy governed this pick) or `model_only` (below threshold, or no fresh captured line -- see "Observed-movement pick policy" above) |
| `movement_delta` | Signed points the locally-captured line moved from `decision_home_spread`, home-oriented; blank/null when no fresh line was available this pass |
| `movement_pick_side` | The side the market moved toward, whenever `movement_delta` is not null -- the candidate side even on rows where `movement_policy` did not select it |
| `model_only_pick_side` | The recomputed production-policy pick (post coach-fade and frozen arrest policy, pre movement-policy override) -- always present; the counterfactual the `model_only_refresh_incumbent` challenger tracks |
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
had not yet passed were eligible. "Policy" is `movement_ge_1.0` when the
captured market line moved >=1.0 point from the frozen Tuesday line and the
pick followed it, or `model_only` otherwise -- see "Observed-movement pick
policy" above. This is research output, not a wagering recommendation.

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

- **Wiring a research overlay (injury value-lost, or any other) into the
  refresh's actual decision path.** The mechanism exists; the promotion
  decision does not, and this build does not make it unilaterally (see
  "Overlays" above).
- **Settling `final_pick_per_game` against outcomes.** The function exists
  and both the Tuesday and final pick are recoverable per game, but wiring
  this into `prospective-score`'s settlement pass (so a season can report
  "Tuesday-graded accuracy" vs. "refresh-graded accuracy" side by side) is a
  natural next step, left for when there is a real week of refresh data to
  settle.
- **Re-running the per-overlay challenger recorders from `refresh-picks`
  itself.** Those recorders are first-write-wins per game
  (`nfl_ats.prospective_scoring`), the same anti-backdating discipline this
  ledger uses, so calling them again mid-week from `refresh-picks` would be
  a safe no-op for any game already recorded on Tuesday -- not useless, but
  not load-bearing either, and adding the wiring without a clear use for it
  is unjustified scope for this build.
