# Prospective 2026 evidence (POL-10)

Written 2026-08-17, three weeks before Week 1 locks. Everything else in this
repository grades models on history. This is the machinery that grades them on
the future.

## Why it had to exist before Week 1

Two frozen results are unresolved, and prospective scoring is the only way to
settle either without spending one of the **two opener confirmation windows
left in the entire project**:

| result | where it stands | window |
|---|---|---|
| `mod07_weak_signal_stack` | +1.97 points at the opener (53.29% vs 51.32%, 456 games), `probability_positive` **0.8745** against a pre-fixed 0.90 bar → `unresolved`, not promoted | [2020, 2021] spent |
| `best_pick_ranker` | `confirmed` at 60.0% top-1 — on **35 top-1 picks**, interval [−7.00, +22.88] | [2013, 2015] and [2020, 2021] spent |

Prospective scoring needs no registry window at all
(`docs/rotation_registry.md`, "what this deliberately does not do"). It costs
nothing but patience. But it only produces evidence if the picks are *recorded
before kickoff*, every week, starting with Week 1 — a season that goes
unrecorded is simply gone.

Before this work, three gaps meant that would have happened:

1. **Nothing computed whether a 2026 pick won.** The MKT-04 ledger recorded
   picks and scored `clv_points` (how far the line moved), but never joined
   against `result`, so it could not say win or loss. `opener_pick_evaluation`
   could, but only as a historical backtest over archived snapshot pairs.
2. **The weekly Best Pick was never persisted.** It was recomputed on each page
   render from the current week's parquet. `docs/index.html` is one
   current-week page that every publish overwrites, so Week 1's nomination
   would have stopped existing the moment Week 2 published.
3. **The MOD-07 stack was not registered**, so no challenger card was being
   produced or preserved at all.

## What runs now

`weekly-run` gained four **optional** trailing steps (8-11). They run *after*
the publish and their failure is reported but never aborts the run — the pool
card is the deliverable, and a missed week of research evidence must not take
it down. Everything on the path to a published card stays fail-closed.

```
8  build-learned-availability-features   rebuild game_features_weak_stack.parquet
9  margin-predict --feature-profile weak_stack     the challenger's own card
10 prospective-record --challenger mod07_weak_signal_stack --season S --week W
11 prospective-score                     settle everything recorded so far
```

`--skip-prospective` opts out. Step 9 cannot disturb the published card: no
evaluation matches the `weak_stack` configuration, so `activate_matching_ats_model`
returns `None` and the active manifest is left exactly where the publish left
it (verified on the real tree: the manifest digest was byte-identical after a
challenger run).

**Since 2026-08-18, step 10 additionally requires `weekly-run --record-decisions`**
(and step 7's own ledger write requires the same flag) — recording is opt-in,
not opt-out, so an ordinary or rehearsal `weekly-run` builds and scores the
challenger's card (steps 8, 9, 11 still run) but does not append anything to
either ledger. See "Known divergence" below for the incident that made this
necessary and `nfl_ats.clv.RECORDING_LOCK_WINDOW` for the second, function-level
guard that backs it up.

Step 10 finds its card by **configuration fingerprint**, not by directory name
or recency. The active model's card and the challenger's card land in the same
`artifacts/margin_predictions/` namespace, and picking the newest directory
would silently record the baseline's picks as the challenger's.

### Two grades, and which one is primary

`prospective-score` reports forced-pick ATS accuracy twice:

- **`decision_line` — PRIMARY.** The spread the pick was actually made at. The
  pool freezes its number on Tuesday and grades entries against it, so this is
  the number that matters (`docs/pool_edge_plan.md`).
- **`close_line` — secondary.** The *same* pick settled against the close.

Note the difference from `clv.opener_pick_evaluation`, which re-forms a pick at
each line. Here the pick is a fact on the ledger and only the settlement line
changes: "would this recorded pick have won at the close?" is the question the
entry price answers, and re-picking at the close would score a model we never
published.

Push handling is not reimplemented. Settlement calls `clv.pick_correct`, the
same function the opener evaluation uses: `result - line`, strictly positive is
a home cover, a zero margin is a push excluded from accuracy (FND-04,
`docs/modeling.md`). A game can be a push at one grade and settled at the
other; both are tracked separately.

## The anti-backdating guarantee

Prospective evidence is worth exactly as much as the proof that the picks
existed before the games did. That proof is enforced twice.

**On write.** Both recorders refuse any game at or past kickoff and never
rewrite an existing row. A republished card with a moved line does not relocate
the anchor; a challenger card with flipped picks cannot overwrite what is
already recorded.

**On read.** `settle_prospective_picks` re-checks every row's
`recorded_at_utc` against its own `kickoff` and raises rather than score a
ledger that was edited afterwards. A hand-written row cannot be laundered into
evidence just by running the scorer.

The test that pins it is
`tests/test_prospective_scoring.py::test_scoring_refuses_a_pick_recorded_at_or_after_its_own_kickoff`
(covering after-kickoff, exactly-at-kickoff, and unprovable-timing rows), with
`test_record_challenger_records_dedupes_and_refuses_started_games` covering the
write side.

### The Best Pick's own rule

The weekly Best Pick is now a durable `is_best_pick` column on
`artifacts/clv_ledger/decisions.parquet`, written by `publish-predictions`
under three rules:

1. **Whole-week pre-kickoff.** The flag is written only while *every* game on
   the card is still in the future. The pool locks all picks before any game;
   nominating one after Thursday night has been played would be choosing with a
   result in hand, so once any game has started the week gets no Best Pick at
   all. Fail-closed: no evidence beats corrupt evidence.
2. **First write wins.** A week that already carries a flagged row is never
   re-flagged.
3. **Exactly one per week**, enforced on read too, so a hand-edited ledger
   fails loudly.

Rule 1 is why the flag may land on a row an *earlier* run appended: the
decision rows are append-only, but the nomination is a separate, one-time,
still-pre-kickoff write about that week.

The nomination also now appears on the published card itself, marked `★` with a
one-line note. The ledger answers "what did we choose?" months later; only the
card answers "which one do I enter today?".

## The `--freeze` gap: closed, but not the way the note assumed

`artifacts/prospective/challengers.json` carried a standing `known_gap`:
`margin-predict` has no `--freeze` flag, so challenger cards do not go through
`prospective.freeze_forecast`. Assessment, and the decision:

**Adding `--freeze` to `margin-predict` would not have bought what it looks
like it buys.**

- `freeze_forecast` enforces the **classification** prediction-safety contract.
  It requires `pick`, `market_home_no_vig_probability` and `market_hold`, none
  of which a margin card has. They would have to be manufactured onto the card
  purely to satisfy a validator — and a release-blocking safety contract
  running over synthesized inputs is worse than no contract. The alternative,
  generalizing `validate_prediction_card` to a second card shape, is a
  substantial refactor of a release-blocking contract, undertaken for a
  guarantee we can get more cheaply.
- The margin card is **already safety-audited**: `margin-predict` runs
  `validate_outcome_prediction_card` and writes the result to
  `prediction_safety.json` beside the card.
- Margin artifacts are **already write-once** in practice: every run writes a
  fresh `{season}-week-{NN}-{run_id}` directory and never overwrites a previous
  one.
- The digest argument does not survive scrutiny either. `freeze_forecast`'s
  sha256 lives in `manifest.json` **in the same directory as the file it
  hashes**. That is evidence against corruption, not against a motivated
  rewrite — edit both and it verifies.

So the substantive gap is not "no frozen directory". It is three specific
claims a challenger's evidence needs, and the ledger makes all three, more
strongly than a frozen forecast would:

| claim | how the ledger proves it |
|---|---|
| the pick existed before kickoff | asserted at write, re-asserted at scoring |
| the pick was never changed | append-only, first-write-wins per (challenger, game) |
| the pick came from *this* configuration | `config_fingerprint` must equal the registration's, or the recorder refuses; `source_sha256` pins the exact card file |

The last row is the one `--freeze` would **not** have given at all, and it is
the realistic failure mode: a challenger quietly re-tuned in week 6 and
accumulating picks under the same id would convert a re-tune into "prospective
evidence". `test_record_challenger_refuses_a_retuned_configuration` pins it.

The fingerprint deliberately compares the feature table by **file name**, not
content digest: the tables are rebuilt every Tuesday to cover the new week, so
the digest changes weekly by design. The observed digest is recorded on every
ledger row instead, so the actual input to each week's picks stays auditable
without being a false alarm.

## Known divergence — RESOLVED 2026-08-17: Week 1 was reset

> **This resolution did not hold. Kept verbatim below, not deleted or
> reworded, because the repo's rule for negative results applies here too:
> record what was claimed and let the correction sit next to it, in the
> open.** The claim "both ledgers... empty... matter... closed" was false
> again within hours of being written. See "What actually happened
> (2026-08-18)" immediately below for the full correction, the timeline, and
> what now actually prevents this from recurring — deleting the rows never
> did.

> **Decision taken (option 1 below): the 2026 week-1 rehearsal rows were
> deleted from both ledgers on 2026-08-17**, so the first write for Week 1
> will be the real Tuesday-lock card on 2026-09-08. Both files removed rather
> than truncated, because every row they held was a week-1 rehearsal row and
> both loaders already return an empty frame when the file is absent —
> `load_paper_decisions` and `load_challenger_decisions` treat "no ledger yet"
> as the normal first-run state. Verified after: both ledgers load as 0 rows
> with their column contracts intact (16 and 18 columns), the challenger
> registration for `mod07_weak_signal_stack` is untouched, and
> `prospective-score` runs clean against empty ledgers. A copy of the deleted
> rows was kept outside the repo before the reset. Nothing else in Week 1 was
> hand-recorded, so the Tuesday run writes the pick sides, the lines, and the
> Best Pick nomination together, from one card.

### What actually happened (2026-08-18)

The 2026-08-17 reset above deleted the rehearsal rows, but deletion was never
a fix for the actual defect: nothing stopped the same *ordinary, documented*
command from repopulating the ledger the same way. It did, within hours.

A live readiness check on 2026-08-18 found `artifacts/clv_ledger/decisions.parquet`
holding 16 real 2026-Week-1 rows again — `recorded_at_utc`
**2026-08-18T01:24:56Z**, `model_id` `4b01f055b684e27e` (the pre-promotion
`player`-profile id, activated roughly seven minutes before the `weak_stack`
promotion at `01:31:39Z`), `is_best_pick=True` already locked onto
`2026_01_ARI_LAC`. This was not a hand-edited row and not a bypass of any
guard that existed at the time: it was `nfl-ats publish-predictions` — the
ordinary Tuesday command — run during that session's own live testing, with
its ledger-recording flag not passed. The flag existed
(`--skip-clv-ledger`), but it was **opt-out**, so recording was the default
and skipping it required remembering to ask. Nobody did, because the person
running it did not think of that moment as "the real lock" — and the system
had no way to know it wasn't, either.

That is the actual defect the 2026-08-17 reset never touched: **the real
ledger's integrity depended entirely on every future session remembering to
pass a flag during ordinary testing.** A safeguard with that shape fails
exactly once and then fails silently forever after, because the ledger
doesn't reject a legitimate-looking write — it just accepts it.

**What now prevents recurrence, fixed and verified 2026-08-18:**

1. **Recording is opt-in, not opt-out.** `--skip-clv-ledger` is gone.
   `publish-predictions` and `weekly-run` both take `--record-decisions`
   (default `False`); without it, no paper-decision or challenger row is
   ever attempted. The real Tuesday lock is the one time this flag is
   passed on purpose.
2. **A second guard lives inside the recording functions themselves, not
   only at the CLI layer** — `nfl_ats.clv.refuse_if_outside_recording_lock_window`,
   used by both `record_paper_decisions` and (via the shared import)
   `record_challenger_decisions`. It refuses to write whenever a week's
   earliest kickoff is more than `RECORDING_LOCK_WINDOW` (7 days) away
   from the recording instant. This is the guard that would have caught
   the actual incident regardless of which flag was or wasn't passed:
   replayed against the real 2026-08-18T01:24:56Z instant and the real
   Week 1 kickoffs, it refuses, naming the 22-day gap explicitly. Because
   the check lives in the functions, not the CLI, it also protects
   `clv-ledger` (whose own `--skip-record` is still opt-out) and
   `prospective-record` (which has always recorded unconditionally when
   invoked) — every path into either ledger is covered, not just the one
   that caused this incident.
3. Regression tests pin both: `tests/test_clv.py::test_record_paper_decisions_refuses_a_recording_weeks_before_kickoff`,
   `tests/test_prospective_scoring.py::test_record_challenger_refuses_a_recording_weeks_before_kickoff`,
   and `tests/test_cli.py::test_publish_predictions_does_not_record_by_default` /
   `test_publish_predictions_records_with_the_explicit_flag`.

**The 16 contaminating rows themselves were left untouched.** They were not
created, deleted, or modified by the session that found them; a backup was
taken outside the repository before any further work. Their disposition —
reset again so the real Tuesday publish is genuinely the first write, or
accept them as a second rehearsal artifact — is the owner's decision, tracked
in `docs/week1_readiness.md`. Either choice now holds: the guards above stop
the same command from silently repopulating the ledger a third time.

The reasoning that led to the original (non-holding) reset is kept below,
because the same trap recurs every time a week is published early — the
guards above are the actual fix for it, not the deletion.

## The divergence itself

The paper-decision ledger anchors each game at the line and pick of the
**first** publication of that week, by design (MKT-04 wants the entry price).
The pool, though, grades what the user enters at the **Tuesday lock**. Those
are the same thing only if the week's first publish *is* the Tuesday one.

As of the 2026-08-20 arrest-policy promotion, `pick_side` is the final played
side after coach fade then the player-arrest policy. The same append-only row
also stores `model_pick_side`, `pre_arrest_pick_side`, frozen home/away arrest
flags, flip markers, and the arrest snapshot ID/hash. The active paired
`player_arrests_recent_14d_no_overlay_incumbent` challenger records the
coach-only arm. This identity replaced the former candidate registration,
which remains in the registry as `SUPERSEDED_BY_PROMOTION`.

For the existing head-coach overlay evaluation, `prospective-score` derives
`base_model_no_pick_overlays` from that same row's frozen `model_pick_side`.
The registered `hc_year_one_fade_overlay` rows remain the coach-only arm, and
`active_model` now correctly denotes the final policy that was actually
played. No later prediction or mutable source is used to reconstruct any arm.

The current Week 1 ledger was written during the 2026-08-17 rehearsal, three
weeks early. Its 16 pick sides and lines are August's. If the model's picks
move by 8 September, the ledger will score picks the user did not play.

Two clean options, both the owner's call:

1. **Reset Week 1** — delete `artifacts/clv_ledger/decisions.parquet`'s 2026
   week 1 rows (and the challenger ledger's) before the Tuesday run, so the
   first write is the real locked card. This also frees the Best Pick
   nomination to be made from the Tuesday card.
2. **Accept it** and note that Week 1's ledger is a rehearsal artifact.

**Option 1 was taken on 2026-08-17** (see the resolution note at the top of
this section). The recording rules were written so either choice worked, and
the Best Pick for Week 1 was deliberately never hand-recorded, which is what
kept option 1 available. `publish-predictions` writes it automatically on the
Tuesday run.

## Commands

```powershell
# Settle every recorded prospective pick at both grades (safe to run any time;
# unplayed games simply stay "pending").
.\.tools\uv.exe run python -m nfl_ats prospective-score

# Record a registered challenger's weekly card (weekly-run step 10 does this).
.\.tools\uv.exe run python -m nfl_ats prospective-record `
    --challenger mod07_weak_signal_stack --season 2026 --week 1
```

`prospective-score` flags: `--features` (default the canonical table),
`--start-season` (default 2026 — earlier seasons are backtests, not pre-kickoff
decisions), `--skip-challengers`, `--bootstrap-samples`, `--bootstrap-seed`.
Output lands in `artifacts/prospective_scoring/<run_id>/` as
`settled_decisions.parquet`, `week_summary.csv` and `metadata.json`, the last
carrying per-entrant accuracy at both grades plus week-blocked intervals once
anything has settled.

## State as of 2026-08-17

- `mod07_weak_signal_stack` registered `ACTIVE_PROSPECTIVE`, fingerprint
  `bc77638d47e2748c`.
- Its Week 1 card is generated and its 16 picks are recorded pre-kickoff. The
  two arms **disagree on 3 of 16 games** (`ATL/PIT`, `GB/MIN`, `NE/SEA`), which
  is where all of the paired evidence will come from.
- The active model's 16 Week 1 picks were already on the ledger; `is_best_pick`
  is False on all of them and will be written at the Tuesday publish.
- `prospective-score` runs clean and reports 16 pending decisions per entrant,
  which is the correct answer three weeks before kickoff.

## Tuesday-visibility audit (2026-08-19)

> **Owner-corrected 2026-08-20:** this audit's data-availability facts stand
> exactly as measured -- the injury report files Wednesday, official rows
> are 0.43% Tuesday-visible, `home_qb_name`/`away_qb_name` are 100% null
> pregame, etc. What is corrected is the CLASSIFICATION those facts feed.
> The premise that our picks are forced at the Tuesday build was wrong: only
> the pool's LINE locks Tuesday, and picks are editable up to each game's
> real deadline (**refined 2026-08-20: min(kickoff, Sunday 16:00 ET) --
> SNF/MNF lock early at Sunday 4pm, not at their own kickoff**)
> (`docs/pool_edge_plan.md`). So class **(c) "Tuesday-impossible"**
> below means "needs a late-week refresh pass to fire correctly," not
> "unplayable" or "permanent no-op" -- the Tuesday-built card is the opening
> statement, and a pre-kickoff refresh (rebuilding `game_features_player` at
> a Saturday-ish cutoff and re-applying the overlay) is the answer this pool
> actually plays. Two exceptions this correction does NOT touch:
> `backup_qb_fade_overlay`'s deactivation stands unchanged -- its data
> source (`home_qb_name`/`away_qb_name`) is populated post-game regardless
> of when in the week the card is built, so no refresh, however late,
> fixes it; that is a genuine "impossible at any pregame timestamp" finding,
> not a timing artifact. And `injury_value_lost_tilt_overlay`'s "near-total
> no-op" finding below is correct and stays correct for the TUESDAY PUBLISH
> specifically -- what changes is that the Tuesday publish is no longer the
> pool's final word: on a late-week refresh, with the current week's
> official report filed (features rebuilt at production's own ~Saturday
> `decision_hours_before_kickoff=24` cutoff), the overlay reads real
> `value_lost_diff` and fires exactly as `docs/injury_value_lost.md`'s
> historical evidence describes (+1.316 pts, P+ 0.8875). See
> `docs/injury_news_sourcing.md` §5.1 and `docs/pool_edge_plan.md` for the
> parallel correction, and `artifacts/prospective/challengers.json`'s
> `injury_value_lost_tilt_overlay` entry for the corrected
> `tuesday_visibility_caveat`.

**Trigger.** `docs/injury_news_sourcing.md` section 5.1 (read) measured that
`injury_value_lost_narrowed`'s registered +1.316 accuracy-point edge
(`probability_positive` 0.8875, `registry/weak_signals.json`) is built on a
Saturday decision cutoff (`decision_hours_before_kickoff=24`, the default
everywhere in `players.py`), and that under a true Tuesday-noon cutoff — the
pool's actual lock (`docs/pool_edge_plan.md` line 80) — the same 456 games
read **+0.000 accuracy points**, `probability_positive` **0.3965**. The
Saturday-minus-Tuesday channel delta (+1.3158 pts, `probability_positive`
0.9003) accounts for essentially the whole originally measured effect. This
is an information-**timing** finding, not an effect-existence one: per
AGENTS.md, an interval or point estimate crossing/reaching zero is never
grounds to reject or close a signal, and nothing below closes anything —
`injury_value_lost_narrowed` stays `unresolved_below_power` in the registry,
exactly as `docs/injury_news_sourcing.md` §5.1 already recorded it. What this
audit adds is: does the same problem apply to the other overlay
challengers, and specifically, what does the live Tuesday `weekly-run` build
actually feed each one?

**Method.** For every `ACTIVE_PROSPECTIVE` entry in
`artifacts/prospective/challengers.json` (read in full), traced the overlay
module (`src/nfl_ats/*_overlay.py`, `best_pick_nomination.py`) back to the
concrete column(s) and table it reads, then checked whether that data exists,
in full, by the Tuesday-noon lock of the week the live `weekly-run` sequence
actually runs (`src/nfl_ats/weekly.py:1-14`, read: ingest is step 1,
`build-player-features` is step 3, `publish-predictions` is step 7, all
"every regular-season Tuesday ... before the pool locks"). Two claims below
are **measured**, not inferred: run directly against the newest local
schedule snapshot, `data/raw/20260817T235649Z/schedules.parquet`.
`player_qb_continuity` is `CLOSED_BEFORE_ACTIVATION`, not
`ACTIVE_PROSPECTIVE`, so it is out of scope and excluded below.

### Classification table

| Challenger | Classification | Live-time behavior at the Tuesday build | Evidence-cutoff match? | Action needed |
|---|---|---|---|---|
| `injury_value_lost_tilt_overlay` | **(c) Tuesday-impossible** for the current week's rows | Near-total no-op: `value_lost_diff` reads 0.0 for essentially every current-week game | **NO** | See below — fix options only, not applied |
| `hc_year_one_fade_overlay` | (a) Tuesday-safe | Fires normally | Yes | None |
| `division_revenge_tilt_overlay` | (a) Tuesday-safe | Fires normally | Yes | None |
| `backup_qb_fade_overlay` | **(c) Tuesday-impossible** — in fact pregame-impossible at any hour, not just before Tuesday noon | **Permanent no-op**: 0 live flips, ever, under the current data source | **NO** | Flag as dead on arrival; no fix attempted (see below) |
| `surface_switch_tilt_overlay` | (a) Tuesday-safe | Fires normally | Yes | None |
| `spread_gap_zone_fade_overlay` | (a) Tuesday-safe | Fires normally | Yes | None |
| `smooth_cdf_mapping` | (a) Tuesday-safe | Fires normally | Yes | None |
| `best_pick_nomination_v2` | (a) Tuesday-safe | Fires normally | Yes | None |
| `best_pick_nomination_v3` | (a) Tuesday-safe | Fires normally | Yes | None |
| `mod07_weak_signal_stack` (base model, not a pick-level overlay) | (a) on a **read**, not independently measured this session | Not directly evaluated | Not evaluated | Recommended as the next audit target (see below) |
| `specialist_absence_fade_refresh_v1` | read (2026-09-05, `specialist_absence_fade_refresh_overlay.py`): refresh path, ACTIVE_PROSPECTIVE | Injury report consumed by `refresh-picks`; absent source produces a skip placeholder | Weekly LS/P Out component only; historical IR-wire component is not reproduced | Track both played and would-be sides in `specialist_absence_fade_refresh_decisions.parquet`; no rotation window spent |
| `low_total_div_home_dog_challenger` | read (2026-09-05, `low_total_div_home_dog_challenger.py`): publish path, ACTIVE_PROSPECTIVE | Divisional home dog with decision total <=42; absent source skips | Card decision lines replace the historical opener archive; frozen pick rule differs from the fitted feature screen | Track challenger and baseline together in `low_total_div_home_dog_challenger_paired_decisions.parquet`, plus standard challenger ledger; no rotation window spent |
| `rain_on_grass_dog_challenger` | read (2026-09-05, `rain_on_grass_dog_challenger.py`): publish path, ACTIVE_PROSPECTIVE | Grass plus forecast precipitation probability >=60%; shares existing weather fetch; absent forecasts skip | Live forecast replaces frozen archive proxy; frozen pick rule differs from the fitted feature screen | Track challenger and baseline together in `rain_on_grass_dog_challenger_paired_decisions.parquet`, plus standard challenger ledger; no rotation window spent |

### `injury_value_lost_tilt_overlay` — traced in full, first as instructed

**What it reads at publish time.** `apply_injury_value_tilt_overlay` computes
`value_lost_diff` from `data/processed/game_features_player.parquet`'s two
columns, `diff_injury_skill_epa_value_lost` +
`diff_injury_defense_disruption_value_lost`
(`src/nfl_ats/injury_value_tilt_overlay.py:26-36`, `:100-119`, read). That
table is rebuilt every Tuesday by `weekly-run` step 3
(`build-player-features`,
`injury_value_tilt_overlay.py:91-96`, read), which calls
`enrich_with_player_features` → `_injury_rows_asof`
(`src/nfl_ats/players.py:793-809`, read): it looks up
`injuries_by_game[(season, week, team)]` — **only that game's own week's
report rows** — and filters to `date_modified <= decision_at`, where
`decision_at = kickoff - decision_hours_before_kickoff` (default 24 hours,
i.e. Saturday; `players.py:1046`, `:1210`, read).

**Why that's Tuesday-impossible, not just Tuesday-degraded.** The
`decision_hours_before_kickoff` filter is irrelevant to what the LIVE Tuesday
build sees, because the underlying rows don't exist in the ingested data at
all yet at that point: `docs/injury_news_sourcing.md` §1 measured (read,
that document's own session) that the NFL's mandated first practice report
of the week is filed **Wednesday**, and Monday+Tuesday combined are under 1%
of `date_modified` rows in every one of 16 seasons (2009-2024). §5.1 (read)
then measured this directly on the exact rows the tilt's construct uses:
only **328 of 76,784** official injury-report rows (0.43%) have
`date_modified` at or before their own game's Tuesday noon, rising to 8.13%
even crediting every PFT news article as "known." Weekly-run's ingest (step
1) and `build-player-features` (step 3) both run Tuesday, before Wednesday's
filings exist — so for the current week specifically,
`injuries_by_game.get((season, week, team))` is essentially always empty at
build time, independent of the 24-hour parameter.

**Live-time behavior, precisely.** `_injury_rows_asof` returns `None` when no
row survives the filter (`players.py:806-809`); `_injury_features(None, ...)`
then returns NaN for every injury-state metric
(`players.py:816-817`, read); `raw_value_lost_diff` coerces those NaNs to
0.0 (`injury_value_tilt_overlay.py:118`, read: `fillna(0.0)`); and the flip
condition requires `value_lost_diff` to be **strictly nonzero**
(`injury_value_tilt_overlay.py:196-199`). Net effect: the overlay is a
**near-total no-op on the live weekly card** — it will flip essentially
nothing, essentially every week, because the signal it needs does not exist
yet when `weekly-run` builds Tuesday's card.

**Evidence-cutoff divergence, stated exactly.** The registered evidence in
`artifacts/prospective/challengers.json:226-243` (read) still cites the
Saturday-cutoff numbers verbatim — `effect_accuracy_points: 1.316`,
`probability_positive: 0.8875` — with no reference to the Tuesday-cutoff
collapse. `docs/injury_news_sourcing.md` §5.1 measured, the same day
(2026-08-19), that under a true Tuesday cutoff on the identical 456 games the
contrast is **+0.000 pts** (official-only) to **-0.219 pts** (official +
PFT-foreshadowed), and that the Saturday-vs-Tuesday channel itself is
**+1.3158 pts** (`probability_positive` 0.9003) — i.e. numerically the
*entire* originally measured effect. So: **yes, there is a real divergence**
between what the challenger's registration cites as its evidence and what
its live Tuesday arm can actually do. This does not refute the underlying
`injury_value_lost` mechanism (reliability 0.87-0.93, still real information)
and nothing here is bounded by a positive control — both AGENTS.md admissible
closing grounds are absent, so the family correctly stays
`unresolved_below_power`, per `docs/injury_news_sourcing.md` §5.1's own
verdict, which this audit does not revisit or overturn.

**Fix options — reported, not implemented (per this task's scope).**

1. **Re-derive the tilt from genuinely Tuesday-visible inputs.** Rebuild
   `diff_injury_skill_epa_value_lost`/`diff_injury_defense_disruption_value_lost`
   (or a parallel construct) restricted to rows visible by the game's own
   Tuesday noon — which, per the 0.43%-8.13% coverage measured in
   `docs/injury_news_sourcing.md` §5.1, means the construct would need to run
   almost entirely on the PFT-news-augmented signal (`scripts/ingest_injury_news.py`)
   rather than the official report, and would need its own fresh effect
   measurement before being trusted — the existing +1.316/0.8875 figures do
   not transfer.
2. **Document the expected live no-op explicitly, change nothing else.**
   Update the challenger's `status_reason`/`evidence` block and
   `injury_value_lost_tilt_overlay.py`'s module docstring to state plainly
   that the live 2026 arm is expected to record close to zero flips per week
   (not because the underlying mechanism is wrong, but because its input
   data does not exist yet at Tuesday build time), so a future reader of the
   prospective ledger does not mistake "near-zero flips, near-identical
   accuracy to the baseline" for a null result on the construct itself.

Neither option is applied here — this document reports the divergence and the
two paths out of it; the choice is an owner/orchestrator decision.

### `backup_qb_fade_overlay` — a second, stronger divergence found in the same pass

Not the requested-first challenger, but the audit surfaced a sharper version
of the same class of problem, so it is reported here in full rather than
buried in the table.

**What it reads.** `backup_qb_flag_by_game` reads `home_qb_name`/`away_qb_name`
directly from the newest local schedule snapshot
(`src/nfl_ats/backup_qb_fade_overlay.py:148-169`, read) to determine each
game's own starter, compared against that team's modal starter over its
strictly-prior starts this season (`>=3` prior starts required,
`MIN_PRIOR_STARTS`, `:109`).

**Measured directly this session:** `data/raw/20260817T235649Z/schedules.parquet` —
`home_qb_name`/`away_qb_name` are **272/272 null (100%) for every 2026
REG-season game, every week 1-18**, including games many months away, while
the same two columns are **0/272 null for the complete 2025 season**. This
column is populated only after a game is played (sourced from the box
score/PBP), never before — there is no pregame "projected starter" field
here, unlike `home_coach`/`away_coach` or `surface` (both measured 0/272
null even for 2026's full future schedule).

**Live-time behavior, measured by simulation.** Because
`_modal_backup_flag_for_group` only adds a start to its running counter when
`isinstance(qb_name, str)` (`backup_qb_fade_overlay.py:143`) but still
evaluates `qb_name != modal_qb` for the flag itself
(`:140`), a missing (`NaN`) current-week starter compares **unequal** to the
modal starter in Python — so once a team clears the 3-prior-starts
eligibility floor, its *own, not-yet-played* game is always flagged `True`,
regardless of who is actually starting. Simulated this session by masking
each target week's own `home_qb_name`/`away_qb_name` to `NaN` (reproducing
exactly what the live Tuesday build sees) and re-running
`backup_qb_flag_by_game` across all of 2023, 2024, and 2025, all weeks 1-18
(816 team-weeks tested):

| outcome | count | mechanism |
|---|---:|---|
| neither side flagged | 144 | weeks 1-3, eligibility floor (`>=3` prior starts) not yet reached |
| **both** sides flagged | 672 | weeks 4-18, **every single eligible game** — both teams' own-game `NaN` reads as "differs from modal" |
| exactly one side flagged | **0** | never observed |

`apply_backup_qb_fade_overlay`'s flip rule requires the pick's side to be
flagged **and** the opponent not flagged (`backup_qb_fade_overlay.py:298-303`,
the "clean case" — both-flagged games are excluded, mirroring
`coach_fade_overlay`). Since the measured data shows every eligible week
lands every game in the "both flagged" bucket, **the overlay produces zero
live flips in every one of the 816 team-weeks tested** — not a Tuesday-noon
problem specifically, a **pregame** problem: no hour of the week before
kickoff has this field populated, so this is not merely
"Tuesday-degraded", it is "impossible at any pregame timestamp" under the
current data source.

**Evidence-cutoff match: NO.** The registered evidence
(`registry/weak_signals.json:bias_battery_backup_qb_start` /
`_opener`, -0.2731 / -2.3578 pts) was computed on the fully resolved
historical archive, where every game's actual starter is known after the
fact. The live card can never reproduce that input before kickoff.

**No fix option is offered for this one**, unlike the injury tilt: the gap
is not a cutoff-tuning problem (there is no earlier cutoff that would help —
the field simply does not exist until after the game), it would need an
entirely different, genuinely pregame data source (a depth-chart/starter
report feed) to ever fire at all. That is a build task, not a documentation
fix, and is out of this audit's scope.

### The Tuesday-safe challengers, briefly

- **`hc_year_one_fade_overlay`** (`coach_fade_overlay.py:84-195`): reads
  `home_coach`/`away_coach` for the current game plus the prior season's
  modal coach. Coaching hires are public knowledge months before the season;
  measured 0/272 null on `home_coach`/`away_coach` for all of 2026 REG.
- **`division_revenge_tilt_overlay`** (`division_revenge_tilt_overlay.py:99-182`):
  reads the *first* meeting's final score between the same two teams this
  season. A same-season rematch only occurs in a later week, by which point
  the first meeting is a completed game (result populated post-game, same as
  `home_qb_name`, but the relevant game is weeks in the past by
  construction, not the current one).
- **`surface_switch_tilt_overlay`** (`surface_switch_tilt_overlay.py:159-217`):
  reads `surface`, a stadium-level structural fact, aggregated over the
  **whole season's** schedule (not just prior games) — the module's own
  docstring states this is deliberate, since surface is "not a cover
  outcome" (`:173-183`). Measured 0/272 null on `surface` for all of 2026
  REG, every week including week 18.
- **`spread_gap_zone_fade_overlay`** (`spread_gap_zone_fade_overlay.py:74-81`):
  reads only the card's own `spread_line` — no external table at all.
- **`smooth_cdf_mapping`** (`smooth_cdf_mapping_overlay.py:19-36`): refits
  the active card's exact recipe on the same leak-safe training cutoff the
  card itself already used; introduces no new external data source.
- **`best_pick_nomination_v2`/`v3`** (`best_pick_nomination.py:41-51`,
  `:175`): the eligibility pool is read from `tuesday_opener_quotes`, the
  project's own local capture of the Tuesday-opener market — the same line
  the pool itself locks against, by definition not later-arriving
  information.

### `mod07_weak_signal_stack` — flagged, not fully audited

This is the ACTIVE model itself, not a pick-level overlay — every other
overlay above transforms *its* picks. Its `weak_stack` feature table is
enriched from `game_features_pbp.parquet` with, in the project's own words,
"LEARNED availability semantics" under the same column names the fixed-prior
`game_features_player.parquet` table uses
(`injury_value_tilt_overlay.py:94`, `artifacts/prospective/challengers.json`
known_gap text, both read). `src/nfl_ats/availability.py`'s module docstring
(read, `:1`) describes "season-lagged empirical player-availability
probabilities" — a prior probability conditioned on
season/report-category/position, not a read of the current week's actual
filed designation — which on its face looks structurally different from,
and less exposed to, the current-week-report gap that breaks
`injury_value_lost_tilt_overlay`. **This was not independently measured to
the same depth this session**: `weak_stack` draws on several feature
families beyond injuries (QB continuity, roster continuity, expected
lineups) that were not individually traced for Tuesday visibility, and the
learned-availability build path itself (`build-learned-availability-features`)
was only read, not run. Recommended as the next audit target, since every
overlay's baseline arm — the thing every tilt/fade is scored against — rides
on this same table.

### Provenance

- **Measured this session:** the schedule-snapshot null counts for
  `home_qb_name`/`away_qb_name`/`home_coach`/`away_coach`/`surface`/`result`
  (2025 vs 2026 REG), and the 816-team-week backup-QB flag simulation across
  2023-2025, all run directly against
  `data/raw/20260817T235649Z/schedules.parquet` this session.
- **Read this session:** every file:line citation above
  (`src/nfl_ats/injury_value_tilt_overlay.py`, `players.py`, `weekly.py`,
  `backup_qb_fade_overlay.py`, `coach_fade_overlay.py`,
  `division_revenge_tilt_overlay.py`, `surface_switch_tilt_overlay.py`,
  `spread_gap_zone_fade_overlay.py`, `smooth_cdf_mapping_overlay.py`,
  `best_pick_nomination.py`, `availability.py`), `docs/injury_news_sourcing.md`
  in full, `docs/pool_edge_plan.md` line 80, and
  `artifacts/prospective/challengers.json` in full.
- **Reported (from `docs/injury_news_sourcing.md`, itself measured that
  session, not re-verified here):** the +1.316/0.8875 Saturday-cutoff
  figures, the Tuesday-cutoff collapse to +0.000/0.3965, and the
  Saturday-minus-Tuesday channel-delta figures.
- **Inferred:** none of the classifications above rest on inference — every
  (b)/(c) call is backed by either a direct code trace to a table whose
  as-of semantics were already measured (injury tilt) or a fresh measurement
  run this session (backup-QB).
- **Not done:** no overlay logic, registration, or challenger status was
  changed. No `weak-signals record` or `rotation record-look` call was made
  — this audit adjudicates information timing, and the one family it
  touches (`injury_value_lost_narrowed`) was already correctly left
  `unresolved_below_power` by `docs/injury_news_sourcing.md` §5.1; nothing
  here provides grounds to reclassify it.

### 2026-09-05 prospective registration completion

Read (`artifacts/prospective/challengers.json`, the three entries above): all
three remain `unresolved_below_power` and `ACTIVE_PROSPECTIVE`. Their positive
expected-value leans are tracked; these registrations spend no rotation window
and do not change the published card. The historical fitted-feature effects
are not measurements of these frozen pick-level rules.

Reported (registration evidence, unverified by a new historical run): specialist
absence +0.8772 accuracy points, week interval [-0.4357, +2.2422],
`probability_positive=0.87905`; low-total divisional home dog +0.4386,
[-0.6608, +1.7544], `probability_positive=0.68955`; rain-on-grass dog +0.6579,
[-1.7058, +2.8446], `probability_positive=0.69175`.

Measured (`.\.tools\uv.exe run --no-sync python scripts\lockday_rehearsal.py`):
34 active challengers (27 publish, 6 refresh, 1 weekly-run), zero static wiring
errors. This static audit does not execute live recorders or prove live input
availability. Read (`low_total_div_home_dog_challenger.py`,
`rain_on_grass_dog_challenger.py`): paired companion files preserve the baseline
when a recorder is called directly; standard prospective scoring still consumes
the shared challenger ledger and production paper ledger.
