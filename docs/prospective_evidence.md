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
