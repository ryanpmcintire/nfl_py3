# Veteran rest-day BACK overlay: a no-window-cost prospective challenger

Written 2026-09-11, as a direct follow-up to LEAD-11 (`docs/veteran_rest_day_tell.md`,
`scripts/veteran_rest_day_tell.py`). Follows the `backup_qb_fade_overlay` /
`division_revenge_tilt_overlay` / `injury_value_lost_tilt_overlay` /
`hc_year_one_fade_overlay` precedent (`docs/backup_qb_fade_overlay.md` and
siblings) for wiring a pick-level, post-prediction transform into the
prospective challenger ledger at zero rotation-registry window cost.

## Binding verdict taxonomy (verbatim, per session instructions)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## This is a post-hoc reversal, stated plainly

**The direction built here was chosen AFTER seeing LEAD-11's sign, not
predeclared.** LEAD-11 predeclared a FADE of teams with rest-tagged 30+
veteran starters (rest = hidden decline the market has not priced) and
measured the opposite: **read**, `registry/weak_signals.json:
veteran_rest_day_tell_on_production` (rotation family
`veteran_rest_day_tell_on_production`, assigned and now permanently spent
window `[2020, 2021]`, opener grade) records effect **-2.6316 accuracy
points, week-blocked 95% [-5.765, +0.222], probability_positive 0.038225**,
466 paired games / 35 weeks, 39/466 forced picks flipped. **Read**, the same
write-up's independent on-field comparison (starter-matched, 2021-2025,
n=790 flagged vs n=8,381 non-flagged 30+ starters) found flagged veterans
show a SMALLER production drop from their own trailing baseline (-0.0154 vs
-0.0273) and a LOWER 4-week missed-game rate (0.195 vs 0.251), replicating
in both odd (2021/23/25) and even (2022/24) season halves. Both readings
point the same way: a rest day for a 30+ starter reads like load management
that works, not a hidden decline. Team rest-tag propensity is itself a
reliable team-season trait (**read**, Pearson r=0.5633, 95% [0.285, 0.768],
probability_positive 0.99985), so `no_split_half_reliability` is
inadmissible as a closing ground, and the opener-window interval crosses
zero so `wrong_sign_resolved` is inadmissible too. LEAD-11's own
classification is `unresolved_below_power` -- it did not close the FADE
construct, it declined to spend the card on it at this power.

**This overlay does not close that ledger entry either.** It is a separate,
independent BACK rule, registered as a fresh `ACTIVE_PROSPECTIVE`
challenger, graded exclusively on 2026+ games. It never touches, re-scores,
or re-uses the spent 2020-2021 rotation-registry window: doing so would
spend the same window twice on the same family, exactly what the rotation
registry's window discipline exists to prevent.

### The "+2.63 / P+ 0.96" figure this lane's task brief cited

This lane's task brief described the reverse direction as "BACKING those
teams at +2.63 points, probability_positive 0.96, on the same flips." **Read
and checked, this is the exact arithmetic negation** of LEAD-11's own
recorded numbers: -2.6316 -> +2.6316, and 0.038225 -> 1 - 0.038225 =
0.961775 (rounds to 0.96). It is not an independently re-run backtest, and
it is not treated as one here -- no new bootstrap was run against the
2020-2021 window for this document. It is read only as the DIRECTION and
rough MAGNITUDE the reversal is expected to carry, for two reasons stated
plainly:

1. It is a pure sign-flip of ONE existing measurement, not a new sample.
2. As built below, this overlay's flip RULE is not literally "the same 39
   games with the opposite pick." A literal same-games reversal would just
   be "do not fade," i.e. play production's original pick unchanged on
   those 39 games -- that reproduces production exactly and would measure a
   flat **0.0** delta, not +2.63. The true behavioral mirror of a rule that
   always moves picks OFF a flagged team is a rule that always moves picks
   ONTO a flagged team, which is a different population of games (see "The
   rule" below). Whether that rule's own effect size lands near +2.63 on
   2026+ games is exactly what `nfl-ats prospective-score` will tell us --
   it is not assumed here.

## The construct: reused, and one deliberate simplification (disclosed)

**Reused verbatim from LEAD-11 (measured, `scripts/veteran_rest_day_tell.py`
section 1):** a player-week is flagged when (a) the injury row's reason text
(`report_primary_injury`, `report_secondary_injury`,
`practice_primary_injury`, `practice_secondary_injury` on nflverse
`injuries.parquet`) matches `\brest(?:ed|ing)?\b|load management`
(case-insensitive); (b) the player is chronologically **30 or older** at
that game's kickoff, using `nflreadpy.load_players()`'s cached birth dates
(`data/players/raw/<snapshot>/players.parquet`, 0.13% null league-wide,
100% coverage on the 2019-2026 rest-tagged rows LEAD-11 measured); (c) the
row's own final `practice_status` is **not** "Did Not Participate In
Practice." Implemented identically (same regex, same age threshold, same
DNP exclusion) in `src/nfl_ats/veteran_rest_back_overlay.py`'s
`rest_tagged_veteran_flag_by_team_week`, reading the raw nflverse injuries
snapshot directly (`data/raw/nflverse_injuries/<snapshot>/injuries.parquet`,
which carries the reason-text columns; the canonical
`data/players/raw/<snapshot>/injuries.parquet` copy LEAD-11 also used for
`observed_at_basis` diagnostics is not needed here since only the flag
itself, not its proxy-timestamp provenance, feeds this overlay).

**Deliberately NOT reused: LEAD-11's trailing-snap-share starter gate.**
LEAD-11's own production screen (section 3 of `docs/veteran_rest_day_tell.md`)
restricted its team-value feature to flagged players whose trailing
snap share (a strictly-prior, up-to-4-game mean, computed only from games
already played) was also >= 0.5 -- a "starter" gate. **Measured, 2026-09-11**
(direct computation against the current data caches): reapplying that gate
to any season's Week 1 makes it a **permanent no-op**, because trailing
share is `NaN`/0 before any game has been played that season -- there is no
"prior 4 games this season" in Week 1, ever, for any team, in any year. LEAD-11's
screen never needed to confront this because it was scored over 35 whole
weeks per season, where the effect is dominated by weeks 2+; this overlay's
task explicitly required confirming a real 2026 Week 1 value, which the
starter-gated construction cannot produce by design. The team-week flag
used here is instead the plainer **"this team has at least one rest/NIR-tagged
30+ veteran on this week's injury report,"** with no workload weighting.
This is a real, disclosed deviation from LEAD-11's own screened
population -- it is not the identical 39-game construct, and its own
prospective performance is untested until 2026+ games settle.

## The rule, exactly as built

```
team_week_flagged(team) = any 30+ veteran on that team's injury report this
                           week matches the rest/NIR flag above

home_pick    = home_cover_probability >= 0.5
picked_side_is_flagged   = home_flagged   if home_pick else away_flagged
opponent_side_is_flagged = away_flagged   if home_pick else home_flagged

flip when:
    (home_flagged != away_flagged)   AND   NOT picked_side_is_flagged
```

In plain language: **when exactly one side carries a rest-tagged 30+
veteran this week, and the active model's raw pick is NOT already on that
side, flip onto it.** This is the structural mirror of the ORIGINAL fade
rule (`home_flagged != away_flagged AND production_pick_home ==
home_flagged`, i.e. flip AWAY whenever production already sits on the
flagged side): fade always moves a pick off a flagged team, back always
moves a pick onto one. REG season only, matching LEAD-11's REG-only
measurement. A both-flagged game is reported (`both_flagged_games`) but
never flipped, following the same "clean case" pattern as
`backup_qb_fade_overlay` and `coach_fade_overlay` -- there is no measured
direction for flagged-vs-flagged.

Implemented in `src/nfl_ats/veteran_rest_back_overlay.py`:
`veteran_rest_flag_by_game` derives the per-game flag from the schedule
snapshot plus the newest injuries and players caches;
`apply_veteran_rest_back_overlay` applies the rule at pick level;
`overlay_disclosure_note` produces the plain-English provenance sentence
(not currently surfaced anywhere); `record_veteran_rest_back_overlay_decisions`
records the overlay's own arm into the shared prospective challenger
ledger.

**Pregame-safe.** The injury report row itself is the league's own pregame
disclosure (final practice-participation status filed before kickoff); the
overlay reads the current live snapshot at record time and only ever
computes a flag for games whose kickoff has not yet passed (enforced by the
existing `refuse_if_outside_recording_lock_window` / pre-kickoff filtering
machinery every sibling overlay already uses). Unlike `backup_qb_fade_overlay`
(which was found, and left, `DEACTIVATED_STRUCTURAL_NO_OP` because its
`home_qb_name`/`away_qb_name` source is populated only post-game), the
injury-report source used here is confirmed populated pregame: **measured,
2026-09-11**, the live 2026 Week 1 injuries snapshot
(`data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet`) already
carries 139 report rows for Week 1 before any Week 1 game has kicked off,
including the one rest-tagged 30+ veteran described below.

## Confirmed live against the current week (2026 Week 1)

**Measured, 2026-09-11**, against the active forecast
(`artifacts/margin_predictions/2026-week-01-20260910T210852Z`, model
`d49194e04945a5e5`): exactly one team carries the flag this week -- the
Rams (`LA`), home team in `2026_01_SF_LA`, via rest-tagged veteran
`gsis_id 00-0033110` (rest reason text, age-eligible, cleared to Full
Participation by the week's final practice report). No other 2026 Week 1
team is flagged, and there are zero both-flagged games. The active model's
raw pick on that game was SF (`home_cover_probability` 0.4442, i.e. the
away/non-flagged side), so the overlay's one eligible flip flips the pick
onto LA:

```
flip_count: 1
flipped_game_ids: ['2026_01_SF_LA']
BackFlip(game_id='2026_01_SF_LA', matchup='SF at LA', backed_team='LA', opponent_team='SF')
both_flagged_games: []
```

`record_veteran_rest_back_overlay_decisions` was run once this session
(`artifacts/prospective/challenger_decisions.parquet`, challenger_id
`veteran_rest_back_overlay`) to confirm the full recording path end to end.
By the time of that run, two of the week's sixteen games (`2026_01_NE_SEA`
and `2026_01_SF_LA`) had already kicked off, so both were correctly
excluded from the ledger by the existing pre-kickoff gate -- **the one game
this overlay would flip is one of those two**, so this week's ledger rows
(14 of 16 games) all happen to match production's original pick 1:1; the
overlay's first real divergence from production will show up starting
Week 2 or on a future week where the flagged game is recorded before its
own kickoff. This does not change what was confirmed: the overlay computed
a correct, non-trivial value (one real flag, one real flip) for the current
week, and the recording path wrote real rows to the shared ledger without
touching `recommendations.csv`, `CURRENT_PREDICTIONS.md`, or any served
page.

## Important caveat (stated up front, not buried)

Same double-counting question every sibling overlay in this family states:
the active `weak_stack` model already carries QB-continuity and
injury/availability features (`docs/injury_value_lost.md`). If the model
already prices the specific "rest-tagged 30+ starter" effect, this
overlay's flips should show little edge over the model's own raw picks; if
it does not, the overlay's flips should show a positive edge. Prospective
scoring settles this empirically, not this document.

## What is and is not wired in

- `src/nfl_ats/veteran_rest_back_overlay.py`: the transform
  (`apply_veteran_rest_back_overlay`), the signal reader
  (`veteran_rest_flag_by_game`, `rest_tagged_veteran_flag_by_team_week`),
  the disclosure sentence (`overlay_disclosure_note`, not currently
  surfaced anywhere), and the recorder
  (`record_veteran_rest_back_overlay_decisions`).
- `src/nfl_ats/cli_commands/publishing.py`'s `orchestrate_publish_predictions`:
  one more purely additive, fail-open `try`/`except` block (mirroring the
  existing ones) that calls the recorder when `--record-decisions` is
  passed, plus a `PUBLISH_CHALLENGER_RESULT_KEYS` entry
  (`veteran_rest_back_overlay` -> `veteran_rest_back_overlay_challenger_ledger`).
  This writes ONLY to `artifacts/prospective/challenger_decisions.parquet`;
  it never touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`,
  `README.md`, or the public site. **The production pick path
  (`publish_active_predictions`) is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `veteran_rest_back_overlay`, status `ACTIVE_PROSPECTIVE`, `model` block a
  snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring every other overlay
  challenger).
- **Not wired anywhere:** there is no switch that applies this overlay to
  the published card. Playing this on the real card is a separate owner
  decision this document does not make -- it is explicitly a post-hoc
  reversal with no historical confirmation of its own, so there is even
  less basis for playing it than for the (also unplayed) sibling
  challengers.
- **Tracked independently against the active model's own card, not
  stacked on the other overlays**, matching every sibling overlay's
  pattern exactly: each recorder reads the SAME un-flipped active-model
  card and applies its own transform independently.

## No new tests (moratorium)

Per the standing test moratorium, no test file or test function was added
for this module. Verification here is the direct run reported above
(`record_veteran_rest_back_overlay_decisions` against the live 2026 Week 1
card), not a pytest fixture.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free, honestly-labelled
evidence stream. Once 2026 accrues enough weeks, `nfl-ats prospective-score`
reports this challenger's `probability_positive` at both grades, paired
against the active model, exactly like every other overlay challenger. That
number, and only that number, will say whether backing rest-tagged 30+
veterans holds up -- not the arithmetic mirror of LEAD-11's own spent
window quoted above.
