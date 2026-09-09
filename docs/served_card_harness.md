# Reading candidates against the card that is actually played

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

## The gap this closes

Every through-the-card measurement taken before 2026-09-09 evening
(`docs/spread_hole_target.md`, `docs/era_weighted_on_card.md`,
`docs/rookie_crew_reconciliation.md`, `scripts/spread_hole_arms.py`) scored its
candidate against the **three-member** card — coach fade OR division revenge OR
player arrests — because that is the only card the archive harness in
`nfl_ats.overlay_composition` knew how to build. Since commit `b5f60b7` the
served policy is
`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`,
a **nine-member** joint OR
(`src/nfl_ats/four_overlay_composition.py:71`, `COMPOSITION_ORDER`).

`nfl_ats.unserved_tilt_marginals.served_card_flip_set` now builds either card
from each member module's own `apply_*` function and its own data, reusing the
loaders `build_candidate_flip_sets` already used verbatim.
`scripts/spread_hole_arms.py` and `scripts/unserved_tilt_marginals.py` both take
`--card served|three`, with `served` the default.

## Replay proof

Measured this session, `artifacts/served_card_harness/20260909T233923Z/replay_proof.txt`,
on `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` (active model
`c657058903f3232b`, `gaussian_median`), 1,537 games, 34 opener pushes, 1,503
scored:

| Card | Union flips | Accuracy | Published |
| --- | ---: | ---: | ---: |
| raw model, no card | 0 | 54.558% | 54.558% |
| three-member | 279 | **55.888%** | 55.888% |
| **served nine-member** | **487** | **56.886%** | 56.886% |

Per-member flip counts on the archive: coach fade 107, division revenge 155,
player arrests 24, bye edge 73, cold visitor 57, protection mismatch 111,
interim coach 6, tank zone 17, precipitation-on-a-high-total 15. The nine-member
union is the three-member 279 plus the 208 incremental flips
`docs/unserved_tilt_marginals.md` attributes to the six added members, and it
reproduces both published figures to the digit.

## Re-scored on the true card

Paired week- and season-blocked bootstrap, 20,000 samples, seed 20260821,
within-week correlation zero. Artifact
`artifacts/served_card_harness/20260909T233923Z/results.json`.

### Rookie-crew reconciled rule — verdict unchanged

| Card | Window | n | Card % | With it % | Changed | Delta (pts) | 95% week | P+ week | P+ season |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **nine-member** | 2020-2025 | 1,503 | 56.886 | 57.019 | 6 | **+0.133** | [-0.197, +0.466] | **0.7904** | 0.7970 |
| three-member | 2020-2025 | 1,503 | 55.888 | 56.021 | 6 | +0.133 | [-0.198, +0.467] | 0.7904 | 0.8018 |
| nine-member | 2020-2021 | 456 | 57.237 | 57.456 | 1 | +0.219 | [+0.000, +0.667] | 0.8197 | 0.8758 |
| nine-member | 2022-2023 | 514 | 59.144 | 58.949 | 3 | -0.195 | [-0.811, +0.392] | 0.2863 | 0.1242 |
| nine-member | 2024-2025 | 533 | 54.409 | 54.784 | 2 | +0.375 | [+0.000, +0.943] | 0.9352 | 0.8745 |

The headline is identical on both cards: **+0.133 accuracy points,
`probability_positive` 0.790**, six changed picks. What the true card changes is
where the gain sits. On the three-member card the whole six-season gain was in
2020-2021 (+0.439) with 2022-2025 dead heats; on the served card it is
+0.219 / -0.195 / +0.375 across the three eras. Same six picks, different card
underneath them, so the same rule now helps most in the most recent era. That is
a magnitude statement per era, never an absence.

### Arm T2 at exactly 7 — verdict CHANGES

| Card | Scope | n | Card % | With it % | Changed | Delta (pts) | 95% week | P+ week | 95% season | P+ season |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| **nine-member** | line 7 | 74 | 50.000 | 44.595 | 4 | **-5.405** | [-11.111, -1.266] | **0.0080** | [-9.091, -1.429] | 0.0080 |
| three-member | line 7 | 74 | 50.000 | 48.649 | 1 | -1.351 | [-4.348, +0.000] | 0.1820 | [-3.409, +0.000] | 0.1698 |
| standalone (no card) | line 7 | 74 | 51.351 | 59.459 | 8 | +8.108 | [+1.333, +16.000] | 0.9893 | [+2.353, +14.706] | 0.9994 |
| nine-member | overall | 1,503 | 56.886 | 56.487 | 48 | -0.399 | [-1.330, +0.526] | 0.1958 | [-1.630, +0.883] | 0.2703 |
| three-member | overall | 1,503 | 55.888 | 55.023 | 63 | -0.865 | [-1.796, +0.067] | 0.0347 | [-1.793, +0.068] | 0.0353 |
| nine-member | 0-6.5 | 1,104 | 58.786 | 58.877 | 33 | +0.091 | [-1.002, +1.130] | 0.5756 | [-0.952, +0.938] | 0.5999 |
| nine-member | 7.5-10 | 194 | 52.577 | 51.031 | 7 | -1.546 | [-4.211, +1.053] | 0.1273 | [-4.950, +1.075] | 0.1781 |
| three-member | 7.5-10 | 194 | 51.546 | 47.938 | 11 | -3.608 | [-6.863, -0.510] | 0.0141 | [-5.319, -1.942] | 0.0000 |
| nine-member | 10.5+ | 131 | 51.145 | 51.145 | 4 | +0.000 | [-3.030, +3.077] | 0.5001 | [-3.175, +2.976] | 0.4974 |

Two things move, in opposite directions.

**At exactly 7 the sign resolves.** On the three-member card T2 moved one pick
and read -1.351 with an interval touching zero. On the true card it moves four
picks, all four wrong, and both the week-blocked interval [-11.111, -1.266] and
the season-blocked interval [-9.091, -1.429] sit entirely below zero. That is
the one admissible closing ground, so
`served_card_harness_t2_line7_2020_2025` is recorded `refuted_mechanism` /
`wrong_sign_resolved`. What is closed is exactly that cell — T2's marginal on
the served nine-member card at lines of exactly 7. **T2's standalone +8.108 at
exactly 7 (`probability_positive` 0.9893) is not closed and neither is arm T2
overall**; the standalone gain is a real thing that the served card, which
already moves those games, converts into a loss. And 74 games with 4 changed
picks is a small cell, said out loud.

**At 7.5-10 a recorded closure does NOT reproduce.**
`spread_hole_t2_card_7p5_10_2020_2025` was recorded terminal
(`wrong_sign_resolved`) yesterday on the three-member card at [-6.863, -0.510] /
[-5.319, -1.942]. On the card actually played the same arm reads -1.546,
[-4.211, +1.053], `probability_positive` 0.1273 — unresolved, not closed. The
same is true of `spread_hole_t1_card_7p5_10_2020_2025`'s sibling reasoning,
which this lane did not re-score. **A closure taken against the three-member
card is not a closure against the played card**, and the two terminal cells in
`mod18_spread_hole_target_v1` should be revisited on that basis.

Overall, T2 is less bad on the served card (-0.399, P+ 0.196) than on the
three-member card (-0.865, P+ 0.035), and it is still the wrong side of an
expected-value call. The incumbent card stays.

### Late-week move-follow rule — it does interact, and it reverses more than it endorses

The rule can be replayed on 2023-2025, with one deviation stated up front.
`nfl_ats.late_week_move_follow_refresh_overlay` loads quotes with
`capture_kind="live"` and its module docstring says "Only live captures are
prospective inputs, never historical backfills." Measured this session
(`artifacts/served_card_harness/20260909T233923Z/scan_market.py`), the 8,795
snapshot directories under `data/market/raw` hold **44** live captures and all
of them are 2026 (first `20260817T230004Z`). So the SERVED path cannot reach
2023-2025 at all.

What can: `late_week_follow_frame` is a pure function over a quotes frame and
does not itself filter on capture kind, and the archive holds **6,966
`intraday_hourly` historical-backfill snapshots**, 2,322 per season for
2023, 2024 and 2025 — 3.95 million spread rows, all twelve books of the
`LEADERSHIP_WEIGHTS` universe, covering every weekday including
Wednesday-through-Saturday, over all 816 archive games in those seasons. The
replay below feeds those rows to the frozen function with each game's cutoff set
to its own pick deadline `min(kickoff, Sunday 16:00 ET)`, i.e. the maximum
late-week information the rule could ever see. Zero quote rows were refused by
the function's own backdating guard.

Overlap with the six members added to the card on 2026-09-09, 2023-2025:

| | Games |
| --- | ---: |
| Archive games 2023-2025 | 816 |
| ...with replayable late-week exposure | 816 |
| ...with a qualifying move (at least half a point) | 329 |
| Games ONLY the six new members flip | 118 |
| — follow rule **reverses** the new flip (opposite side) | **20** |
| — follow rule **endorses** the new flip (same side) | **14** |
| — no qualifying late-week move | 84 |

Per new member over 2023-2025: protection mismatch 67 flips, bye edge 41, cold
visitor 19, tank zone 11, precipitation 4, interim coach 3.

Graded at the frozen Tuesday opener:

| Cell | n | Card % | With follow % | Changed | Delta (pts) | 95% week | P+ week | P+ season |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| follow on the served card, 2023-2025 | 799 | 56.070 | 55.695 | 145 | -0.375 | [-3.270, +2.532] | 0.3997 | 0.3008 |
| follow on the six new members' own flips | 114 | 50.877 | 49.123 | 20 | -1.754 | [-9.174, +6.307] | 0.3255 | 0.2034 |
| served nine-member vs three-member card | 799 | 55.820 | 56.070 | 114 | +0.250 | [-2.141, +2.605] | 0.5873 | 0.7406 |

So the answer to "does any of the six new members flip games the follow rule
then flips back" is **yes, 20 of 118**, against 14 it endorses. Neither the
overall stack nor the overlap subset resolves — both are
`unresolved_below_power` and both lean against stacking the follow rule on the
nine-member card (`probability_positive` 0.400 and 0.326). Nothing is closed.

**What a faithful replay would need.** To grade the rule as it is actually
served, rather than as arithmetic: live captures (`capture_kind` absent, which
the loader reads as `live`) at the actual Wednesday-through-Saturday refresh
timestamps of a season, plus the recorded Tuesday card for that week
(`artifacts/prospective/original_card`, which supplies `pick_side`,
`decision_home_spread` and `recorded_at_utc`), plus a real `RefreshResult` plan
so per-game eligibility and deadlines come from `nfl_ats.pick_refresh` instead
of being reconstructed. None of that exists before 2026-08-17. The backfill
replay above is the honest substitute and is labelled as one.

### Side effect: the unserved-tilt table, re-run on the true card

`scripts/unserved_tilt_marginals.py --card served` exercised end to end this
session; artifact
`artifacts/served_card_harness/unserved_tilt_marginals_on_served_card/20260909T234904Z/`.
Baseline 56.886% on 1,503 scored games, union 487 flips.

The seven of the fifteen candidates that the nine-member card already contains
now read exact dead heats — 0 incremental flips, delta 0.000,
`probability_positive` 0.5000, which is the correct value for a no-op under the
2026-09-08 fix to `probability_positive_from_draws`. Of the eight that remain
unserved, **none has `probability_positive` above 0.5** on the true card:

| Adjustment | Flips | Already on card | New flips | Delta (pts) | P+ (week) |
| --- | ---: | ---: | ---: | ---: | ---: |
| spread_gap_zone_fade_overlay | 195 | 72 | 123 | -0.532 | 0.2396 |
| third_down_reversion_fade_overlay | 312 | 85 | 227 | -0.998 | 0.1751 |
| pace_mismatch_dog_tilt_overlay | 177 | 62 | 115 | -1.065 | 0.0713 |
| backup_qb_fade_overlay | 174 | 62 | 112 | -1.198 | 0.0429 |
| special_teams_return_tilt_overlay | 275 | 92 | 183 | -1.730 | 0.0283 |
| surface_switch_tilt_overlay | 225 | 60 | 165 | -1.597 | 0.0191 |
| injury_value_tilt_overlay | 711 | 217 | 494 | -4.325 | 0.0041 |
| turnover_luck_rebound_tilt_overlay | 305 | 98 | 207 | -3.127 | 0.0003 |

On the three-member card `spread_gap_zone_fade_overlay` read -0.399 at
`probability_positive` 0.3149; on the true card it is -0.532 at 0.2396. Nothing
in that pile closes — every one of these members was measured positive on the
bare model, so what is being read here is an interaction with a card that
already moves 487 games, not a refuted mechanism. These rows are not re-recorded
here; `docs/unserved_tilt_marginals.md`'s three-member cells stand and this table
is the same measurement against the card that is played.

## Decision

- **Rookie-crew reconciled rule: still positive on the card that is played.**
  +0.133 accuracy points, `probability_positive` 0.790, six changed picks. The
  pool is forced picks, so a rule that is 79% likely to be better is a rule to
  play; the only thing stopping it is that crew assignments are not published
  until Wednesday-Thursday, after the Tuesday lock
  (`docs/referee_assignments_capture.md`). Nothing here is wired.
- **Arm T2 at exactly 7: closed on the served card, open everywhere else.**
- **Late-week follow: real interaction, 20 reversals against 14 endorsements,
  unresolved either way.**

### What serving the rookie-crew rule would take (NOT wired here)

1. **It is a model feature, not an overlay tilt.**
   `nfl_ats.crew_tilt_refresh_overlay` is the refresh-path vehicle — it already
   reads the Wednesday `data/players/referee_assignments` snapshot, checks each
   game's own `pick_deadline`, and writes a paired challenger ledger
   (`artifacts/prospective/crew_tilt_refresh_decisions.parquet`) without ever
   touching the played pick — but the rule it currently serves is the additive
   `crew_tilt_flags` probability tilt on two `penalty_crew_tendencies` cells.
   The rookie-crew rule changes the fitted ridge instead. Serving it means
   either a refresh-time refit on profile `weak_stack_rookie_crew_underdog`,
   restricted to games whose deadline is still open, or converting the learned
   coefficient into a third tilt cell in that module. Only the first preserves
   the +0.133 measured above; the second is a different rule and needs its own
   read.
2. **`ROOKIE_ELIGIBLE_SEASON_FLOOR` must move from 2016 to the loaded
   population's own first season** (`src/nfl_ats/officials_flag_features.py:78`).
3. **The flag's line source must accept the close proxy before 2020.** A serve
   built on the Tuesday-opener store alone is the -0.599 arm, not this one.

### A defect this lane found on that same path

Measured this session: `nfl_ats.crew_tilt_refresh_overlay.run_stacked_backtest`
raises

```
DataContractError production chain members not reconstructed:
['bye_edge_fade', 'forecast_cold_visitor_tilt', 'pbp08_protection_mismatch_tilt',
 'interim_hc_first_game_tilt', 'tank_zone_fade_tilt', 'precip_high_total_tilt']
```

on the current archive. It builds a three-member `members` dict at
`src/nfl_ats/crew_tilt_refresh_overlay.py:1050` and then asserts every name in
`COMPOSITION_ORDER` is present at line 1055 — which became a nine-name tuple in
`b5f60b7`. The guard is marked `# pragma: no cover - defensive` and no test
exercises it, so the nine-member promotion broke the crew-tilt backtest
silently. Not fixed here; this lane was scoped to the archive harness.

## Registry

Twelve cells, family `served_card_harness_nine_member`, named
`served_card_harness_<arm>_<window>`. Eleven are `unresolved_below_power` and
report `probability_positive`; one,
`served_card_harness_t2_line7_2020_2025`, is `refuted_mechanism` on
`wrong_sign_resolved` because both blockings sit entirely below zero. The exact
argv lists are saved as
`artifacts/served_card_harness/20260909T233923Z/record_commands.json` and were
run one at a time against `registry/weak_signals.json`.

## Commands run

```
python artifacts/served_card_harness/20260909T233923Z/replay_proof.py
python artifacts/served_card_harness/20260909T233923Z/laneT_rescore.py \
  --out artifacts/served_card_harness/20260909T233923Z
python scripts/unserved_tilt_marginals.py \
  --per-game-artifact artifacts/opener_evaluation/20260909T183120Z/per_game.parquet \
  --data-root data --repo-root . --card served \
  --output-root artifacts/served_card_harness/unserved_tilt_marginals_on_served_card
nfl-ats weak-signals record ...   (12 cells; argv lists in record_commands.json)
```
