# Registry reliability sweep, 2026-09-01 (ORCH-D)

**Goal.** Every construct in `registry/weak_signals.json` gets a *measured*
split-half reliability, so that one of the only two admissible closing
grounds stops being unusable for two thirds of the pile.

**Measured this session** (`registry/weak_signals.json`, read 2026-09-01):
624 signals total; 543 are NFL `accuracy_points`; **365 of those carry
`reliability: null`** and 178 carry a number. Of the 178 already measured,
18 sit at or below 0.10 and 93 at or above 0.80.

**Why it matters, stated before what is wrong with it.** AGENTS.md allows a
line of work to be closed on exactly two grounds: a RESOLVED wrong sign, or
"the trait has no split-half reliability". With `reliability: null`, that
ground can neither be used on a construct that genuinely has no trait, nor
*ruled out* on one that does — so 365 entries currently sit in a state where
the strongest available out-of-sample predictor is simply unknown. Filling
them in is what lets the ranked agenda be sorted by something better than
point estimates that are all individually below power.

---

## Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never the
binary "contains zero".

**This sweep closes nothing.** Reliability is one of the two closing grounds,
so a measured reliability near zero is a real terminal finding *only when the
measurement is itself sound* — adequate split sizes, the same construct
definition the registry cell used, and the seasons the cell used. A
reliability measurement never by itself changes any cell's classification
here. Workers RECORD reliabilities; §2 lists which cells end up at ≤ 0.10 so
the owner can decide on a separate, explicit closure step.

Within-week correlation is ZERO (owner mandate); nothing here estimates it.

---

## §1 — Grouping, method, and the entry list

### The three quantities, and why they are never interchangeable

All three land on the same [-1, 1] correlation scale, which is exactly why
every recorded number carries its method string into the entry's audit note.
The estimator is `nfl_ats.cfb_qb_dependence.split_half_reliability` — reused,
never reimplemented — the same function behind the FluView, team-style,
PBP-08 and CFB role-continuity reliability precedents, wrapped for this sweep
in `scripts/reliability_lib.py` with one seed (`20260901`) and 4,000
bootstrap draws for every group.

| Tag | Unit | Halves | What it means | Admissible as a closing ground? |
| --- | --- | --- | --- | --- |
| `METHOD_TRAIT` | team-season | odd/even weeks | the construct is a continuous per-team-week quantity (an EPA rate, an injury value-lost, a rating divergence) | **yes** — this is the quantity `NO_SPLIT_HALF_RELIABILITY_MAX = 0.10` was calibrated against |
| `METHOD_VENUE` | venue-season | odd/even weeks | venue-level constructs (roof state, altitude, playing surface, weather at the stadium) | no, not on its own — a venue trait can be perfectly stable and say nothing about either team |
| `METHOD_EXPOSURE` | team-season | odd/even weeks | a per-game FLAG with no continuous parent trait; measures the flag's *exposure rate* per team-season | **no** — a schedule quirk with no stable team structure can still move covers, and closing on it would be the crossing-zero mistake in a new hat |

This follows the registry's own established precedent, not a new convention:
the six `attention_battery_*` cells all share `reliability = 0.1316`, which is
the reliability of the *attention-gap trait* the battery thresholds, not a
per-cell number (read: `registry/weak_signals.json`, and
`scripts/attention_battery_screen.py:437`). A battery's cells inherit the
reliability of the trait they are built on.

### What is deliberately NOT written to the registry

A flag cell's **effect replication** — does the cover-rate gap survive on
held-out seasons — is a different question from all three above. Every worker
reports it (`half_season_replication`: odd-season gap vs even-season gap, in
percentage points, with counts and sign agreement, plus a battery-level
correlation across cells) into its artifact and into §2. It is *reported, not
recorded*: it is not a correlation, so it does not belong in a
correlation-scaled field that a validator reads as a closing ground, and a
sign disagreement on a small subset is the expected shape for a
real-but-small effect at this resolution.

### Never write an unmeasurable reliability as a number

`split_half_reliability` returns NaN when too few units survive its
≥2-observations-per-half floor. `scripts/reliability_lib.py` returns an
explicit `status` (`measured` / `insufficient_split_units` /
`constant_or_all_missing`) and only `measured` may be recorded;
`nfl-ats weak-signals set-reliability` refuses a non-finite value as a second
line of defence. A construct with too few units is reported as **unmeasured**,
never as reliability 0 — writing NaN through as a number would manufacture the
appearance of a closing ground out of nothing.

### A conserved-total quantity has no split-half reliability to measure

Found mid-sweep, and it is the sharpest trap in this whole exercise. A quantity
whose **season total is fixed** — days of rest is the canonical case, because a
team-season spans a fixed number of days — is forced to correlate NEGATIVELY
between any two halves: more rest in one half mechanically means less in the
other. The estimator dutifully returns a large negative number, and because a
reliability at or below 0.10 is a legal `no_split_half_reliability` ground in
the validator, recording it would plant a booby trap that a later session could
use to close an entire family on a mathematical artifact.

Measured 2026-09-01 on 2009-2025 REG schedules, team-week `rest`
(544 team-seasons): the odd/even-week split gives **r = -0.9766, 95%
[-0.9816, -0.9713]**, and a RANDOM within-team-season half split over 20
reseeds gives **mean r = -0.8514** (range -0.8813 to -0.8066). The negative
survives randomising the split, so it is not an odd/even alternation effect —
it is the conserved total, and split-half reliability simply does not apply to
that quantity.

**Rule for the sweep**: before recording any NEGATIVE trait reliability, re-run
the same measurement with the week column replaced by random integers. If it
stays strongly negative, the quantity is compositional; report
`not_applicable_compositional_constraint` and do not record it as a trait. Three
entries recorded before this was found (`travel_rest_away_off_bye`,
`travel_rest_short_week_road`, `travel_rest_home_off_bye`, all at ~-0.741) were
re-recorded the same day with a corrected method string that opens "NOT
ADMISSIBLE AS A `no_split_half_reliability` CLOSING GROUND" and carries the
proof above; their values are unchanged and retained only as the audit trail of
what the estimator returned.

### Positive control (mandatory per group)

A near-zero reliability is uninterpretable until the instrument has been shown
able to find reliability that is genuinely there, at *these* unit counts. Every
group script runs `reliability_lib.positive_control`, which plants traits with
a known per-unit variance share (0.0 / 0.2 / 0.5 / 0.8) on that group's own
unit structure and reports what the estimator recovers.

Measured this session on the 2009-2025 REG schedule's team-week frame (544
team-seasons, `scripts/reliability_lib.py` smoke run): planted share 0.0 →
recovered −0.077; 0.2 → 0.839; 0.5 → 0.942; 0.8 → 0.985. At these unit and
observation counts the instrument detects even a modest true share easily, so
a near-zero measured value on a well-populated trait is an informative,
control-bounded reading rather than a power failure. Groups with far fewer
units must report their own control table before any low number is read.

### The new recorder

`nfl-ats weak-signals set-reliability --name --reliability --reliability-low
--reliability-high --method --source --reason` (added this session,
`src/nfl_ats/weak_signals.py`, wired in `src/nfl_ats/cli.py`, tested in
`tests/test_weak_signals_set_reliability.py`). It writes the `reliability`
field and appends ONE audit line to `notes` carrying the interval, the method
and the measuring artifact. The schema has no reliability-interval field, so
the interval lives in that note; the entry's own `source` is never
overwritten. Effect, interval, classification, closing_ground,
probability_positive and every other field are carried over byte-for-byte, and
recording a low number never reclassifies anything.

### Groups (disjoint, exhaustive: 365 of 365)

Full per-entry manifest with seasons, sample sizes and sources:
`<scratchpad>/orchD_manifest.json`.

| Group | n | Construct(s) | Builder(s) reused by import | Primary method |
| --- | --- | --- | --- | --- |
| `modeling_overlay` | 59 | model-vs-model paired deltas, overlay compositions, era weighting, best-pick rankers | `scripts/overlay_*`, `era_weighting_lib.py`, `best_pick_ranker.py`, `mod07_weak_stack.py` | trait where an overlay thresholds a trait; otherwise `not_applicable` (no trait to be reliable) |
| `health_roster` | 46 | injury value-lost, NFL.com/PFT report channels, player/availability bundles, arrests, ADP, interim HC | `nflverse_injuries_*`, `nflcom_friday_*`, `player_arrests_policy_eval.py`, `ffc_adp_divergence_screen.py` | trait |
| `graph_team_stat` | 41 | graph-rating team-stat inputs | `graph_team_stat_screen.py` + `game_features_weak_stack_v4.parquet` | trait |
| `weather` | 33 | actual/forecast weather + total interaction | `nfl_weather_battery_screen.py`, `nfl_forecast_weather_screen.py`, `weather_total_interaction_screen.py` | venue |
| `schedule_clock` | 30 | body clock, travel/rest, DST transition | `body_clock_screen.py`, `body_clock_night_screen.py`, `nfl_travel_rest_battery_screen.py`, `dst_transition_battery_screen.py` | exposure (traits where one exists, e.g. travel distance) |
| `opener_eval` | 28 | production-rule error-mining slices of one opener evaluation | `artifacts/opener_evaluation/.../per_game.parquet` | trait for continuous slicers; exposure for categorical |
| `env_venue` | 27 | altitude, roof, surface, AQI/drought, venue milestones | `altitude_screen.py`, `roof_decision_screen.py`, `surface_familiarity_screen.py`, `environmental_exposure_battery.py`, `venue_milestone_screen.py` | venue |
| `market_micro` | 26 | odds microstructure, public betting, Sagarin divergence, SBR opener, CDF mapping | `odds_microstructure_battery.py`, `public_betting_battery_screen.py`, `sagarin_divergence_battery.py`, `sbr_era_opener_eval.py` | trait |
| `movement` | 26 | line-movement attribution, observed-movement channels, expansion battery | `movement_attribution.py`, `observed_movement_channel.py`, `movement_expansion_battery.py` | trait (movement magnitude) |
| `bias_battery` | 25 | NFL bias battery cells | `nfl_bias_battery_screen.py` (`build_long_table`, `add_history_features`, `build_hypotheses`) | trait/exposure per cell |
| `situational` | 24 | bye weeks, divisional rematches, motivation ladder, primetime | `bye_overvaluation_screen.py`, `divisional_rematch_screen.py`, `motivation_ladder_screen.py`, `primetime_cells_screen.py` | exposure |

Deliberately-leaked oracle controls (`*_oracle_*`, `weather_oracle_ceiling_*`,
`observed_movement_oracle_*`) stay in their groups for coverage accounting but
are measured only for their *inputs'* reliability; their effect numbers are
ceilings by construction and no reliability is read as evidence about a
playable rule.

<!-- §2 and §3 are appended by the integrating orchestrator after the workers land. -->
