# Graph ratings v2 — the NFL screen: predeclaration

Written **before any ATS number is produced by the `team_stat` arm**, per the
same rule that governs `docs/graph_ratings_v2.md` (the engine) and
`docs/graph_input_screen.md` (the input list). This document declares the one
comparison that joins those two lanes, and it declares it before the outcome
evaluator is pointed at it.

**Sections 1-7 are the predeclaration** and contain no accuracy, cover rate,
Brier, or `probability_positive` against NFL outcomes -- only design decisions
and **measured, outcome-free** feature-space diagnostics. **Section 8 was added
after the look** and reports what it found; it changes nothing above it.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". A promotion threshold governs only what the docs may
CLAIM; it never governs which card is played, which is expected value.

## 1. What was missing, and what this closes

The two lanes finished on 2026-08-26 did not meet. The engine
(`src/nfl_ats/graph_ratings_v2.py`) accepted exactly two edge signals,
`residual` and `raw_margin`, and raised on anything else (**read**, its
`validate()` before this change). The input screen's entire purpose — the
owner's framing, **read** from `docs/graph_input_screen.md` — is that "each
team statistic is scored separately before being fed to the graph engine".
There was no way to feed one. The ranked list of 38 cluster representatives
had nowhere to go.

This session added the `team_stat` arm: `edge_signal="team_stat"` with
`signal_column="<family>"` builds edges from `home_<family> − away_<family>`,
one graph per screened statistic. Two conventions exist in the feature table
(**measured**, `reliability_map.discover_family_pairs`): a prefix form
(`home_def_takeaway_rate`) and a suffix form (`gap_division_revenge_home`), so
`signal_column_pair` names the columns explicitly when the prefix convention
does not apply — five of the 38 representatives are suffix-form, and without
this they would have been silently unreachable rather than merely inconvenient.

## 2. What this arm IS, stated precisely

The screened families are **pregame rolling team values**, not realized
in-game outcomes (**measured**: `home_def_takeaway_rate` is populated in week 1
of 2023 with zero NaNs, so it carries prior-season information). The edge is
therefore knowable before kickoff, and the graph is **opponent-adjusting a
statistic** rather than absorbing an outcome: a team whose statistical edge was
built against opponents who themselves hold that edge is endorsed more than one
whose identical raw edge came against opponents who do not.

Leak-safety is stricter than this arm needs, not looser. A week's own
differentials are folded into the accumulator only AFTER every game that week
has been assigned its ratings, so a `team_stat` rating for week `w` reads the
graph through week `w−1`, exactly like the outcome arms. Pinned by
`test_future_team_stat_values_cannot_change_prior_ratings`.

## 3. The mechanism check that had to come first (measured, outcome-free)

Before declaring anything against outcomes, the cheap decisive question: is a
graph-propagated statistic actually *different* from the raw `home − away`
differential the input screen already tested? If they correlate at ~0.99 the
whole lane is a no-op and no window should be spent on it.

**Measured this session** over the full merged feature table (4,902 rows,
seasons 2009–2026), one graph per cluster representative, correlating the
propagated `*_katz_diff` against the same family's raw differential. Results
are in section 7. This touches no outcome column, spends no rotation window,
and is reported as what it is: a feature-space diagnostic, not evidence about
accuracy.

## 4. The comparison being declared

**Question.** Does opponent-adjusting a screened statistic through the graph
beat using that statistic raw, on forced-pick ATS accuracy?

**Three arms, one evaluator, one window, per family:**

| arm | feature fed to the market-residual model | role |
|---|---|---|
| baseline | none (market only) | reference, as in the input screen |
| **control** | `home_<family> − away_<family>` (the raw differential) | **comparator of first resort** |
| treatment | `<family>_katz_diff` from the `team_stat` graph | the candidate |

The primary quantity is the **paired treatment-minus-control** delta. This
mirrors the discipline the engine document already declared for its own arms:
a treatment that does not beat its control on the same engine, same structure,
same window is not evidence the construction did anything. Baseline-relative
numbers are reported too, but the graph is not credited for what the raw
statistic already earned.

**Model class, evaluator, and blocking** are reused unchanged from
`scripts/graph_input_screen.py` (`fit_market_baseline`,
`fit_single_feature_market_residual_model`, `pick_correct`, and the week- and
season-blocked bootstrap) so treatment, control and the input screen's own
published numbers are commensurable by construction rather than by assertion.

## 5. Frozen structural hyperparameters, and the honest reason

`alpha = 0.85`, `half_life_weeks = 8.0`, `max_row_l1 = 1.0`,
`prior_weight = 1.0`, `min_games = 16`, `propagation = "signed_katz"`,
`injury_beta = 0.0`. Frozen here, before scoring, and not retuned on NFL.

The reasoning is stated rather than implied, because the convenient choice and
the honest one differ. The CFB grid (**read**, `docs/graph_ratings_v2.md` §6)
returned residual-arm coherence of −0.000287, −0.001013, −0.005681 and
+0.004119 across its four settings — every reading within ±0.006 of zero, so
**that grid did not resolve `half_life_weeks` for market-relative edges**;
picking its +0.004119 "winner" (`half_life_weeks = 16.0`) would be ranking
noise. The single structural setting CFB actually resolved is the raw-margin
control at `alpha = 0.85`, `half_life_weeks = 8.0` (+0.531023). Those are also
the module's own defaults. They are frozen on that basis.

CFB cannot fit a `team_stat` arm directly — the NFL statistic columns have no
CFB counterpart — so `cfb_structural_coherence` now **refuses** the arm with a
stated reason rather than returning a self-correlation between the rating diff
and the quantity its own edges were built from. The `team_stat` arm inherits
frozen structure; it does not refit it.

## 6. Window, grade, and recording

- **Grade.** The screen stage runs **close-graded**; no play/no-play or
  promotion decision may be settled on it. Any decision that could move a card
  is graded at the **opener**, per the binding "grade the decision at the
  opener" rule.
- **Window.** A rotation family is declared and its window **assigned by
  `nfl-ats rotation assign`**, never hand-picked, so the CLI computes the
  earliest eligible block at declaration time. The family is a variant of a
  prior line of work — MOD-15's PageRank/HITS schedule graph, recorded as
  `graph_schedule_rating_brier` and reclassified `unresolved_below_power` in
  `docs/closure_audit.md` — and the declaration says so.
- **Selection contamination, disclosed.** The 38 cluster representatives were
  chosen **by holdout `probability_positive` on 2020–2025 at the opener grade**
  (**read**, `docs/graph_input_screen.md`, "Representative selection per
  cluster"). That window has therefore already served as a selection surface
  for exactly these families, and no headline number for this screen may be
  reported on it. This is a disclosed discount, not a prohibition: windows
  retire per-family, and a reused window carries a stated discount, not a ban.
- **Instrument checks before the window is spent.** Two of them, run first,
  because a harness bug found after the look would have cost a window for
  nothing.
  - *Positive control*: the treatment feature is replaced by the realized
    `ats_margin`, a deliberate leak. An instrument that cannot see this is
    blind, and "no effect" from a blind instrument would mean nothing.
  - *Null*: settle margins shuffled **within week**, over **200 permutations**.
    Model fitting never sees the grading outcome, so the fitted models are
    reused across all 200 permutations and only the grade changes, which is what
    makes this affordable at all.

    Two corrections belong here, because both were wrong before they were
    measured. First, the original check used a **single** permutation and read
    its one draw of −2.53 accuracy points as a broken harness; one draw is not a
    test, and every family in a run shares it, so their deltas move together.
    Second, the 200-permutation version does **not** centre on zero, and that is
    a property of the design, not a defect: within-week permutation preserves
    each week's realized home-cover rate `c_w`, so an arm picking home at rate
    `h_w` has expected null accuracy `1 − h_w − c_w + 2·h_w·c_w`, making the
    expected null delta `2·mean_w[(h_w^treat − h_w^control)(c_w − 0.5)]`.
    **Measured** on the 2012–2014 smoke window, that closed form reproduces the
    Monte-Carlo null means to within ~0.3 points (−1.856 vs −1.892, −1.400 vs
    −1.266, −1.111 vs −0.842, −0.189 vs −0.216) — these arms carry large and
    differing home-pick rates (55–67% home against a 49.67% cover rate).

    The permutation null is therefore the **conservative** reference: it treats
    week-level home-tilt as noise. Every scored family reports it **alongside**
    the bootstrap-versus-zero interval the rest of the project uses, never
    instead of it, so this screen's numbers stay commensurable with every other
    entry in the registry while carrying the sharper reference too.
- **Recording.** Every family gets one `nfl-ats weak-signals record` entry
  carrying the paired treatment-minus-control effect in `accuracy_points`, its
  week-blocked interval, and `probability_positive`. `unresolved_below_power`
  unless the family's own interval is resolved entirely on the wrong side of
  zero at the grade that decides, in which case
  `refuted_mechanism`/`wrong_sign_resolved`. No promotion threshold is drawn.

## 7. Measured feature-space diagnostic (no outcome column touched)

Run at the section 5 frozen config over all 38 cluster representatives, on the
full merged feature table (4,902 rows, seasons 2009-2026). `r` is the Pearson
correlation between the propagated `*_katz_diff` and that family's own raw
`home - away` differential, over every game the graph rated. **No outcome column
is involved.** `P+ (screen)` is carried across from the input screen's holdout for
orientation only -- it is that screen's number, not this one's.

| family | r (graph vs raw) | n rated | P+ (screen) |
|---|---:|---:|---:|
| `bias_week2_anchor` | +0.004 | 4612 | 0.572 |
| `bias_playoff_holdover` | +0.037 | 4614 | 0.681 |
| `qb_start_probability` | +0.054 | 4598 | 0.502 |
| `gap_division_revenge` | +0.066 | 4614 | 0.455 |
| `gap_post_blowout_loss_bounce` | +0.087 | 4614 | 0.503 |
| `gap_post_blowout_win_letdown` | +0.097 | 4614 | 0.686 |
| `active_roster_continuity` | +0.144 | 4043 | 0.804 |
| `secondary_lineup_continuity` | +0.184 | 3514 | 0.591 |
| `special_teams_lineup_continuity` | +0.187 | 3514 | 0.518 |
| `skill_lineup_continuity` | +0.224 | 3514 | 0.723 |
| `front_lineup_continuity` | +0.230 | 3514 | 0.567 |
| `offense_lineup_continuity` | +0.236 | 3514 | 0.751 |
| `injury_secondary_unavailability` | +0.263 | 4322 | 0.551 |
| `injury_offensive_line_unavailability` | +0.276 | 4322 | 0.380 |
| `team_games` | +0.280 | 4598 | 0.737 |
| `injury_special_teams_unavailability` | +0.284 | 4322 | 0.422 |
| `injury_skill_unavailability` | +0.287 | 4322 | 0.702 |
| `injury_defense_disruption_value_lost` | +0.291 | 3239 | 0.602 |
| `injury_skill_epa_value_lost` | +0.300 | 3239 | 0.426 |
| `ats_residual` | +0.436 | 4554 | 0.600 |
| `def_sack_rate` | +0.509 | 4554 | 0.559 |
| `drive_seconds_per_drive_allowed` | +0.513 | 4554 | 0.766 |
| `pbp_start_yardline_100` | +0.514 | 4554 | 0.470 |
| `def_takeaway_rate` | +0.522 | 4554 | 0.849 |
| `drive_turnover_rate` | +0.544 | 4554 | 0.737 |
| `def_rush_epa_per_play` | +0.547 | 4554 | 0.325 |
| `pbp_drives` | +0.559 | 4554 | 0.691 |
| `def_yards_per_play` | +0.583 | 4554 | 0.711 |
| `qb_starter_experience_log` | +0.583 | 4598 | 0.353 |
| `pbp_def_explosive_rate_allowed` | +0.588 | 4554 | 0.487 |
| `off_rush_epa_per_play` | +0.600 | 4554 | 0.828 |
| `off_cpoe` | +0.648 | 4554 | 0.781 |
| `off_sack_rate` | +0.671 | 4554 | 0.695 |
| `pbp_pressure_allowed_rate` | +0.708 | 4554 | 0.840 |
| `pbp_off_pass_rate` | +0.723 | 4554 | 0.626 |
| `pbp_off_success_rate` | +0.734 | 4554 | 0.709 |
| `pbp_matchup_explosive_rate` | +0.773 | 4582 | 0.501 |
| `active_roster_mean_experience` | +0.779 | 4057 | 0.376 |

**All 38 representatives built** (the five suffix-form families are why
`signal_column_pair` exists). Median |r| = **0.368**, range **0.004 to 0.779**.

**What this settles, and what it does not.** It settles the only question that
could have made the whole lane pointless: the graph is not reproducing the raw
differential. At the median the propagated rating shares about 14% of its variance
with the statistic it came from, and for eight families less than 5%. Whether that
different information is *useful* information is exactly what the declared
comparison in section 4 measures, and nothing here anticipates its sign. The
families where propagation changes the most are, in general, the sparse situational
and continuity ones; the play-by-play rate statistics move the least, which is what
I would expect from quantities that are already close to schedule-neutral -- an
inference, not a measurement.

## 8. Results (added after the look, 2026-08-26)

One look, close-graded, on the rotation-assigned window **[2011, 2013]** -- 746 games, 51 weeks, all 38 cluster representatives. Recorded as 38 `weak-signals` entries under family `graph_ratings_v2_team_stat` and as one `rotation record` look (verdict `unresolved`), which spent that window. Artifact: `artifacts/graph_team_stat_screen/20260826T175934Z/results.json`.

### The answer

**Against zero** -- the reference the rest of the registry uses -- the graph arm leans slightly negative: mean paired delta **-0.123** accuracy points, median -0.134, positive in **12 of 38** families. The random-effects pool of the 38 commensurable cells is **-0.107 points, 95% [-0.424, +0.209]**, `excludes_zero: false`, tau-squared 0.133.

**Against each family's own permutation null** -- the reference section 6 built, which removes the home-tilt artifact -- the arm is a **coin flip**: median percentile **50.2**, **19 of 38** above their own null's centre, sign test **p = 1.000**, null-adjusted mean **+0.007** points. The mean null offset across families is -0.131 points, which is essentially the whole of the -0.123 apparent negative lean.

So the honest reading is not that schedule-adjusting a statistic hurts. It is that **at this window's ~2-point resolution the transform is indistinguishable from using the statistic raw**, and the small negative lean a naive zero-reference would have reported is an artifact of the arms' differing home-pick rates, not of the graph.

### What this implies for the decision, before what is wrong with it

Nothing here earns an opener window on its own. The family stays **open** (`unresolved_below_power` for all 38 entries, two eligible close windows remaining) because neither admissible closing ground is met: no interval is resolved wrong-sign at the grade that decides, and no positive control has been shown able to detect an effect this size and found it absent. If a future session does spend an opener look here, the predeclared candidates are the three cells that lead by the conservative reference, not the one that leads by the naive one.

### Per-family, ranked by percentile against its own null

`delta` is treatment minus control in accuracy points; `P+` is the week-blocked bootstrap fraction favouring the graph arm **against zero**; `null mean` is where that family's within-week permutation null centres; `pctile` is where the observed delta sits inside its own null.

| family | delta | P+ (vs 0) | week 95% CI | null mean | pctile |
|---|---:|---:|---:|---:|---:|
| `def_yards_per_play` | +2.145 | 0.965 | [-0.134, +4.589] | +0.279 | 95.5 |
| `injury_skill_unavailability` | +1.340 | 0.949 | [-0.134, +3.146] | +0.200 | 93.5 |
| `off_sack_rate` | +2.949 | 0.987 | [+0.401, +5.707] | +1.227 | 92.5 |
| `pbp_matchup_explosive_rate` | +1.609 | 0.879 | [-0.941, +4.320] | -0.105 | 92.5 |
| `pbp_def_explosive_rate_allowed` | +1.340 | 0.884 | [-0.799, +3.495] | -0.113 | 92.0 |
| `active_roster_mean_experience` | +1.609 | 0.954 | [-0.134, +3.577] | +0.532 | 83.0 |
| `gap_division_revenge` | -0.536 | 0.257 | [-2.310, +1.319] | -1.233 | 83.0 |
| `pbp_off_success_rate` | +0.670 | 0.698 | [-1.757, +3.196] | +0.078 | 68.5 |
| `drive_seconds_per_drive_allowed` | -0.402 | 0.420 | [-3.705, +3.003] | -1.167 | 67.0 |
| `pbp_pressure_allowed_rate` | +0.670 | 0.762 | [-1.199, +2.594] | +0.185 | 67.0 |
| `pbp_start_yardline_100` | +0.000 | 0.488 | [-3.729, +3.822] | -0.506 | 64.0 |
| `off_cpoe` | +0.000 | 0.477 | [-2.378, +2.775] | -0.296 | 60.0 |
| `qb_starter_experience_log` | +0.938 | 0.716 | [-2.041, +4.540] | +0.765 | 55.5 |
| `off_rush_epa_per_play` | +1.609 | 0.911 | [-0.669, +3.963] | +1.450 | 53.5 |
| `skill_lineup_continuity` | +0.268 | 0.729 | [-0.528, +1.064] | +0.121 | 53.5 |
| `qb_start_probability` | -0.938 | 0.232 | [-3.964, +1.854] | -0.987 | 52.5 |
| `pbp_off_pass_rate` | +0.268 | 0.595 | [-1.852, +2.394] | +0.115 | 52.0 |
| `front_lineup_continuity` | +0.000 | 0.442 | [-0.942, +1.051] | -0.056 | 51.5 |
| `def_rush_epa_per_play` | -1.475 | 0.139 | [-4.156, +0.945] | -1.474 | 51.0 |
| `gap_post_blowout_loss_bounce` | -0.670 | 0.233 | [-2.677, +1.217] | -0.675 | 49.5 |
| `ats_residual` | -0.938 | 0.289 | [-4.112, +2.168] | -0.960 | 48.0 |
| `bias_week2_anchor` | -0.268 | 0.435 | [-3.059, +2.443] | -0.214 | 45.0 |
| `pbp_drives` | -1.340 | 0.229 | [-5.266, +2.375] | -1.162 | 43.0 |
| `bias_playoff_holdover` | +0.000 | 0.492 | [-2.699, +2.527] | +0.082 | 41.5 |
| `team_games` | -0.134 | 0.352 | [-1.203, +0.821] | -0.092 | 37.5 |
| `def_takeaway_rate` | -1.340 | 0.252 | [-4.973, +2.397] | -0.896 | 36.0 |
| `gap_post_blowout_win_letdown` | -0.402 | 0.359 | [-3.106, +2.854] | +0.009 | 32.5 |
| `secondary_lineup_continuity` | -0.268 | 0.347 | [-1.845, +1.207] | -0.160 | 32.5 |
| `injury_secondary_unavailability` | +0.000 | 0.474 | [-1.995, +2.087] | +0.421 | 27.5 |
| `special_teams_lineup_continuity` | -0.134 | 0.358 | [-1.089, +0.806] | +0.267 | 27.0 |
| `injury_offensive_line_unavailability` | -0.268 | 0.355 | [-2.674, +1.992] | +0.394 | 21.5 |
| `offense_lineup_continuity` | -0.134 | 0.390 | [-1.221, +1.070] | +0.241 | 20.0 |
| `active_roster_continuity` | -3.485 | 0.002 | [-6.768, -0.798] | -1.498 | 4.5 |
| `injury_skill_epa_value_lost` | -0.670 | 0.031 | [-1.463, +0.133] | +0.174 | 4.5 |
| `def_sack_rate` | -1.743 | 0.028 | [-3.774, +0.132] | -0.199 | 2.5 |
| `injury_defense_disruption_value_lost` | -1.072 | 0.013 | [-2.234, +0.000] | +0.326 | 1.0 |
| `injury_special_teams_unavailability` | -1.743 | 0.089 | [-4.321, +0.922] | +0.006 | 1.0 |
| `drive_turnover_rate` | -2.145 | 0.025 | [-4.228, +0.003] | -0.050 | 0.5 |

### The three readings worth carrying forward

- **`off_sack_rate`** is the only family whose interval excludes zero on the positive side (**+2.949** points, P+ 0.987, [+0.401, +5.707]). Against its own null it sits at the 92.5th percentile, because that null centres at +1.227 -- so roughly 40% of the headline number is the artifact, not the transform.
- **`def_yards_per_play`** leads by the conservative reference (**95.5th** percentile, +2.145 points against a null centred at +0.279) despite ranking second against zero. This is the ordering the null reference changes.
- **`off_rush_epa_per_play`** reads P+ 0.911 against zero and sits at the **53.5th** percentile of its own null (+1.609 observed, +1.450 null centre). Essentially all of its apparent edge is the artifact. This single row is the clearest argument for having built the permutation reference at all.

### Disclosed limitations

- **38 uncorrected cells.** No multiplicity correction is applied. Four families sit at or below the 2.5th percentile of their own null and one at or above the 95th; with 38 draws, roughly one in each tail is what chance alone produces, so the individual tails are not findings.
- **The cells are not independent.** All 38 are scored on the same 746 games against the same market baseline, so the pooled interval overstates precision. The per-family rows are the safer read, exactly as the registry's own pooling note warns.
- **Two degenerate arms.** `special_teams_lineup_continuity` and `injury_skill_epa_value_lost` hit training folds with no observed values (sklearn skipped the feature), so those arms partly collapse to the market baseline and their near-zero deltas are partly mechanical. Disclosed rather than excluded.
- **A disclosed prior look.** A one-family, one-season plumbing smoke test (`def_takeaway_rate`, 2012) ran before the window was assigned, to verify the harness end to end. Its number changed no design decision -- every one of them was frozen in sections 1-7 first -- and the fact is carried in the rotation family's declaration.
- **The window overlaps one season of the input screen's own close-graded selection window** (2013, of 2013-2019). Windows retire per-family and a reused season carries a stated discount, not a ban; this is that statement.
- **Close grade only.** Nothing here may settle a play/no-play decision, and no terminal classification is drawn at this grade.
