# Should the played overlay policy change? (twelve-member re-run, 2026-08-25)

Owner question: *"why aren't we currently making the highest EV bet based on our
data — surely there are challengers that add meaningful EV at confidence >= our
current model?"*

Right question, and it forced a correction. The short answer is that the played
policy already **is** the best composition this project has evidence for, and
nothing in the current challenger pool survives an honest forward test as an
addition to it.

## Correction recorded first

Mid-analysis this session I asserted that the four-member subset
`coach + division_revenge + arrests + spread_gap_zone` was measured best but
was **not played**, and recommended adding `spread_gap_zone_fade_overlay` to
production. **That was wrong.**

`nfl_ats.clv._FOUR_OVERLAY_POLICY_ID` is
`overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`, and
`record_paper_decisions` resolves it whenever
`require_fresh_arrest_overlay=True` — which `cli._cmd_publish_predictions`
always passes. **Measured** by running the real recorder against the real
active model in this session's lock-day rehearsal: the recorded policy is that
four-member union, `spread_gap_zone_flip_count = 1` on the live Week 1 card.

The error came from reading `a_incumbent_chain` in
`artifacts/max_ev_composition/.../arms_summary.csv` as "what ships". It is the
**former** coach→arrests chain — which is exactly why
`overlay_production_chain_coach_arrest_incumbent` exists as its paired control.
Lesson, and it is the one `AGENTS.md` already states: verify what production
does by running production's own code path, not by reading a study's label for
its baseline.

## What is actually played (measured, n=1,503 opener games)

| policy | opener accuracy | vs raw model |
|---|---|---|
| raw model (`weak_stack`, probability rule) | 53.3599% | — |
| **played four-member union** | **55.4225%** | **+2.063 pts** |

That matches `docs/overlay_subset_composition.md`'s own row for
`coach+divrev+arrest+sgz` exactly (55.4225%, +2.063), i.e. production is the
in-sample argmax of that study's 127 subsets.

## Re-run with twelve members (4,095 subsets)

Five pick-flipping overlays have been registered since the 2026-08-21 study, so
`scripts/overlay_subset_holdout_v2.py` re-runs the same design over twelve
members. Predeclared decider, stated in that script before it produced output:
**choose on 2020-2022 only, apply unchanged to 2023-2025.** The reverse split is
a stability check, never a second chance to pick a winner.

### Wholesale subset selection got WORSE, not better

| | 7 members / 127 subsets (2026-08-21) | 12 members / 4,095 subsets (this run) |
|---|---|---|
| shrinkage factor (OLS slope) | 0.636 | **0.593** |
| rank stability (Spearman) | 0.721 | **0.617** |

Forward split, this run: the selected subset scores **+0.626 pts** on the
holdout while the *former* two-member chain scores **+0.876** on the same 799
games. Widening the candidate pool bought more selection noise than signal —
more subsets means a higher in-sample maximum and a worse out-of-sample one.

The naive "play everything" control is decisive against itself: all twelve
members on the holdout score **−2.128 points** (501 of 799 games flipped).

## The decision-relevant test: add ONE member to the PLAYED policy

Ten candidates instead of 4,095, so far less selection noise. Ranked on
2020-2022 **only**:

| candidate | selection half | holdout |
|---|---|---|
| `interim_hc_first_game_tilt_overlay` (**frozen choice**) | +0.142 | **+0.000** (0 games changed) |
| `forecast_weather_kn_warm_team_cold_late_tilt` | +0.000 | +0.000 |
| `forecast_weather_kn_precip_high_total_tilt` | +0.000 | +0.000 |
| `forecast_cold_visitor_tilt` | −0.142 | +0.501 |
| `surface_switch_tilt_overlay` | −0.284 | −1.627 |
| `pbp08_protection_mismatch_tilt_overlay` | −0.284 | +1.001 |
| `backup_qb_fade_overlay` | −0.994 | −1.377 |
| `injury_value_lost_tilt_overlay` | −3.835 | −1.252 |

**No addition helps.** The only candidate with a positive selection-half
marginal fires on five games in the whole archive and zero in the holdout.

**Two numbers deliberately not acted on.** `pbp08_protection_mismatch` reads
+1.001 (week P+ 0.8978) and `forecast_cold_visitor` +0.501 (P+ 0.9039) on the
holdout — but both ranked NEGATIVE on the selection half. Choosing either now
would be selecting on the holdout, the exact error this design exists to
prevent. Both stay challenger-tracked; 2026 supplies an independent read. They
are recorded here so the temptation is visible rather than buried.

## Three active challengers are negative IN COMPOSITION

Consistent across both halves, as additions to the played policy:

| challenger | selection | holdout | holdout week P+ |
|---|---|---|---|
| `injury_value_lost_tilt_overlay` | −3.835 | −1.252 | 0.2061 |
| `surface_switch_tilt_overlay` | −0.284 | −1.627 | 0.0220 |
| `backup_qb_fade_overlay` | −0.994 | −1.377 | 0.1317 |

This closes **nothing** about any of them standing alone: no interval sits
entirely on the wrong side of zero on both blockings, no split-half reliability
was measured here, and no positive control was run. Per the binding taxonomy
these remain `unresolved_below_power`. What is measured is narrower and still
useful — *as additions to this particular four-member policy, on this archive,
they subtract.* They cost nothing today because they are recording-only.

## Disclosures

* **Attribution ceiling.** Every member was registered on windows overlapping
  this archive, so even the holdout half is virgin only for the SUBSET CHOICE,
  never for the components. These are honest estimates of the selection
  procedure, not fresh confirmations of any overlay.
* **Grade mismatch on one member.** `forecast_weather_kn_precip_high_total_tilt`
  reads `total_line`, and the schedules snapshot's `total_line` is the CLOSING
  total while the spread leg here is opener-graded. Inherited from the
  registered signal, which reads the same field; disclosed, not introduced.
* No rotation-registry window was spent.

## Decision

**No production change.** The played four-member union stays. That is not
caution and not a threshold refusal — it is the measured result: the best
available addition changes zero holdout games, and the wholesale re-selection
underperforms what is already shipped.

The owner's premise was right and the project's own rule stands: a signal with
`probability_positive` above 0.5 is worth playing. What this run adds is that
**composition is a separate decision from the signal.** An overlay can be
positive alone and negative on top of four others that already flip overlapping
games, and choosing among many compositions is itself an act with its own
error — measured here at a 0.59 shrinkage factor.
