# Big-spread Best-Pick eligibility challenger

## Decision

`best_pick_big_spread_eligibility` is registered `ACTIVE_PROSPECTIVE` as a
side-ledger-only challenger; the published/played Best Pick remains the live v2
nomination [read: `artifacts/prospective/challengers.json`, challenger
`best_pick_big_spread_eligibility`]. The challenger is invoked only by
`publish-predictions --record-decisions`; ordinary publishing records neither
this challenger nor any other decision ledger [read: `src/nfl_ats/cli.py`,
`_cmd_publish_predictions`].

I think this is the smallest honest action implied by the source result: accrue
an independent 2026 comparison without spending another retrospective window
or presenting a mined subgroup as a confirmed policy effect [inferred from the
source/result mismatch described below].

## Source evidence and its limit

The decision-relevant project baseline is **53.36%** under the production
probability rule at the opener grade on 1,503 push-excluded games from the
1,537-game paired 2020-2025 archive [read: `ROADMAP.md:34-45` and
`docs/opener_error_analysis.md`, “Method”]. The 52.10% historical field in
`artifacts/active_ats_model.json` is a different grade/quantity and does not
supersede that baseline [read: `artifacts/active_ats_model.json`,
`historical_evaluation`; `ROADMAP.md:34-45`].

The source battery measured the active production-rule read at **45.45%** in
the `abs(opener spread) >= 10` cell: 154 games, **-7.9054 accuracy points**
versus 53.36%, week-blocked interval **[-15.1434, -0.5298]**, and
`probability_positive = 0.0200` for that cell beating the baseline [read:
`registry/weak_signals.json:opener_error_mining_spread_magnitude_10plus`]. The
same battery measured the 7-9.5 cell at -4.38 points with
`probability_positive = 0.090`, while the two smaller buckets leaned positive
[read: `docs/opener_error_analysis.md`, “Full results table”].

The source result is `unresolved_below_power`, from one subgroup in a mined,
uncorrected-multiplicity battery [read:
`registry/weak_signals.json:opener_error_mining_spread_magnitude_10plus`]. It
does **not** measure the weekly Best-Pick policy head-to-head; translating a
game-level error cell into a nomination rule could yield a smaller, zero, or
opposite week-level effect [inferred]. The challenger therefore makes no
historical improvement claim and awaits prospective paired evidence [read:
`artifacts/prospective/challengers.json`, this challenger’s
`prospective_protocol`].

## Frozen rule

The rule starts with v2’s existing below-median cross-book-dispersion pool and
its alpha=2000 candidate probabilities [read:
`src/nfl_ats/best_pick_nomination.py`, `nominate_v2`]. It then:

1. excludes every v2-eligible game with
   `abs(decision_home_spread) >= 10.0`;
2. uses v2’s unchanged `candidate_dist` ranking and unchanged
   dispersion/game-id tie-break among the remaining games; and
3. falls back to the unmodified v2 pool if all v2-eligible games are 10+, so
   the forced weekly Best Pick is never omitted.

Those three behaviours are read directly from
`src/nfl_ats/best_pick_big_spread_challenger.py` and are pinned at both signs
of the 10-point boundary in
`tests/test_best_pick_big_spread_challenger.py` [read]. The 10.0 cutoff is the
source battery’s existing bucket edge; it was not retuned after examining a
new outcome window [read: `docs/opener_error_analysis.md`, “Predeclared cells”
and “Ranked leads,” item 2].

## Leakage and decision safety

The selector reads only `game_id`, the card’s pregame `spread_line`, and v2’s
already-computed pregame eligibility/ranking table; changing synthetic
postgame `result` and `home_cover` columns cannot change its nominee [read:
`src/nfl_ats/best_pick_big_spread_challenger.py`,
`apply_big_spread_eligibility`; measured this session:
`test_postgame_columns_cannot_change_nomination_and_inputs_are_unchanged`
passed]. Non-finite spreads, duplicate games, incomplete v2 tables, and
dropped/duplicated joins fail closed [read: the same function].

The recorder requires an active registration, matching base-model
configuration fingerprint, synchronized forecast, finite decision spread,
valid kickoff, the repository’s seven-day recording window, and every game in
the week still pre-kickoff [read:
`record_big_spread_nomination_challenger_decisions`]. Existing rows under this
challenger id are append-only and never rewritten [read: the same function;
measured this session:
`test_recorder_writes_only_the_alternative_nominee_and_never_rewrites` passed].

The recorder writes exactly one row containing the active card’s unchanged
pick side and decision line; `bet_side` is `PASS` and `edge` is missing because
this is a forced-pick nomination comparison, not an invented betting edge
[read: `record_big_spread_nomination_challenger_decisions`]. Neither
`nfl_ats.publishing` nor `nfl_ats.card_view` imports the challenger, so it
cannot alter the published mark or played card [read:
`src/nfl_ats/publishing.py`, `src/nfl_ats/card_view.py`, and the challenger
module’s import sites].

## Prospective read

At each lock-day run, use the existing command [read:
`artifacts/prospective/challengers.json`, this challenger’s
`weekly_recording_command`]:

```powershell
.\.tools\uv.exe run nfl-ats publish-predictions --record-decisions
```

Score the challenger’s one weekly row against `best_pick_nomination_v2` on
paired prospective weeks, with the recorded decision/opener line primary and
close secondary [read: `artifacts/prospective/challengers.json`, this
challenger’s `prospective_protocol`]. Report `probability_positive` at every
checkpoint; do not turn low resolution into a terminal negative without one
of the closing grounds admitted by `AGENTS.md` [read: `AGENTS.md`, “An interval
crossing zero is NOT grounds for rejection”].

A read-only 2026 Week 1 dry-run against synchronized forecast artifact
`artifacts/margin_predictions/2026-week-01-20260820T005017Z` nominated
`2026_01_MIA_LV` under both live v2 and this challenger; the v2 pool contained
no 10+ candidate, so `excluded_game_ids=[]` and `fallback_to_v2=false`
[measured this session: direct `nominate_big_spread_challenger` call using the
artifact’s recorded feature table and local Tuesday market store]. No ledger or
published-card write occurred in that dry-run [measured this session: the
read-only command loaded artifacts and printed the result only].
