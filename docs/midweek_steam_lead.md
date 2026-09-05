# LEAD-04 midweek steam refresh overlay

## Predeclaration — 2026-09-05, frozen before outcome scoring

Construct (inferred): coordinated spread moves across at least three books may
carry information playable by following their direction after Tuesday. This is
a refresh overlay on the production card; the grading line stays frozen at the
Tuesday opener. No coefficients, thresholds or direction are fitted.

Population: 2023–2025 regular-season games in the local intraday_hourly archive,
joined by game ID to the active-model chronological opener evaluation. Retain
all paired non-push games, including games without steam (production no-op).
Use the existing production four-member union of coach fade, division revenge,
player arrests and spread-gap fade, with each evaluated against the original
probability-rule pick (HOME when probability >= 0.5), complementing once on the
union. Reuse the historical arrest adapter from overlay_composition. Preserve
chronological prediction-level output and report each season separately. This
is a reused-window screen, not a new held-out confirmation; production subset
selection on the archive inflates its own retrospective baseline.

Event definition: spread-line changes relative to the same book's prior
observed quote, deduplicated across outcome rows. Require at least three
distinct books with the same direction within a trailing inclusive 60-minute
observation window. Each counted provider update must be later than its prior
observation, no later than capture, and within the final capture's preceding
60 minutes. This imposes both observation and provider-clock bounds; sampled
quotes cannot recover intervening unobserved reversals. A qualifying current
arrival emits one event per game/time/direction, even if a prior qualifying
window overlaps. Opening quotes do not count. Conflicting duplicates error.
Only Wednesday–Saturday Eastern in the game's own week qualify; earlier-week
quotes for upcoming games are excluded. No minimum move size beyond nonzero.

FOLLOW direction: home_spread_line is the required home winning margin, the
negative of the home handicap (read: src/nfl_ats/market_data.py:153–157).
An increase follows HOME; a decrease follows AWAY. Latest observable event
wins; simultaneous opposing qualifying directions cause a no-op. Apply only
events strictly before min(kickoff, that week's Sunday 16:00 Eastern), and no
later than the refresh's as_of. Monday night uses the preceding Sunday.
Missing steam is a no-op, never backfilled with future quotes. Historical
screen uses the final permissible refresh. No live publication wiring here.

Primary comparison: production-plus-follow against production, paired on all
eligible games, units accuracy_points (100 times mean correctness difference).
Descriptive: steam-side opener cover rate on flagged non-push games, with its
own interval and probability_positive against 50%; never pool that subset
quantity with the full-population overlay increment.

Uncertainty: resample season/week blocks, 20,000 draws, seed 2026090516,
percentile 95% intervals, bootstrap standard error and probability_positive
(fraction of resampled effects > 0), using the existing
overlay_composition.blocked_bootstrap_matrix implementation. Also report
raw-model probability-rule and opener home-favorite correctness as baselines.

Oracle positive control: replace steam with the CLOSING move direction relative
to Tuesday, HOME on a positive standardized close-minus-open move, AWAY on a
negative move, production unchanged for zero/missing move. Score at the opener
on the same game population and report its paired interval/probability_positive.
This uses deliberately future information and is never a feature or live rule;
it tests the movement instrument, not a proven candidate-sized detection bound.

Reliability: for each team-season, proportion of its games exposed to qualifying
pre-cutoff steam (any direction, including conflicting events), separately for
odd and even scheduled weeks; Pearson correlation across team-seasons with
both halves. No outcome enters exposure. Undefined correlation is reported as
undefined, not zero reliability and not a closing ground. Report per-season
correlations too. Coverage includes all 272 scheduled games per season.

Binding taxonomy for every verdict (verbatim):

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: (1) refuted mechanism - a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is unresolved_below_power: record it, report probability_positive, never the binary "contains zero". If a record command errors, the verdict is wrong, not the validator. Decisions are expected value: probability_positive above 0.5 favours playing it as a refresh overlay; state what the result implies for the DECISION before what is wrong with it. Never say a lead needs more games; the data is fixed and the project is model-limited.

## Coverage measured before scoring

Measured (`python scripts/midweek_steam_on_production.py --mode coverage`,
`artifacts/experiments/midweek_steam/coverage.json`): 6,966 hourly manifests,
zero missing quote files; each season has 432 Wednesday, 414 Thursday, 432
Friday and 414 Saturday manifests. Game coverage refers to the game's own week.

| Season | Midweek games | Game/snapshot pairs | Events | Before cutoff | Games with pre-cutoff events | Latest unambiguous flag |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 272 | 24,574 | 1,215 | 1,132 | 244 | 239 |
| 2024 | 272 | 24,412 | 565 | 476 | 195 | 194 |
| 2025 | 272 | 24,475 | 568 | 477 | 197 | 196 |

Measured (same artifact): 269 otherwise qualifying events preceded the game's
own midweek and were excluded. Repeated event windows are not independent
game observations. Per-day distinct games Wednesday/Thursday/Friday/Saturday:
2023 272/272/253/252; 2024 272/270/251/249; 2025 272/272/251/249.

## Measured results — 2026-09-05

Decision (inferred): I think the pooled expected-value comparison favours keeping
production rather than applying this frozen refresh universally; 2025's separate
estimate favours the refresh, so the screen leaves era dependence unresolved.
Measured (`artifacts/experiments/midweek_steam/results.json`, generated by
`python scripts/midweek_steam_on_production.py --mode score`): production gets
431/799 (53.9424%) correct, the refresh 413/799 (51.6896%), with 278 non-push
flips. Increment **-2.252816 accuracy points**, 95% [-6.091371, +1.639344],
standard error 1.984387, **probability_positive=0.11835**, 54 week blocks.
All 816 hourly-covered games paired; 17 opener pushes excluded from accuracy.

| Season | Non-push games | Production | Refresh | Increment, accuracy points | 95% interval | probability_positive |
|---|---:|---:|---:|---:|---|---:|
| 2023 | 266 | 58.6466% | 49.6241% | -9.022556 | [-15.730337, -2.290076] | 0.00205 |
| 2024 | 266 | 50.0000% | 48.8722% | -1.127820 | [-6.463878, +5.303030] | 0.31115 |
| 2025 | 267 | 53.1835% | 56.5543% | +3.370787 | [-2.952030, +9.523810] | 0.83730 |

Measured (same results artifact): latest unambiguous steam side covers on
51.4563% of 618 flagged non-push games, 95% [48.1132%, 55.0000%],
probability_positive versus 50%=0.79365. This descriptive cover rate has a
different denominator and estimand from the production increment.

Measured (same artifact): the closing-direction oracle scores 55.3191%,
increment +1.376721 accuracy points versus production, 95% [-2.405063,
+5.334988], probability_positive=0.73905. Raw model probability-rule accuracy
is 52.8160%; opener favorite accuracy is 52.6909% on the same paired population.
These are retrospective measurements, not prospective performance estimates.

Measured (`reliability.parquet` and `results.json`): odd/even team-season steam
exposure reliability is 0.385943 over 96 team-seasons; per season 0.340701 /
0.108657 / 0.240855. Exposure includes any qualifying pre-cutoff event, even
when simultaneous conflicting directions prevent a pick flag.

Measured (`nfl-ats weak-signals record`, success): recorded
`midweek_steam_follow_refresh_on_production_cx16` in
`registry/weak_signals.json`, units accuracy_points, family
`midweek_steam_refresh_on_production`, classification `unresolved_below_power`.
No whole-population resolved wrong sign, zero reliability, or candidate-sized
positive-control bound was measured; the per-season table does not redefine
the predeclared pooled family. The descriptive subset result is not pooled
as an independent overlay increment.

Read (results provenance): active model `ab29832a4e099766`, opener input
`artifacts/opener_evaluation/20260905T194919Z/per_game.parquet`; SHA256
`a0e9561d66297d734cde79b29a18cdc7d63b7470aebd26620b5067a9efc5e7ed`.
The harness verifies the opener evaluation matches the active recipe and
feature-table digest. The historical production union reuses the original
arrest adapter; no Tuesday snapshot is fabricated from later steam.

Measured (targeted leakage fixtures): the initial DST fixture caught adding
16 elapsed hours across the autumn clock transition; localization now occurs
after constructing Sunday 16:00 wall time. The test also covers future refresh
observations, events at/after early kickoff, earlier game weeks, Sunday events,
duplicate books, stale/future provider updates, and conflicting directions.
