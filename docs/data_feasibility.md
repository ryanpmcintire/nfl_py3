# Historical data feasibility

A source is not promoted merely because it has many rows. NFL games and weeks
are the effective experimental units; plays and players within the same game
are correlated. A feature also needs a defensible availability time and enough
outer seasons to distinguish a repeatable gain from one favorable season.

## Admission tiers

- **High:** at least eight usable seasons and roughly 2,000 games, with a
  defensible pregame or strictly lagged construction. Suitable for primary
  walk-forward and outer-season evaluation.
- **Medium:** five to seven seasons, a material source-regime change, or limited
  player coverage. Suitable for a frozen exploratory candidate, not broad
  tuning.
- **Low/blocked:** fewer than five seasons, no defensible historical timestamp,
  or an effective sample that is sparse at the modeled interaction level.
  Collect prospectively or simplify the hypothesis before modeling.

These are minimum research gates, not guarantees of predictive value.

## Verified August 2026 inventory

| Lead/source | Actual usable coverage | Rows audited | Availability semantics | Tier and decision |
|---|---:|---:|---|---|
| Schedules, closes, team stats, PBP | 2009–2025 | 4,703 canonical games; 781,712 stored PBP rows | Completed-game state is strictly lagged; schedule line is a historical close | High for closing-line and football-state research |
| Injury/practice reports | 2009–2024 | 84,684; every advertised season nonempty | `date_modified` is UTC and varies within the game week; require `<=` decision cutoff | High historically; live source replacement required after 2024 |
| Weekly rosters | 2002–2025 | 906,378; 24 nonempty seasons | Week-level, without an observation timestamp | High for conservative prior-week continuity; not last-minute status |
| Historical depth charts | 2001–2024 | 869,185 rows before the source change | Week-level, without an observation timestamp | High only with a conservative prior-week rule |
| Timestamped depth charts | 2025 | 554,215 appended snapshots | ISO-8601 observation timestamp | Low for retrospective estimation; valuable prospectively |
| Player-game snap counts | 2013–2025 | 324,611; advertised 2012 file is empty | Realized game outcome; may affect later games only | High for lagged role/value and roster-continuity models |
| Weekly player production | 2009–2025 | 291,747 canonical player-game rows | Realized game outcome; joined to snap-weighted state only after its game | High for low-dimensional lagged injury-value proxies; box scores do not isolate causal value |
| Play participation/personnel | 2016–2025 | 478,989 plays; ten nonempty seasons | Retrospective outcome data; NGS through 2022, FTN from 2023 | High for low-dimensional unit/formation effects; medium for player interactions |
| Next Gen Stats | 2016–2025 | 5,933 passing; 14,731 receiving; 6,059 rushing weekly rows | Published after games and thresholded to qualifying players | Medium; use as lagged player priors with availability indicators |
| Free multi-book opener/close sample | 2025 | One season | Stage is known, but full quote timestamps are unavailable | Low/blocked for edge claims; sufficient only to test ingestion and normalization |
| Locally archived live odds | Collection era forward | Grows prospectively | Book-specific `observed_at` timestamp | Initially low; becomes the correct line-movement dataset over time |
| Final observed weather | Historical schedule era | Game-level | Retrospective realized weather, not a forecast snapshot | Blocked as a decision-time weather feature |

The row totals above were read from the actual nflverse releases rather than
inferred from advertised start years. In particular, the 2012 snap-count file
loads successfully but contains zero rows, and the 2025 depth-chart source has
a different timestamped schema that must not be concatenated silently with the
older weekly schema.

## Highest-priority external audit: college football

College football is the most promising way to add independent evidence and
learn shared player mechanisms, not a license to append CFB rows to the NFL
training table. The maintained
[cfbfastR archive](https://github.com/sportsdataverse/cfbfastR-data) advertises
play-by-play from 2002 onward. The newer
[SportsDataverse CFB pipeline](https://github.com/sportsdataverse/cfbfastR-cfb-data)
defines play participants, game rosters, player/team boxes, betting, injuries,
recruiting, returning production, and resolved NFL draft links. The
[CollegeFootballData API](https://api.collegefootballdata.com/getting-started)
offers a free-key route to games, plays, lines, rosters, player usage,
recruiting, returning production, and draft data.

Those are advertised surfaces, not yet verified local coverage. XLG-01 must
measure every season and source regime before ingestion is admitted. In
particular, it must establish whether:

1. betting rows represent an opener, close, or unidentified resolved line;
2. injury observations carry a genuine pregame timestamp and revision history;
3. an absent injury record means healthy, uncovered, or unavailable source;
4. game rosters and play participants identify non-participation rather than
   merely omit players below a recording threshold;
5. player identities can be linked through college seasons and into NFL draft
   and roster IDs without ambiguous name-only joins; and
6. historical redistribution and API retention are allowed.

If injury timestamps fail, CFB can still estimate realized position-specific
role loss, player/replacement value, matchup mechanisms, and rookie priors. It
cannot then train a defensible historical probability-of-playing model. CFB
benchmarks remain league-specific, and all claims of NFL improvement are scored
on matched NFL-only outer weeks.

## Lead-level implications

### Build now

1. **Injury and practice state.** Sixteen seasons are enough for a restrained
   position-weighted model and outer-season evaluation through 2024. Preserve
   `date_modified` and exclude observations after the chosen decision cutoff.
2. **Snap-weighted player and unit value.** Thirteen seasons support shrunk QB,
   offensive-line, receiver, secondary, pass-rush, and special-teams priors.
   Current-game snaps are outcomes and never enter their own pregame features.
3. **Roster continuity and offseason priors.** Twenty-four seasons of weekly
   rosters can measure returning snaps, unit turnover, rookie/experience mix,
   and position-group continuity.
4. **Joint score/total distributions.** The existing 2009–2025 game sample is
   sufficient for low-dimensional distributional comparisons now.

### Build only after simpler player state

- Participation and personnel provide about ten seasons and 2,700 effective
  games. That is enough for shrunk position-unit and formation effects, but the
  478,989 plays are not independent observations.
- A universal receiver-versus-corner interaction is too high-dimensional as a
  first model. Start with speed/usage/coverage-style archetypes or hierarchical
  position effects, then admit named player pairs only when repeated-matchup
  counts pass a declared threshold.
- Next Gen Stats begins in 2016 and excludes players below publication
  thresholds. It should complement, not replace, PBP and snap-based priors.

### Collect rather than backtest broadly

- One free season of opener/close data cannot establish a modern line-movement
  edge or public-bias effect. Preserve new book quotes prospectively and use the
  existing sample only for pipeline validation.
- Precisely timestamped depth revisions begin in 2025. They can power frozen
  future forecasts, but not a convincing multi-season historical comparison.
- Forecast-time weather needs an archived forecast source. Realized game
  weather would leak information about what was known at the decision time.

## Required audit before each experiment

Every new feature proposal must record:

1. first and last nonempty season;
2. nonempty seasons and source/schema regimes;
3. games and weeks with both teams covered;
4. repeated observations at the modeled player/unit interaction level;
5. the earliest defensible availability timestamp;
6. whether missingness means healthy/absent, below a publication threshold, or
   an unavailable source;
7. the outer seasons reserved for final evaluation.

If those facts do not support the proposed model complexity, reduce the model
or collect more data rather than treating thousands of correlated rows as
independent evidence.

## First player-value screen

The August 2026 player-family screen makes an important distinction. The
original 52.05% full-player result was not primarily an injury-report result:
injury-only reached 51.28%, continuity 51.95%, and QB plus continuity 52.34%
over the same 2,075 games. The injury source has only two duplicate
player/team/week rows in the canonical download and 99.6% of observations are
at least 24 hours before kickoff; it behaves as a weekly final observation,
not a recoverable sequence of intraday revisions.

Weekly player production was therefore added as a separate immutable source.
Reported absence probabilities are multiplied by prior snap share and
reliability-shrunk lagged offensive EPA or defensive disruption per 100 snaps.
The full player-value profile reached 52.14%, versus 52.05% without value, but
the 0.10-point paired increment was unresolved under both week and season
blocking. Two-season nested profile selection reached 52.47% over 2020–2025,
with worse probability scores and a 49.82% 2025 outer season. This admits
regularization/calibration and participation-based player ratings as the next
experiment; it does not admit broad tuning of injury weights on the same test
seasons.

The frozen regularization/calibration gate subsequently evaluated 48 declared
configurations. Its prior-two-season selector scored 50.70% over 1,582
2020–2025 games, slightly below the matched fixed-base 50.88%; the blocked
interval crossed zero. The top pooled QB+continuity/alpha-1 row reached 52.63%
and the full-player/alpha-1/beta row paired 52.34% classification with 0.24965
Brier, but both were found after viewing the grid. This strengthens the case
for a lower-variance, newly specified participation-rating hypothesis; it does
not create more independent seasons or justify another broad grid on the same
outcomes.

## Participation-rating screen

The complete 2016–2025 source was archived as 478,989 immutable rows. After
joining to the competitive PBP contract and requiring two unique 11-player
lineups, the rolling three-season fits used between 23,906 and 85,972 plays and
rated 1,758–2,872 players per target season. Those play counts support a
strongly shrunk low-dimensional screen; they do not turn plays within the same
game into independent ATS samples.

The fixed offense/defense adjusted-plus/minus extension reduced matched
2018–2025 ATS accuracy from 52.14% to 51.71% and worsened Brier error. That
rejects promotion for this formulation. It does not imply that participation
data is unusable for narrower unit/formation effects, but any such follow-up is
a new development hypothesis and cannot be described as independent evidence
on the already-viewed seasons.

## Learned availability screen

The injury/snap overlap contains 62,206 visible player-games from 2013–2024;
57,294 have a season-lagged learned rate for out-of-time scoring. Learned
report/practice/position probabilities improved the direct availability Brier
score from 0.09500 to 0.09056. This validates the transformation target and the
importance of practice status, not an ATS edge by itself.

Replacing the fixed status weights throughout the matched `player_value`
features moved ATS accuracy from 52.14% to 52.24%, with a week-blocked interval
of [-0.63, +0.78] percentage points. Probability and margin diagnostics moved
slightly in the right direction, but 2025 accuracy was 49.08%. Retain this as a
low-variance refinement lead. The next data-supported target is expected
offense/defense role delivery relative to a player's strictly prior snap share;
it can distinguish a full return from a one-snap appearance without adding a
new source.
