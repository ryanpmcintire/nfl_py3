# NFL ATS

NFL ATS is a reproducible research and prediction pipeline for estimating which
side of an NFL point spread will cover. It uses maintained
[nflverse](https://nflverse.nflverse.com/) datasets instead of scraping HTML,
constructs every feature strictly from information available before kickoff,
and evaluates model choices with nested chronological walk-forward tests.

<!-- CURRENT_PREDICTIONS:START -->
## Current ATS forecast: 2026 Week 1

> **Early, mutable research preview.** Lines, injuries, depth charts, and model inputs may change before kickoff. Regenerate and republish this card as the week approaches.

Active model: `market_residual` with `player` features (`80e458040e48b926`). Its chronological 2018-2025 evaluation classified **1,080 of 2,075 non-push games correctly (52.05%)**. The week-blocked 95% interval was 49.85%-54.25%.

| Date        | Matchup    | ATS prediction   | Model estimate   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | SEA -3.5         | 51.5%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 54.3%            |
| Sun, Sep 13 | ARI at LAC | ARI +10.5        | 61.6%            |
| Sun, Sep 13 | ATL at PIT | PIT -3           | 51.2%            |
| Sun, Sep 13 | BAL at IND | BAL -3.5         | 52.9%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 60.0%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 55.5%            |
| Sun, Sep 13 | CLE at JAX | JAX -7.5         | 56.8%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 51.4%            |
| Sun, Sep 13 | GB at MIN  | GB -1.5          | 51.4%            |
| Sun, Sep 13 | MIA at LV  | MIA +3.5         | 55.1%            |
| Sun, Sep 13 | NO at DET  | DET -7           | 52.1%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +3           | 51.0%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 52.1%            |
| Sun, Sep 13 | WAS at PHI | WAS +4.5         | 59.0%            |
| Mon, Sep 14 | DEN at KC  | KC -3            | 53.8%            |

[Open the standalone card](CURRENT_PREDICTIONS.md) for provenance and interpretation.
<!-- CURRENT_PREDICTIONS:END -->

This repository is a ground-up successor to the original 2018–2023
`spank_vegas` project. The untouched legacy implementation is preserved in the
Git tag `legacy-2023-w6`.

## Principles

- One canonical row per game, keyed by nflverse `game_id`.
- Immutable, timestamped source snapshots with provenance manifests.
- A single sign convention: positive margins and spreads favor the home team.
- Explicit chronological development, validation, and outer-test periods; a
  model may only train on games before the week it predicts.
- The market is a baseline, not a feature we quietly declare victory over.
- Probability calibration, vig-aware decisions, passes, and ROI are first-class
  outputs.
- Every published card passes an independent, fail-closed prediction-safety
  contract before it is written.
- Raw data and model artifacts never belong in Git; only the deliberately
  published, human-readable current card is tracked.

## Data semantics

```text
home_margin = home_score - away_score
ats_margin  = home_margin - spread_line
```

In nflverse, a positive `spread_line` means the home team is favored. Therefore:

- `ats_margin > 0`: home covered
- `ats_margin < 0`: away covered
- `ats_margin == 0`: push

Historical nflverse lines are closing lines. Live lines can move, so every
refresh creates an immutable snapshot with a UTC fetch timestamp. A backtest
must not pretend a closing line was available earlier than it really was.

## Quick start

The project requires Python 3.12 and uses
[uv](https://docs.astral.sh/uv/) for Python and dependency management.
This working copy already has a portable uv executable at `.tools/uv.exe`; it
does not add `uv` to the system `PATH`. From PowerShell in the repository, run:

```powershell
.\.tools\uv.exe sync --all-groups
.\.tools\uv.exe run nfl-ats doctor
.\.tools\uv.exe run nfl-ats dashboard
```

On a fresh clone where `.tools/uv.exe` does not exist, install uv first (for
example, `winget install --id astral-sh.uv -e`), restart the terminal, and use
`uv` in place of `.\.tools\uv.exe` below.

To reproduce and publish the currently active full-player pipeline:

```powershell
.\.tools\uv.exe run nfl-ats ingest --start-season 2009 --end-season 2026 --stats-end-season 2025
.\.tools\uv.exe run nfl-ats build-features
.\.tools\uv.exe run nfl-ats pbp-ingest --start-season 2009 --end-season 2025
.\.tools\uv.exe run nfl-ats build-pbp-features
.\.tools\uv.exe run nfl-ats player-ingest
.\.tools\uv.exe run nfl-ats player-value-ingest
.\.tools\uv.exe run nfl-ats build-player-features
.\.tools\uv.exe run nfl-ats margin-backtest --features data\processed\game_features_player.parquet --feature-profile player
.\.tools\uv.exe run nfl-ats margin-predict --features data\processed\game_features_player.parquet --feature-profile player --season 2026 --week 1
.\.tools\uv.exe run nfl-ats publish-predictions
```

The backtest establishes the matching historical classification result. The
prediction command then trains on completed games before the earliest kickoff
in the requested week and writes a synchronized outcome card. The publisher
updates the two tracked GitHub Markdown views only after those identities match.
Each run also writes `prediction_safety.json`. It independently recomputes
probability bounds, picks, vig and no-vig math, edges, prices, training cutoffs,
and model-input coverage. An inconsistency aborts the command instead of
publishing a plausible-looking card.
It also writes a forced-pick `pool_card.csv`/Markdown file ranked by confidence
for ATS pools; this is separate from the vig-aware PASS/bet policy.
Ordinary prediction runs are mutable research snapshots and should be
regenerated as lines and inputs change. The optional `--freeze` mode is reserved
for a deliberately declared prospective evaluation window near the decision
time; it is not part of the normal early-preview workflow. Frozen records are
immutable, refuse retrospective or unverifiable rows, and are revalidated when
read.

The dashboard opens locally at `http://127.0.0.1:8501` and reads generated
artifacts without modifying them. The main navigation is organized around five
plain questions: this week's predictions, whether the model works, why it made a
specific pick, whether the data/system are healthy, and advanced research. Model
selection, feature ablations, archived tests, and other researcher diagnostics
remain available without crowding the prediction workflow.

Historical and weekly headline pages share `artifacts/active_ats_model.json`.
That atomic manifest links one exact evaluation artifact to one matching weekly
forecast, method, feature profile, regressor, Ridge strength, calibration policy,
and feature-table hash. A forecast with no exact historical match is marked
`UNLINKED` and cannot silently replace the synchronized dashboard default.

After regenerating a synchronized weekly outcome card, update the GitHub-facing
README table and standalone card with:

```powershell
.\.tools\uv.exe run nfl-ats publish-predictions
```

The publisher refuses an unlinked card or model-ID mismatch. It updates the
marked README section and `CURRENT_PREDICTIONS.md` from the same active artifact,
so both tracked files move together in one commit.

Current book-specific odds are optional. After obtaining a provider key, set it
for the current PowerShell session and archive a timestamped observation:

```powershell
$env:THE_ODDS_API_KEY = "your-key"
.\.tools\uv.exe run nfl-ats odds-ingest
.\.tools\uv.exe run nfl-ats odds-summary
```

The key is never written to an artifact. A separate no-key command downloads
the public Spreadspoke/Kaggle archive, preserves its source ZIP and license,
normalizes its reported closing spreads, and audits them against nflverse:

```powershell
.\.tools\uv.exe run nfl-ats market-backfill
```

That archive has one reported close per game, not timestamped quote movement.
It is useful for source comparison but is never presented as a 24-hour or
Tuesday-morning line. Local timestamped collection supplies that future history.

A second no-key command downloads a free CC BY-NC 2025 sample containing the
market opener plus reported closing lines and prices from nine named books for
spreads, moneylines, and totals:

```powershell
.\.tools\uv.exe run nfl-ats market-open-close-backfill
```

It normalizes 17,100 outcome quotes across 285 games. The source distinguishes
open from close but supplies no quote timestamps, so it is suitable for
open-to-close movement research—not a claim about the line available at an
arbitrary Tuesday or 24-hour decision point.

## Current benchmark

The primary research target is **binary ATS classification**: select the team
that covers more accurately than a 50% coin flip in chronological out-of-sample
games. The point spread is required to define that label. Betting prices, vig,
ROI, and Kelly sizing are not part of this definition; they remain an optional
paper-betting analysis. Brier score and log loss diagnose probability
calibration, but poor calibration does not erase an observed classification
advantage.

The first full player-profile result is therefore meaningful. Across 2,075
non-push games from 2018 through 2025, the market-residual model selected the
covering team **1,080 times (52.05%)**, 2.05 percentage points above chance. Its
95% accuracy interval is 49.85%–54.25% when resampling whole NFL weeks and
50.19%–54.14% when resampling whole seasons. Against the exact paired base
model, accuracy improved from 51.08% to 52.05%; that 0.96-point incremental
improvement remains unresolved under week blocking. The 0.2532 Brier score says
the confidence magnitudes are poorly calibrated, not that the 52.05% binary
classification result disappears. This is the strongest current ATS lead and
deserves focused replication and refinement.

The player layer has now been decomposed instead of treating that 52.05% as one
indivisible result. In a fixed 2018–2025 comparison, the coarse injury-only
profile reached 51.28%, lineup continuity reached 51.95%, QB plus continuity
reached 52.34%, and the original full bundle reached 52.05%. The original gain
therefore came mostly from continuity/QB state, not the first injury-severity
index. None of the paired week- or season-blocked intervals resolved a fixed
profile improvement over base.

A second leak-safe layer archives 291,747 weekly player-stat rows from
2009–2025 and weights reported absences by reliability-shrunk, strictly lagged
offensive EPA and defensive disruption per snap. Adding those two value fields
to the full player profile classified 52.14% correctly, 0.10 percentage points
above the original player model; its paired interval crossed zero. A
nested policy selecting among base/injury/player/value profiles on the prior
two seasons reached 52.47% over 1,582 games from 2020–2025, versus 50.88% for
the fixed base profile. Its 1.58-point paired accuracy improvement had a
week-blocked interval of 0.06–3.10 points, but Brier score was worse and the
latest outer season (2025) was only 49.82%. This is the strongest refinement
lead, not a promoted model. It required a frozen nested regularization and
calibration gate before any promotion decision—not more tuning of feature
definitions against the same seasons.

That frozen gate is now complete. The budget was declared before scoring: four
profiles (`base`, `player`, `player_qb_continuity`, `player_value`), Ridge alpha
values 1/10/100, and no/Platt/isotonic/beta calibration. Each calibrator learned
only from earlier out-of-sample weekly predictions beginning in 2016; every
outer season from 2020 through 2025 selected one of 48 configurations using only
the prior two seasons. The resulting selector classified 50.70% of 1,582 games,
versus 50.88% for fixed base/alpha-10. Its paired change was -0.19 percentage
points with a week-blocked 95% interval of -2.48 to +2.21 points. **It is not
promoted.**

The pooled table still produced useful hypotheses. QB plus continuity with
alpha 1 and no calibration led classification at 52.63%; its +1.54-point change
over base had a week-blocked interval of -0.29 to +3.26 points and its Brier
score was worse. Full-player alpha 1 with beta calibration reached 52.34% and
improved Brier from 0.25208 to 0.24965. Both were identified after comparing 48
rows, so they are development leads rather than independent confirmation. The
active 52.05% model remains unchanged. Participation-based player ratings are
the next attempt to add new signal instead of further selecting among these
same rows.

That participation experiment is now complete and negative. An immutable
2016–2025 snapshot contains 478,989 source rows. For each target season, one
regularized adjusted-plus/minus fit used only the preceding three seasons of
competitive, valid 11-on-11 plays; EPA was clipped at five points, team effects
absorbed broad team strength, Ridge alpha was fixed at 1,000, and player
coefficients received a 500-play reliability prior. The resulting offense and
defense ratings added only two injury-value contrasts to the prior player-value
profile. On the exact same 2,075 non-push games, ATS classification fell from
52.14% to **51.71%**. The -0.43-point change had a week-blocked 95% interval of
-1.53 to +0.63 points, and Brier error worsened by 0.00083. The active model is
unchanged. The code and result remain available as a failed fixed hypothesis;
the next availability work should learn actual play probabilities from
historical report status and snaps rather than retune these player ratings.

That follow-up is also complete. Across 57,294 player-games scored strictly
with earlier-season rates, replacing the hand weights with report × practice
rates plus a strongly shrunk position refinement improved the availability
target's Brier score from 0.09500 to 0.09056 and classification from 87.11% to
87.88%. When those learned rates replaced the old weights throughout the same
QB, injury-burden, and player-value feature columns, matched ATS classification
moved from 52.14% to **52.24%**—1,084 rather than 1,082 correct games. Brier,
ECE, margin error, and straight-up scores also improved slightly. The ATS gain
was only 0.10 points with a week-blocked interval of -0.63 to +0.78 points, and
2025 scored 49.08%, so this is promising but unresolved and is not promoted.
The next refinement should predict expected role/snap delivery, not merely
whether a player logged any snap. A replacement live injury source is still
required because the historical nflverse injury feed ends in 2024.

The first expanding-window development benchmark is intentionally recorded even
though it is not a winning strategy. Because its 2018 through 2025 results were
examined during model development, it is not described as an untouched final
test. Retraining the default logistic model before every week produced:

- 2,075 evaluated non-push games and 50.2% side accuracy;
- Brier score 0.2508 and log loss 0.6947;
- 538 selected bets at a 2% minimum edge, returning -1.60 units (-0.3% ROI);
- quarter-Kelly paper bankroll 100 → 94.97 with an 18.1% maximum drawdown.

Week-blocked 95% intervals are 48.1%–52.2% accuracy, 0.2496–0.2521 Brier,
and -8.7%–8.4% ROI. Those intervals make the conclusion clearer: the current
result is statistically compatible with no classification advantage.

The research bar is out-of-time ATS accuracy above 50% that remains stable by
week and season. Probability quality is a secondary diagnostic and paper
returns are reported separately; neither defines the classification target.

A 2022–2025 feature-set ablation also found that the nine-feature
`market_context` model outscored the full 58-feature model on Brier score
(0.25018 vs 0.25051). This is an exploratory comparison, not a newly selected
production model; it indicates that the present team-form features have not yet
shown stable incremental value.

The next outcome-model benchmark (2018–2025, 2,127 games) separates winner,
margin, and ATS questions. The independent fair-margin ridge model predicted
the winner correctly 65.2% of the time, but the closing market was better on
winner probability and absolute margin error (9.84 vs 10.15 points). A model
that predicted a correction to the market reached 67.2% winner accuracy, versus
66.4% for the market favorite, but had worse Brier score and margin error; its
week-blocked winner-accuracy improvement interval crossed zero. Its +2.9% paper
ROI also had a 95% week-block interval of roughly -2.1% to +8.3%, so it is not
evidence of an ATS edge.

The versioned PBP experiment added 48 early-down EPA, success, explosive-play,
pressure, pass-rate/PROE, drive, and field-position states (106 total inputs).
It did not improve the outcome models: fair-margin winner accuracy fell from
65.2% to 64.7%, and direct-ATS Brier was essentially unchanged. This negative
result is retained as an experiment artifact instead of being tuned away.

A stricter six-candidate nested comparison subsequently selected a PBP variant
in four of eight outer seasons. PBP modestly improved the logistic candidates'
two-season validation Brier score in five or six folds, depending on the base
feature set, but was effectively tied for histogram boosting. The selected
outer predictions reached 50.36% ATS accuracy, 0.25084 Brier, and -1.66% paper
ROI; week-blocked 95% intervals were 48.40%-52.56% accuracy and -9.30%-6.79%
ROI. PBP is therefore a weak lead for refinement, not demonstrated edge.

The legacy PageRank/HITS idea has also been rebuilt with strict weekly cutoffs,
continuous scores, temporal decay, and a ridge/SRS opponent-adjustment
comparator. Across 2018–2025, graph candidates were selected in zero of eight
nested outer seasons. The simpler schedule rating was selected three times but
did not produce a statistically resolved improvement. Both remain available as
research feature sets and neither is a default.

Adding those ratings to the fair-margin and market-residual models also made
both probability and margin error worse. The graph experiment is therefore a
completed negative result, not an unfinished candidate awaiting promotion.

The next PBP iteration separates six observed offensive efficiencies into
time-decayed, ridge-shrunk offense and opposing-defense effects before each NFL
week. Its 18 matchup fields modestly improved full-PBP direct-ATS Brier from
0.250803 to 0.250738, but the season-blocked 95% improvement interval crossed
zero (-0.000050 to 0.000199). It worsened fair-margin MAE (10.175 to 10.194),
market-residual MAE (9.935 to 9.957), and straight-up Brier (0.22262 to
0.22358). Opponent adjustment is therefore implemented and available for
research, but is not a default feature family.

The drive layer is also complete: points, yards, plays, duration, scoring, and
turnover rates per possession are carried forward for both offenses and
opponent-allowed defenses. Adding its 36 fields to raw PBP raised full-model
ATS Brier from 0.250803 to 0.250891. The season-blocked improvement was
-0.000088 with a 95% interval of [-0.000481, 0.000217], so drive state remains
available as a research family and did not advance to the more expensive
outcome-model stage.

The nonlinear histogram-boosting margin challenger was also worse than Ridge:
fair-margin MAE was 10.30 vs 10.15 points, and market-residual MAE was 10.00 vs
9.91. Its residual-model paper ROI was -2.1% rather than Ridge's exploratory
+2.9%. Ridge therefore remains the simpler research default; neither model has
demonstrated an ATS edge.

A nested rolling-origin evaluation now chooses among nine logistic/HGB and
feature-set candidates using only the preceding two seasons before scoring each
outer season from 2018 through 2025. Across 2,075 non-push outer-test games it
produced 49.7% accuracy, 0.25064 Brier error, and -3.0% selected-bet paper ROI.
Six different configurations won the eight validation folds, so the current
model-selection signal is unstable. This corrects the evaluation protocol and
leaves substantial feature/model research open; it does not imply that
historical research is exhausted.

Nested evaluation computes one leak-safe chronological prediction stream per
candidate and reuses immutable season slices for overlapping validation folds.
This is equivalent to repeating the same weekly fits inside every fold, while
removing redundant computation and making larger frozen candidate budgets
practical.

An explicit team-error dependence audit found pooled lag-one correlation of
-0.018, inside the season-preserving shuffle null range (-0.031 to 0.032;
two-sided p=0.238). The current residuals therefore do not show detectable
team-level serial correlation, but week/season blocked intervals remain the
conservative reporting default. Independence affects uncertainty; it does not
prohibit validation and test partitions.

The [historical data-feasibility audit](docs/data_feasibility.md) now gates the
research backlog. It verifies actual nonempty releases, source/schema changes,
availability semantics, and effective game-level sample size before a lead is
implemented. The first historical player layer now uses timestamp-filtered
2009–2024 injury reports, lagged 2013–2025 snap counts, and strictly prior-week
rosters. Its value extension uses 2009–2025 weekly player production only after
each game is complete. Player participation and NGS provide ten seasons for
restrained unit-level work. The one-season opener/close sample and one season
of precisely timestamped depth history are not sufficient for retrospective
edge claims.

## Commands

```text
nfl-ats doctor
nfl-ats smoke-source [--schedule-season YEAR] [--stats-season YEAR]
nfl-ats ingest [--start-season YEAR] [--end-season YEAR] [--stats-end-season YEAR]
nfl-ats build-features [--snapshot SNAPSHOT_ID] [--ewm-span 8]
nfl-ats pbp-ingest [--start-season YEAR] [--end-season YEAR]
nfl-ats build-pbp-features [--snapshot SNAPSHOT_ID] [--opponent-half-life 16]
nfl-ats depth-ingest [--start-season YEAR] [--end-season YEAR]
nfl-ats build-qb-features [--decision-hours 24] [--max-depth-age-days 14]
nfl-ats player-ingest
nfl-ats player-value-ingest [--start-season YEAR] [--end-season YEAR]
nfl-ats build-player-features [--value-span 16] [--value-prior-snaps 200]
nfl-ats participation-ingest [--start-season 2016] [--end-season YEAR]
nfl-ats build-participation-features
nfl-ats participation-ablation
nfl-ats build-learned-availability-features
nfl-ats availability-ablation
nfl-ats player-ablation [--profiles base,player,player_value]
nfl-ats player-model-selection
nfl-ats odds-ingest [--regions us] [--markets spreads,h2h]
nfl-ats odds-summary
nfl-ats market-backfill
nfl-ats market-open-close-backfill
nfl-ats backtest [--start-season YEAR] [--end-season YEAR] [--model logistic|hgb]
nfl-ats nested-evaluate [--first-test-season YEAR] [--last-test-season YEAR]
nfl-ats dependence-audit --predictions PATH
nfl-ats experiment [--start-season YEAR] [--feature-sets market,market_elo,full]
nfl-ats margin-backtest [--feature-profile PROFILE] [--methods METHODS] [--ridge-alpha FLOAT]
nfl-ats margin-predict --season YEAR --week WEEK [--feature-profile PROFILE] [--ridge-alpha FLOAT]
nfl-ats publish-predictions [--destination PATH] [--readme PATH]
nfl-ats handoff [--destination PATH]
nfl-ats predict --season YEAR --week WEEK [--model logistic|hgb] [--feature-set SET] [--freeze]
nfl-ats dashboard [--port 8501] [--no-browser]
```

Use `.\.tools\uv.exe run nfl-ats <command> --help` for all options in this
working copy.

Reproduce the evaluator positive-control audit—which first verifies the exact
active 52.05% prediction stream and then measures recovery of known synthetic
0.5-, 1-, and 2-point signals—with:

```powershell
.\.tools\uv.exe run python scripts\sensitivity_audit.py
```

The output is a timestamped, ignored directory under
`artifacts/sensitivity_audits/`; it cannot activate or publish a model.

`predict` defaults to the compact `market_context` model and produces two
distinct outputs: a forced estimated ATS winner and probability for every game,
and a vig-aware paper action that may remain `PASS`. The former answers “which
side does the model predict will cover?” even when the estimated advantage is
too small to treat as evidence of an actionable edge.

## Repository layout

```text
src/nfl_ats/       application and modeling code
tests/             deterministic unit and integration tests
docs/              architecture, data, and modeling decisions
data/raw/          ignored immutable source snapshots
data/pbp/raw/      ignored, season-partitioned PBP snapshots
data/players/participation/raw/ ignored participation snapshots
data/market/raw/   ignored timestamped book/line/price observations
data/market/historical/raw/ ignored source archives and normalized reported closes
data/market/historical/open_close/raw/ ignored 2025 opener/multi-book close snapshots
data/processed/    ignored feature tables
artifacts/         ignored backtests, predictions, and fitted models
```

## Development

```powershell
.\.tools\uv.exe run ruff format --check .
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run mypy src
.\.tools\uv.exe run pytest
```

See [docs/architecture.md](docs/architecture.md),
[docs/data.md](docs/data.md), [docs/modeling.md](docs/modeling.md), and the
[performance contract](docs/performance.md) for the design, leakage boundaries,
and evaluator budgets. The full prioritized backlog, including simulation and
pool moonshots, lives in [ROADMAP.md](ROADMAP.md). New development sessions
should begin with [HANDOFF.md](HANDOFF.md). The agent and repository hooks refresh
and stage it automatically before commits and enforce freshness before a
`master` push; `nfl-ats handoff --check` is available for diagnostics.

## Responsible use

Sports markets are noisy and efficient. Backtest performance is not a promise
of future profit. Track the exact line, price, book, and decision timestamp;
include pushes and vig; report uncertainty; and never wager money you cannot
afford to lose.

## License and attribution

Project code is MIT licensed. Most nflverse data is distributed under CC-BY
4.0, while some upstream datasets use different terms. See
[docs/data.md](docs/data.md) and the upstream documentation before redistributing
data or using it commercially.
