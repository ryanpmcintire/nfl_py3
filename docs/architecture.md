# Architecture

The project is deliberately a batch pipeline. Each stage has a narrow contract
and writes a durable result that can be inspected independently.

```text
nflverse schedules + weekly team stats + optional PBP/depth
                 |
                 v
       immutable raw snapshot
                 |
                 v
    one-row-per-game feature table
                 |
          +------+------+
          |             |
          v             v
  weekly backtest   weekly prediction
          |             |
          v             v
 metrics + rows    model + recommendations
          |             |
          v             v
 uncertainty +       optional immutable
   model card        pre-kickoff record
```

## Boundaries

`data.py` owns network acquisition and upstream schema validation. It does not
clean data in memory and silently continue when required columns disappear.

`snapshots.py` writes Parquet files and SHA-256 provenance into a timestamped
directory. The manifest is written last and acts as the commit marker. Existing
snapshots are never mutated.

`features.py` owns the definitions of the target and predictors. Team form is
calculated after each completed game and joined to future games with a strict
earlier-than lookup. Elo values are captured before the current game updates
the ratings.

`pbp.py` owns the narrowed, season-partitioned PBP contract, v1 situation
filter, drive table, team-game efficiencies, and strict pregame state join.
`opponent_adjustment.py` fits time-decayed ridge offense and opposing-defense
effects at weekly cutoffs, then produces matchup expectations without allowing
the week being scored into the fit.
`players.py` owns immutable injury, roster, snap, and weekly player-production
snapshots. It emits every game before updating snap or production state from
that game. Injury burden is available as both prior-snap share and a
reliability-shrunk, lagged value-weighted form. The latter is a research proxy,
not a claim that public box-score statistics isolate causal player value.

`quarterbacks.py` owns timestamped depth snapshots, as-of starter selection,
and prior player states. It never substitutes an actual starter for a missing
pregame observation.

`market_data.py` owns the optional current-odds provider boundary. It archives
raw and normalized timestamped quotes, attaches nflverse game IDs, and derives
same-book pre-kickoff closes, consensus, and CLV without mutating observations.

`historical_market.py` owns the no-key historical closing-line cross-check. It
preserves the public source archive and license, normalizes favorite-oriented
spreads, and audits overlapping rows against nflverse without inventing quote
timestamps.

`open_close_market.py` owns the free 2025 opener/multi-book close sample. It
converts the wide source into long market outcomes, labels stage without
inventing timestamps, matches nflverse games, and produces one-game movement
and consensus-close audits.

`margin.py` and `outcomes.py` compare market, independent fair-margin,
market-residual, straight-up, and direct-ATS forecasts on identical weekly
cutoffs. Ridge strength is explicit model identity. The predictive margin
distribution uses a chronological residual holdout.

`calibration.py` transforms chronological market-residual probabilities with
none, Platt, isotonic, or beta calibration. It fits each target week only from
earlier out-of-sample predictions, then records the raw probability, history
size, and latest eligible date. `experiments.py` owns the frozen 48-candidate
player-model budget and the prior-two-season/next-season selection policy.

`modeling.py` accepts only the explicit feature allowlist in `constants.py`.
Scores, results, labels, identifiers, and timestamps cannot become predictors
by accidentally selecting every numeric column.

`graph_ratings.py` owns temporal opponent adjustment. Before each NFL week it
computes continuous PageRank and offense/defense HITS scores from a decayed
graph containing only earlier weeks. It also fits a time-weighted ridge/SRS
schedule rating on the identical history. Current-week games are added only
after every game in that week receives its pregame features.

`backtest.py` is the only path used for historical evaluation and weekly
scoring. For a week beginning on date `D`, all training rows have a game date
strictly before `D`. This also prevents a Sunday result from leaking into a
Monday prediction in the same NFL week.

`prediction_safety.py` is the release boundary for scored cards. It does not
trust derived output columns: it independently recomputes sides, edge,
break-even probability, no-vig probability, and market hold, and validates
probability bounds, spread/price plausibility, game identity, method coverage,
calibration identity and strictly earlier calibration cutoffs,
margin identities, input missingness, and strict training cutoffs. Any hard
failure aborts publishing. Tests mutate otherwise-valid cards to prove each
class of corruption fails closed.

`evaluation.py` wraps that scorer in nested rolling-origin selection. Candidate
models see only prior validation seasons; each selected configuration is fixed
before its outer test season is scored. It computes each candidate/week
prediction once and slices the resulting chronological stream into overlapping
validation folds, avoiding statistically redundant refits.

`cli.py` composes those stages. Generated data and artifacts are ignored by
Git; code, contracts, tests, and dependency locks are versioned.

`portfolio.py` converts prediction probabilities and prices into constrained
paper stakes, supports conservative probability haircuts, and simulates
conditional bankroll paths. `experiments.py` compares named feature sets without changing the
evaluation windows. `reporting.py` contains read-only summaries consumed by the
local Streamlit dashboard; the UI never trains models or mutates source data.
`active_model.py` atomically links the active ATS method, exact historical
evaluation, and matching weekly card. Dashboard pages resolve their default
artifacts through this one manifest instead of independently choosing whichever
directory happens to be newest. Exact feature-table hashes prevent a forecast
from inheriting metrics from a superficially similar but different evaluation.
`publishing.py` turns that linked card into the tracked README table and
`CURRENT_PREDICTIONS.md`. Publication fails on an unlinked forecast or model-ID
mismatch, keeping the GitHub landing page on the same model as the dashboard.
`handoff.py` combines read-only Git inspection, the active-model manifest, the
tracked weekly publication, local feature availability, and the roadmap's
ordered priorities into `HANDOFF.md`. Root `AGENTS.md` makes that handoff and
the repository's research invariants part of every new Codex session.
`prospective.py` is the only path that can freeze a forecast. It requires known
future kickoffs and writes a new hashed record rather than updating a prior
forecast. A frozen record stores its safety certificate and re-runs the full
contract when read; a matching file hash alone is not sufficient.

## Intentional limitations

- PBP and opponent-adjusted PBP are available, but neither full-history
  ablation improved the baseline reliably. Timestamped QB depth coverage begins
  in the 2025 era and is not yet sufficient for historical promotion.
- nflverse schedule spreads are treated as historical closing lines. They are
  suitable for research but do not reproduce the information set at an
  arbitrary earlier decision time.
- Current recommendations are useful only when their stored line and price
  match the line and price actually available to the user.

Those constraints keep the first revived system auditable. Injury, roster,
weather-forecast, and line-movement sources can be added only after they have a
timestamped point-in-time contract.
