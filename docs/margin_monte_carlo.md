# Margin Monte Carlo (SIM-01)

Status: simulation plumbing implemented; no research experiment or verdict has
been run from this module.

## Contract

**Read** from [`src/nfl_ats/margin.py`](../src/nfl_ats/margin.py) and
[`src/nfl_ats/margin_simulation.py`](../src/nfl_ats/margin_simulation.py): for
fair-margin and market-residual models, `fit_margin_model` estimates the stored
residual distribution on an out-of-time calibration partition, and
`simulate_margin_distribution` resamples it with replacement. The sampler adds
each game's predicted center and uses the same integer-line three-way
settlement as `MarginModel.predict`; it also retains both latent and
integer-rounded margins.

**Read** from the same module: `home_cover_probability` uses the repository's
smoothed two-way ECDF convention. Its separate three-way outputs --
`home_cover_probability_excluding_push`, `push_probability`, and
`home_loss_probability` -- sum to one. The conditional home-cover probability
removes pushes for comparison; the code creates no wagering action or stake.

**Read** from the same module: simulation refuses a target at or before the
model's `training_max_gameday`, requires a finite spread for every game, and
stores the fixed seed and training cutoff beside every probability row. The
raw latent and integer-settled draws remain available through
`MarginMonteCarloResult.sample_frame()` for audit or downstream diagnostics.

## Usage

**Read** from [`src/nfl_ats/outcomes.py`](../src/nfl_ats/outcomes.py): the weekly
fitter preserves the repository's strictly-prior training cutoff:

```python
from nfl_ats.margin_simulation import simulate_margin_distribution
from nfl_ats.outcomes import fit_margin_models_for_week

target, models = fit_margin_models_for_week(
    features,
    season=2026,
    week=1,
    feature_profile="weak_stack",
    methods=("market_residual",),
)
simulation = simulate_margin_distribution(
    models["market_residual"],
    target,
    samples=10_000,
    seed=20260902,
)
probabilities = simulation.probabilities
draws = simulation.sample_frame()
```

**Inferred use, not evidence:** calibration plots, key-number summaries, or
comparisons against realised outcomes may be built from these outputs, but
doing so is a separate research experiment. This implementation neither scores
such an experiment nor records a weak-signal classification.
