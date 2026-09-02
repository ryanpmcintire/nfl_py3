# Drive-level simulator

`src/nfl_ats/drive_simulation.py` provides simulation plumbing at the
possession level. It fits empirical drive rows from the repository's canonical
play-by-play filter and simulates regulation games as alternating possessions.

Each sampled drive retains its starting field position, outcome, points,
sampled pace, regulation-capped elapsed clock, offense/defense, score state,
and profile fallback. Sampling the
field position, outcome, and pace from one historical row preserves their
observed dependence. In the final 15 minutes, leading and trailing offenses use
separate empirical pools; outside that window the neutral pool applies. When
available, offense-team and opponent-defense pools contribute symmetrically,
then fall back to the named league-state or all-drive pool.

The API has two explicit point-in-time boundaries:

- `fit_drive_simulator(..., training_max_gameday=...)` excludes every game
  after the cutoff before constructing any distribution.
- `simulate_drive_distribution(...)` refuses every target game at or before
  the model cutoff.

The same seed produces identical score and drive tables. The drive table is the
audit record behind every simulated final score. This first version is a
regulation-only empirical environment: possessions alternate, overtime and
halftime kickoff policy are omitted, and starting field position is sampled
from the applicable historical drive pool rather than transitioned from a
play-level punt or turnover model. Those transitions belong to SIM-04's full
play-by-play scope. No wagering action or experiment decision is part of this
module.
