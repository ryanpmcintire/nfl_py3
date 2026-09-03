# Open benchmark suite foundation (SKY-07)

Status: **foundation implemented; publication remains open**.

`nfl_ats.open_benchmark` defines a deterministic exchange contract for an NFL
against-the-spread benchmark without publishing a dataset or claiming a
leaderboard result. It is deliberately a library rather than a CLI or scheduled
job while source licensing, an external host, private test labels, and leaderboard
governance remain owner decisions.

## Dataset contract

Each release is a two-file directory:

- `observations.csv` contains one game per row, in canonical chronological order;
- `manifest.json` pins the schema, task, split counts, column order, byte count,
  SHA-256 digest, source/license declarations, and a dataset content identity.

The exporter accepts exactly the declared columns. Core rows include game identity,
teams, kickoff, the benchmark decision deadline, the timestamp through which inputs
were observed, the spread, and a split. A release can declare additional numeric
feature columns. The contract enforces all of the following before writing:

- every input observation is at or before its decision deadline;
- every decision deadline is strictly before kickoff;
- game IDs are unique;
- train, validation, and test are all non-empty and chronologically disjoint;
- train and validation labels agree with the sign of `ats_margin`;
- public test rows contain neither `ats_margin` nor `cover_side`;
- positive `spread_line` means the home team is favored, and
  `ats_margin = home_score - away_score - spread_line`.

The benchmark task is a forced HOME/AWAY pick plus the home cover probability
excluding pushes. Pushes are represented in the labeled development data and are
declared as excluded from an eventual accuracy score. The foundation does not
implement that score, inspect model output against private outcomes, or select a
winner.

## Submission contract

`export_open_benchmark_submission` writes a canonical `submission.csv` and
`submission.json`. A submission must cover every test game exactly once, choose
HOME or AWAY, provide a probability consistent with that pick, and prove that both
the prediction and its newest input precede the benchmark deadline. Its metadata
pins the exact dataset and manifest hashes. Validation never reads a test outcome
and returns integrity metadata, not a score.

## Publication blockers

The manifest reports publication readiness mechanically. `NOASSERTION`, an empty
source-URL list, or a missing public URL is a named blocker. This is intentionally
fail-closed: code must not turn an unreviewed source into a supposedly open dataset.

SKY-07 therefore remains open until all of these are resolved outside this module:

1. choose public inputs whose redistribution terms cover the normalized release
   (**scoped 2026-09-03** from tracked `config/source_policies.json`: nflverse
   CC-BY-4.0 raw and derived with attribution is the only clean input family —
   game identity, kickoff, spread/scores, and in-repo features computed solely
   from nflverse inputs; per-dataset upstream checks still owed since some
   nflverse datasets carry different terms; Odds API raw, weather vendors,
   Sagarin, PFR, Sportradar, and CFBD are excluded by policy);
2. publish the exact license and source URLs;
3. choose durable external hosting and pin its release URL;
4. define custody and audit rules for withheld test outcomes;
5. predeclare leaderboard metrics, submission limits, tie handling, and versioning;
6. export and independently verify the first real dataset release.

No generated dataset, private label file, model artifact, score, or leaderboard is
tracked by this foundation.
