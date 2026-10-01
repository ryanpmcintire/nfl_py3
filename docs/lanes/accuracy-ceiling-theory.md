# Why accuracy is stuck, and how to get past it (MOD-24 parent)

## Goal
Raise the model's own record against the opener (802-701, 53.36%, 2020-2025)
by fixing the grading instrument and adding information the opener lacks.

## State
2026-10-01 theory: (1) inputs are public team data the opener prices; the
team-quality ceiling is 0.013 pts; (2) 1,503 games give an SE of about
1.3 pts, so win-loss can't see gains of 0.5-1 pt; (3) noise of 13 pts vs an
edge of 1 pt; (4) about 7,700 tests reuse 2020-2025.
Correction: the extended population is not new. docs/proxy_opener_replication.md
(2026-08-19) graded the served model on SBR opens: 2011-2019 50.38%, against
53.36% for 2020-2025. The SBR open differs from the Tuesday line by 1.36 pts
on average, so it was kept out of the headline. XLG-09 already used a
2011-2025 population. New here: paired candidate-vs-base grading on it.
Owner approved all four units 2026-10-01. ROADMAP MOD-24.

## Next (one lane each)
- U1 docs/lanes/acc-u1-extended-grade.md
- U2 docs/lanes/acc-u2-proper-scores.md
- U3 docs/lanes/acc-u3-ngs.md
- U4 docs/lanes/acc-u4-forward-log.md (logs MOD-23 unit-5a man/zone first;
  arms that U1-U3 favor get added later)

## Hazard
Editing any pinned source (src/nfl_ats: names in PINNED_NAMES, *features.py,
*_overlay.py, names containing margin or model, cli_commands/prediction.py)
stops the v2 validation capture. All MOD-24 code goes in scripts/.

## Open
Why does the 2011-2019 proxy grade sit 3 pts below 2020-2025: line noise,
era, or 2020-2025 selection reuse? U1 reports it per era.
