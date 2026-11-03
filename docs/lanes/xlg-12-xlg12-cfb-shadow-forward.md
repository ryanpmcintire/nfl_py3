# XLG-12 College shadow forward ledger

## Goal
Make forward evidence 4-5x faster by scoring frozen candidates on college openers each week. ROADMAP row XLG-12.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: NFL forward logs need until January for a few hundred games; college supplies about 60 graded opener games per week in the same weeks. A frozen common-feature candidate scored forward on both leagues gives pooled forward evidence several times faster. Data: CFBD odds and games (CFBD_API_KEY in user env), scheduler capture of Tuesday college lines.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: mod24_forward_log.py and POL-10 (NFL-only forward), lead69_cfb_replication (retrospective replication). Different: concurrent college forward scoring as a pooled shadow.

## Next
Check CFBD line history for 2026 openers and freeze the candidate coefficients. First unit: First unit: confirm the college opener source and timestamp (first-observed quote versus Tuesday), freeze one declared candidate from the mod24 pooled levers (RidgeCV), and score it for the games already played in 2026 as a dry run.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
College opener definition differs from the Tuesday NFL opener; mod24 measured the proxy gap as unresolved.
