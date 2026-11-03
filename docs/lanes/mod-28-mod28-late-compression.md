# MOD-28 Team late-game margin compression

## Goal
Test whether a per-team late-game margin compression trait is reliable and, if so, adds to the cover probability on large spreads. ROADMAP row MOD-28.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Some teams and coaches run the clock and shorten the game when leading, others keep scoring; that moves the final margin across the cover without changing who wins. Per team-season trait = mean change in margin over the final quarter given the team entered it ahead by 9 or more, built only from prior games; a fitted pregame term on favourites of large spreads. Data: nflverse play-by-play 2009-2025 already local (data/raw/nflverse_pbp_full, game_features_pbp).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: mod25e-e175 and e131 (league-level leader Q4 scoring in the simulator, not a team trait), big_spread_diagnosis.md and MOD-18 (spread-size regime, not a team trait), mod23-unit10r garbage-time label (target denoising, not a feature). No grep hit for a per-team compression or backdoor-cover trait.

## Next
Build the trait table from pbp (entered-Q4-ahead-by-9+ games, prior games only) and run split-half reliability with a team-permutation null. First unit: First unit: split-half reliability of the trait across seasons and within season (permutation null over teams), before any ATS term; only if reliability is above its null does a LOSO term get fitted and graded at the opener.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Large-spread favourites are about 25 percent of games, so the term may be below power; record it unresolved rather than closing it.
