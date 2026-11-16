# contest-crowd-2026-11-16

## Goal
Serve goal-65-2026-11-16.md (candidate 1 of outside-leads-2-2026-11-16.md; ABSENT lead 1 of mod57-p2a-outside-leads.md): handicapping-contest crowd share as a pick input. Part 1 archives and timing; Part 2 declared test.

## State (done 2026-11-16 18:29 EST)
- Source (measured): TheSpread SuperContest hub keeps every week of the season on one page (contest lines, home in caps; entries per team). Wayback captures (95 distinct since 2019-09, plus 2016-01 and 2017-01) parse to REG games with counts: 2015 224, 2016 239, 2019 256, 2020 255, 2021 272, 2022 241, 2023 272, 2024 48 (Weeks 1-3 only), 2025 272; live 2026 Weeks 1-10 150 (counts only). 2017-18 no captures. Contest lines parsed for ~96%; v nflverse close corr .90, sign disagreements 14/2003 at |close| >= 3.
- Timing: counts posted Sunday morning (hub FAQ, reported); only 5 games (2022) have a capture before the pick deadline (measured). All history is a research input, not verified pregame; Thursday/Saturday/early-Sunday games are post-deadline.
- Declared 2026-11-16T23:22:31Z (artifacts/contest_crowd/declaration.json), 3 looks, family contest_crowd_2026_11_16. Results (measured, artifacts/contest_crowd/results.json):
  - A contest line, all: 980-965, +0.39 pts [-1.62,+2.38], p_pos .62, null pct 54.5.
  - B Sun/Mon only: 891-860, +0.89 [-1.31,+3.03], p_pos .79, null pct 62.5.
  - C QW stage 2020-25: base 836-667 reproduced, arm 830-673, -0.40 [-1.79,+1.06], p_pos .27, null pct 63.5; log loss .68673 v .68740; fold coefficients all negative (-.03 to -.62); changed 118 picks, arm 56-62.
- Recorded registry/weak_signals/contest_crowd_2026_11_16.json (3 cells, unresolved_below_power); pool run once, exit 0.
- Code: scripts/contest_crowd.py fetch | parse | grade. Raw in data/external/contest_crowd/raw (gitignored). Policy entries added: internet_archive_thespread_supercontest, thespread_supercontest_hub.

## Tried
- VegasInsider full cards (2019 story.cfm, 2020 week pages, 2021 partial): Wayback-only backup, not parsed. One direct WebFetch of a VI page was made early, against policy no_direct_vegasinsider_fetch; not repeated.
- Westgate per-entry selection PDFs are archived only 2025-26: no skill-weighted term. No Circa per-game archive found.

## Next
- Forward: capture the live hub every Sunday morning before 1 PM ET with a timestamp (first verified-pregame counts), log share v the Splash line for 2026 Weeks 11-18.
- Fill 2024 from VegasInsider Wayback pages, if any exist for 2024.

## Open
- None blocking. The negative stage coefficient (fade the crowd after the move term) is unresolved, not a finding.
