# Outside leads scout, pass 2 (2026-11-16)

## Goal
Serve goal-65-2026-11-16.md: find new mechanisms or free data sources not tried here and
not on the outside-leads-2026-11-16.md exclusion list. Research only: no code, no commits.

## State
Done (one subagent pass, 33 tool calls). Novelty checked with one case-insensitive Grep
over registry/weak_signals and ROADMAP.md (alternation of every keyword below).
1. Skill-weighted handicapping-contest crowd. Paying entrants pick sides against a line
   frozen Thursday (Circa Millions, 2019+) or the SuperContest card (NFL; College edition).
   Term: share of entrants on each side at the frozen line, weighted by each entrant's
   prior-season record (weights fit LOSO). Evidence (reported): RotoWire 2022, 251 games,
   majority side beat the closing line 113-52, top decile 17-8 at the contest line, tail ROI
   -5.79%; 2023 consensus 54.88% through Wk13 versus 51.96% for the whole pool. Metz & Jog,
   Applied Econ Letters 30(18) 2023 (abstract only): SuperContest picks 2013-18 are about a coin
   flip and show recency bias; skilled entrants show less of it. Data: Circa posts picks Saturday
   evening (reported), which is before the Sunday deadline (not TNF or Saturday games); Westgate
   selection PDFs for the current season only (read: 2026 College wks 1-11 at
   westgateresorts.com .../2026-supercontest-college-selections). data.vsin.com/procontests
   returns 403 to the fetcher. 2020-25 history is unconfirmed (Wayback, Circa X posts).
   Grep `supercontest|circa.?million|contest.?consensus`: ws 0, ROADMAP 0.
2. FantasyPros expert consensus rankings (ECR) summed to team offense, minus the market
   implied team total. Data: github.com/dynastyprocess/data files/db_fpecr.parquet;
   nflreadr load_ff_rankings(type="all") = "full archive" with scrape_date (read). Start
   year is unconfirmed. No college source (inferred). No ATS evidence (inferred). Grep
   `fantasypros|\becr\b|ff_rankings|fantasy.?projection|dynastyprocess`: ws 0, ROADMAP 1
   (LEAD-55: injury source only).
3. Expert ATS consensus (Pickwatch, FantasyNerds): the archives found track straight-up
   picks only (reported). Weak. Grep `pickwatch|expert.?picks?`: 0/0.

## Tried
Dead or already done: playoff leverage, motivation and bowl eligibility (mod39_playoff_leverage,
motivation_ladder_*), roster continuity (graph_ratings_v2_team_stat), total-conditioned key
numbers (total_conditioned_lattice). Big Data Bowl winners are play-level; their metrics feed
NGS (on disk). ESPN win rates are overwritten season-to-date tables with no weekly history, so
they cannot be shown to be pregame. No free timestamped splits archive (SharpAPI keeps 48 h;
Sportradar and VSiN are paid). Literature: arXiv 2410.21484 review puts credible OOS ATS at
53-57%; Hanken 2025 thesis finds the market efficient. Neither is a 2022-26 peer-reviewed edge.

## Next
Candidate 1, as a source unit: confirm that 2020-25 Circa Millions picks by entry exist (Circa
X posts, Wayback captures of the VSiN and Westgate pages). If they do, build it college-first
from the SuperContest College PDFs. Test it net of the Tuesday-to-deadline move term, because
picks that beat the close may only repeat the line move.

## Open
Does db_fpecr keep weekly snapshots back to 2020? Are Circa per-entry picks archived for past seasons?
