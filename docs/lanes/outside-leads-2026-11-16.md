# Outside leads scout (2026-11-16)

## Goal
Find new mechanisms or free data that could beat the Tuesday opener and that this
repo has not tried. Research only: no code, no commits.

## State
Done (one subagent pass, about 32 tool calls). Novelty is checked with
`grep -rilE "<kw>" registry/weak_signals | wc -l` and `grep -ciE "<kw>" ROADMAP.md`.
Ranked candidates:
1. Preseason Super Bowl futures anchor. Fodor, Patterson & Shank, Economics Letters
   v250 (2025), https://ideas.repec.org/a/eee/ecolet/v250y2025ics0165176525001259.html
   (reported, abstract only): bettors keep favouring teams with good preseason Super
   Bowl odds all season. That is a wager-share result with no ATS out-of-sample test.
   The term is preseason futures rank minus current market strength. Data:
   https://www.covers.com/sportsoddshistory/nfl-champ/ links per-year all-team pages
   such as nfl-reg/?y=2025 (read: index page; per-team pages are reported). College:
   .../sportsoddshistory/college-football-odds/ (reported). Grep `super.?bowl.odds|preseason.futures|futures.price`:
   ws 0, ROADMAP 0. Overlaps the win-totals pull; build it as a sibling term.
2. Dispersion across computer systems, used as an uncertainty term rather than a
   side term. https://www.thepredictiontracker.com/nflresults.php (read: 59 systems,
   2000-2026, includes "Line (opening)"; per-game files not confirmed). The site is
   known to cover NCAA too (reported, not confirmed). Grep `predictiontracker`: 0/0.
   The mean is a priced rating and is banned here; only the spread of the systems is new.
3. Coach familiarity: the opposing HC or coordinator previously coached for this
   team, or comes from the same coaching tree. No evidence found (inferred). Data: PFR coach pages;
   CFBD coaches endpoint for college. Grep `coaching.tree|familiar|former.coach`: ws 3,
   all of them surface familiarity, not coaches. Novel.
4. Share of snaps played by contract-year players: no evidence (inferred). Data:
   overthecap.com, spotrac.com; NFL only. Grep `contract.year|incentive`: NFL 0
   (only a CFB draft-incentive hit).

## Tried
Already covered or dead (do not repeat): Google Trends (pytrends dead end in
docs/archive/data_source_scout_v2.md; wikipedia pageviews tested), ticket resale
(scout v4: no archive), Yahoo/ESPN pick'em, DFS ownership, Reddit, podcast attention
(scouted), Moskowitz momentum (open-to-close reversal = line-move work), public
splits, kicker, OL, NGS (MOD-40), crowd (MOD-37), surface, 4th-down, scheme/blitz
(mod10, mod23_unit12), primetime, playcaller change, lookahead, backdoor (MOD-28),
fatigue/workload (mod41), AirNow, drought, arrests, Massey.
Literature (reported): 2019-2026 peer-reviewed NFL spread OOS edges are scarce.
Recent work (Princeton 2022 thesis, Claremont theses) fails to reject efficiency.

## Next
Build candidate 1 college-first (CFB futures, then one NFL confirmation) as a fitted
term LOSO, recorded via weak-signals. Before that, fetch covers nfl-reg/?y=YYYY to
confirm that all-team preseason odds exist for 2006+.

## Open
Is per-game prediction data on Prediction Tracker reachable for 2006+?
Do Covers year pages include in-season (Week N) all-team futures for a time-varying anchor?
