# NFL.com Friday-refresh composition eval

Status: **predeclared and frozen 2026-08-22, before any scoring run.** The arm
definitions, overlay rule, tie rule, and bootstrap configuration below were
written down before `scripts/nflcom_friday_refresh_feature.py` ever executed;
the script is an implementation of this document, not a source of it. Measured
results are appended at the bottom after the run.

## Question (owner task, 2026-08-22)

The red-team-proofed lead — NFL.com Friday Out>=2 designations restricted to
starter-caliber players (`redteam_nflcom_out2_starters_only`, −1.2748 pts
full-slate scaled, [−2.0550, −0.5122], P(neg)=0.9995; see
`docs/edge_audit_redteam.md` Claim 2) is challenger-tracked only. This study
measures the signal AS THE LATE-WEEK REFRESH PATH WOULD ACTUALLY PLAY IT:
composed on top of the production chain (raw model -> coach fade ->
player-arrests policy) at the Saturday cutoff, on the same paired opener
archive used by `scripts/movement_composition_eval.py`, graded at the frozen
Tuesday line.

## Population and effective n

Paired opener archive `artifacts/opener_evaluation/20260819T174244Z/
per_game.parquet` (1,537 REG games, 2020-2025), RESTRICTED to seasons
2022-2024 — the only seasons where the immutable NFL.com snapshot
(`data/raw/nflcom_injuries/20260821T222602Z/`, 54/54 pages, seasons 2022/2023/
2024 REG weeks 1-18) has coverage. **Disclosed effective n: 799 archived games
before push exclusion; each arm's scored n equals its non-push games at the
Tuesday grade (~3 seasons only).** Three seasons is small; every interval below
must be read with that in mind. No window spend: attribution on
already-looked-at data only, Saturday-cutoff attribution exclusively.

## Incumbent chain (arm a)

Sequential reproduction of card_view's live order, exactly as in
`scripts/movement_composition_eval.py`: raw
`pick_home_at_open_probability_rule` -> coach-fade flip set (frozen
`apply_coach_fade_overlay` on predictions rebuilt from the archive) ->
player-arrest policy back-side flip (frozen `apply_frozen_policy` machinery).
Reproduction gate BEFORE restriction: chain accuracy on the full 2020-2025
archive must match the published sequential figure 0.541583499667332 within
1e-9, or the script fails.

## Team-game flags (frozen)

Computed per team-game from the latest `data/raw/nflcom_injuries` snapshot,
using the IDENTICAL name normalization, starter proxy, and join machinery as
`scripts/nflcom_friday_designation_screen.py` (imported, not reimplemented):

- `starter_out_count`: number of **Out** designations on STARTER-CALIBER
  players (most recent prior REG game of the same season with
  max(offense_pct, defense_pct) >= 0.50 in snap_counts.parquet). Week 1 has no
  prior-week snaps: forced 0 and counted as missing-required-data (disclosed).
- `total_out_count`: number of Out designations, any players.

## Overlay rule (frozen before scoring)

Applied at the Saturday cutoff to the arm-a chain pick of each game. Let
`picked` = the team the chain backs, `opp` = the opponent.

**Flip to the opponent iff `picked` is flagged AND `opp` is NOT flagged. If
both teams are flagged, KEEP the incumbent pick (tie rule, frozen now). If
neither, keep.**

Arms:

- **(b) `chain_plus_out2_starters`**: flag = `starter_out_count >= 2`
  (the red-team lead as played).
- **(c1) `chain_plus_out1_starter`**: milder variant, flag =
  `starter_out_count >= 1`.
- **(c2) `chain_plus_net_out_diff_ge1`**: net differential variant, flag =
  `picked.total_out_count - opp.total_out_count >= 1`.

No other variants are scored. The differential arm's threshold is frozen at 1;
no post-hoc threshold search.

## Bootstrap (frozen)

`nfl_ats.clv.week_blocked_bootstrap`, 20,000 samples, seed **20260823**, block
= "week" primary and "season" secondary, paired accuracy deltas vs arm a in
accuracy points, full slate, graded with `pick_correct` against
`margin_vs_open` (pushes NaN and excluded identically from every arm).

## Classification discipline

Mined-family composition of an already-recorded category-3 signal: every arm
is predeclared to record `unresolved_below_power` unless a terminal AGENTS.md
ground applies (whole interval strictly below zero would be admissible
`wrong_sign_resolved`; an interval crossing zero never is). These arms are
compositions/decompositions of `redteam_nflcom_out2_starters_only` — never
pool as independent. Recording happens ONLY via explicit
`nfl-ats weak-signals record` lines returned for central recording; this
script writes a provenance stamp under
`registry/experiments/nflcom-friday-refresh/` and NEVER touches either
registry JSON.

## Refresh-path integration contract (frozen spec)

1. **Which pass computes it.** NOT the Tuesday prediction pass. A new
   late-week refresh pass runs Saturday morning UTC, AFTER the week's final
   NFL.com league injury page is ingested and BEFORE card regeneration. It
   reads the already-computed production-chain picks for the week, computes
   `starter_out_count` / `total_out_count` per team from the newest
   `data/raw/nflcom_injuries` snapshot plus snap_counts through the prior
   week, applies the frozen overlay above, and emits the refreshed card. The
   model never refits; the Tuesday grading line is never re-picked.
2. **Snapshot freshness required from `ingest_nflcom_injuries.py`.**
   - During season the ingester must support incremental scope: fetch ONLY the
     current (season, reg week) page(s) into a fresh timestamped snapshot dir
     (immutable convention retained), rather than the 54-page historical
     backfill; `--seasons` must accept the live season.
   - Timing gate: the current week's page fetch timestamp must satisfy
     fetched >= Friday 16:00 ET of that game week AND ~~fetched < earliest
     kickoff among that week's games~~ **fetched < each GAME's own pick
     deadline** (see "2026-08-25 correction" below: the struck clause is
     unsatisfiable in any week containing a Thursday game, i.e. 17 of 18).
     The refresh pass FAILS OPEN: if the current week's page is absent or
     fails the timing gate, it keeps the incumbent chain pick for every
     affected team-game and records the skip.
   - Starter proxy input: prior-week `snap_counts.parquet` must already be
     present in the players snapshot (existing weekly ingest covers this);
     Week 1 = proxy unavailable = forced keep, matching the frozen rule.
3. **Leakage statement.** Every flag derives solely from pages fetched-as-of
   their own week (that week's FINAL report, published Fri/Sat, predating
   kickoff); the starter proxy consumes prior-week snap shares only. The
   refresh pass itself consumes nothing dated after its Saturday run time.

---

## Measured results (2026-08-22 run)

Artifact: `artifacts/nflcom_friday_refresh/20260822T231604Z/` (registry stamp
`registry/experiments/nflcom-friday-refresh/20260822T231604Z.json`). All
figures below measured from that artifact. Chain reproduction gate PASSED
before restriction: full-archive chain accuracy 0.541583499667332 exactly
matches the published sequential figure; 107 coach flips, 25 arrest flips
after coach. Coverage: all 54 archive season-weeks present in the NFL.com
snapshot, zero pages missing; every archived game has at least one Out row on
one side.

Population: 799 games restricted to 2022-2024; 780 scored per arm after push
exclusion. Effective n is THREE SEASONS ONLY.

| Arm | n | Accuracy | Picks changed | Paired delta vs chain | Week-blocked 95% | P+ (wk) | Season-blocked 95% | P+ (se) |
|---|---|---|---|---|---|---|---|---|
| (a) incumbent chain | 780 | 55.13% | 0 | — | — | — | — | — |
| (b) chain + out>=2 starters fade | 780 | 57.31% | 67 | **+2.1795 pts** | [+0.5222, +3.8911] | 0.9954 | [+0.0000, +4.8872] | 0.9623 |
| (c1) chain + out>=1 starter fade | 780 | 55.77% | 151 | +0.6410 pts | [-2.3257, +3.6176] | 0.6447 | [+0.0000, +1.1278] | 0.9623 |
| (c2) chain + net out-diff >=1 | 780 | 55.51% | 305 | +0.3846 pts | [-3.4048, +4.0868] | 0.5709 | [-3.0075, +4.8872] | 0.5984 |

Overlay diagnostics: arm b flagged the picked team in 76 games and kept 9
both-flagged ties under the frozen rule; arm c1 flagged 268 picked teams with
117 both-flagged keeps; arm c2 flipped 305 games (it has no tie rule beyond
the differential itself, which makes it the least selective and weakest
arm — measured as such here). Week-1 starter-proxy-unavailable games: 48
(forced keep, disclosed above).

Honest small-n framing (3 seasons): arm b's composed delta is POSITIVE with
P+ 0.9954 week-blocked and the season-blocked point estimate sits on the
positive side too (its 3-block interval lower edge lands exactly at 0.0000 —
at three season blocks that secondary interval is near-degenerate and the
week-blocked primary with 54 blocks is the read), but this is attribution on already-looked-at data — the parent
signal was selected AND red-teamed on these same 2022-2024 seasons, so no
fresh confirmation is claimed and none should be. Per-season paired deltas
(arm b vs chain, measured from season_summary.csv): 2022 +0.000 pts (14
flips, net wash), 2023 +1.504 pts (16 flips), 2024 +4.888 pts (37 flips) —
the sign holds or is flat in every season but the effect concentrates in
2024, and n per season is ~250 games. The milder variants dilute exactly as
the parent signal's threshold structure predicts: relaxing to out>=1 or going
to a net differential roughly triples the flip count and collapses the effect.
All three arms remain category 3 `unresolved_below_power`; an interval or a
season split never grounds rejection, and no terminal ground was met.

Record lines for central recording (never written to either registry JSON by
this session) are printed by the script and stored verbatim in
`metadata.json` under `proposed_weak_signal_records`.

## 2026 prospective challenger registration (frozen rule text, ready to paste)

```text
Challenger: nflcom_friday_refresh_out2_starters_v1
Status: PROSPECTIVE — registers a 2026-season prospective test; nothing here
  is claimed as confirmed. The historical composed figure (+2.1795 pts,
  P+ 0.9954 week-blocked, artifacts/nflcom_friday_refresh/20260822T231604Z/)
  is attribution on already-looked-at data (the parent signal was selected
  AND red-teamed on the same 2022-2024 seasons) and is an upper bound.
Rule (frozen): For each REG game on the weekly card, let CHAIN be the
  production pick (raw model -> coach fade -> player-arrests policy), PICKED
  the team CHAIN backs, OPP the opponent. From the current week's FINAL
  NFL.com league injury page (ingested before the Saturday refresh pass),
  count OUT designations on STARTER-CALIBER players for each team, where
  starter-caliber = played >=50% of offensive or defensive snaps in that
  team's most recent prior REG game of the same season (snap_counts.parquet;
  Week 1 = proxy unavailable = no flag). If PICKED carries >=2 such Out
  designations and OPP carries <2, replace CHAIN with the opposite side;
  otherwise keep CHAIN unchanged (both-flagged keeps; page absent or failing
  the freshness gate keeps).
Timing: computed once per week by the late-week refresh pass, Saturday UTC,
  after the week's final injury page is ingested and before card
  regeneration; the Tuesday grading line is never re-picked.
Grading: opener-grade (frozen Tuesday line) vs the incumbent chain's picks,
  paired accuracy deltas in accuracy points, full slate, pushes excluded;
  week-blocked primary / season-blocked secondary bootstrap.
Prospective decision rule: track all season; evaluate after the season with
  the predeclared taxonomy only — an interval crossing zero never grounds
  rejection; promotion claims require the owner's declared threshold; card
  play remains expected value at the opener grade.
Fails open: any ingest or freshness failure leaves the incumbent card
  untouched for affected games and logs the skip.
```

---

## 2026-08-25 correction: the freshness gate was unsatisfiable, and the
## published figure scored two days the corrected gate excludes

Two defects, found together while wiring the in-season injury capture. Both
were measured this session; neither closes anything.

### 1. The week-wide gate could never open

The gate as frozen (integration contract item 2, struck above) demanded the
page be fetched at or after **Friday 16:00 ET** and strictly before the
**earliest kickoff of the week**. Every NFL week opens with a Thursday night
game, so that second clause refers to a moment roughly 20 hours *before* the
first clause allows. The window is empty by construction.

Measured on real schedules (`data/raw/20260824T115346Z/schedules.parquet`,
true kickoff times from `gameday` + `gametime` in ET):

| Week | Earliest kickoff (ET) | Gate opens (ET) | Satisfiable |
|---|---|---|---|
| 2026 wk 1 | Wed Sep 9, 20:20 | Fri Sep 11, 16:00 | no (+43.7h) |
| 2025 wk 1 | Thu Sep 4, 20:20 | Fri Sep 5, 16:00 | no (+19.7h) |
| 2025 wk 5 | Thu Oct 2, 20:15 | Fri Oct 3, 16:00 | no (+19.8h) |
| 2025 wk 10 | Thu Nov 6, 20:15 | Fri Nov 7, 16:00 | no (+19.8h) |
| 2024 wk 5 | Thu Oct 3, 20:15 | Fri Oct 4, 16:00 | no (+19.8h) |
| 2024 wk 12 | Thu Nov 21, 20:15 | Fri Nov 22, 16:00 | no (+19.8h) |
| 2023 wk 8 | Thu Oct 26, 20:15 | Fri Oct 27, 16:00 | no (+19.8h) |

Satisfiable in **0 of 7** weeks checked. Across the 2026 REG schedule only
week 18 has no Thursday game, so the arm could have recorded at most 1 week
of 18 — and in practice zero, since no 2026 page had been captured at all.
The consequence was not a wrong number; it was **silence**: the challenger
carrying this project's strongest measured composed figure would have
produced no prospective evidence for an entire season, failing open into
permanent no-op.

The intent was always per-game — the contract's own leakage statement
(item 3) says "pages fetched-as-of their own week ... predating kickoff".
The week-wide minimum was a proxy for that and is simply wrong once a week
holds games on different days. **Corrected** in both implementations
(`src/nfl_ats/prospective.py`, `src/nfl_ats/nflcom_refresh_overlay.py`) to
the boundary the codebase already encodes, `pick_refresh.pick_deadline` =
min(own kickoff, the week-wide Sunday 16:00 ET lock). A Friday page now
scores the Sunday/Monday slate and drops only the Wed/Thu games it genuinely
post-dates. Pinned by
`test_a_thursday_game_no_longer_silences_the_whole_week` in both test files.

This is strictly *more* leakage-safe per admitted game than the old rule was
for the games it silently dropped, because each game is now judged against
its own deadline rather than an unrelated game's kickoff.

### 2. The published +2.1795 includes games the corrected gate excludes

`scripts/nflcom_friday_refresh_feature.py` joins Out counts on
`(season, week, team)` with no kickoff filter, so the archival study applied
the week's **final Friday** page to Wednesday and Thursday games whose
kickoff had already happened. Re-scored from the same frozen artifact
(`artifacts/nflcom_friday_refresh/20260822T231604Z/per_game.parquet`) with
the study's own machinery and constants (`week_blocked_bootstrap`, 20,000
samples, seed 20260823). The full-population reproduction gate matched the
published figure to 1.3e-5 before the restriction was applied.

| Population | n | Paired delta vs chain | Week-blocked 95% | P+ (wk) | Season-blocked 95% | P+ (se) |
|---|---|---|---|---|---|---|
| ALL (as published) | 780 | +2.1795 | [+0.5222, +3.8911] | 0.9954 | [+0.0000, +4.8872] | 0.9623 |
| **Gate-admitted (corrected)** | **719** | **+1.9471** | **[+0.1416, +3.7635]** | **0.9827** | **[+0.4367, +4.0984]** | **1.0000** |
| Excluded (Wed/Thu/Fri only) | 61 | +4.9180 | [-3.2787, +13.5593] | 0.8333 | [-5.2632, +13.6364] | 0.8518 |

61 of 799 games are excluded (57 Thursday, 2 Wednesday, 2 Friday). Of the 67
picks the rule actually changed, 7 sat on excluded games and ran 5/7 for the
arm against 2/7 for the chain — the excluded slice was unusually favourable,
which is why removing it costs anything at all.

**What this implies for the decision:** the arm is still worth playing
prospectively and still worth recording. The production-reachable estimate is
**+1.95 accuracy points, P+ 0.983 week-blocked**, with both blockings' intervals
above zero on the admitted subset; the correction costs about 0.23 points, not
the finding. Quote +1.95, not +2.18, for anything that describes what the wired
rule can actually earn. Recorded as
`nflcom_refresh_out2_starters_on_chain_gate_admitted` in
`registry/weak_signals.json` (`unresolved_below_power` — three seasons, mined
family, no admissible closing ground and none sought). The original
`nflcom_refresh_out2_starters_on_chain` entry is left in place unchanged as the
record of what the archival study measured.
