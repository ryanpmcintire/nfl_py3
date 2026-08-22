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
     fetched >= Friday 16:00 ET of that game week AND fetched < earliest
     kickoff among that week's games. The refresh pass FAILS OPEN: if the
     current week's page is absent or fails the timing gate, it keeps the
     incumbent chain pick for every affected team-game and records the skip.
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
