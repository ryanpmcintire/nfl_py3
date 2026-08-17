# XLG-04: Cross-league role-delivery replication — predeclaration

Predeclared: 2026-08-16 (US), before any delivery or absence result was
computed in either league. The frozen constants below are mirrored in
`src/nfl_ats/cfb_roles.py`; the experiment runner records
`hypothesis_frozen_before_scoring: true` and this document's definitions in its
artifact metadata. Results, when they exist, are appended in a separate section
below the predeclaration and never edit it.

## Hypothesis under replication

PER-12 (NFL, snap-share based) found that injury-listed players who play at all
deliver approximately their full prior role: the median realized-to-prior
snap-share ratio was 1.011, and 55.8% of played rows met or exceeded their
prior role. XLG-04 asks whether the same mechanism — **"a player who
participates at all delivers roughly their accustomed share"** — holds in
college football, where no injury, lineup, or snap data exists and
participation must be measured through credited actions in play-by-play text.

This is a participation-level replication. It touches no ATS outcomes, no
spreads, and no game results in either league. Any eventual ATS-relevant
candidate built on this mechanism must go through the frozen XLG-03 CFB
benchmark separately.

## Measured quantity

For each game, team, and **action type**, a player's share of the team's
credited actions:

| Action type | CFB definition (play-by-play credits) | NFL definition (official weekly stats) |
|---|---|---|
| `dropback` | `pass == True` plays crediting `passer_player_id` | pass attempts + sacks taken |
| `carry` | `rush == True` plays crediting `rusher_player_id` | carries |
| `reception` | `type.text == "Pass Reception"` crediting `receiver_player_id` | receptions |

Positions are deliberately defined by football function (the action type), not
roster labels: CFB game rosters carry no position column.

**Prior role state**: per (team, player, action type), an exponentially
weighted average with span 8 (alpha = 2/9) over the player's *appearance*
games only — no zero-imputation for missed games, no offseason decay, and a
fresh state after a team change. This mirrors the NFL role-state update in
`src/nfl_ats/players.py` exactly.

## Frozen configuration

- Seasons: 2013–2025 inclusive, regular season only, both leagues. CFB games
  restricted to the XLG-03 canonical FBS-vs-FBS table.
- Qualification: prior share ≥ 0.50 (dropback), ≥ 0.20 (carry), ≥ 0.15
  (reception), with ≥ 3 prior appearances.
- Team-game validity: ≥ 10 team dropbacks, ≥ 10 carries, ≥ 5 receptions;
  smaller team-games are excluded from states and outcomes.
- CFB measurement gate: a (season, action type) cell enters only if ≥ 95% of
  eligible plays carry a credit; failing cells are excluded and logged, never
  patched. (Motivating fact: receiver credits on *incompletions* are
  non-stationary — 77% in 2014 down to 43% in 2023 — which is why the frozen
  receiving measure is reception share, not target share.)
- **Delivery sample (primary, gated)**: qualifying player-games with ≥ 1
  credited action. Ratio = game share ÷ prior share.
- **Replication gates (per action type, all three required)**:
  1. CFB median ratio ∈ [0.90, 1.10];
  2. |CFB median − matched NFL median| ≤ 0.10 (NFL computed with the same
     action-share definitions, not PER-12's snap shares);
  3. CFB severe under-delivery rate (ratio ≤ 0.5) ≤ 15%.
- **Absence/replacement analysis (secondary, ungated, descriptive)**:
  qualifying players with zero credited actions in a valid team-game, plus how
  the vacated share redistributes and whether team volume changes.

## Declared limitations

1. **Absence is a proxy, not ground truth.** The CFB participation contract is
   explicit: credited actors are positive evidence only; absence of credit is
   never evidence of absence. The gated claim therefore lives entirely in the
   delivery sample, which conditions on positive evidence (≥ 1 action). The
   absence analysis is descriptive groundwork for XLG-05/XLG-07 and asserts
   nothing.
2. **Kneels are included as carries in both leagues.** NFL official weekly
   stats cannot exclude them, so the CFB side keeps them too; matched
   definitions were preferred over a one-sided filter.
3. **Reception share is outcome-contaminated relative to target share** (a
   catch depends on the throw); it is the stable, matched measure available in
   both leagues.
4. **Measurement sources differ**: CFB counts come from play-text credits,
   NFL counts from official stats. The CFB coverage gate bounds, but does not
   eliminate, this asymmetry.
5. The delivery sample conditions on playing; it says nothing about
   *availability* (who plays), which remains the NFL-side PER-11 model's job
   and is unlearnable in CFB (XLG-07 fails closed historically).

## Relationship to the roadmap

- Replicated action types license predeclaring a CFB-side role-loss feature
  for the XLG-03 benchmark (the week-blocked-interval yardstick) and inform
  the XLG-05 transfer design.
- Non-replication is a real result: it would mean the NFL role-delivery
  mechanism is league-specific (or snap-share-specific) and weakens the case
  for CFB-pretrained role features. Either way the result is retained.

---

## Results (run 2026-08-17T03:12Z, artifact `artifacts/cfb_role_experiments/20260817T031206Z`)

Recorded once, against the frozen gates above. No gate was adjusted after
seeing data.

| Action type | CFB n | CFB median | NFL median (matched) | CFB severe-under (≤0.5) | Gates | Verdict |
|---|---|---|---|---|---|---|
| dropback | 14,436 | **1.043** | 1.009 | 8.1% | 3/3 | **Replicated** |
| carry | 29,541 | **0.995** | 0.970 | 14.9% | 3/3 | **Replicated** |
| reception | 32,849 | 0.911 | 0.906 | 19.1% | 2/3 | **Not replicated** (severe-under gate) |

- **The mechanism replicates cleanly for dropbacks and carries**: a college
  player with a material prior role who participates at all delivers
  approximately the full prior share, in both leagues, with matched
  definitions. The NFL matched dropback median (1.009) also independently
  agrees with PER-12's snap-share median (1.011), tying the action-share and
  snap-share measurements together.
- **Reception failed exactly one gate, and the failure is symmetric across
  leagues**: CFB severe under-delivery was 19.1% against the frozen 15%
  ceiling — but the matched NFL value is 17.2%, which would also have failed.
  The league medians differ by only 0.005 (0.911 vs 0.906) and the gap gate
  passed easily. The honest reading: reception shares have fat under-delivery
  tails in *both* leagues (a catch depends on the throw), the cross-league
  agreement is strong, and the frozen absolute ceiling was calibrated for
  snap-like stability that reception volume does not have. Per protocol the
  verdict stands as **not replicated**; any retest with a re-calibrated gate
  is a new predeclaration, not a rerun.
- Both leagues under-deliver relative to prior at the median for receptions
  (~0.91) — the prior is an over-estimate for receiving volume generally,
  which is itself a replicated cross-league fact.
- The 2013 CFB reception cell was excluded by the coverage gate (the 2013
  play-by-play uses different play-type text and credits zero receptions);
  the gate failed closed as designed.

### Absence/replacement descriptives (ungated)

- When a qualifying QB is absent (proxy label), the top replacement takes a
  median **100% of dropbacks in both leagues**; team play volume barely moves
  (median team-volume ratio ~0.98 in both leagues, all action types).
- Median top-replacement carry share after a lead-back absence: CFB 0.45,
  NFL 0.57.
- **Composition caveat (important):** absence events conflate single-game
  absences with *permanent departures* (graduation, transfer, draft, release)
  because a qualifying state persists until the player reappears. CFB event
  counts (e.g., 140,630 reception events vs 32,849 delivery rows) are
  therefore dominated by departures, not injuries. Any future use of absence
  events as a feature must first separate departures from temporary absences;
  these descriptives characterize redistribution, not injury response.

### Roadmap consequence

Dropback and carry role delivery are now cross-league-replicated mechanisms:
a CFB-side role-continuity/role-loss feature family may be predeclared for
the XLG-03 benchmark (the week-blocked yardstick), and the XLG-05 transfer
design can treat "delivered ≈ prior when participating" as league-general for
those two action types. Reception-based features carry the recorded
non-replication.
