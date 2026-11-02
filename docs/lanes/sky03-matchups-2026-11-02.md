# Lane: SKY-03 matchup interactions

## Goal
Pair-level matchup effect beyond main effects, from pbp: (A) pass protection (pressure = sack or QB hit per dropback) x opposing pass rush; (B) receivers x coverage (EPA per non-sack dropback). One fitted term on the four-term probability, LOSO 2020-2025, opener.

## State
Done 2026-11-02. scripts/sky03_matchups.py; artifacts/sky03/ (reliability.json, fitted_term.json, record_batch.json, team_games.parquet cache). Recorded 6 cells, family sky03_matchups, unresolved_below_power.
Traits: EB-shrunk as-of team rates (current season prior weeks plus previous season), shrink constant from odd/even variance components of earlier seasons. Interaction c fit on all earlier games (expanding), x = expected dropbacks * (cA*zA + cB*zB*non-sack share), home minus away.
Reliability (odd v even weeks, 2016-2025, team-level interaction slopes, 32 teams): A offense r 0.07 [-0.27,0.40] p+ 0.66; A defense 0.15 [-0.25,0.54] p+ 0.78; B offense 0.34 [-0.08,0.64] p+ 0.95; B defense 0.27 [-0.11,0.57] p+ 0.93. Not zero.
Term (1,503 games): OOS Brier gain -0.00043 [-0.00088,+0.00003] p+ 0.03; log loss -0.00088; accuracy -0.67 pts [-1.53,+0.13]; decisive 38, term 14-24. In-sample +0.00002 p+ 0.56; gap +0.00045. Fold coefficients -0.081,-0.109,-0.030,+0.017,-0.033,+0.021 (unstable, mostly negative). c_A about -5, c_B about -1 EPA per unit product (mechanism sign correct for pooled c).

## Tried
Single summed term, two units. Not tried: separate A/B terms, play-level pressure/air-yards data (not in on-disk pbp subset), player-level OL/CB ratings (MOD-22/23 territory).

## Next
Split A and B as separate declared looks only if owner wants; richer pressure data (PFR/NGS pressure, coverage) would raise trait reliability.

## Open
ROADMAP SKY-03 row not updated (orchestrator). Reliability is not zero, so no closing ground applies.
