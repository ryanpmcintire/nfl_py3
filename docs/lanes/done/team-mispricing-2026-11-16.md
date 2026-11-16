# Team mispricing (goal-65 wave), 2026-11-16

## Goal
Persistent team-level mispricing at the opener, most of all on quiet games (line moves < 1 pt after Tuesday; 59% of games, served ~50%). Script scripts/team_mispricing.py, artifacts artifacts/team_mispricing/.

## Novelty (read, grep of registry/weak_signals, ROADMAP.md, docs/lanes)
- (b) team-specific multi-season ATS error: NOT novel. The served base already carries state_ats_residual (features.py:313, 357-386): EWMA of each team's ATS residual across seasons with offseason retention toward the league mean, i.e. a shrunk multi-season team error (closes, not openers). Also public_claim_ats_streak_regress (pp .919).
- (a) public-team premium: mechanism touched by bias_battery_marquee_favorite (hand-picked DAL/GB/PIT/NE list, favourites only, pp .099), tv_attention_fade_on_production (trailing 3 audiences, pp .717), attention_battery (Wikipedia pageview z, short-term), public_betting_battery (game bet share), futures_anchor (college only, futures-minus-opener gap, sign reversed). Never tested: a measured, persistent, league-revealed popularity level as a continuous term on all games and on quiet games. Instrument chosen: prior-season national-window count (the league allots primetime to the teams that draw audiences). Kalshi/Polymarket volume only 2024-26 (too short for a 2011-25 screen); futures rank already used by futures_anchor.

## Declaration (stamped 2026-11-16T22:33:39Z, before any outcome joined)
- pop(team, s) = count of the team's season s-1 REG games in a national window (weekday Thu, Sat or Mon, or Sunday gametime >= 20:00; repo definition from attention_followup_both_cold_non_primetime), franchise-mapped (OAK->LV, SD->LAC, STL->LA), z-scored across teams within s-1.
- T1 pop_diff = pop_z(home) - pop_z(away). T2 pop_resid: pop_z residualised (within season s-1, OLS across teams) on prior-season point differential per game, then home minus away (removes the quality part, which the opener prices).
- Expected sign NEGATIVE on home_cover (more popular side overpriced), larger on quiet games.
- Screen vs the close, REG 2011-2025, pushes dropped: OLS slope of home_cover on the standardised term (accuracy points per SD); populations ALL and QUIET (|close - opener| < 1, opener = Tuesday capture else SBR open; games without an opener leave QUIET only). 4 looks (2 terms x 2 populations), one family.
- Null: 200 within-season shuffles of the term; survivor = slope beyond the null 95th percentile in the declared direction AND same sign in halves 2011-18 / 2019-25. pp from 2000 season-block bootstraps.
- Served look only for a survivor: one term added LOSO 2020-25 on the QW served logit at the opener, pushes kept, ALL and QUIET (Tuesday-to-deadline move < 1), 200 shuffle null, halves 2020-22 / 2023-25, accuracy points and log loss vs QW 836-667.

## State (done 2026-11-16 17:37 EST; measured, artifacts/team_mispricing/screen.json, screen.log exit 0)
- Screen vs close, REG 2011-25, 3,818 non-push games (all have pop), quiet 1,809 (opener coverage 3,803; opener-close corr .951; SBR sign +1). Effect = accuracy points per SD in the declared direction (fade popular).
- pop_diff ALL: -0.62 [-2.55, +1.37], pp .262, null pct 15.5 (p95 +1.18); halves -0.47 / -0.79; 7/15 seasons declared; fade record 1586-1620.
- pop_diff QUIET: -0.21 [-2.70, +2.43], pp .425, pct 47.5; halves +0.73 / -1.16; fade 740-745.
- pop_resid_diff ALL: -0.74 [-2.45, +0.92], pp .189, pct 18.5; halves -0.07 / -1.58; fade 1890-1927.
- pop_resid_diff QUIET: -0.40 [-2.81, +2.28], pp .377, pct 35.5; halves +0.75 / -1.65; fade 912-896.
- No survivor (gate: null p95 and declared sign both halves), so the served look was NOT run. 4 looks, family team_mispricing_2026_11_16, recorded unresolved_below_power (record --batch exit 0; pool --league nfl --effect-units accuracy_points exit 0).

## Tried
- Run 1 crashed: data/raw_schedules snapshots hold only 2026; fixed to data/raw/2026*/schedules.parquet (2009-2026, as v4_candidate.py). No outcome read from run 1.

## Next
- (b) is already served (state_ats_residual, cross-season EWMA with offseason retention); an opener-based variant would be a near-duplicate, low value.
- Point estimates lean the other way (popular teams cover slightly more; pp of the reverse ~.74-.81 on ALL), matching futures_anchor (futures info the openers lag, 13/13 folds). A declared reverse-sign "popular/preseason-favoured underpriced" term is a NEW family look, not a re-read; screen before any served look.

## Open
- None.
