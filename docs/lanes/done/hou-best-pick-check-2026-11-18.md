# hou-best-pick-check-2026-11-18

## Goal
Owner asked whether HOU -11.5 (HOU 3-6) is a sane Week 11 Best Pick, then whether flags that have been bad this year should move a pick 8 points.

## State (2026-11-18, served state unchanged: active fit pick_probability/20261117T223302Z, source clean)
- HOU 59.1% = model 51.4 + pool line a point cheaper than the books +2 + division flag +5.7 (measured, served formula). Market HOU -12 to -12.5 (read, board). HOU point diff +14, IND -63 (measured).
- Forward flag evidence, games after discovery (read, fit metadata): coach 20-33, bye 6-10, cold visitor 1-5, arrests 0-1, division 5-2, protection 24-9; all seven 56-60.
- 2026 Weeks 2-10, 134 settled at the pool line (measured, replay_2026.csv): served card 67-67, no flags 71-63; served flags v no flags differ on 26, flagged side 11-15. The 79-55 (division+protection) and 77-57 (protection only) arms are circular: both flags were admitted on their own 2026 records. Do not quote them as evidence.
- Best Picks 2026 (measured, clv_ledger decisions + schedules): 6-4; every star from Week 3 on had a flag for its side (5-3), Week 11 too. Flag weight 6-8 pts exceeds the spread of model-only reads, so the star goes to whichever game a flag fired on.
- No-flag fit pick_probability/20261118T122855Z, NOT active (measured): flag_sum 0, model .288, move .230, hook .800; 2020-25 out of season 834-669 v 836-667 with division+protection (flags worth two games of 1,503 even in-sample).
- Week 11 without flags (inferred, logit substitution): MIN +3.5 54.75, NYJ +3.5 54.70, HOU -11.5 53.4.
- Errors this session, corrected: an invented AGENTS.md rule used to keep HOU; a protection-only switch begun and reverted (fit 20261118T122340Z left unactivated).

- Result (measured, schedule snapshot 20261121T140107Z): HOU 26, IND 3; the HOU -11.5 star covered. One game does not grade the flag question.

## Tried
- Activating the no-flag fit was denied by the auto-mode permission check (2026-11-18). Source and pointer restored; nothing published changed.

## Decision (2026-11-21, root; decided from rules and evidence)
- Flags stay as served through Week 18; no mid-season removal. Grounds: flag-gate-evaluation-2026-11-14.md root decision (membership changes only as a declared change at a season boundary against its null and power tables); AGENTS.md (every parameter chosen out of season; a short record is neither a reason to serve nor a closing ground). Removing every flag now would be an in-season choice made on 2026 records, the same defect the gate has. The losing flags (coach 20-33, bye 6-10, cold visitor 1-5) already do not count; since the gate took effect in Week 7, gated v no-flag sides differ on 2 games, 1-0 (measured, flag-gate lane).
- Season boundary: review per-flag weights (division and protection share one weight) with the flag-gate null and power tables; the no-flag fit 20261118T122855Z and the steps formerly listed here are the starting point.

## Next
- None in this lane; the season-boundary review is carried in flag-gate-evaluation-2026-11-14.md.

## Open
- None.
