# mod25e E107 Q4 clock-ended drives (state-aware half-end)

## Goal
Why Q4 last play <=5s is .212 sim v .128 real when Q2 now matches (.270 v .263). Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhgh (CKH=1), seeds 11-13. Name the mechanism from real behaviour; hook default off.

## State (2026-10-09, measured)
- Decomposition (tests/scratch/e107/q4.py; clock-ended Q4 drives by score at last play, real v sim): P(last<=5s) t9+ .196 v .405 (share .185/.181), t4-8 .534/.526, t1-3 .542/.495, tie .241/.274, lead 1-8 .082/.106, lead 9+ .025/.079. Within-state gaps: t9+ -.209 x .18 = .038, lead9+ .054 x .36 = .019, lead1-8 .024 x .28 = .007 of the .084. Real t9+ drives: 3.95 plays, hl 20, tou .15 v sim 5.0 plays, hl 15, tou .47. Leader kneel share ~.95 both.
- Hazard table (haz.py; P(play is last play of game | snap hs band x score band), real v sim): run t9+ hs(20,30] .73 v .43, (30,45] .74 v .33, (45,60] .61 v .22; lead 9+ run (30,60] .62 v .21; lead 1-8 (30,45] .91 v .22; tie/trail 1-8 runs real ~0-.19. Pass t9+ (10,20] .28 v .12. Kneel and timeout use are not the gap (kneel P(last) .86-1.0 both).
- Carrier: CKH pool draw shrinks cells toward the hs-bin marginal (fit_h.json a=128 run, 512 pass; ckc.smooth_pm), chosen by pmf ll which the bulk elapsed dominates. Held-out (LOSO) CKH P(end) for run, hs(20,45]: t9+ .35 v obs .49; t1-8 .28 v .13; tie .29 v .12; lead 9+ .32 v .43 (hier.py: a two-level pmf hierarchy does not help pmf ll, -.0006). Real behaviour: decided games (down 9+ or leading) let the clock run, contested ones (within 8) hurry; state is washed out of P(end).
- Hook CKS=1 (needs CKH=1; scripts/mod25e_cks.py, branch in mod25e_ckc.py pol, label suffix ke): P(play ends the half | hs, qtr, score band, timeouts) from a 3-level shrinkage ladder (hs x qtr -> x band -> x timeouts), replaces the pool tail mass; non-end elapsed drawn from the existing CKH pmf truncated below the remaining time. Fit (cks fit, 972 looks over run+pass, grid edges/sign/timeouts/qtr-split/a1/a2): run LOSO binary ll -.5127 v best hs-only -.5955 (+.083, 9/9 seasons), pass -.2599 v -.2641 (+.004, 6/9). Held-out P(end) by band matches obs (run hs(20,45]: .50/.13/.14/.29/.41 v obs .49/.13/.12/.31/.43; hs-only .34/.30/.33/.26/.29). Fit artifacts/mod25e3/cks/fit.json.
- Smoke s31 1x4 launched (log artifacts/mod25e3/ke_smoke_s31.log, label ...tbkhghke).

## Tried
- Kneel eligibility/timeouts/CDR/DECIDE as carriers: kneel P(last) and class shares match real in every state; tou differences follow from the longer sim drives.

## Next
Score smoke v artifacts/mod25e3/smoke/e5_..tbkhgh_s31 (clock2 as script with CLK_LABEL; Q4 <=5s, clock-ended hs0/hl/n/yl0 by state, P(final3|tied last 5)). If direction right, 3-seed run seeds 11-13 + bootstrap (E100_BASE/E100_CAND). Record run ll as a generator lane result.

## Open
Leader start-time (l9+ hs0 114 v 92 sim) and start spot (yl0 63 v 59) are separate composition gaps. Hook not yet in the 3-seed base.
