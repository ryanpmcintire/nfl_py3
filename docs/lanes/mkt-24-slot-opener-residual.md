# MKT-24

## Goal
Kickoff-slot opener residual: is the opening line systematically off by Thursday, Sunday early/late/night, Monday, Saturday, holiday or international slot (ROADMAP MKT-24). Nothing here is served.

## State
Unit 1 DONE 2026-11-03: no slot mispriced; 20 cells unresolved (scripts/mkt24_slot_residual.py, no arguments -> artifacts/mkt24_slot_residual/report.json).
Unit 2 DONE 2026-11-14 17:17 (local): `--split era parity parity_opener` -> artifacts/mkt24_slot_residual/reliability.json; 38 cells recorded `unresolved_below_power` (family `mkt24_slot_opener_residual`, names `mkt24_u2_*`, batch weak_signals_batch_u2.json); `pool` loads. All numbers below **measured** this session.

Declared 17:09 before computing: residual = final home margin minus the line, pushes kept. Line by era, never mixed in a game: 2020-25 Tuesday multi-book opener (1,534 games); 2011-19 SBR archive opening number, a proxy and not the Tuesday opener (2,303 games, 2 flagged ambiguous); no closing line. Splits: `era`, `parity` (odd/even 2011-25), `parity_opener` (odd/even inside 2020-25, 3 blocks a half, sensitivity). 113 looks. A slot repeats only if its sign agrees in era and parity. Zero-reliability ground needs the whole weighted-correlation interval at or below zero in both primary splits plus a planted control that sees a detectable miss.

Mean residual in points [season-block 95%], home record at the line, exact binomial p:

| slot (n 2011-19 / 2020-25) | 2011-19 SBR open | 2020-25 Tuesday opener | sign era / parity |
|---|---|---|---|
| Thursday 121/89 | +1.44 [-0.50,+3.35] 67-52 p.20 | -0.88 [-2.54,+0.81] 39-50 p.29 | no / no |
| Sun early 1221/754 | -0.53 [-1.24,+0.06] 570-610 p.26 | +0.86 [-0.03,+1.65] 367-368 p1.0 | no / no |
| Sun late 558/379 | -0.20 [-1.30,+0.73] 263-278 p.55 | -0.06 [-1.03,+0.71] 192-176 p.43 | yes / yes |
| Sun night 150/102 | +3.34 [+1.59,+5.21] 79-65 p.28 | -0.56 [-2.06,+1.24] 51-49 p.92 | no / yes |
| Monday 149/108 | -0.26 [-2.06,+1.69] 70-77 p.62 | -0.06 [-1.53,+1.50] 50-57 p.56 | yes / no |
| Saturday 43/45 | -0.60 [-3.87,+0.74] 16-25 p.21 | -1.33 [-6.22,+2.52] 18-26 p.29 | yes / no |
| Holiday 32/29 | -3.94 [-7.52,-0.02] 9-20 p.06 | -2.64 [-6.98,+2.28] 12-17 p.46 | yes / yes |
| International 29/28 | -1.88 [-6.81,+2.15] 11-18 p.27 | +1.46 [-3.36,+7.67] 17-11 p.35 | no / no |

- Slot-pattern split-half reliability (weighted correlation of the 8 slot means across halves; permutation = slot labels shuffled inside season-week): era -0.27 [-0.65,+0.38], probability_positive .27, perm p .81; parity +0.51 [-0.44,+0.82], .80, perm p .13; opener-only parity +0.03 [-0.45,+0.50], .61, perm p .49. Unweighted +0.20 / +0.48 / -0.36. Signed root of the cross-half covariance (size of a real slot spread): -0.48 / +0.52 / +0.18 points.
- Sign agreement in both primary splits: 2 slots (holiday, Sunday late); 2.76 expected by chance; perm p .83.
- Thursday alone: era product of half means -1.27 pts^2 [-5.64,+1.54], P(same sign) .20, perm p .86, concordance -0.89 [-1.00,+0.94]; parity product 0.00 [-3.18,+3.31], concordance 0.00 [-1.00,+1.00]; seasons positive 7 of 15 (p 1.0); pooled +0.46 points, 106-102.
- Planted control (constant Thursday-only miss on label-shuffled data; cover shift measured on the real residuals): 1/2/3/4/6 points = 4.5/7.5/10.6/13.5/20.0 cover points. Product test sees it .17/.55/.88/.99/1.0; the pattern correlation only .06/.12/.24/.42/.81.
- Holiday is the one slot the rule names: pooled 21-37 at the line (exact binomial p .048, one of 8 slots), -3.32 points, era product perm p .041. It is 1.6% of games, so 0.22 points of card accuracy against about 1.5 detectable by a one-term card test (**read** mod-36 lane). **Inferred**: Thanksgiving hosts are fixed (Detroit, Dallas), so slot and team are confounded.

Implication: the reliability interval does not support zero reliability (it reaches +0.38 and +0.82, and the correlation instrument misses a 4-point Thursday miss more often than not), so the mechanism stays `unresolved_below_power`; nothing closed. For Thursday the control bounds only a constant miss of about 3 points or more; smaller stays unresolved. Sunday night 2011-19 (+3.34, whole interval above zero on both interval types) does not show at the Tuesday opener (-0.56, probability_positive .24): a difference of line source or magnitude, not shown absent.

## Tried
- Unit 1: ridge slot dummies on logit(p_open), inner-LOSO C, intercept-only control.
- Unit 2 post hoc (only widens): season-block intervals for slots with about 3 games a season came out narrower than the exact binomial, so recorded slot cells carry the wider of block and game-level intervals; the table above shows the declared block interval.
- Concordance of two half means is the natural one-slot reliability but spans nearly [-1,+1] under any truth; the product test is the usable Thursday statistic.

## Next
1. Holiday forward check (declared 2026-11-14, one look at 2026 season end): 2026 holiday games' home cover rate at the Tuesday opener, one-sided low, exact binomial.
2. Sunday night line-source question: grade SBR open and Tuesday opener on the same 2020-25 games if sbr_odds covers them, before any era claim.
3. Per-slot planted control for holiday (n 61) to state its own detectable size. Orchestrator decides whether Thursday's 3-point bound merits `bounded_by_control`.

## Open
Zero crossing closes nothing; one fitted probability selects the side; constants chosen leave-one-season-out; every cell is a look; verdicts go through `nfl-ats weak-signals record`; the 2011-19 line is a proxy and is never called the opener.
