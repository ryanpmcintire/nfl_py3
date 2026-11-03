# XLG-13 Screen-to-served transfer audit

## Goal
Does a gain on the pooled 14-input harness (scripts/pooled_screen.py) predict the gain on the 88-input served model, NFL 2020-25 opener (base 786-717 screen, 802-701 served; accuracy points).

## State
Family xlg13_screen_transfer declared before any sign was read; 2 looks (Spearman, Pearson), recorded unresolved_below_power. Table (screen NFL-opener gain, artifacts/pooled_sweep/<arm>.json; served gain):
ridgecv +0.93 (cv.json); -0.73 (mod23-unit16 R, unit17 C)
enet +1.40 (enet.json); -3.19 (unit16 E)
accalpha +0.80 (accalpha.json); -1.80 (unit18 ACC)
huber +0.67 (huber.json); -0.87 (mod23-unit10r-robust-wide H)
n=4. Spearman -0.40 (boot P+ .25, exact permutation P(rho<=obs) .375), Pearson -0.84 (P+ .21), sign agreement 0/4. Bootstrap intervals [-1,1] (n=4).
Not paired: LL-chosen alpha (unit18 LL -1.13) has no screen arm; hgb/recency/ts/winsor served gains lack a same-window number.

## Tried
Tabulation from existing artifacts only; no fits. Script scripts/xlg13_screen_transfer.py (ignored).

## Next
Limit written to mod24-more-games lane: screen not a pre-screen for served changes. More pairs need served runs of slope, ts, hgb on 88 inputs.

## Open
n=4, all shrinkage-type, so rank test has no power; the sign result (0/4) is the usable evidence.

## ROOT CORRECTION 2026-11-03 13:55
- The pooled harness used the closing spread_line as model input while grading at the opener (scripts/mod24_more_games.py:63-73), so these results include close-minus-open line movement unavailable on Tuesday; not evidence for the served model until the opener-input rerun (see docs/lanes/mod24-more-games-2026-11-03.md).

## RERUN 2026-11-03 17:15 (fixed opener-input harness, 35f7f070)
Pre-fix screen effects (artifacts/pooled_sweep, 13:02-13:07, before fix 14:05) are struck: ~~ridgecv +0.93, enet +1.40, accalpha +0.80, huber +0.67; Spearman -0.40, Pearson -0.84~~.
Measured, artifacts/pooled_sweep_fixed/*.json via scripts/pooled_screen.py, NFL opener 2020-25 n=1503, base 754-749 (was 786-717): ridgecv 768 wins +0.93 [-0.26,2.14] P+ .93; enet 763 +0.60 [-1.47,2.47] P+ .71; accalpha 748 -0.40 [-1.55,0.66] P+ .22; huber 765 +0.73 [-1.41,2.78] P+ .76.
Script now reads screen JSONs and served.json as data (scripts/xlg13_screen_transfer.py --screens --served), report artifacts/xlg13_screen_transfer/report_fixed.json. Old data reproduces -0.84/-0.40/.375 exactly.
Served (unchanged, from mod23 units; NOT re-verified as post-fix): -0.73, -3.19, -1.80, -0.87. New Spearman +0.80 (boot P+ .94; exact permutation P(rho>=obs)=4/24=.17), Pearson +0.28 (P+ .85), sign agreement 1/4. Bootstrap intervals [-1,1] (n=4).
Reading (inferred): the "screen anti-predicts served" result was a closing-line artefact; with n=4 the new rank is positive but below power. Served gains were all negative, so the screen ordering is not demonstrated either way. Not recorded as a cell (descriptive n=4).
