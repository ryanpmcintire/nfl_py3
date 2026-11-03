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
