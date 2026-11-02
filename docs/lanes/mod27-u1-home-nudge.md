# mod27-u1-home-nudge
## Goal
Does dropping the home nudge (residual_location median, mass_preserving_lattice.py:152, added to predicted margin before the discrete read) improve the served pick probability at the opener?
## State
Done 2026-11-02. Harness: scratchpad/agentU1/ (inv.py, ns26.py, u1.py, out.txt, results.json, cells.json). Served point = predicted_margin + median(walk-forward residuals); per-week shift recovered by inverting the production reader to the stored discrete read (exact, sd 0.03-0.06 within week). Median shift: 2020 -1.49, 2021 -1.73, 2022 -0.92, 2023 -0.06, 2024 +0.46, 2025 +0.39, 2026 +0.37 (sign change confirmed).
LOSO 2020-25 stack refit (n 1503): served 833-670, no-nudge 819-684; log loss .68719 vs .68706 (no-nudge better by .000135, 95% -.00045..+.00074, P(pos) .68); Brier .24705 vs .24699; disagreements 62, no-nudge 24-38; accuracy diff -0.93 pts (P(pos) .03). 2020-22: LL diff -.00026 (P .26), 23-25: +.00049 (P .87), disagreements 19 with no-nudge 5-14. 2026 fwd (n 103): 57-46 vs 58-45 served; LL +.0028 (P .76), 13 disagreements 6-7.
Looks: 5 recorded cells (family mod27_home_nudge) plus reliability/chain/base-read descriptives (about 8 looks total).
## Tried
No-nudge base only, other terms unchanged; intercept absorbs part (-0.057 vs -0.012).
## Next
Not served: no beat on opener grade. Possible: shift fitted as a stack term rather than fixed median.
## Open
One 2026 game (CHI-SEA wk8) missing from chain.csv, dropped.
