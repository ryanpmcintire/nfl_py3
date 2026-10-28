# Pool hook term

## Goal
Test whether a hook term (logit discrete cover prob at the pool line minus at the Tuesday line) added to the four-term served pick fit improves it when graded at a simulated half-point pool line.

## State
Script tests/scratch/pool_hook/hook.py, output tests/scratch/pool_hook/out.txt (measured). Whole-line Tuesday opens 2020-25: n 661, push 5.1 pct, hook-side cover 52.57 pct (= 0.5+push/2). LOSO +hook vs four-term: accuracy -0.03 pts (P+ 0.47), Brier gain 0.00026 (P+ 0.997), log loss gain 0.00051 (P+ 0.996); hook coef 0.77-0.88 per fold. 2026: 39 whole-line games, hook side 22-17, card took it 22 of 39 (13-9 when took, 8-9 when not). Recorded as family pool_hook_term_2026_10_27 (2 cells).

## Tried
Pooled prior-season push mass within +-1 spread point; hook = logit(0.5(1-/+p0)); 2020 has no prior seasons so hook 0 there.

## Next
Served opener population drops pushes, so the graded fit has no hook information; 34 push games were added with neutral imputed flag/move terms. Rebuild push rows with real features before any serving decision. Hook coef near 1 matches the mechanism (it is partly mechanical).

## Open
Does serving need the pool line captured per game (forward log)? 2020 prior-season p0 gap.
