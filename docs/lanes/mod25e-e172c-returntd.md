# E172c return-TD excess

## Goal
Find why sim returns turnovers for TDs more (def TD/game .458 v .395; share of turnover rows 4.5% v 3.4%).

## State (measured, tests/scratch/e172c/r1-r5.py)
- Sim def-TD share of turnovers 4.52% v real 3.30% at same field mix; INT 13.2% v 11.2%, fumble 11.7% v 9.2%. Excess is within field-position bin (ratio 2.0 to 1.2), not composition, not geometry (sim spot below source 60%).
- E172d: Geo (mod25e_geo.py:344) is NOT the carrier: free-row share 3.36% v 3.32% natural; symmetric-window variant refuted and deleted. 894 mismatched def-TD rows = logging artifact mod25e_gz.py:127-128 (SERVEDIDX fixes); geo.py:395-401 cannot create a def TD.
- E172e (r5.py, r5b.log, 12,000 real run/pass query states, K 200, cumulative weights via mod25e_draw.stage_weights): return-TD share of drawn turnover rows: uniform k-set 3.91%, +team kernel 3.99%, +IPW 4.05%, +tilt (current) 4.04%, +dk 3.95%, TILTL 3.92%; pool-wide 3.52%. No step creates 1.25x; the k-set is already 1.11x pool at matched states (state mix). Turnover share of rows 5.46% to 5.67%.
- Defense gradient (strongest third of def_row v weakest, return-TD per turnover): uniform 4.33 v 3.53; team kernel 5.24 v 3.24 (steepens most, sim04_engine.py:523-526); tilt 5.58 v 3.10 (mod25d_variance.py:956); TILTL 5.54 v 3.28 (barely changes). Right sign, steeper than natural; no real-data gradient yet.
- PASS_W not installed in the offline build (ns lacks it): not replayed.

- E172h (tests/scratch/e172c/h1-h3.py; sim 314,635 turnover rows s11-13, real 20,242): REAL gradient (pregame def_row, strong v weak third) 4.31 v 2.64, diff 1.67 [1.14,2.22] 90%; slope -0.86%/SD [-1.09,-0.63]. SIM (def_sim reconstructed from generator latents, scale 1 assumed; share 4.518 matches) 5.45 v 3.68, diff 1.77 [1.61,1.92]; slope -0.83%/SD. Per-SD gradient matches: sim is NOT too steep; its level is +1.1 pts in both thirds. Held-out opposite-parity EPA rating (corr .77 with def_row, noisier): 3.84 v 3.11, slope -0.24 [-0.44,-0.04], attenuated, uninformative on steepness.
- State mix explains ~0: real rtd share at sim mix (fp x code x down x qtr, 312 cells) 3.44 [3.23,3.65] v real 3.40 v sim 4.52. Within-state excess 1.1 pts (1.31x). Mixes agree (fp mean 33.0 v 32.3, INT 71.9% both).
- Carrier: neither mix nor defense gradient nor kernel bandwidth (axish not needed). Residual is a uniform within-state level lift after the draw (pool 3.52, K-set 3.9-4.05, served 4.52).

## Tried
Gap-conditioned share, bin decomposition, Geo replay, symmetric window, weight-stage replay.

## Next
1. Replay mod25e_gz/geo replacement on the same query states with SERVEDIDX=1 and compare return-TD share of served-source rows to K-set 3.95 (offline, no sim).
2. Done in E172h: gradient and state mix cleared.
3. SAFEG first; crH patch deferred. TILTL does not fix this.

## Open
Where the remaining 1.14x (4.0 v 3.52 pool, then served 4.26) arises after the base draw.
