# acc-u5-denoised-target

## Goal
Train the base model (ridge alpha 10, 88 inputs) on a less noisy target for the same games; grade on the real margin vs the opener. Parent: docs/lanes/accuracy-ceiling-theory.md.

## State
2026-10-01 done, all three arms unresolved_below_power, recorded (family mod24_denoised_target, 3 cells in registry/weak_signals.json). Code scripts/mod24_u5.py, outputs artifacts/mod24_u5/ (report.json, per-game parquets, targets_sd_lambda.json, weak_signals_batch.json). Base reproduced 802-701 (2020-25), 1142-1089 proxy.
Targets (play-by-play, slope refit on seasons before each test season): T1 total EPA diff (R2 0.99 with real margin, so it is the real margin, SD 13.16 vs 13.17); T2 scrimmage EPA + success, fumble-lost plays set to out-of-season down/type mean, wp 0.05-0.95 only, no special teams (R2 0.66, SD 10.86); T3 lambda*real + (1-lambda)*T2, lambda per season from 4 inner seasons.
Results (measured, accuracy diff vs base, season-blocked P+): pooled 2011-25 T1 -0.27 (.20), T2 -0.16 (.39), T3 -0.54 (.14). True era 2020-25: T1 807-696 +0.33 (.87, 4/6 seasons, flips 51-46, temp log loss +0.00045 P+ .95), T2 791-712 -0.73 (.28), T3 787-716 -1.00 (.19). Proxy era: T1 -0.67 (.06), T2 +0.22 (.62), T3 -0.22 (.25). Temperature-recalibrated log loss pooled: T1 +0.0003 (.97), T2 +0.0007 (.88), T3 -0.0001 (.23); true-era lattice RPS T1 +0.0028, T2 -0.0067, T3 -0.0081.

## Tried
Inner lambda scored by logistic-temperature log loss on cover vs spread_line (not the opener; no inner opener before 2020). Looks: 3 arms x 2 eras, 75 inner lambda cells.

## Next
Inference: label noise is not the binding limit; T2 shrinks SD 17 percent but flips 40 percent of picks with no net gain, T1 has no denoising. A neutral variant (T2 with real-margin anchor ridge only on lineup inputs) is not queued. Parent verdict stands: information is binding.

## Open
T1 proper-score gain is small and positive in both eras (P+ .75-.95) while accuracy is flat; one of about 60 looks.
