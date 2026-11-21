# LEAD-107

## Goal
Read dispersion as precision: between-read disagreement of the frozen language reads sets how much weight each game's read term gets (ROADMAP LEAD-107). Mechanism: LEAD-100 measured between-read SD .052 logit against between-game .093; a game where repeated reads disagree carries less information. Fitted form: read term / (1 + disagreement), divisor fitted leave-one-week-out. Forward test of the LEAD-108 variant: lead107-forward-2026-11-16.md.

## Declaration, unit 1 (written 2026-11-21 09:58 ET, before any dispersion-error number was computed)
- Games: every game with at least 3 stored reads of one packet: the Week 9 Tuesday packet (source run plus 3 repeats in artifacts/lead99_llm_read/repeats = named arm, 4 reads; artifacts/lead99_llm_read/blinded/reads mapped by blinded_packet.json = blinded arm, 3 reads). 15 games, 1 week.
- Per game and arm: m = mean read logit, D = between-read SD (ddof 1). Later line L1 = median across served books of each book's last quote after the latest read instant and before the deadline (lead101_delta_read.served_quotes; no new line source). L0 = packet pool line. q = c (L0 - L1), c = served move_toward_home 0.2204 from artifacts/pick_probability/20261111T011859Z/coefficients.json (fitted out of season, not here). Error e = |m - q|.
- Statistic: Spearman rho(D, e), mechanism sign positive; game-cluster bootstrap 95 pct interval, probability_positive; within-week permutation of D, upper p. Decisive record: high-D games with above-median error. Median-split error gap as a second look. Family lead107_read_dispersion_weight: 2 arms x 2 = 4 looks; rho cells recorded (unit correlation).

## State (measured 2026-11-21 10:00 ET, `uv run --no-sync python scripts/lead107_dispersion_error.py`, exit 0, ruff clean; artifacts/lead107_dispersion_error/summary.json, per_game.csv)
- Named (4 reads, 15 games): record 3 of 7 high-D games above median error (low-D 4 of 8). rho -0.084 [-0.602, +0.461] P+ .377, null mean +0.001, upper p .615. Median-split gap -0.001 logit [-0.151, +0.106] P+ .338, p .515. Mean D 0.052, mean error 0.205.
- Blinded (3 reads, 15 games): record 3 of 6 (low-D 4 of 9). rho +0.034 [-0.560, +0.568] P+ .557, upper p .454. Gap +0.013 [-0.124, +0.135] P+ .539, p .421. Mean D 0.041, mean error 0.200.
- Reading (inferred): no sign either way; with between-read SD about a quarter of the mean error, read noise cannot be the main part of the miss on these 15 games, so a dispersion weight would barely move the term. One week, one packet: `unresolved_below_power`, not refuted (no interval wholly on the wrong side, no reliability measure of D yet).
- Registry: 2 cells recorded (`--replace`, exit 0, "recorded": 2), registry/weak_signals/lead107_read_dispersion_weight.json; `pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
- 11-03 novelty check (nearest LEAD-100, LEAD-102, LEAD-106). 11-21 unit 1: scripts/lead107_dispersion_error.py, stored reads only, no model call.

## Next
- More games with at least 3 reads of the same packet are the only way to power this; no stored repeats exist beyond Week 9 Tuesday (measured: repeats/ holds 3 files, all of that packet). Root decides whether to queue K repeat reads of a later packet (API spend). Then rerun the script unchanged; it picks up every source run with repeats.
- Split-half reliability of D itself (reads 1-2 vs 3-4) before any weight is fitted.
- 2026-11-21 11:24 AM ET (LEAD-100 unit 2): Week 11 now has repeat reads for the dispersion test: 3 valid repeats of runs/2026-week-11-deadline-20261119T230145Z.json in artifacts/lead99_llm_read/repeats (4 reads per game; 12 games qualify, IND_HOU kicked off 11-19 before the repeats, so drop it). Measured: Week 11 between-read SD 0.029 logit, max 0.013 prob, so D has little range there. Outcomes not read. Rerun lead107_dispersion_error.py once Week 11 lines settle.

## Open
- The later-line target goes through the served move coefficient; a discrete-margin mapping is the alternative and is not built.
- Zero crossing closes nothing; one fitted probability selects the side; nothing here is served.
