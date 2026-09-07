# Gaussian median promotion ? 2026-09-07

Decision (read: coordinator instruction, 2026-09-07): switch production to
`gaussian_median` before Tuesday September 8 at 09:15 ET. The point model,
residual sample and Gaussian scale stay the same; location becomes the sample
median. This implements the standing expected-value decision.

Reported (unverified here; `docs/residual_offset_study.md`, implementation and
composed-card measurement): model-only accuracy is 53.96% versus 53.36%,
+0.599 accuracy points, week-blocked 95% [-0.734, +1.940],
probability_positive 0.795. Through the played overlay union it is 55.42%
versus 55.22%, +0.200 points, week-blocked 95% [-1.128, +1.501],
probability_positive 0.597 on 1,503 non-push games. The selected, reused
archive remains `unresolved_below_power`; this is not independent confirmation
or a claim of stable profitability. The result was already recorded as
`mod06_residual_offset_opener_v1_gaussian_median_composed_2020_2025`.

Reported (unverified here; same study): Week 1's composed card changes one
pick, `2026_01_NE_SEA`, from SEA to NE.

Implementation (read: `src/nfl_ats/gaussian_mean_mapping_incumbent_overlay.py`):
`gaussian_mean_mapping_incumbent` tracks the former mean mapping on the same
residual draws, requiring the median refit to reproduce the card before
recording forced picks. Its append-only prospective rows use PASS and NaN
for bet side and edge. The existing `ecdf_mapping_incumbent` stays active;
its verification now uses the forecast's recorded probability method.
Both are wired beside the published-card decision recorder with fail-open
error reporting. Research backtests retain their explicit historical defaults.

Release procedure (read: lane G task): prove evaluation, activation, opener
scoring, composition and board provenance under an isolated artifacts root
before updating weekly steps 4?5. The coordinator performs the real refresh;
this lane does not publish or activate against production artifacts.


Measured (fleet scratch `laneG/06-verification.json`, 2026-09-07): the full
isolated CLI chain passed, including matching median activation and all
three number-provenance checks. The rendered index shows NE +3.5 and 55.4%,
read from the matching composition (0.5542248835662009). Measured
(`laneG/07-tests.log`): 234 tests passed. Measured (lockday rehearsal):
40 active paths, 0 errors. Weekly steps 4?5 were changed only after the
isolated chain passed. The production manifest was not edited by this lane.
