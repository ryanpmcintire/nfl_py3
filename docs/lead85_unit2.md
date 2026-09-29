# LEAD-85 unit 2: weekly Best Pick replay

**Measured:** Command: .tools/uv.exe run --no-sync python scripts/lead85_unit2.py. The unit-2 declaration was saved in the lane before nomination outcomes; its unchanged snapshot and hashes are in tests/scratch/codex/lead85_unit2/. All intervals below are 95%.

## Changed nominations first

**Measured:** records are wins-losses-pushes. Wilson intervals describe cover rate among nonpush nominees; pool reward credits pushes 0.5. Probability_positive is abbreviated P+ in tables.

| Season | Changed weeks | Served W-L-P | Candidate W-L-P | Better/worse/equal | Reward change pp [interval]; P+ | Exact p |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 0 | 0-0-0 | 0-0-0 | 0/0/0 | n/a | 1.000000 |
| 2024 | 0 | 0-0-0 | 0-0-0 | 0/0/0 | n/a | 1.000000 |
| 2025 | 0 | 0-0-0 | 0-0-0 | 0/0/0 | n/a | 1.000000 |
| pooled | 0 | 0-0-0 | 0-0-0 | 0/0/0 | n/a | 1.000000 |

**Measured:** the following identifiers locate changed weeks; full nomination/prediction rows remain in scratch.

| Season/week | Served nominee (side) | Candidate nominee (side) | Served/candidate reward |
| --- | --- | --- | --- |

## Nominee cover rates

**Measured:** Wilson intervals treat weekly nonpush outcomes as Bernoulli observations; season dependence is handled separately by the paired hierarchical intervals. Unresolved selection ties have equal weights.

| Season | Population | Weeks | Served W-L-P | Served Wilson | Candidate W-L-P | Candidate Wilson | Exact p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | all | 18 | 12-5-1 | 70.59% [46.87,86.72] | 12-5-1 | 70.59% [46.87,86.72] | 1.000000 |
| 2023 | changed | 0 | 0-0-0 | n/a | 0-0-0 | n/a | 1.000000 |
| 2024 | all | 18 | 10-7-1 | 58.82% [36.01,78.39] | 10-7-1 | 58.82% [36.01,78.39] | 1.000000 |
| 2024 | changed | 0 | 0-0-0 | n/a | 0-0-0 | n/a | 1.000000 |
| 2025 | all | 18 | 8-10-0 | 44.44% [24.56,66.28] | 8-10-0 | 44.44% [24.56,66.28] | 1.000000 |
| 2025 | changed | 0 | 0-0-0 | n/a | 0-0-0 | n/a | 1.000000 |
| pooled | all | 54 | 30-22-2 | 57.69% [44.19,70.13] | 30-22-2 | 57.69% [44.19,70.13] | 1.000000 |
| pooled | changed | 0 | 0-0-0 | n/a | 0-0-0 | n/a | 1.000000 |

## Primary nominee Brier and weekly reward

**Measured:** positive improvements favor the candidate: served minus candidate for Brier, candidate minus served for reward. Reward is in percentage points. Brier conditions on each arm's nonpush nominees; paired resampling retains every week. Gap means OOS minus optimistic IS. Pooled IS repeats earlier training weeks across folds and resamples them together.

| Panel | Split | Metric | Served | Candidate | Improvement [interval]; P+ |
| --- | --- | --- | --- | --- | --- |
| 2023 | IS | brier | 0.26663692 | 0.26679109 | -0.00015417 [-0.00077178,+0.00056417]; P+ 0.3130 |
| 2023 | IS | reward_points | 50.00000000 | 50.00000000 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2023 | OOS | brier | 0.22253112 | 0.22177823 | +0.00075289 [-0.00002622,+0.00181386]; P+ 0.9695 |
| 2023 | OOS | reward_points | 69.44444444 | 69.44444444 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2023 | gap | brier | -0.04410580 | -0.04501286 | +0.00090706 [-0.00017078,+0.00213573]; P+ 0.9503 |
| 2023 | gap | reward_points | 19.44444444 | 19.44444444 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2024 | IS | brier | 0.22586547 | 0.22599647 | -0.00013100 [-0.00064222,+0.00033210]; P+ 0.3049 |
| 2024 | IS | reward_points | 64.28571429 | 64.28571429 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2024 | OOS | brier | 0.25845006 | 0.25822962 | +0.00022044 [-0.00041290,+0.00087568]; P+ 0.7382 |
| 2024 | OOS | reward_points | 58.33333333 | 58.33333333 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2024 | gap | brier | 0.03258459 | 0.03223315 | +0.00035144 [-0.00044140,+0.00118527]; P+ 0.7941 |
| 2024 | gap | reward_points | -5.95238095 | -5.95238095 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2025 | IS | brier | 0.23000519 | 0.22999749 | +0.00000770 [-0.00023661,+0.00025783]; P+ 0.5086 |
| 2025 | IS | reward_points | 63.20754717 | 63.20754717 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2025 | OOS | brier | 0.25828117 | 0.25858094 | -0.00029977 [-0.00056865,-0.00004234]; P+ 0.0105 |
| 2025 | OOS | reward_points | 44.44444444 | 44.44444444 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| 2025 | gap | brier | 0.02827598 | 0.02858345 | -0.00030747 [-0.00067456,+0.00004919]; P+ 0.0474 |
| 2025 | gap | reward_points | -18.76310273 | -18.76310273 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| pooled | IS | brier | 0.23437143 | 0.23443536 | -0.00006393 [-0.00036350,+0.00025405]; P+ 0.3787 |
| pooled | IS | reward_points | 61.42857143 | 61.42857143 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| pooled | OOS | brier | 0.24664887 | 0.24643443 | +0.00021444 [-0.00032115,+0.00089586]; P+ 0.7375 |
| pooled | OOS | reward_points | 57.40740741 | 57.40740741 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |
| pooled | gap | brier | 0.01227743 | 0.01199907 | +0.00027836 [-0.00036190,+0.00100375]; P+ 0.7659 |
| pooled | gap | reward_points | -4.02116402 | -4.02116402 | +0.00000000 [+0.00000000,+0.00000000]; P+ 0.5000 |

## Population, calibration and audit

**Measured:** 1531 source-complete historical opener games; 816 outer-season games; 714 post-cutoff games before dispersion filtering; 54 held-out weeks. 1 dispersion fallback weeks; 0 unresolved arm/week ties. Maximum row-versus-discrete-PMF error 3.33e-16; no game side changes.

**Read:** historical openers are the frozen pool-line proxy; no pre-2026 Splash captures are used. Eligibility is kickoff after Sunday 12:45 Eastern from unit 1, followed by the shared Tuesday dispersion filter. Ranking is unconditional calibrated cover + 0.5 push, as frozen in docs/lead85_protocol.md:14 and docs/lanes/lead72.md:29. Tie order is score rounded to 12 decimals, lowest dispersion, smallest absolute spread, earliest kickoff, then equal weights. The one calibrated conditional cover probability chooses each game's side.

**Measured:** no fitting or parameter selection occurred here. Unit 1 coefficients and separate-year positive temperatures are reused unchanged. Fit through Y-3, reserve Y-2, calibrate Y-1, score Y. The archived training cutoff and saved discrete PMFs passed replay checks. Source hashes and coefficient intervals are copied to the scratch summary.

| Outer | Fit seasons | Model logit | Composition | Market move | Availability | Intercept | Served/candidate inverse temperature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 2020 | -0.006681 | 0.363213 | 0.048984 | 0.000000 | -0.113420 | 1.052180/1.081192 |
| 2024 | 2020,2021 | 0.370792 | 0.260210 | 0.061742 | 0.000000 | -0.030925 | 1.226058/1.252461 |
| 2025 | 2020,2021,2022 | 0.362850 | 0.287314 | 0.090751 | 0.000000 | -0.041454 | 0.988894/1.000000 |

**Inferred:** coefficient variation is descriptive, not a new fit or stability test; the source-complete availability term is constant and unidentified by the likelihood. Its zero coefficient comes from ridge identification. Only three outer seasons exist.

**Measured:** five equal-width reliability bands of the nominated side's conditional cover probability, OOS nonpush only:

| Arm | Band | Nonpush weight | Mean probability | Observed cover |
| --- | --- | --- | --- | --- |
| four_term | 0.0-0.2 | 0.0 | n/a | n/a |
| four_term | 0.2-0.4 | 0.0 | n/a | n/a |
| four_term | 0.4-0.6 | 14.0 | 0.570985 | 0.500000 |
| four_term | 0.6-0.8 | 38.0 | 0.644737 | 0.605263 |
| four_term | 0.8-1.0 | 0.0 | n/a | n/a |
| integrated | 0.0-0.2 | 0.0 | n/a | n/a |
| integrated | 0.2-0.4 | 0.0 | n/a | n/a |
| integrated | 0.4-0.6 | 14.0 | 0.571480 | 0.500000 |
| integrated | 0.6-0.8 | 38.0 | 0.644857 | 0.605263 |
| integrated | 0.8-1.0 | 0.0 | n/a | n/a |

## Inference, look accounting and limits

**Measured:** 10,000 paired hierarchical season/week bootstrap draws, base seed 85 with deterministic panel/split offsets, percentile intervals and half credit for zero draws. Bootstrap intervals condition on the saved fits; they do not refit models. The exact reward null swaps arm labels independently within each week; rational reward differences are enumerated by convolution, using the absolute total as the two-sided statistic. Equal-reward weeks contribute no randomization information. With no changed weeks, subset estimates are unavailable; a degenerate unchanged-reward bootstrap is not evidence of tight prospective precision.

**Read:** 713 study looks were reserved in the original declaration. Unit 2 reports 72 metric cells (three arm/contrast cells x two metrics x IS/OOS/gap x four panels), 10 reliability cells, 16 Wilson cells, 8 exact-null cells and 4 changed-reward contrast cells: 110. The latter 28 supplemental cells increase charged study looks to 741. No post-outcome variant or threshold was selected. These correlated cells are not independent discoveries.

**Inferred:** unresolved_below_power, pending the orchestrator's serial registry entry. No admissible closing ground has been established under AGENTS.md:65-79; probability_positive, not an interval's zero crossing, informs the estimate. The original five-arm model-only/dated-market/Elo comparison remains open beyond this assigned two-arm replay. Reused historical archives and three seasons do not establish prospective improvement; no serving decision follows. Candidate-versus-served record commands are prepared only in the lane.
