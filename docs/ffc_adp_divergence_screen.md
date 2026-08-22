# FFC ADP divergence screen — predeclaration and results

Written 2026-08-22. Screen owner paths: `scripts/ffc_adp_divergence_screen.py`,
this doc, `artifacts/ffc_adp_divergence_screen/`,
`registry/experiments/ffc-adp-divergence-screen/`.

**This document is the freeze.** Sections 1-6 below were written and committed
to disk BEFORE the script computed any outcome column or bootstrap draw; the
script implements exactly what they specify and asserts its own cell names
against §4. Section 7 (results) was appended only after the run completed.
Measure-only: neither registry JSON (`registry/weak_signals.json`,
`registry/rotation.json`) is written by this work; recording happens via the
explicit `nfl-ats weak-signals record` lines returned in §8.

## 1. Mechanism (frozen)

Crowd enthusiasm diverging from market price in Weeks 1-4, where team-quality
priors are thinnest. FantasyFootballCalculator ADP is a dated, pre-Week-1
crowd expectation of each roster's fantasy talent (exact mock-window stamps;
see `docs/ffc_adp_sourcing.md`). In the season's first weeks the betting line
leans heavily on last year's record while the crowd's ADP already embeds
offseason skill-talent news (free agency, rookies, coaching). The screen asks
whether that divergence is exploitable, and in WHICH frozen direction — each
cell below carries exactly one direction, chosen before scoring (**inferred**
mechanism, not evidence).

**Disclosure (binding framing)**: ADP is a preseason covariate with NO
in-season refresh. Every aggregate is formed in late August / early September
of its own season; there is no weekly crowd state. All cells therefore test a
pre-Week-1 prior against Weeks 1-4 outcomes, never an updating signal.

## 2. Data

- `artifacts/ffc_adp/20260822T004750Z/team_top8_feasibility.parquet`
  (**read**, columns verified: year, scoring, franchise_code, n_top8,
  mean_adp_top8, min_adp_top8, mean_times_drafted_top8; 1,013 rows, 16 years x
  2 formats). Per-response sha256 live in that snapshot's `manifest.json`;
  downstream cites the snapshot, never a refetch.
- Newest `data/raw/*/schedules.parquet` (**measured** this session:
  `data/raw/20260817T235649Z/schedules.parquet`, REG 2009-2026, includes
  `spread_line` home-positive and `result`). Prior-season wins are computed
  FROM THIS TABLE (REG wins, tie = 0.5); no win-total market exists locally
  (**read**: no `win_total` hits in `src/` or `data/`), so prior-season wins
  is the predeclared win-total proxy.
- Team-code normalization: schedule codes through
  `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` (OAK->LV, SD->LAC, STL->LA)
  plus LAR->LA on the FFC side (**measured**: FFC emits current codes incl.
  LAR; nflverse uses LA).

### Known gaps, disclosed up front

- **BUF absent entirely from the ADP aggregates for 2010, 2011, 2013, 2014**
  (both formats), and LAR/NYJ/SF missing in 2012 ppr (**read**,
  `docs/ffc_adp_sourcing.md`). Games involving those team-years drop out of
  the join; per-season coverage is printed and stored in the results payload.
- Draft volume thinnest 2011-2017 (~300-1,400/format/year); early-year ADP
  estimates carry wider sampling noise. Effects are reported with
  season-stratified uncertainty via the season-blocked secondary battery.
- Window-stamp honesty: latest mock window END across the snapshot is
  2011-09-09 (**read**, sourcing doc), i.e. the 2011 aggregate may include
  mocks run during that season's Week 1. Every other year's window closes
  before kickoff week. No other year overlaps play.
- Only 231/1,013 rows carry a full 8 players (mean n_top8 = 5.83);
  `mean_adp_top8` remains well-defined throughout (**read**, sourcing doc).

## 3. Population and feature construction (frozen)

Population: REG games, seasons 2010-2025, weeks 1-4 (weeks 1-2 for the thin-
info cells), `spread_line` and `result` non-null, both teams' ADP aggregate
rows present for (season, format). Pushes (ATS margin exactly 0) drop out of
scoring, counted honestly.

Features, per season x format independently:

1. `adp_quality_rank`: ascending rank of `mean_adp_top8` (rank 1 = richest
   fantasy roster). Top tercile = rank <= ceil(N/3) with N = franchises
   present that season-format (N=32 -> 11 teams).
2. `adp_wins_residual`: OLS residuals of `adp_quality_rank` on prior-season
   REG wins (previous calendar season, from the same schedules table;
   tie = 0.5 win), standardized within season x format to mean 0 / sd 1 ->
   `z`. Teams without a prior season (2010) are excluded from this family.
   |z| > 1 defines the extreme-tercile divergence cell.

Outcome target: `add_ats_outcomes`' `home_cover` (pushes NaN). A cell's value
column is always the FORCED pick's cover indicator (1.0/0.0), so effect =
(mean - 0.5) * 100 accuracy points on the full qualifying slate.

## 4. Cells and FROZEN directions

Chosen before scoring; rationale shown is mechanism reasoning (**inferred**),
not evidence. Primary format = ppr (modern fantasy default); standard-format
replications of the two primary cells are predeclared robustness checks.

| # | Name (weak-signal name) | Construction | FROZEN direction |
|---|---|---|---|
| A | `ffc_adp_cellA_highadp_underdog_back_ppr_w14` | weeks 1-4, exactly one side is top-tercile ADP AND priced underdog by `spread_line` (spread_line < 0 home dog / > 0 away dog; pick'em excluded) | **BACK the crowd**: take the high-ADP-roster underdog to cover |
| B | `ffc_adp_cellB_adpwins_residual_pos_back_ppr_w14` | weeks 1-4, exactly one team has \|z\| > 1 AND its z > 0 | **BACK the crowd-hot side**: take the lone positive-residual team (crowd ranks it far above its win baseline) to cover |
| C | `ffc_adp_cellC_highadp_underdog_back_ppr_w12` | cell A restricted to weeks 1-2 | same as A |
| D | `ffc_adp_cellD_adpwins_residual_pos_back_ppr_w12` | cell B restricted to weeks 1-2 | same as B |
| E | `ffc_adp_robust_std_cellA_highadp_underdog_back_w14` | cell A under standard scoring | same as A |
| F | `ffc_adp_robust_std_cellB_adpwins_residual_pos_back_w14` | cell B under standard scoring | same as B |

Direction rationale, frozen: (A) the market prices early-season lines off
prior-season reputation; when it still doubts a roster the crowd has priced
at the top of the league, the frozen bet is that the crowd's offseason
information beats the stale line. (B) is the same mechanism against the
wins-based prior: a lone |z|>1 POSITIVE residual means the crowd moved far
ahead of last year's record; the frozen bet backs the crowd. The negative-
residual tail is the OPPOSITE hypothesis and is deliberately NOT part of any
cell — it would double the shots on goal without having been chosen a priori.

## 5. Standard battery (frozen)

- Primary: week-blocked bootstrap (block = season*100 + week), 20,000 draws,
  seed 20260822, joint resampling of whole weeks; 95% percentile interval;
  `probability_positive` = share of draws > 0. Reported in accuracy points,
  full qualifying slate, forced picks.
- Secondary: identical draws reblocked on season alone (season stability).
- Baselines: 0.5 random-picker is the arithmetic baseline of the accuracy-
  points scale. Honest vig note (predeclared): ATS breakeven at standard -110
  is ~52.4%, so a forced-pick effect should clear ~+2.4 accuracy points
  before it is a wagering-grade edge at all; historical accuracy figures are
  NEVER read as a stable profit claim (AGENTS.md).
- Chronology: evaluation is chronological by construction (all games pooled
  forward, blocks respected); no validation/selection/calibration step exists
  in a measure-only screen — nothing is promoted from it.

## 6. Classification policy (binding, decided before scoring)

Per AGENTS.md: an interval crossing zero is NOT grounds for rejection; at
this evaluator's ~2-point resolution it is the EXPECTED shape for a real-but-
small signal. Only whole-interval-wrong-sign (`wrong_sign_resolved`, and only
if the entire interval sits below zero against a FROZEN direction — cells
here do have frozen signs, so this ground is reachable in principle) or a
positive-control bound close a line. Everything else records
`unresolved_below_power`. Every cell reports `probability_positive`, never
the binary "contains zero". Recording flows through explicit
`nfl-ats weak-signals record` command lines returned to the owner (§8);
nothing is written to either registry JSON by this session.

## 7. Results (measured, appended after the run)

Run: `artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json`
(**measured** this session; deterministic rerun reproduced identical numbers).
Inputs: `data/raw/20260817T235649Z/schedules.parquet` +
`artifacts/ffc_adp/20260822T004750Z` (sha256s in the payload). Coverage:
51-64 scored Weeks 1-4 games per ppr season with both ADP rows present
(**measured**, `coverage_by_season`); the BUF-gap seasons 2010-2014 sit at the
low end, as disclosed. Bootstrap: 20k draws, seed 20260822; zero dropped draws.

All numbers **measured**. Effects are forced-pick cover rate minus 50%, in
accuracy points; the ~+2.4-point vig bar applies before any wagering read.

| Cell | Frozen direction | n | effect pts | week-blocked 95% | P+ | season-blocked P+ |
|---|---|---|---|---|---|---|
| A `...cellA_highadp_underdog_back_ppr_w14` | back high-ADP dog | 275 | **+4.182** | [-0.951, +9.410] | **0.942** | 0.983 |
| B `...cellB_adpwins_residual_pos_back_ppr_w14` | back lone z>+1 team | 226 | -1.327 | [-8.482, +5.814] | 0.344 | 0.319 |
| C `...cellC_highadp_underdog_back_ppr_w12` | back high-ADP dog | 140 | +3.571 | [-2.941, +10.305] | 0.844 | 0.824 |
| D `...cellD_adpwins_residual_pos_back_ppr_w12` | back lone z>+1 team | 113 | -3.097 | [-14.078, +7.658] | 0.268 | 0.265 |
| E `...robust_std_cellA_highadp_underdog_back_w14` | back high-ADP dog | 273 | +1.282 | [-4.182, +6.929] | 0.664 | 0.669 |
| F `...robust_std_cellB_adpwins_residual_pos_back_w14` | back lone z>+1 team | 230 | -2.609 | [-9.289, +4.054] | 0.215 | 0.210 |

Reading (**inferred** interpretation of measured numbers): cell A is the only
direction that moved — P+ 0.942 week-blocked, 0.983 season-blocked, and it
replicates directionally in the thin-info weeks-1-2 cut (+3.571, P+ 0.844)
while the standard-format robustness cut attenuates to +1.282 (P+ 0.664),
consistent with part of the raw effect being a scoring-format artifact.
Cell B's frozen crowd-hot direction went the OTHER way (-1.33 / -3.10), i.e.
the enthusiasm-premium tail did not cover more than coin-flip; its interval is
wide and not resolved either way. No cell's whole interval sits below zero
against its frozen sign, so no cell admits `wrong_sign_resolved`; no positive
control was run, so none admits `positive_control_bound`. Per §6 every cell
records `unresolved_below_power`, including A despite its favorable lean — a
P+ of 0.94 on n=275 is a lead worth following up (format decomposition,
opener-grade line data, split-half reliability of the tercile instrument), not
a promotion claim, and it does NOT clear the ~+2.4-point vig bar as an
interval (lower bound -0.95).

Experiment-provenance stamp (run log, not a verdict):
`registry/experiments/ffc-adp-divergence-screen/20260822T130232Z.json`.

### Gates (measured this session)

- `ruff format --check` + `ruff check` on the owned script: pass.
- `mypy src scripts/ffc_adp_divergence_screen.py`: pass (102 files).
- `pytest --basetemp C:\Users\Ryan\AppData\Local\Temp\opencode\pt_ffcs`:
  1707 passed, 1 failed — the failure is
  `test_every_script_writing_artifacts_json_uses_the_provenance_helper`
  flagging `scripts/recurrence_hazard_features.py`, an UNTRACKED script from
  another concurrent session (present in `git status` before this work began);
  this screen's script passes the same contract directly (`_writes_
  artifacts_json_without_helper` -> False, measured).

## 8. Record commands (returned, not executed)

Neither registry JSON was written by this session. Run these exactly
(`--replace` if re-running after a prior partial record):

```powershell
nfl-ats weak-signals record --name ffc_adp_cellA_highadp_underdog_back_ppr_w14 --description "weeks 1-4, ppr scoring: back the top-tercile mean_adp_top8 roster priced as underdog (frozen direction: crowd side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect 4.1818181818 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -0.9505703422 --interval-high 9.4095940959 --probability-positive 0.9420500000 --sample-games 275 --sample-blocks 64 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Frozen direction BACK the crowd: high-fantasy-talent (top tercile mean_adp_top8, within season) roster priced underdog by spread_line, take the dog to cover. n=275, n_week_blocks=64, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.5418. Season-blocked secondary: +4.18pts, P+=0.98335. Preseason covariate, NO in-season refresh (latest mock-window end 2011-09-09). BUF absent from aggregates 2010/2011/2013/2014; LAR/NYJ/SF missing 2012 ppr. Vig framing: breakeven ~+2.4 accuracy points at -110; interval lower bound is negative, so no wagering-grade claim. Standard-format robustness attenuates to +1.28pts (P+ 0.664) -- format artifact share unresolved." 

nfl-ats weak-signals record --name ffc_adp_cellB_adpwins_residual_pos_back_ppr_w14 --description "weeks 1-4, ppr scoring: back the lone |z|>1 positive ADP-vs-prior-wins residual team (frozen direction: crowd-hot side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect -1.3274336283 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -8.4821428571 --interval-high 5.8139534884 --probability-positive 0.3440000000 --sample-games 226 --sample-blocks 60 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Frozen direction BACK the crowd-hot side: lone team with standardized OLS residual of adp_quality_rank on prior-season REG wins beyond +1 sd (sign convention frozen: positive = crowd ranks better than wins justify). n=226, n_week_blocks=60, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.4867. Season-blocked secondary P+=0.31915. Negative-z tail deliberately excluded (opposite hypothesis, not frozen). Preseason covariate, no in-season refresh." 

nfl-ats weak-signals record --name ffc_adp_cellC_highadp_underdog_back_ppr_w12 --description "weeks 1-2, ppr scoring: back the top-tercile mean_adp_top8 roster priced as underdog (frozen direction: crowd side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect 3.5714285714 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -2.9411764706 --interval-high 10.3053435115 --probability-positive 0.8438500000 --sample-games 140 --sample-blocks 32 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Thin-info replication of cell A restricted to weeks 1-2. n=140, n_week_blocks=32, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.5357. Season-blocked secondary P+=0.82365. Directionally consistent with cell A; power-limited. Preseason covariate, no in-season refresh." 

nfl-ats weak-signals record --name ffc_adp_cellD_adpwins_residual_pos_back_ppr_w12 --description "weeks 1-2, ppr scoring: back the lone |z|>1 positive ADP-vs-prior-wins residual team (frozen direction: crowd-hot side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect -3.0973451327 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -14.0776699029 --interval-high 7.6576576577 --probability-positive 0.2680000000 --sample-games 113 --sample-blocks 30 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Thin-info replication of cell B restricted to weeks 1-2. n=113, n_week_blocks=30, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.4690. Season-blocked secondary P+=0.2646. Leans against the frozen direction but wide; not resolved. Preseason covariate, no in-season refresh." 

nfl-ats weak-signals record --name ffc_adp_robust_std_cellA_highadp_underdog_back_w14 --description "weeks 1-4, standard scoring robustness: back the top-tercile mean_adp_top8 roster priced as underdog (frozen direction: crowd side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect 1.2820512821 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -4.1818181818 --interval-high 6.9289756418 --probability-positive 0.6639500000 --sample-games 273 --sample-blocks 64 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Standard-scoring robustness replication of cell A (predeclared). n=273, n_week_blocks=64, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.5128. Season-blocked secondary P+=0.66890. Attenuation vs ppr (+4.18 -> +1.28) suggests part of the headline effect is scoring-format-specific; decomposition unresolved. Preseason covariate, no in-season refresh." 

nfl-ats weak-signals record --name ffc_adp_robust_std_cellB_adpwins_residual_pos_back_w14 --description "weeks 1-4, standard scoring robustness: back the lone |z|>1 positive ADP-vs-prior-wins residual team (frozen direction: crowd-hot side covers)" --source "scripts/ffc_adp_divergence_screen.py; artifacts/ffc_adp_divergence_screen/20260822T130232Z/results.json; docs/ffc_adp_divergence_screen.md" --effect -2.6086956522 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2010 --season-end 2025 --interval-low -9.2886249956 --interval-high 4.0543287190 --probability-positive 0.2148000000 --sample-games 230 --sample-blocks 59 --classification-evidence "Predeclared freeze (docs/ffc_adp_divergence_screen.md sections 1-6, written before scoring): one frozen direction per cell, standard battery, binding taxonomy. Whole interval does NOT sit below zero against the frozen direction, so wrong_sign_resolved is inadmissible; no positive control was run, so positive_control_bound is inadmissible; unresolved_below_power per AGENTS.md." --notes "Standard-scoring robustness replication of cell B (predeclared). n=230, n_week_blocks=59, bootstrap_samples=20000, dropped_draws=0, seed=20260822. cover_rate=0.4739. Season-blocked secondary P+=0.21045. Consistent negative lean across formats for the enthusiasm-premium direction; still not whole-interval wrong sign. Preseason covariate, no in-season refresh." 
```
