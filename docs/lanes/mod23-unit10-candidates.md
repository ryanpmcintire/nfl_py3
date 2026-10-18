# MOD-23 unit 10: new base-model candidates (inventory, no fits)

## Goal
Up to 3 untested base-model candidates, each attacking a named limit, after a prior-work check against registry/experiments, registry/weak_signals.json, ROADMAP.md, docs/lanes/done. Base 802-701 (2020-25), 1142-1089 (2011-19, SBR open).

## State (2026-10-18, read-only greps; no fits)
Honest result: the new-information space is nearly exhausted (about 190 experiment families, 4,068 pool cells, all of unit 8/9 tested). Two weak survivors, one near-duplicate. Expected value of all three is small; none should displace the armb-forward-log prospective test.

## Candidates (ranked)
1. Winsorized/robust base target (cap |ats_margin| at a cut chosen LOSO, or Huber with LOSO scale). Limit: label noise in tails (blowouts, non-offensive TDs). Prior work: grep huber/winsor/robust-loss over src, scripts, registry, docs, ROADMAP finds none on the base fit (only unrelated hits). Nearest: mod23 unit8 cand 1 (strata variance weights; 5 looks, d_acc -1.13 in 2020-25, 400+ flips), acc-u5 (changes target to EPA). Differs: caps magnitude rather than reweighting by strata; but overlap with unit 8 is real, so expect the same outcome. Data: ats_margin in data/processed/game_features_weak_stack.parquet (exists). Leak: none (training labels only, cut from seasons < test). Declaration: arms = cap at LOSO-chosen percentile; Huber with LOSO scale; eras 2011-19 and 2020-25; grade as unit 8; 4 looks, family mod23_unit10_robust_target. Cost: about 25 single-thread min.
2. Starter draft-capital talent prior as base inputs. Limit: new pregame info for first/second-year starters whose rolling stats do not exist. Prior: draft_round is used only for rookie week-1 share (src/nfl_ats/rookie_prior_surplus_tilt_overlay.py, scripts/xlg06_rookie_priors_screen.py); mod06 position prior and team-quality-priced rule bound it near zero. Differs: all starters, team-level snap-weighted draft_ovr and years-in-league, not an overlay. Data: data/raw/combine/20260822T143152Z/combine.parquet (verified columns: draft_year, draft_round, draft_ovr, draft_team, pos, pfr_id, forty, wt, vertical); starters from data/players/raw snap_counts (2013+) or prior-week snaps; joined by pfr_id/name, known at draft (April), so pregame safe. Leak: starter set must use only snaps before kickoff. Declaration: arms A draft_ovr share of offence/defence snaps, B plus athletic composite; add as 4 diff_ columns to the 88; 2 arms x 2 eras = 4 looks (2011-12 lack snaps, use 2013+ with 2011-12 zeros declared). Cost: about 40 min including the join.
3. Not found. Rolling ATS residual (already inputs: diff_ats_residual, bias_prior_week_ats_*), sandwich/lookahead, TV attention, props (2026 only), span, two-sided labels, availability history all fail the prior-work check. No third candidate offered.

## Next
Orchestrator decides; if run, one agent per candidate, fresh, with this file.

## Candidate 1 run declaration (2026-10-18, before running; script scripts/mod23_unit10_robust.py, since mod23_unit10.py is taken)
Arms: W = ridge alpha 10 (base pipeline) on ats_margin winsorized at +-c; H = sklearn HuberRegressor on the same imputed+scaled 88 inputs, alpha = 10/sigma0 (Huber quadratic zone is r^2/sigma, so this equals ridge alpha 10; sigma0 = 1.4826*MAD of in-sample base-ridge residuals of the same training set), epsilon = quantile q of |resid|/sigma0.
Grid q in {0.80,0.90,0.95,0.99,1.00}; c = quantile q of |label| in the training set; q=1.00 is plain base ridge. q chosen by inner LOSO over seasons prior to the latest training season (same convention as unit 9b), criterion held-out MAE on raw labels. Never touches the scored season.
Looks: 2 arms x 2 eras = 4, family mod23_unit10_robust_target, all unresolved_below_power. wrong_sign_resolved will be applied to a cell only if its whole season-block bootstrap interval is below zero; otherwise nothing closes.
Grade: paired decisive accuracy v base (802-701 / 1142-1089 reproduction), season-block bootstrap pp, log loss and Brier v base and market-only, deciles reliability for the better arm, per-season d_acc, flips W-L, chosen q per fold. Leak check: corrupt latest-season outcomes, selection unchanged.

## Candidate 1 RESULT (done, recorded)
Artifacts artifacts/mod23_unit10r/ (report_all.json, W_q.json, H_q.json, run.log). 4 cells recorded, names mod23_unit10_robust_target_{W,H}_{sbr_2011_2019,tuesday_2020_2025}_accuracy_diff, all unresolved_below_power.
d_acc (pp, season-block pp): W 2011-19 -0.81 (0.077), W 2020-25 +0.13 (0.707), H 2011-19 -0.31 (0.306), H 2020-25 +0.07 (0.516). Chosen q sat at grid edge 0.80 in most folds (grid too narrow; lower cuts untested). Positive-control check in the script (labels x3) is scale-invariant for quantile cuts so it cannot differ: not a valid control. Next: optional wider grid (0.5-0.8) as a new family.
