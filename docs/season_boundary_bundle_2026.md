# Season-boundary pinned bundle 2026

Apply after Week 18 (2027-01-10) in one session with one re-enrolment. Pinned source (PINNED_NAMES in `src/nfl_ats/independent_validation.py`, any src name with model or margin, `*features.py`, `*_overlay.py`, `pool_hook.py`, `pick_probability.py`, `cli_commands/prediction.py`, `pick_refresh.py`) cannot change before then. Lines are LF/CRLF-sensitive: keep each file's existing endings. Status labels are read from lanes; verify against `git log` before applying (the 11-17 v4 bundle may already have landed).

## Items

1. **ENG-58 rest_diff repair.** `src/nfl_ats/features.py`: new `_played_rest(games, schedules)` called before `rest_diff = home_rest - away_rest` (~line 678-694). Patch: `tests/scratch/eng58/eng58_rest_diff.patch` (apply --check 0; rerun `make_patch.py` if features.py changed). Evidence: 16 games change (2019: 6, 2021: 8, 2022: 2), 0 in 2026; opener replay 2020-25 changes 0 picks (record 0-0; 804-699 both); max probability shift 0.0013. Root decision: the abandoned 2022 BUF-CIN game counts as a rest appearance (Week 18 rest stays 6, rest_diff -1), never graded; add it before rebuilding. Lane: `docs/lanes/eng-58-flexed-and-rescheduled-games-audit.md` (Unit 5 and Root decision).
2. **ENG-51 unit 2b.** `src/nfl_ats/mass_preserving_lattice.py:163` `prior_pool` (add `line_source`, `assert_opener_pool`) and `src/nfl_ats/clv.py` opener path. Patch: `tests/scratch/eng51/eng51_unit2b.patch` (written; lane still says "not implemented", so treat as pending verification: `git apply --check`). Evidence: six leaky scripts; LEAD-76 rerun with opener pool stays unresolved_below_power (P+ .657 to .233). Root decision 11-03: bundle with the next pinned change. Lane: `docs/lanes/eng-51-opener-input-provenance-guard.md`.
3. **Validation hash LF-normalised.** `independent_validation._sources`: hash LF-normalised bytes so line-ending churn cannot stop a cohort (v10 stopped 11-18 to 11-21 from an LF rewrite of pick_probability.py). Changes every stored hash, so only at re-enrolment. Lane: `docs/lanes/done/validation-v10-crlf-2026-11-21.md`.
4. **Flag weights inside the counted set (flag-gate).** `pick_probability.py` fit: review per-flag weights (division .148 vs protection .332 split lost held-out, Brier -0.00047, P .086) with the lane's null and power tables. Owner settled 11-21: winning flags stay on, losing flags stay off; no removal or freezing; only weights inside the counted set. Lane: `docs/lanes/flag-gate-evaluation-2026-11-14.md`.
5. **ENG-55 / MOD-26 hook anchor test.** `pool_hook.py:85-118`. Root kept the median anchor; the FanDuel diff is NOT applied. Declared test only: the term's value at quarter-point consensus anchors by anchor type on 2026 forward games; change only if it differs from the fitted weight. Lane: `docs/lanes/eng-55-pool-line-inference-audit.md`.
6. **MOD-27 fenced repairs.** Pool-offset two-stage term (no 2026 side changes; 2020-25 LOSO log loss +0.00238 [0.00099, 0.00379]) and single-book baseline guard, bundled with MOD-17. Thin-book move repair has no support (P+ 0.60): do not apply. Script `scripts/mod27_fenced_repairs_replay.py`; lane `docs/lanes/mod27-better-picks.md`; ROADMAP MOD-27.
7. **Model-improve post-season batch.** Root check 2026-11-21 (read): the v4 bundle landed on 2026-11-17 in 62d554ea (tests/scratch/tuesday_bundle/bundle.patch: QB repair, pregame MOS weather via pregame_weather_features.py, locked-score in pick_refresh.py; docs/lanes/done/week11-lock-2026-11-17.md). Before the boundary, run `git apply --check -R` on card_logit_store.patch and on the ledger move column change to confirm whether they landed too; apply only what fails that reverse check. Original list, per `docs/lanes/model-improve-2026-11-16.md`: QB input repair (`tests/scratch/qb_input_repair/repair_ab.patch`); weather temp/wind pregame forecast or `weather_drop.patch` (`tests/scratch/serve_parity/`; v4 836-667 v 829-674, P+ .997); ledger move column (`pick_refresh.py:629-645`); `tests/scratch/board_pct/pick_refresh_locked_score.patch`; `tests/scratch/card_logit_store/card_logit_store.patch`. Apply order v4, locked_score, card_logit_store (no conflicts per `docs/lanes/v4-landing-2026-11-17.md`). Lane: `docs/lanes/v4-pinned-batch-2026-11-16.md`.
8. **Card fidelity divergence.** Nothing pinned: the stale-card gate is `cli_commands/publishing.py:1593-1595` (not pinned); no bundle change.

## Order and re-enrolment

1. `git status` clean of pinned files; `git apply --check` every patch above together in sequence (v4 items, then ENG-51, ENG-58); rerun a patch's generator if its anchor moved.
2. Apply patches; hand-edit items 3-6 (weights, hook, MOD-27) and run ruff check and format on touched files.
3. Rebuild features and weak-stack tables (`nfl-ats weekly-run`), confirm 16 rest_diff rows changed.
4. Fit before enrol: refit the active model, opener-evaluation, and the pick-probability fit.
5. One re-enrolment: copy the latest `registry/studies/combined_vs_raw_2026_vN.json` (currently v10) to the next vN, set `DEFAULT_STUDY` in `independent_validation.py:30` to it, run `nfl-ats independent-validation enroll` (default), and confirm `status` source_unchanged.

## Checks

- `.\.tools\uv.exe run --no-sync pytest` on durable tests under `tests/` once.
- `scripts/check_served_terms` (served terms sum to logit of the served probability).
- `scripts/card_fidelity_check.py --dry-run`.
