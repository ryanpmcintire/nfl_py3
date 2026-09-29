# Served market-move feature audit

## Goal

Adjudicate LEAD-82's 164 saved-move discrepancies among 799 games without fitting or scoring outcomes.

## State

Completed. **Measured:** 164/799 differences (2023: 61; 2024: 49; 2025: 54), all later `sun_early_close` terminal quotes; 336 changed book moves, no changed book sets or starting lines. LEAD-82 minus saved spans [-2.0, +2.5] spread points. Hourly quotes reproduce 799/799 saved values; all eligible captures reproduce 799/799 LEAD-82 values. Census counts/ranges need no sampling interval or `probability_positive`. Full evidence and feature-only game ledger: `docs/move_feature_audit.md`.

Protocol declared before discrepancy inspection, unchanged: existing LEAD-82 2023–2025 quote-available frozen-fit population; compare numerical feature agreement, book membership, 48-hour freshness, 30-minute coverage, median/mean, sign, and `min(kickoff, Sunday 16:00 ET)`; classify first causal divergence and secondary causes. Read only identifiers, input clocks, quotes, and features. Zero outcome looks; no fitted terms, folds, scoring, or in/out-of-sample metric. Historical opener remains the frozen pool proxy; no standalone side flip.

## Tried

**Measured:** two-thread inline parquet reconstruction through existing extraction functions; 16:00 versus 12:45 changes 0/799 values on either grid. A separate scalar-inversion bug disabled the Sunday snapshot cutoff. Fixed only `src/nfl_ats/sharp_book_movement_features.py:119` (`~include_sunday` to `not include_sunday`). Real quote lines with deliberately delayed snapshot timestamps returned +4.0 before and the required +1.5 after; untouched historical values change 0/799.

Post-fix: `.tools/uv.exe run --no-sync ruff check src` passed; `mypy src` passed (253 files); `pytest -q -n 2 -k 'pick_refresh or market or move' --basetemp <fresh-temp>` passed (56 tests, two unrelated missing-feed warnings). Default UV cache and shared pytest temp were inaccessible; isolated temporary paths resolved both. No tests added, artifacts rewritten, registry commands, publication, or Git mutations.

## Next

Orchestrator reviews the one-line guard fix and report, then commits. LEAD-82's next parity step should name hourly and later-capture inputs separately; use the report's read-only reproduction command. Any richer-source research replay needs its own declared protocol.

## Open

The live Sunday implementation uses kickoff/supplied-now rather than a 16:00 cap; the report distinguishes this from historical clocks. No audit blocker remains. No conclusion about predictive merit follows from feature agreement.

## Record commands

None: this integrity audit makes no signal or rotation verdict.
