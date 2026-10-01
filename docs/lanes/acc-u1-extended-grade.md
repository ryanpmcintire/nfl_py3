# acc-u1-extended-grade

## Goal
Regrade MOD-23 arms paired vs base on 2011-2025 (SBR open 2011-2019, Tuesday open 2020-2025). Parent: docs/lanes/accuracy-ceiling-theory.md.

## State
DONE. scripts/mod24_u1.py (stages select, grade); outputs artifacts/mod24_u1/ (report.json, per-arm parquet, selection_proxy.json, logs, weak_signals_batch.json). Base reproduces 802-701 in 2020-2025; proxy era base 1142-1089 (n 2231, 2011-2019, discrete read plus home-side offset, mirrors clv loop). Four arms recorded in registry via weak-signals record, family mod24_extended_grade, all unresolved_below_power.
Pooled accuracy diff vs base (points, season-blocked 95%, P+): u1 nested 0.24 down [-1.59,1.08] .35; u1 alpha-only -1.02 [-1.73,-0.29] .00; u3 compact net -0.27 [-1.97,1.45] .37; u5a man/zone 0.00 [-0.38,0.40] .47. Log loss improves for u1 nested (-0.0120), u3 (-0.0066), alpha-only (-0.0047), all intervals below 0; u5a +0.0007.
Flips pooled: 877, 186, 604, 140.

## Tried
Proxy-era unit 1 selection redone walk-forward with inner seasons max(2010, s-4)..s-1 (2011 inner is 2010 only: thin). Unit 5a terms reused from mod23_unit5/terms.parquet (zero before 2018, so pre-2018 picks identical to base).

## Next
Alpha-only loses accuracy in both eras with whole interval negative yet gains log loss: not closed (log loss mechanism; not a wrong-sign closure without orchestrator adjudication). Consider reading log-loss gain as a calibration signal, not a pick signal.

## Open
Proxy era uses SBR open not Tuesday open (proxy discount unmeasured here). Season-blocked intervals have 15 blocks only; week-blocked also in report.json.
