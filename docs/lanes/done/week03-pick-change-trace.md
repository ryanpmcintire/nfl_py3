# Week 3 pick-change trace

## Goal

Trace PIT to CIN, WAS to SEA, and NYG to TEN from the saved raw forecast through the fitted probability and pregame lock.

## State

**Measured:** all three published probabilities and all three locked probabilities replay exactly, with maximum absolute error 0.0. Six quote-to-movement checks also match. The serving calculation, not the freeze, changed the sides.

| Raw forecast | Published pick | Measured cause at Markdown publication |
| --- | --- | --- |
| PIT +3.5, 53.9023% | CIN -3.5, 55.8386% | PIT's first-year coach flag is -1: Mike McCarthy versus returning CIN coach Zac Taylor. Its fitted contribution is -0.259974 home log-odds. Removing just that contribution gives PIT 50.6339%. |
| WAS +6.5, 54.0843% | SEA -6.5, 52.0864% | The three leader-book moves toward home are -1.0, -0.5, -0.5 points; their median is -0.5, toward SEA. Its contribution is -0.110889 home log-odds. Removing just that contribution gives WAS 50.6846%. No situational flag fires. |
| NYG -3.5, 50.2098% | TEN +3.5, 50.3893% | No situational flag fires and median movement is zero. The shrunk model contribution is +0.002320 home log-odds; intercept plus quote-availability term is -0.017891. Their sum favors TEN. Both coaches are first-year, so the coach flag cancels. |

**Read:** `pick_probability.py:194-202` computes home log-odds as `-0.052064947 + 0.276509611 * logit(raw_p) + 0.259973644 * flags + 0.221777837 * movement + 0.034174422 * availability`. Full-precision coefficients are in `artifacts/pick_probability/20260927T161329Z/coefficients.json`. The fitted model therefore shrinks raw-model log-odds before adding the other terms.

**Measured:** Markdown publication was 12:16:50 ET. By the 12:25:33 ET publication, movement was +0.5 for PIT and -1.5 for WAS. Locked probabilities became CIN 53.0892%, SEA 57.5733%, TEN 50.3893%; all three sides stayed the same. The frozen record matches this replay. The original grading lines remain unchanged.

**Read:** `card_view.py:401-411` restores raw probabilities before applying the fitted model; `pick_probability.py:868` replaces the served home probability; `publishing.py:112-133` selects its side, then applies the frozen record. The CIN explanation's intermediate "complemented" wording does not describe the final fitted calculation.

## Tried

Replayed production probability functions using the active pregame coefficients, cached schedules, cached Tuesday weather bulletins, arrest/protection inputs, and time-filtered captured market quotes. Verified individual-book direction using actual home handicap quotes. Saved scripts and evidence in `.tmp/week03_trace_probability.py`, `.tmp/week03_trace_market.py`, `.tmp/week03-probability-trace.json`, `.tmp/week03-market-book-trace.csv`, and `.tmp/week03-trace-verification.json`. No fitting, production edits, or publication.

## Next

The requested mechanism trace is complete. Use these exact term contributions and original grading lines in any subsequent evaluation of the combination model.

## Open

This trace explains the three losing changes; it does not establish their future predictive value. No signal is closed and no serving rule is changed. The stored CIN explanation still describes an intermediate legacy transformation.
