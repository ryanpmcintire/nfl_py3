# What has to be revisited, and why

Written 2026-08-18 at the owner's request, after a session that found four
defects in the measurement instrument and one in the decision frame. This is a
triage list, not a set of conclusions. Every claim below carries its
provenance: **measured** (run this session), **read** (from the named file), or
**inferred** (reasoning, not evidence).

Nothing here re-classifies a registry entry. Re-classification requires
re-running the measurement.

## The five defects that drive this list

- **D1 — the calibration step distorts small effects.** *(read:
  `docs/purged_cv.md`, positive-control section.)* Planting a known effect and
  scoring it through the full pipeline: a planted **+1.3** came back
  **−0.7 (wrong sign)**; a planted **+3.0** came back **+1.14 (attenuated)**.
  Bypassing the probability-calibration step (sign-only) recovered both closely.
  **Critical caveat, and it gates everything below:** on real CFB data the
  sign-only-vs-full gap was only **~0.1 points**. So the distortion is
  demonstrated on *planted* effects and is **unverified on real ones**. This is
  the highest-severity defect because it can turn a real positive into a
  recorded negative — and it is also the least confirmed.
- **D2 — reported intervals are 17-58% too narrow.** *(read:
  `docs/estimation_variance.md`.)* The block bootstrap resamples games but never
  refits the model, so every published interval is conditional on one fit.
  Measured coverage **89.5-92.5%** against 95% nominal; refitting flips **19.2%**
  of picks. Direction of the error: **`probability_positive` is overstated** —
  every recorded verdict is more confident than it earned.
- **D3 — bootstrap seed jitter.** *(read: `docs/evaluator_power.md` §4.)* At the
  old `samples=2000`, the seed-to-seed sd of a reported interval edge was
  0.02-0.03 points (~6-7% of the true SE). Decisive only for verdicts sitting
  within ~0.03 of a gate. **Fixed going forward** — defaults raised to 20,000 in
  `experiments.py`, `reporting.py`, `outcomes.py` *(measured: full suite 803
  tests pass, cost 13s)*.
- **D4 — the bootstrap is degenerate below ~4 blocks.** *(read:
  `docs/anytime_valid.md` §6.)* With one block there is one possible resample,
  so the "interval" collapses to a point and guarantees a false alarm.
- **D5 — the decision frame was wrong.** Verdicts were decided against 0.90 /
  0.75 thresholds. Per `AGENTS.md` those govern what the docs may **claim**,
  never which card is **played**. The pool is forced picks. This does not make
  any measurement wrong; it makes the **decisions drawn from them** wrong.

## Tier 1 — the measurement itself is suspect; re-run required

| Item | Why | Source |
|---|---|---|
| `player_qb_continuity_matched_alpha` | **The single worst case.** 997 games on **exactly 4 blocks** — sitting on the D4 floor — with interval `[0.0, 2.2177]`, a lower bound of *exactly* 0.0 being the degeneracy fingerprint. It is classified **`refuted_mechanism`**, a terminal verdict that permanently closes a line, on the strength of an interval that D4 says is not a trustworthy interval. It also has **no `probability_positive` recorded at all**. *(measured: `registry/weak_signals.json`, blocks=4, P+=None.)* | registry |
| `cfb_role_continuity` | `closed_negative` — terminal. Closed the entire XLG-04→XLG-05 cross-league transfer path on −0.67 points. A terminal verdict on a small negative is exactly the shape D1 can manufacture. | rotation registry |
| `pbp_drive_bundle` | `closed_negative`, window `[2013,2017]` spent, **no `probability_positive` recorded** — it predates the continuous-evidence rule. A whole line closed on a bare pass/fail. | rotation registry |
| `player_qb_continuity` | `closed_negative`, window `[2014,2017]` spent, **no `probability_positive` recorded.** Same problem. | rotation registry |
| `residual_location_recency_hl200_cfb`, `..._hl400_cfb` | Both classified **`refuted_mechanism`** (terminal) on P+ 0.014 / 0.0005. D2 says those P+ values are overstated, i.e. the refutation looks stronger than it is. **Inferred:** refutation probably survives — the effects are resolvably wrong-signed on 8,933 games — but a terminal verdict should be verified, not assumed. Lower priority than the rows above. | registry |

## Tier 2 — measurement stands; the decision drawn from it does not

Re-read under the forced-pick frame. **No re-run, no data, no window.**

| Item | Recorded | What changes |
|---|---|---|
| `best_pick_ranker_opener` | **`confirmed`**, P+ 0.865 | **The project's only `confirmed` verdict, and its most fragile claim.** D2 says the P+ is overstated. Independently, `docs/pool_format_levers.md` already found the recorded +8.68 collapses to **+0.92** once the alphabetical tie-break is removed (24 of 35 weeks were ties). A "confirmed" resting on both an overstated interval and an alphabetical artifact deserves the hardest look of anything in Tier 2. |
| `mod07_weak_signal_stack` | `unresolved`, P+ 0.8745 vs a 0.90 bar | Already promoted as a play decision. The registry verdict stays `unresolved` — that is correct and should NOT be changed — but nothing further is owed here. |
| All 11 `unresolved_below_power` entries | P+ 0.0005-0.899 | Classification is unaffected (already unresolved). Their P+ values are overstated per D2 and their point estimates may be attenuated per D1. They are decided on expected value, not on a bar. |

## Tier 3 — already void

- The empirical-Bayes shrinkage work (`docs/decision_rule.md`,
  `src/nfl_ats/decision_rule.py`). Halted and banner-marked by the owner
  2026-08-18. Its posteriors may not be cited. Its forced-pick *framing*
  survives; only the arithmetic is void.

## The one experiment that must run first

**Measure D1 on real effects, not planted ones.** Everything in Tier 1 rests on
the claim that the calibration step can attenuate or invert a small real effect.
That is currently demonstrated only on synthetic plants, and the one real-data
reading disagreed with it (0.1-point gap). Until that is settled:

- If D1 is real at the magnitudes that matter, **every terminal negative in the
  project is suspect** and Tier 1 is understated.
- If D1 is an artifact of how effects were planted, Tier 1 shrinks to the D4
  case (`player_qb_continuity_matched_alpha`) plus the two bare-verdict entries
  that recorded no continuous evidence.

Running the whole Tier 1 list before answering that question would be wasted
work in one branch and insufficient in the other. **Inferred, not measured:**
this is the correct sequencing.

## What this list does NOT say

- It does not say the affected results are wrong. D2 makes intervals wider, not
  point estimates different; most `unresolved` verdicts stay `unresolved`.
- It does not require new data. Every item is a re-analysis of games already
  played and already looked at. **No rotation window is implicated** — per
  `docs/rotation_registry.md` rule 4, windows retire per-family, and rule 6
  makes a re-read a stated discount, not a prohibition.
- It does not license re-running something until it gives a better answer.
  A re-run replaces a verdict once, with the reason recorded.
