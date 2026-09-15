# handle-follow-override-defect

## Goal

Week 1 was 9-6. The one wrong pick was ARI at LAC: a Saturday last-call rule
flipped the card onto LAC -9.5, a side the model priced at 0.362, and the
board displayed it as "50.0%". Done when a rule can no longer quietly override
the model that hard, the card never shows a floored score for an overridden
pick, and the override rules are measured inside a conditioning split on how
far they override.

## State

Diagnosis complete and measured 2026-09-14. Fixes not yet shipped.

**Week 1 record: 9-6, DEN at KC pending. The single pick that differs from the
paper ledger is ARI at LAC** (owner, authoritative): the ledger carries ARI
+9.5 and grades it a win; the played card was LAC -9.5 and it lost. Every
other game on the card matches the ledger.

### The flip

`handle_follow_0_70` fired at the Saturday 12:15 ET last-call pass
(`pick_revisions`, measured; reason "Followed the heavy-money side
(last_call_sat_12:15)", recorded 2026-09-12 16:54 UTC) and flipped four games:

| game | flip | model's prob for the served side | money / tickets |
|---|---|---:|---|
| **ARI at LAC** | ARI -> **LAC** | **0.362** | 74% / 62% |
| CHI at CAR | CAR -> CHI | 0.512 | 70% / 67% |
| DAL at NYG | NYG -> DAL | 0.507 | 77% / 48% |
| DEN at KC | KC -> DEN | 0.509 | 81% / 71% |

Three overrode a near coin flip, which is cheap. ARI at LAC overrode a model
that was firmly the other way, on the biggest spread on the card, with the
tickets heavy too (62%) so it was not a sharp-money signature. On that game
the model, the leader-median line-move rule and the consensus rule all said
ARI; one rule outvoted all three.

**The rule has no guard on how far it is allowed to override the model.** Its
promoted evidence (`docs/handle_follow_on_card.md` H1, read) is 19-18 on 37
flips, +0.38 accuracy points, week-blocked [-3.00, +4.28], P+ 0.5695 — never
conditioned on model-disagreement distance, and negative in the one spread cut
it did take (H3, |spread| <= 3, -0.77, P+ 0.214).

Precedence is NOT the bug: handle-follow only fires when the market rules did
not (`pick_refresh.py`, read), which is what the evidence doc prescribed.

### The display defect

The LAC pick published with `displayed_score` exactly `0.500000`
(`published_picks`, measured). That is `PICK_SIDE_FLOOR`, not a probability.
`attach_displayed_confidence` only raises when calibration pushes a side below
the floor while the stated side is above it; here the policy changed the
**side** without changing the probability, so the stated value was itself
below 0.5 and the guard never tripped. The reader saw a number the model did
not hold.

## Shipped

**The rule is off the served card (2026-09-14, owner call).**
`HANDLE_FOLLOW_SERVED = False` in `pick_refresh.py`. It no longer moves a
pick. It keeps recording: `handle_follow_refresh_overlay` writes
`handle_pick_side` and `off_arm_pick_side` for every eligible game carrying a
money reading whether or not the rule fires, so the arm stays fully scorable
prospectively. Re-enabling is one line.

Why off, stated plainly: its whole promoted evidence was **19-18 across 37
flips**, +0.38 accuracy points, week-blocked [-3.00, +4.28],
`probability_positive` **0.5695** — a 57/43 call, and it cleared the bar only
because the forced-pick rule plays anything above 0.5. Handle exists only for
2023-2025, so 2020-2022 contributed nothing. It was measured at about **0.93
flips per week block**; in 2026 Week 1 it fired **four times in one pass**, so
live it moves several times more of the card than it was ever measured on. And
the gate below was chosen by splitting those same 37 games, which is in-sample
selection, not independent confirmation.

This is a serving decision, not a closure. Nothing here is recorded terminal.

**The gate was shipped and then removed the same day.** The constant was the
in-sample median of the 37 flips it was scored on, and a random 19/18 split
of those flips is at least as lopsided as 12-7 / 7-11 one time in five
(hypergeometric p 0.194, measured). Overfit, per the owner; no gate is in
code. The table below stays as a description only.

The split that was used (`docs/handle_follow_override_distance.md`, artifact
`artifacts/handle_follow_override_distance/20260914T180713Z/`) reproduced H1
exactly first — 73 fires, 37 flips, +0.3846 — then split on override distance:

| Flip group | n | Card's record | Following the handle | Effect vs incumbent | P+ |
|---|---:|---:|---:|---:|---:|
| All flips | 37 | 18-19 | 19-18 | — | — |
| **Shallow** | 19 | 7-12 | **12-7** | **+1.54** [-1.59, +4.68] | **0.834** |
| Deep | 18 | 11-7 | **7-11** | -1.92 [-4.91, +0.78] | 0.091 |

Registry family `handle_follow_override_distance_v1`, three cells, all
`unresolved_below_power`; nothing closed, nothing gated.

Gates: ruff format/check clean, mypy clean, `pytest -q` 4529 passed, 9 skipped.

## Tried

- Graded Week 1 off `paper_decisions` and reported 10-5. Wrong: that ledger
  carries ARI for ARI at LAC. Use the played card, not the paper ledger.

## 2026-09-14, later: the fake 50.0% is fixed forward, and the cap does not generalise

**Display, fixed going forward only.** `board_content._played_pick` no longer
substitutes `PICK_SIDE_FLOOR` when a rule serves a side the model prices below
0.5: it carries the model's real number, the strength word goes empty, and
`GameRow.confidence_label` reads "Against the read". `board_assistant`'s game
sentence reads the label rather than the bare word. The honest number is what
gets frozen into `published_picks` at the next lock.

**Already-published weeks are left exactly as published (owner, binding).** A
first attempt also re-rendered locked Week 1 games (ARI at LAC 50.0% -> 36.2%,
WAS at PHI -> 36.7%, CHI at CAR -> 48.8%). That is rewriting a published card
and was reverted; the frozen ledger still wins for any locked game, verified by
diffing the rendered rows against `HEAD:docs/index.html`. Do not re-apply it.

**The override-distance measurement above was redone from scratch on owner
order; see `docs/leader_median_model_confidence.md` and
`docs/lanes/leader-median-model-confidence.md` for the current numbers.**
The identical 61-49 is confirmed and explained (a mathematical identity
between override distance and model confidence, not a defect), and an
honest leave-one-season-out search finds no confidence gate worth adding to
the leader-median rule.

What the redo says about PHI at 0.368: the firmest quarter of the
leader-median rule's flips (model above 0.58 on its own side, 42 flips) is
20-22 following the rule, the only band that reads negative, but no
confidence cut chosen out-of-season beats the ungated rule, and the rule's
own honest out-of-sample number is +2.13 [-0.51, +4.80] against the
in-sample +3.00 it was promoted on. The bad Week 1 override came from the
handle rule, which is off the served card.

Gates: ruff format/check clean, mypy clean, `pytest -q` 4529 passed, 9 skipped.

## Next

1. Repair the Week 1 paper-ledger row for ARI at LAC so every future grade
   reads the played card.
2. `card-ledger-check` reported "no disagreements" while checking 1 of 10
   week-1 pick-revision rows; widen it to every row.
3. Owner decision: the only remaining lever on a deep override is a floor on
   the model probability a rule may overturn at all. The redo found no
   out-of-season cut that beats the ungated rule, so a floor is a serving
   preference, not a finding.

## Open

- None.
