# served-flip-rules-hold

## Goal

The owner's rule (2026-09-14): no rule flips a served pick on its own
signal; a situational signal enters the model's probability as evidence and
the side is whichever the combined number favours. Done when nothing on the
served card overrides the model outright and the situational signals are
served, if at all, as fitted terms in one probability (lane
`market-updated-model`).

## State

Inventory (read, 2026-09-14, Explore agent; call sites verified in
`four_overlay_composition.py` lines 311-323 and `pick_refresh.py`):
the served card had **ten rules that flip a side outright** and one
continuous tilt. The nine composition members are flips by construction:
the composition's contract accepts only `p` or `1 - p` from every member.

| Rule | Mechanism | Evidence it was promoted on | Status now |
|---|---|---|---|
| late-week leader-median follow | flip on a 0.5-point move | 799 games, thresholds picked on the same games; out of season +2.13 [-0.51, +4.80] | **off** (`LATE_WEEK_FOLLOW_SERVED`) |
| handle follow 70% | flip | 37 flips, 19-18 | **off** (`HANDLE_FOLLOW_SERVED`) |
| rookie crew underdog | flip to a refit's side | two reads with opposite sign (P+ 0.046 vs 0.972) | **off** (`ROOKIE_CREW_SERVED`) |
| interim HC first game | flip | 39 events | **held** (`OWNER_HELD_MEMBERS`) |
| precip + high total | flip | 29-50 flagged games | **held** (`OWNER_HELD_MEMBERS`) |
| coach fade | flip | 762 games / 104 coaches | still served |
| division revenge | flip | 258 flagged opener games | still served |
| player arrests back side | flip | 1,503 games | still served |
| bye edge fade | flip | 498 flagged | still served |
| forecast cold visitor | flip | 245 flagged | still served |
| protection mismatch | flip | 733 flagged team-games | still served |
| tank zone fade | flip | 144 flagged | still served |
| home-side offset, big spreads | continuous margin shift | 1,537 games | still served (already a term, not a flip) |

Held members return `disabled_owner_hold` in the composition provenance and
contribute no flips; re-enabling is removing the name from
`OWNER_HELD_MEMBERS`. All three refresh switches keep recording their arms.

Reader text: the board's refresh note and the assistant's timing entry no
longer say a line move or the money flips a pick.

Gates after the switches (measured): ruff format/check clean, mypy clean,
pick-refresh / composition / board / publishing suites 210 passed.

2026-09-14, owner directive (measured, 1,537 opener-graded games: 50-52%
picks won 57.8%, 61-70% picks won only 52.5%): every per-game probability,
decision score and strength word came off every reader-facing surface
(board meter/ticker/history/assistant/dive-panel/explanation sentence and
`CURRENT_PREDICTIONS.md`'s Decision score column), and the dead
`AGAINST_MODEL_READ_LABEL` path was removed since no served rule can price
a pick below 0.5; the pick, spread, reasons, flip line, Best Pick star and
season-level accuracy are unchanged. Gates (measured): ruff/mypy clean,
full `pytest -q -p no:warnings` 4523 passed / 9 skipped, `publish-board` +
`publish-predictions` regenerated with zero per-game `%` left (grep-verified).

## Tried

- Nothing else; the six larger-sample flips stay served only until the
  combined model in `market-updated-model` gives an out-of-season number
  for serving them as terms instead. The owner can hold any of them now by
  adding the member name to `OWNER_HELD_MEMBERS`.

## Next

1. `market-updated-model` lane: extend the combined model from the line
   move to the seven composition flags as additive logit terms, fitted
   leave-one-season-out, and compare against the flip chain on the played
   card.
2. When that number is in hand, either serve the combined probability in
   place of the composition or hold the remaining members.
3. Week 2 lock is Tuesday 2026-09-15: report which picks the held rules
   would have flipped.

## Open

- Owner decision: hold the remaining seven flip members now rather than
  wait for the combined model.
