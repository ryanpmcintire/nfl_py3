# Lane: sim-finish-amendment6 (done 2026-11-11 11:03)
## Goal
Test whether the spread is the balance point (cover share 1/2), not the mean; grade the simulator on push and half-point pricing under that conditioning.
## State
- Step 1 (real games 2009-25, 4,543 regular-season games, `artifacts/sim08_amendment6/step1.json`): favourite margin minus line is +0.99 at 3.5 [-0.07, 2.11], +1.86 at 7.0 [0.20, 3.43], +1.49 at 4.5, -1.62 at 2.5 [-2.79, -0.43], while decisive cover share is 0.470 to 0.521 on every line 1 to 7.5. Push share 0.090 at 3, 0.060 at 7, 0.085 at 10. Reading supported in direction (mean and balance differ by up to 1.9 pts with cover near 1/2); not tight.
- Amendment 6 appended to docs/sim08_market_anchored_declaration.md (declared before the tilt; 25 cells; third conditioning on these draws).
- `scripts/sim08_amendment6_grade.py` (stage 1 real games, stage 2 balance tilt on main draws) -> `artifacts/sim08_amendment6/{step1,grade,record_batch}.json, balance_weights.npz`. 25 cells recorded, shard registry/weak_signals/sim08_market_anchored.json 84,205 -> 113,905 bytes.
- Results (2026-11-03 engine, 661 whole-line games, 1,322 hook rows): see grade.json. Sim push mass reliable (5 bins within 0.5 pt); W1 served minus sim +0.0014 [-.0086,.0119] P+ .61; W2 table minus sim +0.027 P+ .89; hook rows: sim alone loses to served two-stage (H1 -0.013, P+ .002; decisive 212 v 272), accuracy .526 v .571; stage-one logit plus sim push mass beats stage-one plus empirical table by +0.0005 (P+ .92). K3 balanced .0727 v real .0859 (-1.32 pp, P+ .03), no repair.
## Tried
Mean tilt (A5), shift (inversion lane), balance tilt (A6): third conditioning on the same draws. ESS mean 363, min 212, 929 below the 369.8 floor.
## Next
Driver option (do not edit in this unit): in `scripts/sim08_market_anchored.py` add `amend6()` beside `amend5()` (line 180) reading SIM08_AMEND6, add `--amend6` beside line 898 and env set beside line 910; make `tilt_draw_weights` (lines 184-189) call a per-game bisection on exp(theta x (margin - line)) so weighted cover share among non-push draws = 1/2 (copy `balance_weights` from the A6 script); call sites 245, 467, 732, 797 unchanged apart from passing row["line"] (already passed).
Final-run commands: as sim-finish-amendment5.md Next 1 to 4 with `--amend6 1` replacing `--amend5 1` (smoke check: weighted cover share among non-push within 1e-6 of 1/2; run name amend6; grade with `sim08_amendment6_grade.py` adapted to `--run`).
## Open
- Balanced draws carry no cover opinion at the opener by construction; the served two-stage already beats sim-alone on hook rows. Whether A4 terminal transitions fix the short margin-3 mass (-1.3 pp here; -3.5 pp on 3 to 3.5 lines) is for the final run to show.
