# mod25e-gap-table (authoritative open gaps, base ...gx2faldrg = XFA+LDR+RG; read 2026-10-18 evening)

## Goal
One ranked table so no agent is assigned work a later lane section already did. Sim v real, 3 seeds s11-13, base column of artifacts/mod25e3/e163_ldrg_vsXFA_boot.log and e163_ldrg_boot.log (measured). Zero crossing never closes; all rows are unresolved_below_power unless stated.

## State: ranked gaps
| # | Gap | Sim v real (measured) | Owner lane | Already tried | wk-sig id | Module (state) | Next untried unit |
|---|---|---|---|---|---|---|---|
| 1 | Cross-quarter covariance / persistence / catch-up (drives margin SD 15.17 v gate 14.87, and noise 171 v 165, Q4 slope) | SD 15.17; noise 171.3; xq_cov +4.2 v ~-6.4; Q4 slope -.040 v -.060; E174 2*cross-cov gap +16.5 [+6.0,+28.2] | e168, e174 | strength, carry-over, start-yl channel excluded; carrier = trailer conversion in Q2/Q3/Q4 300-900 (+16.3 [8.9,24.3]) | mod25e-e174 family (e174c RGM, e174d TDR, e174e) | TDR (trailer play mix) built NOT in crH; RGM fit; CTX D covers RGT territory (not in crH) | Install CTX+TDR in crH (sole editor), 3-seed v base; trailer turnover/drive needs a logged sim run (sim 17+ other-end -2.1pp) |
| 2 | Mass at 3 | .1272 v .1412-.1471 (-.020 [-.032,-.008]); path = last score FG (reg -.0105, OT -.0096) | e171, e96, e97 | OT/tie rule, FG make rate (not a defect), FD4 over-smoothing (late punt .097 v .004) | none for e171 (FD5 gap not recorded) | FD5 built, crH hook installed; 3-seed s11,s12 exist, s13 absent | Finish FD5 s13, run tests/scratch/e171/score_fe.py base v fe; then TOH (reach .30 v .423) |
| 3 | Late-clock / final-drive cells (E101/E103 family) | final drive P(score) .224 v .340, clock expiry .162 v .022 (tgt_3seed_ldrg); FG kick <=5s .224 v .355; Q4 last play <=5s .311 v .128 | e103 (done/), e104, e113, e122, e173 | GFL relabel, CKY/CKC (CKC lost LOSO), GSTOP rejected (noise +8.8, r2 -.03) | none | GSTOP2, TOH built default-off, never smoked | Smoke GSTOP2 then GSTOP2+TOH (RNG fixes in 1a7b4803) |
| 4 | Leader 17+ 3rd-down excess (E175) | 3rd conv +5.0pp [+2.3,+7.5]; TD share +3.6pp; tilt carries only +0.6 of +3.9 | e175 | tilt off, TILTL, yard shift, strength, personnel, score-first rerank (E175e FAILS) | mod25e-e175-leader17-3rd-conv-gap | LDR3 in crH default-off (not refit); CTX D handles 9+/17+ leader 3rd FD IN | E175f declared (drop exact-dist filter, 5 terms LOSO, tests/scratch/e175/f1.py) - run it; else CTX in sim |
| 5 | Neutral 3rd&7+ run conversion (E176) | 17.0 v 23.8% all-state; neutral 21.2 v 27.3; CTX D 23.0; E-variant 24.3 (gap -3.0 [-4.9,-1.3]) | e176 | distance pool (not carrier), penalties, CTX D (closes 1.8 of 6.1pp) | e176-neutral-3rd7plus-run-conv-residual-after-ctxD | CTX built, not in crH; CTXDB (E variant) not built | Build CTXDB default-off new module (630x12 LOSO) after CTX; ask what the other ~3pp is (run-type mix, served-yards ~1pp) |
| 6 | Return-TD share (E172c) | 4.52% v 3.40% (B 3.71, g2 4.36) | e172c | gap-conditioned, field mix, defense gradient, bandwidth, Geo candidate replay, paired logs (infeasible) | none | RTD not carrier | p4.py: free-row admission mod25e_geo.py:344 (confirm); then default-off module restricting free rows |
| 7 | Trailer go/turnover (E174) | go -3.5pp [-5.4,-2.0] at 17+; other-end -2.1pp | e174 | E174b-e, TDR, RGM | e174c/d/e ids | TDR not in crH | covered by row 1 |
| 8 | Timeouts (E173) | called rate cells: 60-300s engine over, <=30s under | e173 | logistic + kNN fits | none | TOH built, never smoked | smoke with GSTOP2 |
| 9 | Key numbers 9-15 excess, 14 -.0137, 7 -.0076 | mirror of missing 3/14 | e171 | profile/path only | none | none | none known beyond rows 2-3 |
| 10 | Late r2 | .1427 v .146 (band .132-.166) | - | - | - | - | MET; no work |

## Default-off hooks built, never smoke-tested (measured: no e5 dir/log)
GSTOP2 (smoke_g2s), TOH (smoke_g2sth), STBK (smoke_tk), SAFEG+SAFEGP (smoke_sg), SERVEDIDX (verification sim), LDR3 (not refit), CTX and TDR (not in crH). RGT smoke killed partial; TILTL 3-seed has s11 only; V1AD has s11,s12; FD5 3-seed has s11,s12 (no smoke). All queued in tests/scratch/e178/sim_queue.sh (log empty at read time).

## Lanes finished, move to done/ (not moved)
e91, e92, e95, e97, e99, e100, e101, e102, e106, e107, e108, e111, e112, e113, e115, e116, e117, e118, e120, e121, e122, e123, e124, e126, e129, e131, e160, e161, e162, e166, e167, e169 (superseded by CTX), e172. Keep live: e96, e98, e109, e159, e168, e171-e177, hook-review, generator-fidelity.

## Next
Install CTX (+TDR) in crH via sole editor; then run the queue, one 3-worker sim at a time, thread caps.

## 2026-10-18 evening update (orchestrator; supersedes rows above where they differ)
- Row 1 SD/noise/persistence: CTX F (CTX=2) 3-seed running (tests/scratch/e177/run_cx2_3seed.sh, base ...gx2faldrg). CTXSCR (scramble share, cs) queued on top of CTX F; TDR (tr) 3-seed queued. Trailer turnover per drive still needs a turnover-logging sim.
- Row 2 mass3: FD5 provisional (s11+s13) no gain, OT final-3 worse; s12 contaminated, rerun queued.
- Row 3 late clock: GSTOP2 smoke failed (reach within noise, -4.7 [-16.9,+7.3]); real carrier shared by base and GSTOP2 = trailing offense timeout use on final drives .14 v real .58 (lane hook-review). TOH (timeout calls) init fixed; GSTOP2+TOH smoke queued third.
- Row 4 leader 3rd-down: E175f enlarged pool fails all-state guard; CTX F covers leader FD offline; check in the CTX F run.
- Row 5 neutral 3rd&7+ run: scrambles carry it (real 44.5% scramble share; conv .445 v designed .138). CTXSCR offline: .2845 v real .2901 IN; scramble-flag held-out LL +0.0041 v CTX F, joint +0.0011.
- Row 6 return-TD: static kernel is NOT the carrier (60k-query replay 3.48 v real ~3.40; earlier 5.23 was 300-query noise, record invalidated). CLSDRAW (cl) smoke queued second.
- Hooks: crash-loop guard in crH (worker init failure aborts the run). Simulator gap records use --effect-units rate_pp.
- 21:55 ET: CTX F 3-seed not adopted alone (SD +0.23, noise +5.5, xq_cov +4.6, Q4 slope toward real); it removes a compensating leader-turnover error; Q4 trailer per-state rate is the catch-up carrier (rate part +0.156 pts/drive, occupancy -0.036); penalties ruled out; DECIDE re-picks are the untested suspect (DECLOG smoke queued).
