# week8-postmortem-2026-11-02

## Goal
The owner reported Week 8 as a disaster (4 correct). Grade every layer of the pick chain on the live 2026 season at the Splash opener, name the failing layer, and fix the served side before the Week 9 lock (Tue 11-03).

## State
Scratch scripts (session scratchpad 352ecbdc...): w8_layers.py, chain.py, chain2.py, hist.py, extra.py. Inputs: artifacts/clv_ledger/decisions.graded.parquet (results), published_picks.parquet (card = last publication before each deadline), artifacts/waterfall_feed/*/feed.json (last feed before deadline: residual, smooth, discrete, stack), pick_probability/20261101T171318Z/per_game.parquet (2020-25 opener history).

- Week 8 (13 graded, CHI at SEA Monday): Tuesday card 4-9 (= owner's 4), final card 5-8, Best Pick CIN -10.5 lost (moved from CHI Sun 08:36). Home 10-3, favorites 8-5; Tuesday card took the away side 11 of 14.
- Layers, W8 / W3-8 (88) / 2026 (120): model residual sign 7-6 / 50-38 / 68-52; after +0.43 home probability-rule shift 6-7 / 44-44 / 63-57; discrete read 2-11 / 45-43 / 65-55; fitted stack 7-6 / 45-43 / 68-52; card 5-8 / 41-47 / 64-56.
- Flip records: discrete flips the residual side W8 5 games 0-5, W3-8 7-12, 2026 11-14. Discrete vs smooth disagreements: 2020-25 discrete side 108-131 (239 games), 2026 13-11. Probability-rule flips: 2026 2-9, 2020-25 112-92. Stack flips discrete W3-8 16-16 (W8 6-1). Card != final feed: 8 games 2026, W3-8 0-4 (W8: Sunday 12:43 refresh_trigger_log_sun flips ATL and NYJ, both lost). Late flips Tuesday->final: 11-12 season.
- 2020-25 opener (in-sample to discovery; discrete read first committed 2026-09-08): raw residual 52.8%, smooth 54.6%, discrete uncorrected 53.0%, served base 53.4%, stack (out of season) 55.4%.
- Suspect: discrete read moves far from smooth where no key number sits at the boundary (KC at DEN line 1.5: smooth .539 -> discrete .478; LAC at LA and ARI at DAL 6.5 fall although 7 is on the favourite's cover side).

- Smooth-base challenger DONE (verified: harness reproduces stored LOSO 833-670 exactly). Through the stack the discrete base beats smooth: 2020-25 LOSO 833-670 vs 822-681, disagreements 57-46 discrete, P(smooth better on LL) .146; 2026 forward (104 games) discrete LL .6810 vs .6842, P .008, record 59-45 vs 61-43. Discrete base stays served. Recorded registry/weak_signals/smooth_base_challenger_2026_11_02.json. The pre-stack "discrete 2-11" is a layer reading; the stack is fitted on the discrete logit and absorbs it.

- Discrete-read audit DONE (lane discrete-read-audit-2026-11-02.md): design, not a sign bug. tilted_atoms matches the band MEAN to a point built with the residual MEDIAN shift; band margins are not recentred (KC at DEN band mean margin -1.76, median -3). Shades away from home favourites (home-pick rate .388 vs smooth .475, 2020-25), not located at key numbers. Median-matching tilt: 794-709 / 54-49 vs production 787-716 / 56-47; no variant gains. Recorded discrete_vs_smooth_side_2020_25.
- Recorded (week8_postmortem_2026_11_02): late flips after the last feed 2-6 (P .09); probability-rule median-shift flips 2026 2-9 (P .019, 11 post-hoc games of 18 looks; 2020-25 112-92). Both unresolved_below_power.
- Decision: Week 9 serves the unchanged chain (discrete base + stack). No layer beat the served chain through the played card.

## Tried
- Inputs: source freshness COMPLETE on the card. Move: Saturday Bovada read follows 5-4 on W8 moved games (lane week8-saturday-checks).

## Next
- Pre-register Weeks 9-18 forward tracking of three arms at the opener: served, model residual sign, served minus post-feed late flips; grade weekly.
- Check whether the probability-rule median shift changed sign from 2020-25 to 2026 (+0.36..+0.45 toward home in 2026); a sign change is a mechanism to repair at its source, not a gate.
- Repair the lattice mean/median inconsistency only together with a stack refit, graded through the played card.

## Open
- The multimodal rule (owner directive) requires the discrete read for served probabilities; a smooth read is served only if it beats the discrete read on the opener grade through the played card.
