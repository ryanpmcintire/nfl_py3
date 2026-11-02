# Triage G6 2026-11-02 (read-only)
## Goal
Classify LEAD-66/75/76/80/81/86/89/98/99.
## State
- LEAD-66 CLOSE-NOW: registry/weak_signals/lead66_margin_pmf.json holds 3 contrasts (rps_improvement; 2 refuted_mechanism, 1 unresolved_below_power). Parser now has rps_improvement (weak_signals.py:78). 15 calibration cells not recorded; optional.
- LEAD-86 CLOSE-NOW: registry/weak_signals/lead86-unit2-filtered-regularization.json + LEAD86-fixed-market-penalty.json; unit 2 complete.
- LEAD-98 CLOSE-NOW: 4 cells recorded unresolved_below_power (lane Tried); row stays 🔬 only by marker.
- LEAD-75 FORWARD-ONLY: paired_wind jobs OK (grep OK paired_wind = 75 lines; paired_wind_sun_1445 OK 5 pairs 2026-11-01); sun_1145 FAIL = TimeoutError source read (transient, 1 slot). Score after enough pairs; no wind outcomes read.
- LEAD-99 FORWARD-ONLY: score after Week 18. Tuesday FAIL 10-27 13:30 "no captured pool week has a game still to kick off" (run found no captured week; then MISSED 10-31 on catch-up). Deadline views OK 11-01 (0840/1140/1445). Week 9 Tuesday view absent: check whether Wk9 capture existed.
- LEAD-76/80/81/89 source-gated (inventory done, zero looks).
## Next
See report. LEAD-99 Tuesday FAIL: scripts/lead99_llm_read.py week default picks "captured week with game still to kick off"; on 10-27 week 8 capture missing/past.
## Open
Whether 10-27 week 8 capture existed.
