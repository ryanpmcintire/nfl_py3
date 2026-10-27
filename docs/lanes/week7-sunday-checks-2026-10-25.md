# week7-sunday-checks-2026-10-25

## Goal
Sunday-morning Week 7 check: lineup news and line moves against the pool's Tuesday lines; tell the owner the picks at risk and the final Best Pick.

## State
- 04:53 ET Bovada capture data/market/raw/20261025T085350Z (13 games). Odds Gap skipped (scan >70 min old).
- Moves vs pool line, our side (measured): CIN +2.5 vs market CIN -3 (+5.5 for us); LV +4.5 vs +3 (+1.5); ATL +3.5 vs +2.5 (+1); HOU -9.5 vs -10.5 (+1); SEA -2.5 vs -3, CLE -1.5 vs -2, NO +3.5 vs +3 (+0.5 each); NYJ -4.5 vs -4, CAR -3.5 vs -3, GB +3.5 vs +4 (-0.5 each); ARI, IND, DAL flat.
- Reported: Lamar Jackson ruled OUT Saturday (Huntley starts); Mayfield Q but full practice, expected to start; Nacua Q game-time (calf); Brissett starts for ARI (Beck Q, backup); Breece Hall Q.
- Card on disk is the Sat 17:51Z publish: CIN +2.5 at 51.2% despite the 5.5-pt move; NYJ -4.5 star 67.2%.
- PIT at NO kicks 09:30 ET (international); inactives_sun_intl before it.

## Tried
- Dry-run refresh-picks 08:58Z with the fresh line: 14 considered, NE_CHI post-kickoff, 0 changed picks, 13 games with a current line, no override applied.
- CIN 51.2% diagnosis (read): served p = logit terms model 0.277, flag_sum 0.260, move_toward_home 0.222 per point (pick_probability/20261024T174809Z). Saturday move was -4.0 toward CIN, offset by the model plus the bye flag toward BAL. The board's "model QB Lamar" is the depth-chart display. inactives_refresh_v1 is a challenger (inactives_refresh_overlay.py:32), so Lamar's inactive status will not enter the served number.

- Owner caught it: three Sunday picks differ from the Tuesday card. SF at ATL went to ATL +3.5 and DEN at ARI to ARI +4.5 (Wed 18:20 ET refresh_wed). LA at LV went to LV +4.5 (Sat 12:28 ET lineups_sat). CIN and DAL flipped and came back. NE at CHI went to CHI (Thu, already played). All have NOTIFY-OK, but ntfy does not surface on the owner's phone and email is skipped, so the owner was told in session 2026-10-25 ~05:15 ET.

- 12:48 ET (measured, scheduler_log): refreshes at 08:30, 10:00, 11:55 and 12:45 logged no PICK-CHANGE; last change was Sat 12:32 ET (LV +4.5). Star NYJ -4.5 kept (Sunday nomination nominees_differ false). Card 12:15 ET: CIN +2.5 61.9%, NO +3.5 59.9%.
- Fixed scripts/lead99_llm_read.py (uncommitted): team_names walks back until every slate team is named (KeyError CHI killed the 08:40 and 11:40 reads); weather_block skips empty frames. `packet` builds 14 games. 14:45 run should land.
- draftcap_forward_log FAIL Thu 10-22 and Sun 11:15: refit 2013-2025 coefficients differ from artifacts/draftcap_forward/frozen.json. Week 7 still covered by the 10-18 rows. Cause not yet found (snap file refresh or feature table rebuild suspected).

## Next
- Find which draftcap input changed, then re-enroll v2 for Weeks 8-18 if the change is legitimate. Commit the lead99 fix. Check 14:45 lead99 run and 15:00 refresh_sun_inactives_late for PICK-CHANGE.
- refresh_sun 10:00 ET re-nominates the star; inactives_sun_early ~11:30, refresh_sun_inactives_early 11:55. Grep 'BEST-PICK-LEDGER sunday|PICK-CHANGE|NOTIFY' in data/scheduler_log.txt for 2026-10-25 and report the final star. Lock 16:00 ET.

## Open
- Why the served probability stays ~51% on CIN with the market 5.5 pts through the pool line and Lamar out: diagnose, do not override (one calibrated probability rule).
