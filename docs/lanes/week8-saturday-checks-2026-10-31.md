# week8-saturday-checks-2026-10-31

## Goal
Saturday Week 8 check: scheduler health after the overnight crash, lineup/injury news, line moves vs the pool's Tuesday lines, board update and push; tell the owner every changed pick.

## State
- Machine crashed 01:18 ET (Kernel-Power 41), rebooted 07:04. Ntfs 98: F: needs an online scan (CHKDSK /scan, admin). Scheduler restarted 07:07; ticks failed WinError 1393 on data/scheduler_heartbeat.json until NTFS self-repair 07:09; 07:10 tick ran. data/scheduler_state.json was lost (fresh file), so --status shows NEVER RUN everywhere; scheduler_log.txt history is intact.
- Picks vs Tuesday card (measured, scheduler_log): Wed 18:17 refresh_wed flipped ARI/DAL, ATL/TB, IND/JAX, PHI/WAS; Fri 13:37 flipped ARI, ATL, IND back. Net change since Tuesday: PHI at WAS, PHI +4.5 -> WAS -4.5 (52.4%). Star CHI +3.5 held.
- Card on disk (Fri 13:38 publish, commit 99147994): CIN -10.5 63.0%, DEN +1.5 60.2%, CHI +3.5 59.2% star, LV +1.5 56.9%; the rest 50-53%.

## Tried
- Injury/weather roundup (all reported, unverified web news; nfl.com/injuries/league/2026/reg8, fetched 10-31, Fri-final statuses; QB rows show practice only, no game status). No 2026 weather found (searches returned prior seasons only); no stale items used.
- ARI@DAL [ARI +6.5]: ARI Q Beck full; ARI out G Gaines, DT Nolen; DAL out WR Turpin, LB Winters; DAL Q WR Pickens, DT K.Clark (hurts ARI side if they play) (src: nfl.com, 10-31)
- ATL@TB [ATL +2.5]: ATL out RB B.Robinson, 8 Q (CB Terrell, WR Zaccheaus, DE Ebukam, T C.Williams); TB out WR Egbuka (helps ATL), Mayfield full (src: nfl.com, 10-31)
- BAL@BUF [BAL +3.5]: Lamar Jackson full practice, no designation (coach/Wolfson said he plays, SI 10-26) - plays = good for BAL; BAL out G Ioane, S Hawkins, T Stanley DNP; BUF out WR Shavers, Q WR D.Moore (ltd), TE Kincaid (ltd), T S.Brown (ltd) (src: nfl.com 10-31, si.com 10-26)
- CLE@PIT [CLE +3.5]: PIT out CB Echols, CB Slaughter, Q WR Skowronek; CLE Q T Howard (concussion, DNP), CB Harden, S Thomas; no CLE/PIT QB listing found (src: nfl.com, 10-31)
- IND@JAX [IND +7.5]: IND out WR Gould, CB Walley, Q TE Ogletree (DNP); JAX out S A.Johnson, Q WR Meyers (DNP, ham/ankle) (helps IND) (src: nfl.com, 10-31)
- KC@DEN [DEN +1.5]: Mahomes full; KC out DT Norman-Lott, Q CB Sneed, CB Delane (DNP); DEN out RB Harvey, S Hufanga, Q T Bolles (ltd), DE Z.Allen (ltd), S B.Jones, WR Waddle (ltd chest) (hurts DEN side if out) (src: nfl.com, 10-31)
- LAC@LA [LA -6.5]: LAC out S Derwin James, G Awosika, CB Leonard, WR Thompson (helps LA); LA Q CB McDuffie, C Shelton, TE Parkinson, WR Whittington (src: nfl.com, 10-31)
- LV@NYJ [LV +1.5]: NYJ out T Mitchell, DE Ossai, DT Sweat, DE D.Bailey doubtful, Q RB Hall, G Parham (helps LV); LV out DE P.Johnson, TE Runyon, Q WR Bech, G Powers-Johnson (src: nfl.com, 10-31)
- MIN@DET [MIN +3.5]: DET out S K.Joseph, Q CB McCreary; MIN only Q S Ward; T Darrisaw DNP (no game status listed) (src: nfl.com, 10-31)
- NE@MIA [NE -7.5]: Maye full, Willis full; NE Q WR A.Brown, CB C.Davis, LB Elliss, RB Henderson (hurts NE side if out); MIA out T Proctor, Q CB Brents, CB R.Taylor (src: nfl.com, 10-31)
- PHI@WAS [WAS -4.5]: nfl.com lists this under Mon Nov 2 (schedule check). PHI out WR D.Smith, WR Wicks, WR Covey, CB J.Jones, LB Campbell (helps WAS); WAS out DT Payne, DT Kinlaw, Q DT Newton, TE Bates, WR Burks; Daniels full (src: nfl.com, 10-31)
- TEN@CIN [CIN -10.5]: TEN out RB Spears, WR Tate, P Townsend, Q WR Ridley, DE Franklin-Myers (helps CIN); CIN out S Dugger, Q LB Giles-Harris (src: nfl.com, 10-31)
- CHI@SEA [CHI +3.5]: no CHI/SEA game designations; SEA QB Drew Lock listed Limited (Darnold status unfound, uncertain); SEA T Cross, WR Shaheed, LB Nwosu DNP, no status yet (src: nfl.com, 10-31)
- CAR@GB (Thu): board shows CAR 19 at GB 27, CAR +3.5 lost; week 0-1, season 59-49.
- 07:12 ET Bovada data/market/raw/20261031T111210Z (13 games). Market vs pool line, our side (+ = we hold the better number, measured): CIN -10.5 vs -12.5 (+2.0); WAS -4.5 vs -6 (+1.5); ATL +2.5 vs +2, BAL +3.5 vs +3, DEN +1.5 vs +1, CHI +3.5 vs +3 (+0.5 each); MIN, LV, CLE flat; ARI +6.5 vs +7, LA -6.5 vs -6, NE -7.5 vs -7 (-0.5 each); IND +7.5 vs +10 (-2.5).
- Dry-run refresh-picks 11:13Z (scratch log, not recorded): one change, LAC at LA: LA -6.5 -> LAC +6.5 (home cover 48.9%, line 6.5 -> 6.0 toward LAC). consensus_movement challenger (off) would flip IND/JAX; not served.
- Board defect found: "Why this pick" for ATL@TB, IND@JAX, ARI@DAL argues the Wednesday side after Friday's flip-back. Fix delegated: docs/lanes/why-this-pick-flipback.md.
- independent_validation v4 stopped by its source guard since 2026-10-26 (b95b4c61 edited pick_probability.py 10-25); zero Week 8 captures. v5 protocol written: registry/studies/combined_vs_raw_2026_v5.json (Weeks 8-18). Enroll after the board fix lands, because pick_refresh.py and card_view.py are pinned.
- 07:30 catch-up publish_site_fri_0030 pushed 4daa6b2a built from the live board view (LAC shown before the ledger had it); superseded 6 min later.
- DONE 07:32: v5 enrolled (164 games, 13 Week 8 captured, source_unchanged). Commit 4d6bc8cd (board explanation fix + v5 defaults).
- DONE 07:35: --run-job refresh_sat recorded PICK-CHANGE LAC at LA: LA -6.5 -> LAC +6.5 (51.1%), NOTIFY-OK jiepb3vNFUJj, PICK-CHANGE-PUBLISH 05cf1fcc pushed. Card late-week block: LAC and WAS. Board rows all match the card.
- Lost scheduler state made sweep_missed replay every catch_up job's last window back to back (~50 sync_captures at ~4 min, blocking due jobs). 07:38 wrote data/scheduler_state.json rebuilt from scheduler_log.txt (1,617 runs, 60 days; pre-rebuild copy in the session scratchpad); the daemon merges the file after each job, so the sweep skips the rest.
- Owner asked why IND stays IND after a 2.5-pt move against it. Measured: the served move term read leader_median_net_move 0.0. Bovada moved +1.5 (Tue 8.5 -> Wed 10); William Hill and MyBookie were first captured Wed 22:00Z already at 10, so their only diffs were Friday's flat 10->10 and the median {1.5, 0, 0} = 0. That is why Wed refresh flipped IND/ARI/ATL to the home side (only Bovada had a diff) and Fri flipped them back.
- Fix (sharp_book_movement_features.py): a book counts only if it has a reading before the Wednesday anchor. Training check on artifacts/sharp_book_weighted_movement/spread_quotes.parquet (2023-25): 7 of 2,397 book-game pairs lacked a baseline; the fix changes 2 of 799 training values (2024_14_CLE_PIT -0.5->0, 2025_11_GB_NYG 0->-0.25); frozen training artifact untouched, coefficients stay valid. Live Week 8: only Bovada has a baseline for every game. Dry run: IND at JAX -> JAX -7.5 (58.2%), ARI at DAL -> DAL -6.5 (51.9%); others hold. Gates: ruff/format/mypy clean on src, pytest 1645 passed. Validation v5 -> v6.
- DONE 08:09: --run-job refresh_sat recorded ARI at DAL -> DAL -6.5 (51.9%) and IND at JAX -> JAX -7.5 (58.2%); NOTIFY-OK GAOJKHxz6jAw; PICK-CHANGE-PUBLISH 65fba627 pushed. Card late-week block: DAL, JAX, LAC, WAS. Board rows match. Fix commit 52555216.
- Root of the missing baselines: Tuesday 18:25Z Odds Gap capture (20261027T182541Z-odds-gap-private) listed 1 game per book; Wed 9, Fri 13. 08:11 retry captured 13 games x 3 books; Bovada lines unchanged since 07:12.
- git fsck --connectivity-only exit 0. backup_data --status: E: mirror 100% (1 pending); offsite snapshot 10-25 OK.
- Second crash 08:25, reboot 12:59, scheduler auto-started 13:01; state file survived. 13:27 lineups_sat recorded NE at MIA -> MIA +7.5 (50%), NOTIFY-OK l4JpdyO5ee0o, pushed 8e11397c. Missed while down: odds_private_sat, refresh_last_call_sat_1215, prediction_markets_sat_1200.
- DONE 16:53: Odds Gap root cause was our parser, not the source. scripts/capture_odds_gap_private.py kept only games within 4 days of the scan; the Tuesday response held all 14 games with WH and MyBookie. Fixed (commit 1bec13f4). Re-parsed 13 stored responses since 9/22 into `*-odds-gap-private-reparse` siblings (same observed time, capture_kind live, parse_kind reparse; clv.py:247 loads live only); every previously parsed row matched. Week 8 move term now 3 books (IND/JAX, LV/NYJ 2). --run-job refresh_sat recorded ATL at TB -> TB -2.5 (52.6%) and LAC at LA -> LA -6.5 (51.2%), NOTIFY-OK Z8CL3RtoqhuF, PICK-CHANGE-PUBLISH d0ebee0c pushed.

## Next
- draftcap_forward_log still fails (coefficients differ from artifacts/draftcap_forward/frozen.json since 10-22); Sunday 11:15 will fail again; Week 8 uncovered. Lane: draftcap-forward-log.md.
- Today's scheduled passes keep running (refresh_sat 10:30, lineups_sat 12:00, refresh_last_call_sat_1215, inactives/refresh 15:30-19:15). Grep 'PICK-CHANGE|NOTIFY' in data/scheduler_log.txt for 2026-10-31 to catch later flips.
- Sunday: refresh_sun 10:00 re-nominates the star; grep 'BEST-PICK-LEDGER sunday|PICK-CHANGE|NOTIFY' for 2026-11-01 and tell the owner the final star and any changed picks. Lock Sun 4 PM ET. Then move this lane and week8-lock-2026-10-27 to done/.

## Open
- Tuesday 11-03 14:25 odds_private_tue is the first live check of the parser fix: its JSON should report about 14 games per leader book.
- Weeks 3-7 served the move term on Bovada alone; the recorded picks stand. Any 2026 forward read of the move term should be recomputed from the re-parsed quotes.
- F: online scan needs an elevated prompt: owner runs `chkdsk F: /scan` (read-only online scan) when convenient.
