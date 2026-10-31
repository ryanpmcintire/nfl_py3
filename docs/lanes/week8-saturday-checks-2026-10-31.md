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
- git fsck --connectivity-only exit 0. backup_data --status: E: mirror 100% (1 pending); offsite snapshot 10-25 OK.

## Next
- Today's scheduled passes keep running (refresh_sat 10:30, lineups_sat 12:00, refresh_last_call_sat_1215, inactives/refresh 15:30-19:15). Grep 'PICK-CHANGE|NOTIFY' in data/scheduler_log.txt for 2026-10-31 to catch later flips.
- Sunday: refresh_sun 10:00 re-nominates the star; grep 'BEST-PICK-LEDGER sunday|PICK-CHANGE|NOTIFY' for 2026-11-01 and tell the owner the final star and any changed picks. Lock Sun 4 PM ET. Then move this lane and week8-lock-2026-10-27 to done/.

## Open
- F: online scan needs an elevated prompt: owner runs `chkdsk F: /scan` (read-only online scan) when convenient.
