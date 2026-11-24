# thanksgiving-news-2026-11-24

## Goal
Read-only news scan for the three Thanksgiving games (Thu 2026-11-26): CHI at DET 1:00, PHI at DAL 4:30, KC at BUF 8:20 ET. Covers QB and key injuries, Buffalo weather, and line moves. Feeds the parent lane `docs/lanes/thanksgiving-2026-11-24.md`.

## State (Tue 2026-11-24, ~10:00 ET)
Sources used: beat capture `data/raw/beat_reporters/20261124T123028Z` (posts from 11-22 onward), `data/raw/nflverse_injuries/20261122T140039Z/injuries.parquet` (Weeks 10-11 only), `data/raw/injury_news/20261124T010038Z/current.parquet` (PFT headlines), and `data/raw/forecast_archive/mos_bulletins/{GFS,MEX}/KBUF`. Scratch scripts are tg_scan.py and tg_scan2.py in the session scratchpad. Official team pages and NFL.com still show Week 11 (**reported**, fetched 11-24).

**Lines** (**reported**, Fox Sports 11-23 3:44 PM ET, FanDuel):
- CHI -2.5, total 55.5. Opened DET -2.5 in May, so the side flipped.
- DAL -8.5, total 47.5. Opened DAL -1.5 with a 49.5 total (Covers May lookahead), a 7-point move.
- BUF -3, total 50.5. Opened BUF -3, so the spread has not moved.
- Splash shows CHI -2.5, DAL -8.5 and BUF -2.5 (**read**, parent lane).

**CHI** (Bears are 8-2):
- Monday's estimated walkthrough report lists three DNPs: CB Kyler Gordon (shoulder, also out in Week 11), OL Theo Benedet (quad) and RB Roschon Johnson (ankle, hurt 11-22). LB D'Marco Jackson was full (**measured**, beat capture, 3 reporters, 11-24 01:21Z).
- C Garrett Bradbury left in Week 11 and is not on Monday's list (**measured**).

**DET** (six straight wins):
- S Kerby Joseph had "no red flags" after his debut and should play (**measured**, beat capture, Birkett 11-23 23:04Z).
- C Cade Mays is on IR (**measured**, PFT 11-21).
- Week 11 questionables: CB D.J. Reed, LB Joe Bachie, CB Khalil Dorsey (plus a wrist injury in-game), DL Tyler Lacy, CB Ennis Rakestraw and RB Sione Vaki (**measured**, nflverse).
- LBs wore a No. 46 Jack Campbell wristband on 11-22 (**measured**). That he is out is **inferred**.
- No Week 12 report was found.

**PHI** (3-7):
- QB Jalen Hurts is out for the season with blood clots. RT Lane Johnson retired. Joe Milton III has started since Week 9, and Tanner McKee is limited with a rib injury (**reported**, PhillyVoice 11-23; Inquirer and NBCSP 11-04 to 11-06).
- Monday estimate DNPs: CB Quinyon Mitchell (hamstring, new), CB Jonathan Jones, LB Smael Mondon and EDGE Arnold Ebiketie.
- Monday estimate limited: RB Saquon Barkley (neck, left the Steelers game) and TE Dallas Goedert (shin, new).
- Monday estimate full: DeVonta Smith, Jalen Carter and Jonathan Greenard (**reported**, PhillyVoice 11-23).
- LB Zack Baun and WR Dontayvion Wicks are on IR (**measured**, PFT 11-21).

**DAL**:
- Monday estimate DNPs are all starters: RG Tyler Booker, TE Jake Ferguson, LB Dee Winters and LB Jaishawn Barham.
- QB Dak Prescott (left hand) was full. CB DaRon Bland is on IR (**reported**, PhillyVoice 11-23; Bland's IR **measured**, PFT 11-21).
- Roof status was not found.

**KC**:
- Mahomes was a full participant in Weeks 10-11 and is playing after his ACL recovery (**measured**, nflverse; Newell post 11-22).
- Out in Week 11: RG Trey Smith (concussion), T Jaylon Moore (ankle) and RB Kenneth Walker III (groin) (**measured**, nflverse).
- No Week 12 report was found.

**BUF**:
- Week 11 final report: CB Christian Benford out (groin), LB Dorian Williams out, TE Dalton Kincaid doubtful, and Ed Oliver, Damar Hamlin and T.J. Sanders questionable (**measured**, nflverse; **reported**, Bills site).
- No Week 12 report was found.

**Buffalo weather**:
- NWS point forecast issued 6:57 AM EST 11-24 (**reported**). Thursday: rain likely after 1 PM (70%), high 43°F, NE wind 5-10 mph with gusts to 26. Thursday night: rain (100%), low 36°F, 0.5-0.75 inch, gusts to 26. No snow and no hazards.
- Repo MEX MOS, run 11-22 12Z and two days stale (**measured**): 7 PM ET Thursday at 41°F with 18 kt wind, 63-82% rain and a snow probability of 6% or less.

## Tried
- WebSearch returned mostly 2024-25 pages. Team injury pages and NFL.com reg12 still show Week 11. GFS MAV only reaches 11-25 12Z.

## Next
1. After Wednesday practice, around 4 PM ET, fetch the official Week 12 reports for all six teams.
2. Confirm the Eagles' Thursday starter (Milton or McKee) and Barkley's status.
3. Find Monday estimates for KC, BUF and DET, plus Jack Campbell's status.
4. Refresh the NWS Buffalo forecast Wednesday night and Thursday.
5. Find the AT&T roof decision.

## Open
- Eagles starting QB is unconfirmed for this game.
- KC, BUF and DET Week 12 reports are unknown.
