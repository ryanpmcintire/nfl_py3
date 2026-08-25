# Literature-mined leads (2026-08-21)

Four parallel research agents mined academic/practitioner literature for
documented mechanisms an NFL forced-pick ATS project could replicate. Provenance
tags inline: **measured** (primary source opened this session), **read**
(source opened via search results), **reported** (search snippet, unverified),
**inferred** (reasoning). None of these numbers has been reproduced on local
data; every design below requires its own predeclared screen and registry
record before it can inform a card.

Execution order implied by the four sections combined:

1. Bye-advantage decay / market overvaluation (§2 lead 3) — cheap, clean CBA
   mechanism, directly contradicts a published 73% road-favorite claim.
2. Week-1 totals under bias + holdover-bias re-tests (§1 leads 1-2) — the
   holdover figure already failed to replicate here once (52.5% measured vs
   35.6% published); totals side is untested locally.
3. Body-clock x NIGHT games (§2 lead 1) — Smith et al.'s 5.26-point evening-game
   west-coast ATS effect is the largest published NFL-native number in this
   file, and our body_clock screen tested only EARLY windows.
4. Open-to-close overreaction autocorrelation (§3 lead 1) — Management Science
   2024 negative-autocorrelation finding maps onto the already-built
   observed-movement channel.
5. Availability-science replication ladder (§4) — practice-pattern base rates
   FIRST (pure local replication), then recurrence multipliers; workload-spike
   ratios enter as raw covariates only given the Impellizzeri refutation.
6. Line-move/visibility, primetime slot effects, sentiment shading, backup-QB
   gap valuation — batch with existing families.

---

## Section 1 — documented ATS biases

### Tier 1 — high claimed effect, fully point-in-time replicable

**1. Week-1 totals under bias.** DiFilippo, Krieger, Davis & Fodor, "Early
Season NFL Over/Under Bias," *Journal of Sports Economics* 15(2), 2014
(**read**, ideas.repec.org/a/sae/jospec/v15y2014i2p201-211.html): Week-1
scoring significantly lower 2000-2010 but books fail to cut totals; betting
under every Week-1 total returned **+13.6% per game**, claimed significant.
Test: opening-total archive + schedules, under on all Week-1 openers/closers,
chronological, season-by-season stability, natural post-publication decay
check.

**2. Holdover bias in Week-1 spreads.** Fodor, DiFilippo, Krieger & Davis,
*Applied Financial Economics* 23(17), 2013 (**read**, abstract): prior-year
playoff teams cover only **35.6%** of openers; systematic fade claimed ~22%/game
2004-2012 (**reported**). Follow-ups: "Exploiting Week 2 Bias" (*J. Prediction
Markets*, 2015); Krieger/Davis/Strode, "Patience is a virtue" (*J. Econ. Fin.*
2021) (**reported**). NOTE: the 35.6% figure ALREADY failed to replicate in
this project (52.5% measured on 120 Week-1 holdover favourites, `docs/pool_edge_plan.md`)
— re-test the totals variant and the Week-2 variant instead of the spread fade.

**3. Home-underdog / heavy-favorite bias and its DECAY.** Golec & Tamarkin 1991;
Vergin & Sosik 1999; Dare & MacDonald 1996; Dare & Dennis 2011 (**all
reported** via citation pages). Counter-evidence **read**: Humphreys, Paul &
Weinbach 2013 find home dogs went only **48.35% ATS 2005-2011** — flipped/died —
while bettors pile onto big road favorites. Practitioner: favorites -12+ were
220-275-9 ATS since 1978 (Birnbaum 2013, **read**); Szalkowski & Nelson claim
home-dog bets at **53.5%** 2002-2011 (**reported**). Test: |line| buckets x
home/away at the OPENER, cover rates by 5-year window, estimate the flip year.

### Tier 2 — moderate effect, replicable

**4. Weather mispricing in totals.** Borghesi 2007 (*J. Econ. & Business*,
n=5,463, temperature extremes produce forecast errors consistent with
mispriced acclimatization — **reported**); Borghesi 2008 weather-adjusted
strategy above the 52.38% threshold out-of-sample (**read**). Test with
forecast-time weather only (game-time actuals are upper bounds here).

**5. Post-bye rest differential — DOCUMENTED REVERSAL, ideal decay study.**
Sung & Tainsky 2014 (*JSE*): road favorites off bye covered ~73% over eight
seasons (**read** via Illinois news release). Directly contradicted **read**:
"Bye-bye, bye advantage" (Frontiers in Behavioral Economics, Aug 2026):
state-space models 2002-2023 show bye advantage fell from +2.2 pts pre-2011 CBA
to +0.3 after while market adjustment ROSE +0.39 → +0.97 pts — market now
OVERVALUES the bye by ~0.6 pts; fading bye-advantaged teams won ~52% since
2011. Cheap, high-value, clean mechanism.

**6. Primetime pricing.** Vergin & Sosik: MNF home underdogs 65.6% ATS
1981-96 (**reported**). Modern counter-data **read**: FanDuel Research (Sep
2025): MNF home teams covered only **43.9%** 2019-24 vs 48.8% otherwise; MNF
totals set ~1 pt higher than early Sunday. Sports Insights BetLabs: MNF road
favorites +22.1% ROI 2004-12 (**read**). Test: slot fixed effects on opener
cover rates, split pre/post-2014 TNF regime change.

**7. Teaser key-number pricing.** Stanford Wong's 72.4% leg breakeven vs
historical ~73-74% for cross-3-and-7 pairs (**read**); nflanalytic claims
27-season verification with marginal edge after dynamic pricing (**read**,
unaudited). Test: simulate Wong legs from openers vs closers; the
openers-vs-closers gap is itself the finding.

### Tier 3 — real literature, thinner effects

**8. Line-move predictability / visibility.** Krieger & Davis *J. Econ. Fin.*
48:263-279 (2024) (**read**): low-visibility games show more frequent/larger
line moves (n=3,756, 2007-2021). Costa 2025 thesis (**read**): odds-move
direction predictable; away teams in 0.3-0.7 win-probability games profitable
— HFA overvaluation in close games. Candidate feature family; needs leakage
test.
**9. Sentiment/popularity shading.** Feddersen, Humphreys & Soebbing (WVU WP,
**read**): each 1-pt Facebook-Likes share gap shifts the spread ~0.6 pts toward
the popular home team. Test via popularity proxy strictly pregame.
**10. Divisional-game strategies** (Shank 2019, **reported**) — one flag on
existing pipeline; batch with 5-6.
**11. Playoff pricing** — no strong peer-reviewed NFL gap surfaced; n≈13
games/year keeps any test underpowered for years; record whatever comes out.

Not found: any peer-reviewed claim of >52% long-run ATS against OPENERS
specifically with a modern documented sample (**inferred** from corpus
coverage).

---

## Section 2 — travel/sleep science

**1. Body-clock offset x KICKOFF TIME (night games) — highest priority.**
Smith et al., *Sleep* 36(12), 2013 (**read**, PMC3825451): 40 years of NFL
EC-vs-WC games; west-coast teams beat the spread by **5.26 ± 1.33 pts** in
evening games (n=106, p<0.0001) vs **0.16 ± 0.80** in 1pm games (n=293).
Jehue, Street & Huizenga, *MSSE* 25(1):127, 1993 (**read**): West teams' away
win-% fell -16.3% vs East for day games 1978-87 but held ~68% in night games.
Design: signed tz-offset between team home tz and game local time INTERACTED
WITH kickoff window (>=8pm ET vs 1pm/4pm). All inputs local. Inferred expected
magnitude 1.5-3 pts toward later-body-clock team at night (5.26 is a selected-
class upper bound). OUR EXISTING body_clock SCREEN TESTED ONLY EARLY WINDOWS —
this is the designed-better version.

**2. Directional jet lag with recovery-day decay.** Leota et al., *Front
Physiol* 13:892681, 2022 (**read**; 11,481 NBA games): eastward jet lag
-1.29 pts margin (p=0.015); effect VANISHES when recovery days >= zones
crossed; westward null everywhere. Song et al., *PNAS* 114(6), 2017
(**read**; MLB): resync ~1 h/day. Design:
residual_jetlag = max(0, zones_crossed_eastbound - days_since_travel) from
prior game location/date — exactly the DESIGNED-BETTER version of the naive tz
cell screened near-null in docs/travel_rest_battery.md (averaging over
recovery days dilutes a decayed signal). Inferred NFL magnitude ~0.5-1.5 pts.

**3. Rest-differential extremes.** Front Behav Econ 2:1479832, 2024
(**read**): bye advantage +2.21 pts/gm (CrI 0.61-3.80) pre-2011 CBA collapsing
to +0.31 (-1.01,+1.64) after; MNF-rest market prices +0.37 (CrI 0.14-0.61).
Gitter, JSE (**read**): TNF totals -2.85 pts; home team on 6-vs-7 days rest
wins 14.8pp less. Use as modifier for lead 2 more than standalone.

**4. Altitude acclimatization timeline.** Nussbaum et al., BMJ 335:1278, 2007
(**read**): each +1,000 m ≈ +0.5 goal margin for altitude-native home side;
acclimatization consensus 1-2 weeks moderate altitude. NFL ceiling ~1,600 m ⇒
inferred well under 0.5 pts — cheap add-on term (visitor deficit x days since
arrival), not standalone.

**5. Hotel/away sleep disruption.** AFL actigraphy: NO pregame sleep loss from
interstate travel; match-night sleep falls equally home/away (Richmond 2003,
**read**); Super Rugby overseas sleep down up to ~1h, team-dependent (Shearer
2022, **read**). Fold into lead 2's feature rather than separate screen.

Power honesty (inferred): at σ≈13.1 pts and ~250 games/season, a 1-pt true
effect needs multi-season pooling; leads 1-2 have plausible ≥1-pt effects;
3 borderline; 4-5 sub-point. Predeclare all as category-3-prone and record
regardless of interval position.

---

## Section 3 — analytics community

Filter tags: [PRICING-GAP] escapes the team-quality ceiling; [QUALITY]
re-measures quality (bounded near zero here).

**1. Open-to-close line-movement overreaction [PRICING-GAP].** Simon,
"Inefficient Forecasts at the Sportsbook," *Management Science* 70(12), 2024;
follow-up IJSF 20(4) Dec 2025 finds significant NEGATIVE AUTOCORRELATION in
NFL/NBA/NHL moneyline movement 2019-2023 — books overreact mid-week then
correct (both abstracts read). Feature: (open − current)/|open| signed, plus
days-since-open; both prices known pregame. Maps onto the built
observed-movement channel (+1.863 pts P+ 0.935 challenger).

**2. Net-rest-edge scheduling imbalance [PRICING-GAP].** Sharp Football
Analysis 2026 schedule study + BetIQ trend pages (**reported**): teams with
rest disadvantages show poor ATS splits (small-n, possibly noise). Rest is
known perfectly in advance — calendar-derived family with trivial leakage
tests; evaluate season-by-season.

**3. Week 17-18 motivation/clinch mispricing [PRICING-GAP].** CBS/PFF Week-18
coverage shows lines moving multiple points on rest announcements midweek;
handicappers rate some rest spots 6-10 pts off consensus (**measured read** of
published numbers / **reported**). Design: deterministic clinch/tank-state
classifier (built today in scripts/motivation_ladder_screen.py) driving
"intent-divergent" game flags; small n (~30 games/season x 2 weeks).

**4. Backup-QB replacement-gap valuation [PRICING-GAP].** Sports Insights
oddsmaker survey: elite QBs worth 6-7 pts, non-QB stars ~0.5 pt; books key the
move to starter FAME not starter-minus-backup gap (**reported**). Design:
games with QB change vs prior week; regress margin surprise vs closing spread
on (starter EPA − backup EPA) from prior snaps — does the uniform haircut
misprice good backups?

**5. Fourth-down aggressiveness as VARIANCE identity [PRICING-GAP].**
Aggressiveness shifts margin DISTRIBUTION, not mean — spreads price means
(PFF volume ranges + nfl4th model, both read). Design: go-rate residualized
vs nfl4th recommended rate; interaction residual x |spread|. Split-half
reliability required (already measured +0.320 in PER-07).

**6. Special-teams hidden yardage under new kickoff rules [PRICING-GAP].**
ST-EPA share as CHANGE-from-prior-regime (2024 dynamic kickoff reset stale
priors) — borders QUALITY if it just re-measures goodness; keep the
change-framing.

**7. Weather x offense-style spreads [PRICING-GAP]** — wind x visitor-pass-rank,
cold x dome-team flag (numberFire wind analysis reported: >12mph ≈ 6 fewer
points). Partially covered by ENV-05; the SPREAD-side conditioning remains.

**8. Early-season turnover anchoring [PRICING-GAP]** — weeks 1-3, preseason
turnover proxies (new HC/OC/QB flags) vs lines set off prior-year strength.

Explicitly bounded near zero: opponent-adjusted EPA/DVOA variants, Elo
refinements, player-value aggregation ([QUALITY]).

---

## Section 4 — availability science

Ranked by validated-effect strength x fit to local data (weekly participation
snapshots 2016-2025, injury reports 2009-2024 w/ 24h cutoff, snap counts).

**1. Practice-pattern → game-status base rates (highest fit).**
Footballguys tabulated every team practice report 2017-23 (**read**): final-
practice play rates FP 86%, LP 71%, DNP 29%; hamstring DNP→20%, knee DNP→28%,
concussion DNP→29%. FantasyPros ML model claims Brier 0.135-0.208 by report
day (**reported**, unaudited). Design: player-week target = next-game DNP or
snap-share <10%; features Wed/Thu/Fri status sequence, designation, injury
class, age, DNP-streak. The local 24h-cutoff reports let us REPLICATE the
Footballguys table measured rather than trusting it — do that first; doubles as
the family's reliability check.

**2. Prior-injury recurrence multipliers (strongest measured effects).**
Hamstring meta-analysis, 78 studies / 8,319 injuries (**read**, van Dyk BJSM
2020): any HSI history RR=2.7; same-season recent HSI RR≈4.8; prior ACL RR=1.7;
older age SMD=1.6. de Visser BJSM 2014 (**read**): palpation tenderness at RTP
AOR=3.95; each prior HSI AOR=1.33; 17/64 re-injured within 12 months. Design:
time-since-return hazard flags per body-part class (<60d/<120d post-RTP) plus
career prior counts, predicting next-game DNP; leakage = pre-cutoff rows only.

**3. Concussion protocol → structured availability lag** — mandatory 5-phase
graduated protocol, no fixed timeline (**read** nfl.com fact sheets). Model
next-game availability as f(days-since-listing, phase proxies); validate on
local concussion rows first.

**4. Workload spikes — CONTESTED, carry both sides.** Pro: EWMA-ACWR >2.0 RR
5.9-21.3 in AFL (tiny sample, **read**). Con: Impellizzeri et al., Sports Med
2021 (**read**): ratio artifacts — random denominators reproduce OR≈2; ACWR
does NOT beat intercept-only (identical Brier 0.0351). Design: snap counts as
load proxy; raw acute and chronic terms as SEPARATE covariates (never the
ratio); category-3 until player-level validation speaks.

**5. Age curves & durability persistence.** RB decline sharp at 28→29
(Cockcroft 2023, **read**); high-volume RBs missed FEWER subsequent-season
games — durability persists within player (Orthopedics 2017, **read**); ACL
return ~79% at ~56 weeks with ~⅓ production loss (AJSM 2006, **read**).
Design: age x position splines on games-missed; per-player durability random
intercepts (split-half reliability BEFORE trusting — the 0.933
injury-value-lost reliability suggests real signal lives here).

**6. AGL-style weighting as aggregation target.** Football Outsiders Adjusted
Games Lost weights (**read** methodology): probable ×0.05 ... IR ×1.00.
Reproduce locally as a VALIDATION TARGET for the player-level model's
aggregation, not a new feature family.

Priority: #1 (pure local replication) → #2 (largest published multipliers) →
#5/#6 as targets/aggregators; #4 last, raw-load covariates only.
