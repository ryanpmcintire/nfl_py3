# Era-magnitude report

**Nothing in this document is a result, and nothing here was recorded.**
This is item 6 of the ranked agenda in `docs/pool_edge_plan.md`'s
"2026-08-31 registry state and next shots" section (read,
`docs/pool_edge_plan.md:809-816`): per-era magnitude reporting of the
registry's already-recorded era-split constructs whose sign flips across
the era boundary, plus a screenable mechanism proposal for each.
No `nfl-ats weak-signals record` or `nfl-ats rotation record-look` command
was run to produce this document, no registry entry was changed, closed, or
reclassified, and no screen proposed below was executed. Every mechanism
cell proposed in this document is a predeclaration draft, not a run.

## Binding closing-grounds taxonomy (verbatim, restated because this doc has
no access to a session's AGENTS.md/CLAUDE.md context injection)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism — a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". If a record command errors, the verdict is wrong,
> not the validator.

Also binding: the project's "era magnitude, not presence" rule — effects
vary in MAGNITUDE across eras; a weaker- or opposite-signed era reading is
never evidence of absence, and a sign flip is reported as two magnitudes,
never averaged away.

## 1. Generated tables (measured this session)

Command: `./.tools/uv.exe run --no-sync python scripts/era_magnitude_report.py
--markdown`, run against `registry/weak_signals.json` this session. **9
era-split groups found** (every stem sharing >=2 same-league entries whose
`seasons` ranges are pairwise disjoint). Output pasted verbatim below;
`tau^2` is the DerSimonian-Laird between-era heterogeneity estimator,
computed by calling `nfl_ats.weak_signals.pooled_effect(method="random")`
directly on each group's era members (not reimplemented, so these numbers
are guaranteed to agree with `nfl-ats weak-signals pool`'s own arithmetic).
The pooled point/interval shown under every group is **informational only**
— combining already-visible signs is not a predeclared confirmatory look
(see §3) — and is never an independent vote alongside its own inputs.

### `altitude_deficit_4000ft`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| altitude_deficit_4000ft_era_2009_2017 | 2009-2017 | 2242 | -0.0399 | [-0.4114, +0.3332] | 0.4155 | - |
| altitude_deficit_4000ft_era_2018_2025 | 2018-2025 | 2075 | +0.0905 | [-0.2803, +0.4678] | 0.6873 | + |

- Between-era heterogeneity (DerSimonian-Laird): **tau^2 = 0.0000** (accuracy_points^2); random-effects pooled point (informational, NOT an independent confirmatory look) = +0.0250 [-0.2389, +0.2889], `excludes_zero`=False.
- Sign flips across the era boundary (raw mechanical flag): **True** — but both point estimates sit within ~0.09 points of zero; see §3.
- Full-range parent (OVERLAPPING both eras at once — not an independent vote): `altitude_deficit_4000ft`, seasons 2009-2025, n=4317, effect +0.0230, interval [-0.2443, +0.2889], P+ 0.5662.

### `body_clock_night_west_road_ge2000et`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| body_clock_night_west_road_ge2000et_2009_2016 | 2009-2016 | 48 | -0.1048 | [-0.2647, +0.0642] | 0.1086 | - |
| body_clock_night_west_road_ge2000et_2017_2025 | 2017-2025 | 71 | -0.0640 | [-0.2478, +0.1180] | 0.2469 | - |

- Between-era heterogeneity: **tau^2 = 0.0000**; pooled (informational) = -0.0866 [-0.2088, +0.0357], `excludes_zero`=False.
- Sign flips: **False**. Consistent negative sign, both eras.
- Full-range parent: `body_clock_night_west_road_ge2000et`, seasons 2009-2025, n=119, effect -0.1713, interval [-0.4137, +0.0738], P+ 0.0843.

### `body_clock_west_road_early`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| body_clock_west_road_early_2009_2016 | 2009-2016 | 4317 | -0.0308 | [-0.3268, +0.2738] | 0.4217 | - |
| body_clock_west_road_early_2017_2025 | 2017-2025 | 4317 | -0.1175 | [-0.4583, +0.2272] | 0.2487 | - |

- Between-era heterogeneity: **tau^2 = 0.0000**; pooled (informational) = -0.0685 [-0.2944, +0.1574], `excludes_zero`=False.
- Sign flips: **False**. Consistent negative sign, both eras.
- Full-range parent: `body_clock_west_road_early`, seasons 2009-2025, n=4317, effect -0.1545, interval [-0.6272, +0.3106], P+ 0.2588.

### `bye_overval_home_edge`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| bye_overval_home_edge_pre2011 | 2009-2011 | 744 | +0.2708 | [-0.7434, +1.2010] | 0.7069 | + |
| bye_overval_home_edge_post2011 | 2012-2025 | 3573 | -0.3304 | [-0.7563, +0.0965] | 0.0636 | - |

- Between-era heterogeneity: **tau^2 = 0.0341**; pooled (informational) = -0.1951 [-0.6873, +0.2971], `excludes_zero`=False.
- Sign flips: **True**.
- No exact-stem full-range parent entry exists in the registry for this construct.
- Registry notes (read directly, verbatim): `bye_overval_home_edge_pre2011`: "n_flag=58 underpowered by construction; season secondary degenerate (3 blocks, below 10-block floor); pair-read with post2011 twin only." `bye_overval_home_edge_post2011`: "seed=20260821 samples=20000. Season-blocked secondary [-0.7744,+0.1022] P+=0.0666. Disclosed overlap: subset of venue_milestone_post_bye_home rows, correlated with travel_rest_home_off_bye - do not sign-test-pool together. Battery multiplicity uncorrected (5 cells)."

**See §2.1 for mechanism.**

### `interim_hc_active`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| interim_hc_active_era_2009_2017 | 2009-2017 | 4484 | -0.0456 | [-0.2752, +0.1833] | 0.3333 | - |
| interim_hc_active_era_2018_2025 | 2018-2025 | 4150 | +0.0000 | [-0.2932, +0.2937] | 0.4865 | 0 |

- Between-era heterogeneity: **tau^2 = 0.0000**; pooled (informational) = -0.0282 [-0.2080, +0.1516], `excludes_zero`=False.
- Sign flips: **False** (the later era's point estimate is exactly 0.0, so there is no strictly-positive side to flip to).
- Full-range parent: `interim_hc_active`, seasons 2009-2025, n=8634, effect -0.0239, interval [-0.2044, +0.1648], P+ 0.3859.

### `pt_post_mnf_sunday`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| pt_post_mnf_sunday_era_2009_2017 | 2009-2017 | 4196 | +0.2546 | [-0.1592, +0.6733] | 0.8841 | + |
| pt_post_mnf_sunday_era_2018_2025 | 2018-2025 | 3894 | -0.1367 | [-0.5532, +0.2886] | 0.2566 | - |

- Between-era heterogeneity: **tau^2 = 0.0309**; pooled (informational) = +0.0602 [-0.3232, +0.4437], `excludes_zero`=False.
- Sign flips: **True**.
- Full-range parent: `pt_post_mnf_sunday`, seasons 2009-2025, n=8090, effect +0.0666, interval [-0.2272, +0.3620], P+ 0.6705.

**See §2.2 for mechanism.**

### `sagarin_battery_large_divergence`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| sagarin_battery_large_divergence_era_2010_2016 | 2010-2016 | 498 | +1.8072 | [-2.9955, +6.6855] | 0.7620 | + |
| sagarin_battery_large_divergence_era_2017_2025 | 2017-2025 | 696 | -2.2989 | [-6.0519, +1.5060] | 0.1152 | - |

- Between-era heterogeneity: **tau^2 = 3.5215** — the largest of any group in this report; real detected heterogeneity beyond what sampling noise explains, not just a coin-flip-sized wobble. Pooled (informational) = -0.5359 [-4.5194, +3.4477], `excludes_zero`=False.
- Sign flips: **True**.
- No exact-stem full-range parent entry exists in the registry for this construct (only differently-graded `_open`/`_close` variants; see §2.3).

**See §2.3 for mechanism.**

### `sbr_opener`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| sbr_opener_era_2011_2014 | 2011-2014 | 1024 | -0.2508 | [-3.0348, +2.5845] | 0.4137 | - |
| sbr_opener_era_2015_2019 | 2015-2019 | 1280 | +0.8914 | [-1.8400, +3.6386] | 0.7340 | + |
| sbr_opener_era_2020_2021 | 2020-2021 | 528 | +3.2946 | [-1.3514, +7.6336] | 0.9178 | + |

- Between-era heterogeneity (3-arm DerSimonian-Laird, same formula generalized to k=3): **tau^2 = 0.0000**; naive random-effects pool (informational) = +0.8087 [-0.9889, +2.6062] — matching `docs/pool_edge_plan.md`'s own quoted naive pool of these three arms (+0.809 [-0.989, +2.606], read) to three decimal places, cross-checking that this script's `pooled_effect` call agrees with the number already on record.
- Sign flips: **True** (one negative, two positive arms).
- No exact-stem full-range parent entry exists; instead the registry holds `sbr_opener_pooled_2011_2021`, **measured directly on the 2011-2021 union window**, not built by algebra on these three arms. See §3's precedent discussion — this group is not proposed for a fresh mechanism screen because the project has already run the right kind of confirmatory look for it.

### `surface_familiarity_r3`  (nfl)

| entry | seasons | n games | effect (pts) | 95% interval | P+ | sign |
|---|---|---|---|---|---|---|
| surface_familiarity_r3_era_2009_2017 | 2009-2017 | 973 | +0.5277 | [-2.1556, +3.1865] | 0.6482 | + |
| surface_familiarity_r3_era_2018_2025 | 2018-2025 | 884 | +2.3869 | [-0.3446, +5.0651] | 0.9577 | + |

- Between-era heterogeneity: **tau^2 = 0.0000**; pooled (informational) = +1.4456 [-0.4549, +3.3462], `excludes_zero`=False.
- Sign flips: **False**. Consistent positive sign, both eras — but excluded from §3's candidate list; see there.

## 2. Mechanism proposals for the three sign-flipping groups

Three of the nine groups flip sign across the era boundary with a
substantive magnitude on both sides (not a near-zero straddle):
`bye_overval_home_edge`, `pt_post_mnf_sunday`, and
`sagarin_battery_large_divergence` — the same three named in
`docs/pool_edge_plan.md`'s ranked-agenda item 6 (read, lines 809-816).
`altitude_deficit_4000ft` and `interim_hc_active` also show a raw sign
difference per §1's mechanical flag, but both eras sit within ~0.1 points of
zero for the former and one arm is exactly 0.0 for the latter — treated
under §3 as consistent/low-EV, matching how `pool_edge_plan.md` itself
already groups them (read, lines 705-707). `sbr_opener`'s 3-way split flips
too, but that construct already has a properly-measured union-window
confirmation (`sbr_opener_pooled_2011_2021`); it is discussed as the
precedent in §3, not proposed for a new mechanism screen.

### 2.1 `bye_overval_home_edge` — pre-2011 +0.271 pts vs post-2011 -0.330 pts

**This is not a fresh hypothesis. The mechanism is already fully
predeclared and documented**, and the registry's two era entries here ARE
two of the five cells of that predeclaration. Read,
`docs/bye_overvaluation_screen.md:1`, its own title: "Bye-week overvaluation
screen (post-2011 CBA)"; read, `docs/bye_overvaluation_screen.md:4-7`, its
opening paragraph: "the hypothesis that the market still prices bye rest at
its historical magnitude while the true on-field bye advantage has
collapsed [after the 2011 CBA], leaving a systematic fade-the-bye edge."
Read, `docs/bye_overvaluation_screen.md:49-54`, its "Era boundary (frozen
convention)" section: "Post-CBA era = seasons 2012-2025 ('post-2011');
pre-CBA control = seasons 2009-2011 ... 2011 is assigned to the pre-era
conservatively: the CBA was ratified August 2011, after schedules were
built." The registry's own `bye_overval_home_edge_pre2011.description`
field (read, `registry/weak_signals.json`) already says "pre-CBA true bye
advantage reportedly real" — the CBA, not the overtime rule, is the
hypothesis this construct's own name and predeclaration were built around.

That doc also carries a **reported, explicitly unverified** literature
lead (`docs/bye_overvaluation_screen.md:9-16`): a cited 2026 Frontiers in
Behavioral Economics paper reportedly finds
the true bye advantage collapsing from +2.2 points pre-CBA to +0.3 points
post-CBA, while the market's own bye adjustment reportedly ROSE from +0.39
to +0.97 points — i.e. the market allegedly overvalues byes by ~0.6 points
in the modern era. `docs/bye_overvaluation_screen.md` states plainly that
"none of these numbers are independently verified here; they motivate
direction only," and this document does not verify them either — they are
relayed here as **reported (unverified)**, exactly as that doc already
labels them.

**Verified independently this session (web), because the task asked for
the actual rule-change dates rather than trusting the addendum's inferred
guess:**

- The 2011 NFL lockout ran **March 12, 2011 – July 25, 2011**; the new CBA
  was **signed August 4, 2011**; training camps opened July 27, 2011.
  Reported (web): [2011 NFL lockout — Wikipedia](https://en.wikipedia.org/wiki/2011_NFL_lockout),
  [NFL clubs approve comprehensive agreement — NFL.com](https://www.nfl.com/news/nfl-clubs-approve-comprehensive-agreement-09000d5d820e6311),
  [NFL history in 95 objects: pen, 2011 CBA signing — SI](https://www.si.com/nfl/2014/06/03/nfl-history-in-95-objects-pen-2011-cba-signing).
- The 2011 CBA eliminated two-a-day padded practices, cut offseason OTAs
  from 14 to 10 days, and cut allowed padded training-camp practices from
  as many as 28 to 16. Reported (web):
  [The CBA in a nutshell — PFT/NBC Sports](https://profootballtalk.nbcsports.com/2011/07/25/the-cba-in-a-nutshell),
  [NFL training camps: a re-explanation of CBA rules — Niners Nation](https://www.ninersnation.com/2019/7/29/8934213/nfl-cba-training-camp-agreement-two-a-days-contact),
  [Offseason Rules — NFLPA](https://nflpa.com/active-players/off-season-rules).
- The regular-season "modified sudden death" overtime rule was adopted by
  owners in **2012**, taking effect starting with the **2012** regular
  season; the postseason version of the same rule had already taken effect
  starting the **2010** postseason. Reported (web):
  [What are NFL overtime rules for regular and postseason play? — ESPN](https://www.espn.com/nfl/story/_/id/39111637/what-nfl-rules-regular-postseason-play).
- Kickoffs moved from the 30- to the **35-yard line starting the 2011
  season**, sharply raising touchback rates. Reported (web):
  [Kickoff rule change has a big effect on NFL — ESPN Stats & Info](https://www.espn.com/blog/statsinfo/post/_/id/35597/kickoff-rule-change-has-big-effect-on-nfl),
  [Kickoffs moved to 35, touchbacks stay at 20 — Fox News](https://www.foxnews.com/sports/kickoffs-moved-to-35-touchbacks-stay-at-20).

**Which boundary does the registry split actually use, and does a real rule
change coincide with it?** Read, `scripts/bye_overvaluation_screen.py:59`:
`ERA_POST_MIN_SEASON = 2012` — the split is season 2011 (inclusive) vs
season 2012 (inclusive), i.e. the boundary sits exactly between the
lockout-shortened 2011 season and the first full season played under the
new CBA's practice-time caps. **Inferred:** two of the four verified events
line up exactly at this boundary — the CBA's practice-cap regime (first
full season 2012) and the regular-season OT rule (effective 2012) — while
the other two do not: the postseason OT rule took effect in 2010, one full
season before the "pre" bucket even starts, and the kickoff-to-the-35
change took effect in 2011 itself, which is *inside* the pre-2011 bucket,
not at the transition to it. Of the two boundary-aligned candidates, the
CBA/practice-cap story has the clearer causal path to a HOME bye-rest edge
(less practice time to spend on a bye week's marginal preparation) than the
OT rule does (OT decides a small subset of games and has no obvious
connection to which side benefits more from a bye week) — this is the
project's own already-declared hypothesis, not a new one reached here.

**Predeclarable follow-on screen** (not run; extends
`scripts/bye_overvaluation_screen.py`'s existing 5-cell battery with one new
cell that tests the CBA/practice-cap channel directly, rather than re-slicing
by era again, which is already done):

- **Cell name**: `bye_overval_install_need_moderator`.
- **Population**: NFL REG games, seasons 2012-2025 (the post-CBA population;
  the pre-2011 control is too thin, n=58, to split further), home team off
  strict bye (>=12-day gap, `POST_BYE_GAP_DAYS=12`) AND opponent not off bye
  — the identical base flag as `bye_overval_home_edge_post2011`.
- **Subset (moderator)**: "install-need" home team — new primary starting
  QB this season vs. last season's Week 1 starter, OR a new offensive
  coordinator this season. Complement: neither changed.
- **Direction (predeclared)**: if the CBA practice-cap mechanism is real,
  a bye week's now-capped extra practice time should still carry more
  marginal installation value for install-need teams than for continuity
  teams, so the post-2011 negative home-bye edge should be SMALLER
  (less negative, or flat) for the install-need subset than for the
  complement: contrast (install-need effect) − (no-need effect) > 0.
- **Comparator**: subset vs. complement, difference-in-differences within
  the post-2012 population (same subset-vs-complement convention as the
  existing battery's cells).
- **Grade**: close — matches `bye_overvaluation_screen.py`'s existing grade
  (the pre-release closing `spread_line`); an opener-graded version is a
  distinct future look, not proposed here, because the opener-paired
  archive only starts in 2020 and cannot touch the pre-2011 control era at
  all.
- **Effect units**: `accuracy_points`.
- **Uncertainty**: `nfl_ats.clv.week_blocked_bootstrap`, week-blocked
  primary (20,000 draws, matching `BOOTSTRAP_SAMPLES`/`BOOTSTRAP_SEED` in
  `scripts/bye_overvaluation_screen.py`); season-blocked secondary reported
  as informational only.
- **Positive control**: `docs/sagarin_backfill.md`-style internal
  consistency is not available here, but a genuine external positive
  control exists: run the identical install-need moderator split on
  `travel_rest_home_off_bye` (`home_rest >= 13`,
  `scripts/nfl_travel_rest_battery_screen.py`, n=266 across the full
  2009-2025 window, not bye-specific but far larger and better powered). If a
  practice-time-driven moderator effect is real, it should be visible there
  with more precision even though that population's bye identification is
  looser; if it fails to appear there despite the extra power, that bounds
  the whole install-need-moderator story.
- **Extends**: `scripts/bye_overvaluation_screen.py` (add as cell 6) or a
  dedicated follow-up script modeled on it,
  `scripts/bye_overval_install_need_screen.py`.

### 2.2 `pt_post_mnf_sunday` — 2009-2017 +0.255 pts vs 2018-2025 -0.137 pts

Read, `scripts/primetime_cells_screen.py:43-46`:
`ERA_SPLITS = (("2009_2017", 2009, 2017), ("2018_2025", 2018, 2025))`, with
no comment tying this specific boundary to an external event. Read,
`docs/primetime_cells_screen.md:92-93`: "Era splits (2009-2017 vs
2018-2025) are scored for EVERY cell in the artifact" — this is a fixed,
uniformly-applied convention across the whole primetime battery (seven
cells), not a boundary chosen to test a hypothesized rule change specific
to the post-MNF-Sunday construct. The registry's own
`pt_post_mnf_sunday.description` field (read) already flags this as
"era-instable (fully early-era driven)" without naming any external cause.

**Verified this session (web):** no NFL rule or scheduling-policy change
was found dated at the 2017/2018 boundary that plausibly touches "a team
plays Sunday six days after its own Monday game." The one substantive
short-week/Thursday scheduling change found (allowing two Thursday games
on short weeks, and Thursday flex scheduling) is dated **2023**, five
seasons after this boundary, not 2018. Reported (web):
[NFL owners approve Thursday Night Football scheduling changes — CBS Sports](https://www.cbssports.com/nfl/news/nfl-owners-table-vote-until-may-on-flexing-thursday-night-football-games-for-2023).
A search for sports-betting market-efficiency or algorithmic line-setting
changes specifically dated to 2017 also returned nothing specific. Given
both of those null results, the more parsimonious **inferred** reading is
that this construct's "era instability" is the ordinary mined-battery
pattern of a reading concentrated in one half of an arbitrarily-fixed
split, not a real environmental shift with an identifiable external cause
— which also means the fixed 2009-2017/2018-2025 boundary itself may simply
be in the wrong place to see whatever pattern (if any) is really there.

**Predeclarable follow-on screen** (not run; reuses precedent machinery
already built in this repository rather than inventing a new method):

- **Cell name**: `pt_post_mnf_sunday_changepoint`.
- **Population**: identical to `pt_post_mnf_sunday` — team-game rows where
  the current game is Sunday and the team's own prior game that season was
  Monday (imported from `scripts/primetime_cells_screen.py`'s existing flag
  builder, not redefined), full 2009-2025 window.
- **Direction**: none predeclared — this is a structural/diagnostic cell,
  not a directional hypothesis test, exactly like
  `docs/era_magnitude_profile.md:128` onward ("2a. Free-break changepoint
  (fixed calendar eras are a convenience, not a claim)"). It asks "is there
  a real break at all, and if so where," rather than assuming the fixed
  calendar split already sits at the right place.
- **Comparator**: per-season effect series plus the optimal single-
  changepoint fit (minimizing total sum of squared deviations across every
  candidate break season with >=3 seasons on each side) — the SAME method
  already predeclared and built in `docs/era_magnitude_profile.md:121-128`'s
  "Stage 2" / "2a" sections and its accompanying script, reused rather than
  reimplemented, plus a
  bootstrap distribution of the break season itself (median, [2.5, 97.5]
  percentile spread, modal share).
- **Grade**: close — matches this construct's existing grade (the
  primetime battery is close-graded per `docs/primetime_cells_screen.md`).
- **Effect units**: `accuracy_points`.
- **Uncertainty**: `nfl_ats.clv.week_blocked_bootstrap`, week-blocked
  primary, reused for both the per-season series and the break-season
  bootstrap distribution (same design as era_magnitude_profile's Stage 2a).
- **Positive control**: apply the identical changepoint estimator to
  `hc_year_one_fade`, a construct with an already-known, already-resolved
  large break (read, `docs/era_magnitude_profile.md:84`: "+0.09pts
  2009-2017 vs -8.08pts 2018-2025") as an instrument check — if the
  estimator correctly locates that construct's break at or near where it is
  already known to sit, that validates the machinery before trusting
  whatever break point (or absence of one) it finds for
  `pt_post_mnf_sunday`.
- **Extends**: `scripts/era_magnitude_profile.py`'s Stage 2a changepoint
  code, generalized to accept the `pt_post_mnf_sunday` flag construct from
  `scripts/primetime_cells_screen.py` as an eighth input series.

### 2.3 `sagarin_battery_large_divergence` — 2010-2016 +1.807 pts vs 2017-2025 -2.299 pts

This group has by far the largest between-era heterogeneity in the report
(tau^2 = 3.52, vs. 0.03 for the other two flipping groups) — real detected
heterogeneity, not a coin-flip-sized wobble, which is exactly why
`docs/pool_edge_plan.md` (read, lines 714-717) calls it "the clearest case
in this pass where pooling would misrepresent two eras that are not
measuring the same thing."

Read, `scripts/sagarin_divergence_battery.py:93-99`:
`SEASON_START = 2010`, `SEASON_END = 2025`,
`ERA_SPLITS = (("2010_2016", 2010, 2016), ("2017_2025", 2017, 2025))` —
again no comment tying the specific 2016/2017 cut to an external event; it
is a roughly-midpoint split (7 seasons vs. 9) of the available window.

**Verified this session (web):** no documented change to Jeff Sagarin's own
rating methodology, and no documented sports-betting market-structure or
market-efficiency shift, was found dated at the 2016/2017 boundary.
Reported (web):
[Jeff Sagarin — Wikipedia](https://en.wikipedia.org/wiki/Jeff_Sagarin) (no
methodology-change history given); a search on market-efficiency/
algorithmic-pricing changes around 2017 returned only general discussion of
sharp-money dynamics, nothing dated. The 2018 PASPA repeal that opened
state-by-state legal sports betting in the US postdates this boundary by a
full season and does not align with it either.

**What the registry's own data DOES verify, and it is a stronger, more
mechanical candidate than any "real world" rule change: the early era's
Sagarin coverage is severely and non-uniformly gapped, which is a
measurement-composition confound, not an environmental shift.** Read,
`docs/sagarin_backfill.md` (lines 525-536): "every one of 2012's captures,
and 2013's weeks 1-12, parsed with `era_format='unknown'` and a null
`home_edge_rating`... 2012 contributes zero usable games to this screen and
2013 only contributes usable games from week 13 onward — a real
archive-format gap, not a join bug." Its measured coverage table (read,
`docs/sagarin_backfill.md:538-561`, close-grade population): 2010 = 31.2%, 2011 = 43.0%,
**2012 = 0.0%**, 2013 = 31.2%, 2014 = 87.5%, 2015 = 93.8%, 2016 = 93.8%,
against 2017 = 100.0%, 2018 = 81.2%, 2019 = 75.4%, 2020 = 100.0% in the
later era — i.e. the 2010-2016 arm's games are drawn overwhelmingly from
its better-covered seasons (2014-2016) with essentially nothing from 2012
and only late-season 2013, while the 2017-2025 arm is covered far more
evenly across its own seasons. Read further, `docs/sagarin_backfill.md`
§9 (lines 641-666), this is not merely a hypothetical confound: the exact
same `large_divergence_era_2010_2016` cell was **measured** to move from
+2.926 points (P+ 0.845) to +1.807 points (P+ 0.762) purely from a
coverage-completeness fix (2010's usable games rose 80→240 of 256, 2011's
rose 110→240 of 256) — direct, in-hand evidence that this specific cell's
estimate is sensitive to which seasons/weeks happen to be covered, exactly
the mechanism a coverage-driven confound would predict.

**Predeclarable follow-on screen** (not run; extends
`scripts/sagarin_divergence_battery.py` with a coverage-matched re-read of
the existing early-era cell rather than proposing an unverified "real
world" rule change):

- **Cell name**: `sagarin_battery_large_divergence_coverage_matched_era`.
- **Population**: identical to `sagarin_battery_large_divergence_era_2010_2016`
  (`|divergence| >= LARGE_DIVERGENCE_THRESHOLD (3.0)`, close grade,
  2010-2016), restricted to seasons whose own within-season usable-Sagarin
  coverage is >=80% per `docs/sagarin_backfill.md`'s own measured coverage
  table — which excludes 2012 (0.0%) and 2013 (31.2%) and keeps 2010, 2011,
  2014, 2015, 2016 (all >=80% after the Era-B consolidation fix).
- **Direction**: none predeclared — this is an instrument-composition
  diagnostic, not a directional hypothesis. Report whether the coverage-
  matched subset's point estimate and interval materially change from the
  currently-recorded full-window 2010-2016 read (+1.807, [-2.996, +6.686]);
  if materially unchanged, the coverage-composition explanation is
  disfavored; if it moves substantially, coverage composition is
  implicated as (part of) the mechanism.
- **Comparator**: coverage-matched subset (2010,2011,2014,2015,2016 only)
  vs. the already-recorded full 2010-2016 read, same `|div|>=3` threshold,
  same grade.
- **Grade**: close — matches this construct's existing grade.
- **Effect units**: `accuracy_points`.
- **Uncertainty**: `nfl_ats.clv.week_blocked_bootstrap`, week-blocked
  primary (20,000 draws, matching `BOOTSTRAP_SAMPLES`/`BOOTSTRAP_SEED` in
  `scripts/sagarin_divergence_battery.py`); season-blocked secondary will
  be thin (5 blocks after excluding 2012/2013) and should be flagged
  DEGENERATE below the 10-block floor, the same disclosure convention this
  project already uses elsewhere (e.g. `interim_hc_active`'s own era-split
  notes, §1 above).
- **Positive control**: already in hand, not hypothetical — the Era-B
  coverage-completeness fix documented in `docs/sagarin_backfill.md` §9
  (read, lines 641-666) already demonstrated this exact cell's point
  estimate moving with coverage composition (+2.926 → +1.807 points as
  2010/2011 coverage rose); this is a measured instrument-sensitivity
  precedent, not a new control that needs to be run.
- **Extends**: `scripts/sagarin_divergence_battery.py` (add a
  coverage-filtered variant of the existing `era_2010_2016` cell, reusing
  `ERA_SPLITS`/`LARGE_DIVERGENCE_THRESHOLD` machinery with an added
  per-season coverage filter sourced from `docs/sagarin_backfill.md`'s
  measured coverage table).

## 3. Consistent-sign, low-heterogeneity groups: candidates for a future predeclared union-window confirmation

Four era-split groups show a consistent (or effectively flat/near-zero)
sign across both eras with tau^2 = 0.0000 — exactly the four named in
`docs/pool_edge_plan.md`'s own "Consistent sign across disjoint eras, low
heterogeneity" list (read, lines 695-707):

- `body_clock_west_road_early`: -0.031 (2009-2016) vs -0.118 (2017-2025),
  both negative, tau^2=0.0000.
- `body_clock_night_west_road_ge2000et`: -0.105 (2009-2016, n=48) vs -0.064
  (2017-2025, n=71), both negative, tau^2=0.0000. Thin samples (a rare
  kickoff-slot/geography combination) — a future pooled look here would
  still be sharpening a small effect, not discovering a large one.
- `altitude_deficit_4000ft`: -0.040 (2009-2017) vs +0.090 (2018-2025),
  tau^2=0.0000. §1's mechanical sign-flip flag reads True for this pair,
  but both point estimates sit within ~0.1 points of zero and the pooled
  read (+0.025 [-0.239, +0.289]) is itself indistinguishable from zero —
  this is the low-EV "near zero in both eras" case `pool_edge_plan.md`
  already groups here, not a magnitude-scale flip like §2's three groups.
- `interim_hc_active`: -0.046 (2009-2017) vs 0.000 (2018-2025),
  tau^2=0.0000. Same low-EV, near-zero character; pooled read -0.028
  [-0.208, +0.152].

All four are, per that same read, "technically poolable, low EV" — worth
carrying forward as candidates for one future predeclared union-window
confirmation, not for a mechanism write-up (there is no substantive
magnitude on either side of any of these to explain).

**Excluded despite a consistent sign:** `surface_familiarity_r3` (§1) is
both positive-signed in both eras (+0.528 and +2.387) and tau^2=0.0000 —
mechanically it would qualify — but its own registry note (read, quoted in
full in §1's table) discloses that the 2018-2025 half "lives in the
2018-2025 window — the same window the upstream weather battery was mined
on," so the project's mined-family discipline treats it as reinforcing
evidence for that upstream battery rather than an independent read. Pooling
or confirming it on a union window would launder an already-played
overlay's own decomposition, exactly the exclusion `docs/pool_edge_plan.md`
already applies (read, lines 722-730).

### The precedent for how to confirm one of these correctly

`sbr_opener_pooled_2011_2021` (registry, read) is the standing example of
how a consistent(-ish) family should actually be confirmed. Its three
era-split inputs (`sbr_opener_era_2011_2014` -0.251,
`sbr_opener_era_2015_2019` +0.891, `sbr_opener_era_2020_2021` +3.295) pool
naively — this script's own `pooled_effect` call, measured above in §1's
`sbr_opener` table — to +0.809 [-0.989, +2.606], matching
`docs/pool_edge_plan.md`'s own quoted figure to three decimal places. But
the entry actually on record, `sbr_opener_pooled_2011_2021`, was **measured
directly on the 2011-2021 union window** (2,832 games) rather than
assembled by algebra on the three already-visible point estimates, and it
reads +0.928 [-0.873, +2.754], P+ 0.8406 — a different number from the
naive algebraic pool, because it is a real predeclared look at the union
population rather than a combination of results whose signs were already
known.

**This is the model for confirming any of the four groups above**: when one
is judged worth a look, the right move is ONE predeclared confirmatory
screen measured directly on the union of both eras' seasons (e.g.
`body_clock_west_road_early` on 2009-2025 as a single population), never a
combination of the two already-recorded point estimates after their signs
are visible — exactly the "family declared before signs are seen" rule
this project treats as binding, and exactly what `sbr_opener_pooled_2011_2021`
already did correctly.
