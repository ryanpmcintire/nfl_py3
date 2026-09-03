# Absence dependence: joint non-participation by unit (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any joint-absence number was
computed. Sections 0–7 are frozen; section 8 (Results) is appended after
the run and nothing above it is edited afterwards.

**Owning work package:** PER-10 (owner directive 2026-09-03: comprehensive,
never the independence shortcut) and SIM-03. Files: this document,
`scripts/absence_dependence_screen.py`,
`tests/test_absence_dependence_screen.py`,
`artifacts/absence_dependence/`.

**Parent:** `docs/injury_scenario_mixture.md` (the kernel needs a joint
scenario producer; multiplying marginals asserts an independence nobody
established). This slice measures how wrong independence is, per unit. It
does not build the producer, score ATS, spend a window, or record a
registry verdict (descriptive rates, not effects).

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero".

No verdict is available in this slice regardless of outcome: joint rates are
inputs to a future producer design, not candidate-vs-baseline effects.

---

## 1. What this slice asks (frozen)

For each roster unit (OFF_OL, OFF_SKILL, DEF_FRONT, DEF_SECONDARY — the
frozen `scripts/unit_apm_screen.py` mapping), how much more often do
multi-player absences co-occur than independence implies? The excess is the
dependence mass the producer must represent.

## 2. Population and absence definition (frozen)

Snap-count snapshot `20260817T184901Z`, weekly rosters same snapshot,
seasons 2016–2025 REG. A player-game is an **established contributor**
iff the player logged offensive/defensive snaps (side-appropriate) in at
least one of the trailing 4 team games. Absence = zero side snaps in the
game. (Disclosed: this is non-participation, not a medical diagnosis —
healthy scratches, suspensions, and personal leaves are inside the number
alongside injuries. The producer needs P(lineup), and non-participation IS
the lineup event.)

**Amendment 2026-09-03 (before the valid run; a first run exposed two
false-absence sources and was discarded unrecorded):** a team-game enters
the frame ONLY if (a) the team actually plays that week (nflverse schedule
join — bye weeks have roster rows but no game and must never read as mass
absences) and (b) the team logged ≥100 side snaps in the game (data-presence
gate — the discarded run contained team-games with zero snaps for every
unit, i.e. missing data, which read as simultaneous full sits in all four
units). Both gates fail closed.

## 3. Statistic (frozen; amended before any run — see note)

Per unit, over all team-games with ≥2 established contributors, report
observed shares P(0), P(1), P(2+ absences) descriptively. The PRIMARY
estimand is the **full-sit excess**: observed P(every contributor absent)
divided by the independence-implied product of the pooled marginal,
`Πq / Π` averaged per game as `mean over games of Π_i q` with q the pooled
marginal — with a team-season block bootstrap CI (2,000 samples, seed
20260903).

**Amendment note (2026-09-03, before the first run, from analysis not data):**
as first written this section defined excess on P(2+), which points the WRONG
way under strong coupling: perfectly synchronized full-unit sits produce
FEWER multi-absence games than independent scattering at the same marginal
(all the absences bunch into a few games instead of spreading), so a P(2+)
ratio reads below 1.0 for the most coupled world and above 1.0 for
intermediate ones. The full-sit tail is monotone in coupling strength and is
exactly the whole-lineup state the mixture kernel scores, so it is primary;
P(2+) stays as description only.

## 4. Controls (frozen, diagnostic)

Position-group permutation within team-season (shuffle absence labels
across players holding each player's own total fixed): the observed
full-sit rate must clear the permuted null center, else the tail is an
arithmetic artifact of heterogeneous marginals. Diagnostic only.

## 5. Leakage (frozen)

None applicable: inputs are completed seasons, no pregame application, no
model join. Tests pin the trailing-window definition and the excess-ratio
algebra on synthetic frames.

## 6. Test contract (release-blocking)

`tests/test_absence_dependence_screen.py` covers, without network access:
established-contributor rule, excess-ratio math on a hand-computed frame
(independent case ≈ 1.0, perfectly-coupled case ≫ 1), bootstrap
determinism, and empty-population fail-closed.

## 7. What this slice may therefore claim

At most: per-unit excess joint-absence mass with uncertainty — the
parameter the producer needs and the quantified answer to "why not
independence". It may not claim an injury mechanism, a scenario
probability, or any ATS consequence.

## 8. Results (added after the run, 2026-09-03)

Measured by `scripts/absence_dependence_screen.py` in one valid run
(artifact `artifacts/absence_dependence/20260903T213341Z/results.json`;
an earlier run without the bye/data gates was discarded unrecorded per
the §2 amendment):

- Marginal non-participation q runs 0.20–0.26 per unit (depth churn among
  established contributors, as disclosed — not a medical rate).
- **Full-sit rate is exactly 0.0 in all four units** (0 whole-unit sits in
  ~4,964 team-games each): with 6–15 established contributors per
  team-game, a simultaneous all-out never occurs in ten seasons.
- The primary estimand therefore degenerates (0/0-implied → excess 0.0):
  full-unit sits cannot discriminate coupling from independence because
  neither model produces them at these unit sizes.

What this implies: the coupling question was aimed at the wrong scale.
Whole units never sit together, but the producer cares about PAIRS and
TRIPLES (the QB-WR1-LT joint states the kernel scores) — so the follow-up
is pairwise co-absence excess, predeclared separately in
`docs/absence_pairwise_dependence.md` with this lesson stated as its
motivation. No registry entry was written (a degenerate estimand is not an
effect, and inventing a verdict for it would corrupt the pool).
