# SIM-23

## Goal
Clock-truncated drives: end-of-half and end-of-game drives that stop by expiry or kneel, separated from drive length and scoring rate (ROADMAP SIM-23).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 found leading drives 3.1 s short and scoring low; real leading drives end by kneel and expiry, which cut length and scoring. Measured mechanism: a truncation share by clock and sign of score difference, drawn from the real pool. Data: play-by-play drives and sim trace (local).

## Tried
Novelty check only. Nearest: mod25e-e103 (late clock), mod25e-e104 (half end), mod25e-e107 (q4 drive), SIM-18, SIM-21. Differs: separates truncation from the SIM-21 gap in the resampler.

## Next
First unit for one agent: Classify real drives as truncated and compute score rate and seconds by sign of score difference with and without them, sim versus real.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03 5:25 PM, agent) artifacts/sim23/truncated_drives.json, scripts/sim23_truncated_drives.py (ruff clean)
Looks: 12 sign x time cells x 3 versions + 3 pooled + 9 gap bootstraps (1000 game resamples, real and sim resampled independently). Data: real 2009-2025 Q1-Q4 drives (99,440), sim tl_s11 (384,900). Truncated = drive is last of its half (sim10 he==1 definition, 220-262/1424) = trunc_sim; strict = result End of half or contains a kneel. Real share trail/lead/tied: sim-def 8.1/12.4/3.9%, strict 5.5/10.7/2.2% (kneel-containing 1.7/8.9/1.3%). Sim share 8.3/11.2/3.3%: share is matched.
Real score rate all / excl trunc / in trunc: trail .342/.355/.194, lead .359/.395/.103, tied .374/.376/.328. Sim: trail .304/.328/.039, lead .329/.366/.029, tied .354/.364/.068.
Sim treatment: pool keeps real last-of-half drives with full ev list; draw() reweights only by secs ordering (565-575,1245 area); play-through cuts events past query secs (1424,1432). Scoring inside truncated drives is 4% sim vs 19% real (trail), 3% vs 10% (lead), 7% vs 33% (tied).
Gap (sim minus real, standardised to real cell mix; includes tied, so differs from SIM-21 -.019/-.015/+.001): trail -.040, lead -.033, tied -.022 pooled all drives; excluding last-of-half drives: lead -.0275, tied -.0119. Share of gap from truncated drives (see json frac_explained): trail ~37% [33,43], lead ~27% [23,33], tied ~48% [35,76]. Remaining gap persists on non-truncated drives (prob negative ~1.0).
Caveat: this standardisation gives larger gaps than SIM-21 unit 1; cell mix (secs bins) differs. Inferred: truncation explains a third, not all; unit 2's double count (tilt on untruncated outcomes) is the entry point.
Next: repeat gap on non-truncated drives by deficit bin; test sim last-of-half draws whose pool secs exceeds query secs.
Recorded in ROADMAP 2026-11-11
