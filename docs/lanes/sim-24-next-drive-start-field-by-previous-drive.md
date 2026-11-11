# SIM-24

## Goal
Next-drive start field by previous drive ending and score sign (ROADMAP SIM-24).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 start-field counterfactual: simulated trailing drives start 0.71 yd nearer goal and leading 0.69 yd farther than real, which explains 18% of the leading score gap. Measured mechanism: start yard line conditioned on how the previous drive ended and who scored. Data: drive pool (local).

## Tried
Novelty check only. Nearest: mod25e-e108 (drive start, generator), SIM-21. Differs: inside the SIM-10 resampler kernel.

## Next
First unit for one agent: Tabulate real start yards by previous ending, kicking team and sign of score difference and compare to the kernel candidate pool's start distribution.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03 5:30 PM, agent) artifacts/sim24/start_field.json, scripts/sim24_start_field.py (ruff clean)
Looks: real 10 prev types x 3 signs; matched 3 classes (half start, after score, after non-score) x 3 signs x 4 time bins (edges 0/509/963/1441/1800 s, real quartiles); 3 slopes; 3 bootstrap contributions (500 game resamples). Real restricted to 2009-2017 to match sim seasons (pooling to 2025 made the kickoff start look 1.2 yd off: kick-start mean by season 73-77, 2011-15 touchback rule = 77). Non-truncated drives only. Sim cannot log prior ending type; classes only.
Real start yardline_100 by prev type (2009-17): TD 75.8, FG 76.4, punt 73.7, missed FG 65.1, downs 65.6, INT 50.9, fumble 47.7, safety 62.5. Class x sign real v sim: nonscore trail 70.0 v 69.3, tied 69.6 v 69.2, lead 67.0 v 67.5; after score trail 76.6 v 76.4, tied 76.2 v 76.0, lead 74.0 v 74.4; half start 76.0-76.6 v 75.9-76.5. Matched within ~0.5 yd.
Mechanism (read): sim10_drive.py:1520-1535 carries end yl via kernel drive j: ny = nyl[j] + d (or slope*d / -d) when not scored; after score draws from koseason pool by season (1529-1531); st_shift added (1534). Ending type enters only through the kernel draw j, not logged.
Start-field gap by sign (sim minus real, real cell mix; wtd): trail -0.45 yd [-0.77,-0.15], lead +0.44 [0.05,0.90], tied -0.30 [-0.72,0.09]. Real score slope -0.0078/-0.0084/-0.0087 per yd. Implied gap explained: lead -0.0037 [-0.0076,-0.0004] = 13% of the -0.0275 untruncated lead gap (SIM-21 said 18% with all-era real); trail +0.0035 (sim too short a field, widens trail gap: not an explanation); tied +0.0026 [-0.0008,0.0063]. P(positive): 0.01/1.0/0.92.
Conclusion: start field is not the main residual. Largest cell gap: lead nonscore +0.5 and lead after score +0.35 yd. Candidate mechanism: pooled kernel next-start offset (stslope/st_shift) is not conditioned on lead sign. Not a new look to close; unresolved_below_power, not recorded (rates not admissible units).
Next: split lead nonscore cell by punt vs turnover using kernel j ending type (needs j logged) ; or look at the sim score rate within matched start field (untruncated) to find the remaining ~87% of the lead gap.
Recorded in ROADMAP 2026-11-11
