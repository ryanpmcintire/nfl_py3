# LEAD-73: pre-2023 dated move source inventory

## Goal

Execute ROADMAP.md LEAD-73 unit 1 within the assigned script/report/lane scope.

## State

**Measured:** unit 1 complete; 757 cached Odds API leader-pair games (2020:235, 2021:253, 2022:269); 217 VegasInsider pair games. Three inventory probes; zero outcome looks. Full evidence: `docs/lead73_unit1.md`. Protocol saved before inventory/outcomes; **read**, ROADMAP.md:856 declares:

> **Added 2026-09-29 (unmeasured).** Mechanism: every move-dependent term (MKT-16/18/19, LEAD-67, LEAD-69) is fitted on 799 games because dated intraweek snapshots exist only from 2023; recovering dated Wednesday to Sunday lines for 2009-2022 would roughly double to triple the population at unchanged look count, the only route to resolving a 1-2 point effect. Unit 1 (source inventory, one probe per candidate, `config/source_policies.json` consulted first): candidate dated-line archives (Wayback captures of public odds-history pages, existing VegasInsider 2005-2016 rows for open-to-close only, the Odds API historical endpoint blocked by the cancelled paid plan), stating for each the earliest capture instant relative to the Tuesday-noon freeze and the pick deadline; a source that cannot prove its timestamp precedes the deadline is inadmissible under the point-in-time gate. Unit 2 only if one source clears: refit the move terms on the larger population LOSO by season with the predeclared four-term protocol and report in-sample beside out-of-sample. Checked, not duplicated: LEAD-59 (officials archive, different data), SKY-04 (book leadership, 2023-2025), `free-odds-sources` lane (current-week lines, not history). About 15 tool calls for unit 1.

Unit 1 declaration: population = local candidate dated NFL odds for 2009-2022; target = evidence of an intraweek line captured after Tuesday noon and before the applicable pick deadline; terms/folds = none in this inventory; metric = source coverage and timestamp admissibility; looks = three candidate inventory probes, zero outcome/statistical looks. Consult source policy before each candidate probe. No network acquisition, paid endpoint call, fit, score, or outcome read. Inspect metadata and line/timestamp fields only. Missing timestamps or deadline evidence never clear the gate. Unit 2 remains conditional on a qualifying source and the existing four-term declaration; do not invent missing terms or deadline conventions. An interval crossing zero never closes a signal; one fitted calibrated probability selects the side.

## Tried

Read the row, lane context, research rules, and source policies. **Read:** `scripts/sunday_market_probability_eval.py:22-68` defines the current research deadline as the earlier of kickoff and Sunday 12:45 ET, with Monday history and Wednesday-or-later moves. Use those bounds, Tuesday noon ET for freeze, regular-season schedule matches, and two distinct same-book captures for a possible move. This fixes timestamp accounting before the inventory run. VegasInsider signed home-line orientation remains a separate requirement. No outcome columns will be loaded.

**Measured:** `.tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit1.py` exited 0. Ran twice: the first exposed an overly narrow hostname check; accepting valid `:80` URLs fixed the count without changing the protocol. Final log: `%TEMP%/lead73-unit1-verified.log`. No network, outcomes, registry writes, or tests added.

## Next

**Inferred:** local timestamp gate clears. Next unit: verify model/composition coverage and signed move reconstruction for the 757 cached games, then declare and run the four-term LOSO refit with paired baselines. The assigned unit-1 script does not fit.

## Open

In/out-of-sample, gap, coefficients, decisive record, intervals, and probability_positive are not estimated in an inventory. VegasInsider needs signed home-line/book verification. No signal closed or promoted; population growth and power remain unmeasured.

## Record commands

None: inventory supplies no signal estimate to record. Do not invent an effect or probability_positive. Orchestrator owns later record commands.
