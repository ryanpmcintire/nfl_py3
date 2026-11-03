# LEAD-101

## Goal
News-delta read: an LLM sees only items timestamped after the Tuesday freeze and forecasts the direction and size of the move to the deadline line (ROADMAP LEAD-101).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: LEAD-99 asks for a cover probability from everything, which mostly restates the line. The edge, if any, is in what arrived after the number froze (designations, beat posts, inactives, forecast revisions). Target is line move toward the side, the yardstick ENG-47 shows resolves effects 7-18x smaller than accuracy; it enters the one probability as a fitted term. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 (all information, cover target), SKY-01 (news-to-availability, moonshot), MKT-08 (news-triggered refresh, a trigger not a forecast), MKT-20 (move as target, no news input). Grep of ROADMAP.md, docs/lanes incl. done, docs/*.md and registry/weak_signals for a news-only read forecasting the move found none. Differs: information class and target.

## Next
First unit: Write the delta-packet builder from the stored Week 9 packet timestamps and freeze its prompt by hash; run the Thursday 11-05 and Sunday 11-08 views and write them to a forward log before kickoff; score move agreement and size after the deadline lines are captured.

## Open
Changes: Post-opener information inside the one probability. Data: LEAD-99 packet filtered to post-freeze timestamps; deadline-time pool and market lines. Delta-only prompt; target is signed move in points toward the pool-line side. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03 15:55 ET, 20:55 UTC)
Built scripts/lead101_delta_read.py (reuses lead99_llm_read build_packet, extract_json, sha256_bytes, MODEL; prompt scripts/lead101_prompt.md; no web tools, free capture sources only). Enrolled artifacts/lead101/frozen.json: prompt sha256 4938b8470a756001767184f725685f9947662637c386f701e98b31bb452e98e9, model claude-opus-5-5. Freeze = run_at_utc of the stored Week 9 Tuesday packet = 2026-11-03T18:30:29Z.
Dry run 20:55:41Z: items after freeze per game = 1 group each, all of kind market (later quotes only; no new injuries, headlines, inactives, beat posts or weather yet). One real read made (15 rows, moves -1.5..+0.5), artifacts/lead101/runs/2026-week-09-dry-20261103T205541Z.json. Caveat: a market-quote-only packet lets the model read the move already under way; score view-by-view and report a market-excluded variant before any claim. Look family: 1 prompt, 2 forward views per week, target = signed home-spread move, frozen to deadline line.
## Run commands
`F:\Repos\nfl_py3\.venv\Scripts\python.exe` is not used by the scheduler; command is: uv run --no-sync python scripts/lead101_delta_read.py run --view thursday (Thu 2026-11-05 ~18:30 ET, before the 20:15 ET kickoff) and --view sunday (Sun 2026-11-08 ~11:30 ET, before 12:00 ET). Add --week N after Week 9.
## Proposed scheduler entry (orchestrator adds)
    *(
        Job(
            f"lead101_delta_read_{view}_{day}_{at.replace(':', '')}",
            day,
            at,
            30,
            [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "lead101_delta_read.py"), "run", "--view", view],
            True,
            "LEAD-101 forward log: a no-web LLM read of only the news captured after the Tuesday freeze, forecasting each game's home-spread move to the deadline line, frozen prompt and model, recorded before the deadline.",
            added_on="2026-11-03",
        )
        for view, day, at in (("thursday", "thu", "18:30"), ("sunday", "sun", "11:30"))
    ),
Next: unit 2 scores move agreement and size vs captured deadline lines after Week 9; record via weak-signals record.
- 2026-11-03 15:57 ET (root): scheduler jobs lead101_delta_read_thursday_thu_1830 and lead101_delta_read_sunday_sun_1130 added (capture_scheduler.py, ruff clean; live after the next daemon restart). `--run-job ... --dry` EXECUTES the job: it wrote a Tuesday-stamped thursday record, relabelled to 2026-week-09-dry-20261103T205652Z.json (view dry). Unit 2 scoring must also grade the persistence baseline (move so far, from the quotes in the packet, as the forecast), since the packet includes later market quotes; the read only counts if it beats that baseline.
