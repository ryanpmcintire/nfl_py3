# UI-22 Why the line moved

## Goal
Explain line movement to readers in plain English where news supports it. ROADMAP row UI-22.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Readers see that a number moved but not why. Pair each game's Tuesday-to-lock move with the captured injury and transaction news in that window and print a plain-English reason where one exists. Data: ingest_injury_news and ingest_transaction_news captures.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: SKY-01 news-to-availability model (research term), news-trigger-refresh lane (refresh trigger), UI-20 Books column. No page explains a move.

## Next
Build the window join and decide the wording for no-news moves. First unit: First unit: for Week 10 games with a move of at least a point, list candidate news items in the window and render one sentence in the This Week table.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
A reason implies cause; the sentence must say reported, not proven.
