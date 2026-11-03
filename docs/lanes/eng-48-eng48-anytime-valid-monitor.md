# ENG-48 Anytime-valid evidence monitor for every forward log

## Goal
Anytime-valid evidence monitor for every forward log: a weekly mixture-martingale update on paired log loss against the served model (ROADMAP ENG-48).

## State
Added 2026-11-03, unmeasured. Forward logs are read only once at Week 18; a sequential monitor gives an evidence value after every slate and a pre-registered way to read it, without peeking inflation.

## Tried
Nothing yet. Novelty: docs/lanes/forward-log-audit-2026-11-03.md (coverage only) and the mod23-unit15 lane. Grep for e-process, always-valid and e-value found no sequential-evidence monitor.

## Next
First unit: Implement the monitor as a read-only script, replay it on logged Weeks 4-9, and compare its interval with the plain bootstrap on the same games.

## Open
Data: artifacts/mod24_forward, mod25_forward and armb_forward runs plus the MOD-35, SKY-03, SKY-04 and ENV-02 ledgers. Zero crossing closes nothing; record unresolved results with a closing ground.
