# Pool observables capture (LEAD-52)

**Built 2026-09-04; first real rows land in lock week.** The pool's field
size and prize structure decide every contest-utility question (POL-05),
and each game's pick distribution unlocks at its own kickoff
(`docs/pool_format_levers.md` §8). There is no API (login-walled), so
capture is manual entry into immutable snapshots.

## Command

```powershell
# Once per week, at entry time (Tuesday lock week):
.\.tools\uv.exe run nfl-ats pool-observables --season 2026 --week 1 `
  --entries 100 --paid-places 15 --prize-notes "top 15 paid" `
  --observer "who read the page"

# Per game, any time after its kickoff unlocks the distribution:
.\.tools\uv.exe run nfl-ats pool-observables --season 2026 --week 1 `
  --distribution 2026_01_MIA_LV=0.38,0.62 `
  --unlocked-at "2026-09-13T13:00:00-04:00" `
  --observer "who read the page"
```

Shares are home,away fractions; whole-percent rounding is accepted
within 0.02 of 1.0. A distribution row whose read instant precedes the
unlock instant is refused — pre-kickoff fields are unobservable by
POL-04's closure, so such a row is a contradiction, not data.

## Storage

`data/pool_observables/<UTC-stamp>/observations.json` plus a SHA-256
`manifest.json`, one directory per invocation (write-once; a colliding
stamp is refused rather than merged). `latest_snapshots` inventories
them read-only. Nothing here touches a ledger, a forecast, or a model.

## What this unblocks, and when

- Week 1 entry: field size + prize structure close POL-05's two free
  parameters the same day.
- Sunday unlocks: per-game distributions accumulate next to the MKT-12
  capture log; correlating them against Action Network bet% measures
  whether bet% predicts field lean (the `public_lean` fit).
