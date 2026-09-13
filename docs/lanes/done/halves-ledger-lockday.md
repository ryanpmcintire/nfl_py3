# halves-ledger-lockday

## Goal

LEAD-61 follow-up: `lockday_verify.py` reads the second-half line challenger
ledger (`half_line_2h_underdog_refresh_v1`) like the other refresh ledgers.
Done when `scripts/lockday_verify.py --season 2026 --week 1` lists it.

## State

Done, uncommitted. Opencode lane A (`big-pickle`) added the entry at
`scripts/lockday_verify.py:229-239`. Verified by the coordinator 2026-09-12
19:56 ET (measured): `lockday_verify.py --season 2026 --week 1` prints
`ok half_line_2h_underdog_refresh_v1 126 rows`, no traceback. Committed with the 2026-09-12 evening lanes.

## Tried

- Lane A's staleness check: `crew_tilt_refresh_v1` and `inactives_refresh_v1`
  carry `"wired": False` but the registry command overrides the flag at
  runtime (`lockday_verify.py:405-410`), so both print `ok ... 250 rows`.
  The flag is dead text; owner call whether to flip it (not done).

## Next

(none; finished)

## Open

(none)
