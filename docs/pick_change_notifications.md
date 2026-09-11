# Pick-change notifications

Added 2026-09-10 after the owner asked to be told when a pick flips right
before kickoff while away from the computer.

## What fires

The capture scheduler daemon (`scripts/capture_scheduler.py`) now keeps the
full stdout of every job it runs. After any `refresh_*` job:

- **Pick change.** `nfl-ats refresh-picks` reports a `changed_picks` list
  (game, previous side, new side, new home cover probability, eligibility;
  added to `refresh_summary` in `nfl_ats.pick_refresh`). If any eligible
  pick changed, the daemon logs a `PICK-CHANGE` line and sends an urgent
  notification listing each change.
- **Failed pre-kickoff refresh.** If a `refresh_*_inactives_*` job fails,
  the daemon sends a high-priority notification saying the card was not
  refreshed after inactives, because a silent failure there is exactly the
  case the owner would want to know about.

Nothing fires for routine refreshes with no change, and nothing fires from
the capture-role host: it runs no refresh jobs.

## Transport

[ntfy](https://ntfy.sh): an HTTP POST to `https://ntfy.sh/<topic>`, no
account. The topic is a random name held in the user environment variable
`NFL_ATS_NTFY_TOPIC` (set 2026-09-10; `NFL_ATS_NTFY_URL` overrides the server
for a self-hosted ntfy). The daemon reads it at start, so a topic change needs
a daemon restart. With the variable unset the daemon logs the change and
sends nothing. Anyone who knows the topic name can read the notifications,
so the name is random and lives only in the environment, never in the
repository.

To receive them: install the ntfy app (iOS or Android), subscribe to the
topic name, and set the topic's notification priority to "instant" so a
20:05 change before a 20:15 kickoff arrives on time. A test message was sent
to the topic on 2026-09-10.

## First delivery (measured 2026-09-10)

The 19:15 ET Thursday refresh reported two changed picks and the phone
message arrived at 19:16 with priority urgent: ATL at PIT and WAS at PHI,
both moved to the home side by the served late-week follow rule after the
median book moved 1.0 point toward the home team since the Tuesday line.
The first message printed HOME/AWAY and the raw home-cover probability,
which reads wrong when a follow rule overrides the model (WAS at PHI showed
"HOME, 36.8%"). The text now names the side with its spread, the probability
of the side actually picked, and the reason when a rule overrode the model:
`WAS at PHI: was WAS +5.5, now PHI -5.5 (37%; line moved 1.0 toward PHI,
follow rule)`.

Note on the card: intermediate refresh passes only record to the
pick-revision ledger; the Sunday-morning pass is the one that republishes
`CURRENT_PREDICTIONS.md` (`--publish-card`). Between Tuesday and Sunday
morning, the notification and the ledger are the record of a changed pick,
not the card.

## Timing

The pre-kickoff refreshes already sit inside each game's window: inactives
land about 90 minutes before kickoff, the inactives capture runs at the
next scheduled slot, and its refresh runs 25 minutes after that (Thursday
primetime: capture 18:50, refresh 19:15, kickoff 20:15). A change from that
refresh reaches the phone roughly an hour before kickoff.
