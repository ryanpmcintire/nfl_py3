# Off-site backup and second capture host

Set up 2026-09-10. A friend's Linux server (WireGuard peer `10.66.0.1`,
SSH alias `backup-server`, 100 GiB ZFS quota at `/data`, server-side
daily snapshots kept 14 days) is the second copy of this project's data
and, once ported, the second host for the capture schedule.

## What lives where on this machine

| Path | Purpose |
|---|---|
| `~/.wireguard/friend-wg0.conf` | WireGuard tunnel config (owner-only ACL). Import it into the WireGuard app; the app stores its own encrypted copy. |
| `~/.ssh/backup-server_ed25519` | Bootstrap SSH key (owner-only ACL). Rotate it after the first login, see below. |
| `~/.ssh/config` | `Host backup-server` block. |
| `~/.ssh/known_hosts` | Pinned server host key. |
| `~/.config/restic/password` | Repository passphrase (owner-only ACL). Copy it into a password manager: without it every backup is unrecoverable. |
| `scripts/offsite_backup.py` | The backup. `--status` reads; the bare command backs up, prunes, checks, and logs to `data/offsite_backup_log.txt`. |
| `scripts/capture_scheduler.py` job `backup_offsite` | Sunday 22:30 ET, 30 minutes after the E: mirror. |

None of the secrets are in the repository. `git status` must never list them.

## Bring-up order

1. Install WireGuard for Windows (`winget install WireGuard.WireGuard`,
   needs an admin prompt). Import `~/.wireguard/friend-wg0.conf`, activate
   it, and confirm a handshake within about 30 seconds. No handshake after
   a minute means the server owner has not finished the router port forward;
   nothing on this side is wrong.
2. `ssh backup-server 'df -h /data'` must log in with no password prompt and
   show a 100G filesystem.
3. Rotate the bootstrap key, which travelled through a chat app:
   `ssh-keygen -t ed25519 -f ~/.ssh/backup-server_local -N ""`, append the
   new public key to the server's `~/.ssh/authorized_keys`, point the
   `IdentityFile` line at the new key, confirm login, then delete the
   `friend-backup-bootstrap` line from `authorized_keys` and the old key file.
4. Seed the repository by hand, not through the scheduler: the first backup
   moves about 8.6 GB (measured 2026-09-10: `data` 7.7G, `artifacts` 859M)
   and will outlast the scheduler's 1800 s job timeout on a residential
   uplink. `python scripts/offsite_backup.py --timeout 14400 --no-check`.
5. `python scripts/capture_scheduler.py --run-job backup_offsite` once, so
   the job's first scheduled window is not its maiden run.
6. Restore one file to prove the passphrase and the repository agree:
   `restic restore latest --target <tmp> --include <one file>`.

## Retention and quota

`--keep-daily 7 --keep-weekly 4 --keep-monthly 6`, then `prune`, then
`check`, every run. The log line records `repository_bytes` and appends
`QUOTA-WARN` above 80 GiB. restic deduplicates, so weekly snapshots of a
slowly growing tree cost roughly the week's new captures, not a full copy.

## Second capture host: design

Owner's requirement (2026-09-10): both machines should be able to run the
capture schedule. If this machine is off the server fills the window; if the
server is off this machine does; if both are on, the two results are
reconciled.

### What already makes this tractable

- Every capture is a directory named by its UTC timestamp
  (`YYYYMMDDTHHMMSSZ`) under a per-source folder in `data/`. Two hosts
  capturing the same window write two directories that never collide, so
  merging is a union of directories, never a file overwrite.
- The scheduler's dedupe guard reads the newest snapshot age from disk, so a
  host that has just pulled the other host's capture for a window sees it as
  already captured and skips its own.
- The schedule is version-controlled in `SCHEDULE`, so both hosts run the
  same jobs from the same commit.

### What has to change before the server can run it

- `capture_scheduler.py` hard-codes `.tools/uv.exe` and ten jobs spawn
  `powershell.exe -File scripts/*.ps1`; `odds_capture.ps1` hard-codes
  `F:\Repos\nfl_py3`. The port is: resolve `uv` per platform, spawn `pwsh`
  on Linux, and derive the repository path from the script location.
- The Odds API key, the CFBD key and the Sportradar key would have to live
  on the server for its captures to work. The server owner has root there
  and can read anything the `friend` account holds. Decide which captures
  the server runs: everything, or only the key-free public sources
  (nflverse injuries, player snapshots, injury news, lineups, inactives,
  referee assignments, PFR transactions) with the paid Odds API captures
  staying on this machine.
- Card-writing jobs (`weekly_lock`, every `refresh_*`, `settle_*`,
  `publish`) stay single-writer on this machine. The server captures; it
  never forms or publishes a card.

### Reconciliation rule

- A `sync_captures` job on each host every two hours: list the other host's
  snapshot directories over SSH, copy the ones missing locally, and never
  overwrite an existing directory. Both copies of a doubly-captured window
  stay on disk.
- Each snapshot carries the capturing host in a `capture_host` field in its
  manifest. When a consumer needs one snapshot for a window, it takes this
  machine's capture when present and the server's otherwise. The choice is
  a stated rule, not the timestamp, so the same window resolves the same
  way on both hosts.
- A window that both hosts captured is logged with the size of the
  difference (rows added, rows changed) so a systematic disagreement between
  the two vantage points surfaces instead of being averaged away.

### Status, 2026-09-10 evening (measured)

Backup side is complete: tunnel up (both WireGuard services `Automatic`, so
it survives reboot), bootstrap key rotated out and deleted, repository
seeded (snapshot `56efac8f`, 2.48 GB on the server for 8.6 GB of source,
9 minutes), `--run-job backup_offsite` MANUAL-RUN OK, prune and check
passed, `--restore-test registry/weak_signals.json` hash-matched.

Server facts: Debian 12 LXC on Proxmox, 1 CPU, 512 MiB RAM + 256 MiB swap,
8 GB root (7.4 GB free), `/data` 100 GB ZFS, Python 3.11 system, no git,
no cron, no curl, `wget`/`tar`/`rsync`/`restic`/`systemctl` present,
`loginctl` Linger=no. Outbound reachability from the server: the Odds API,
nflverse on GitHub, nfl.com and ESPN answer; pro-football-reference
returns 403 to `wget` (the PFR capture may need the same browser
user-agent it uses here, unmeasured).

Installed on the server so far: `uv` 0.12.13 at `/data/nfl_py3/.tools/uv`
and CPython 3.12.14 via `uv python install`.

Code shipped in the repository for the second host:

- `capture_scheduler.py`: `UV` resolves to `.tools/uv` off Windows; the ten
  PowerShell wrapper jobs run their underlying `nfl-ats odds-ingest` /
  `public_betting_live_capture.py` argv directly on POSIX;
  `NFL_ATS_SCHEDULER_ROLE=capture` disables every job that is not a
  capture (lineups, weekly_lock, refresh_*, settle_*, backup_*,
  splash_board, verify_full, sync_captures) so the server never forms or
  publishes a card; 56 `sync_captures_<day>_<hhmm>` jobs every three hours
  on the primary.
- `scripts/sync_captures.py`: lists the server's snapshot directories over
  SSH for the twelve capture roots, pulls the ones with no local capture
  inside the job's dedupe window (tar over ssh), writes a `capture_host`
  marker in each pulled directory, verifies the pulled listing against
  the server's, logs `DUP` lines with a file-level size diff for windows
  both hosts captured, and deletes the server copy once accounted for.
  `--dry` lists only (measured: `OK pulled=0 duplicates=0 roots=12`
  against the empty server).
- `scripts/start_capture_scheduler.sh` (nohup, sources
  `~/.config/nfl_py3/env`) and `deploy/nfl-ats-capture.service` (systemd
  user unit, needs `loginctl enable-linger friend` from the server owner to
  start at boot).

Later the same evening (measured): the owner copied the code and wrote
the env file; `uv sync --no-dev` built a 304 MB venv (after fetching the
`LICENSE` the build needs from the public GitHub repo); the systemd user
unit is installed, enabled and active (`systemctl --user enable` started it
before the venv existed, so its first catch-up runs failed on
`Failed to spawn nfl-ats`; restarted afterwards). Under the capture role
the server's schedule shows 81 enabled / 96 disabled jobs.
`--run-job nflverse_injuries_thu` on the server: OK, wrote a snapshot.
`--run-job odds_thu_tnf`: needed two files a fresh copy lacks,
`config/source_policies.json` (untracked in git, copied by hand) and
`data/processed/game_features.parquet` (game-id lookup for quotes; the
sync job now pushes the local copy to the server whenever its size
changes). Memory during the nflverse capture stayed under 200 MB used.

Remaining bring-up on the server, in order (the sandbox refused to copy
code or keys to the server, so the owner runs the first two by hand):

1. Copy the code:
   `scp -r pyproject.toml uv.lock README.md src scripts registry deploy backup-server:/data/nfl_py3/`
2. Write the keys (values from the user environment) to
   `~/.config/nfl_py3/env` on the server, mode 600:
   `THE_ODDS_API_KEY=...`, `CFBD_API_KEY=...`, `NFL_ATS_SCHEDULER_ROLE=capture`.
3. `ssh backup-server 'cd /data/nfl_py3 && ./.tools/uv sync --no-dev'`
   (measure peak memory of one capture afterwards; 512 MiB is tight for
   pandas + scikit-learn imports).
4. `ssh backup-server 'cd /data/nfl_py3 && NFL_ATS_SCHEDULER_ROLE=capture ./.tools/uv run --no-sync python scripts/capture_scheduler.py --status'`
   then `--run-job nflverse_injuries_thu` (key-free) and
   `--run-job odds_thu_tnf` (keyed) as the maiden runs.
5. `ssh backup-server 'sh /data/nfl_py3/scripts/start_capture_scheduler.sh'`;
   ask the server owner for `loginctl enable-linger friend`, then
   `systemctl --user enable --now` the unit copied to
   `~/.config/systemd/user/nfl-ats-capture.service`.
6. Here: `--run-job sync_captures_thu_1530` once a server capture exists.

Reconciliation as built: both hosts capture every window they are awake
for; this machine's capture is the one served whenever it exists; the
server's fills gaps; a doubly-captured window is logged with the number of
files whose sizes differ, never merged or averaged.
