# Off-site backup and second capture host

Set up 2026-09-10. A friend's Linux server (WireGuard peer `10.66.0.1`,
SSH alias `backup-server`, 100 GiB ZFS quota at `/data`, server-side
daily snapshots kept 14 days) is the second copy of this project's data
and the second host for the capture schedule. The September 28 verification
below states its current deployment and remaining limits.

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

`config/source_policies.json` is canonical, non-secret configuration and is
tracked. A fresh checkout therefore includes the fail-closed policy registry
required before any raw capture; API keys remain in the environment files
described below.

## September 28 verification and repairs

**Measured:** the primary scheduler's `--once` check exited 0. Its old summary
reported 230 OK jobs, but the status repair below establishes that this residual
count included failures and inactive jobs. The secondary was running an older
fallback daemon. A code-only archive from checkpoint `916ba64` deployed
499 allowlisted files after SHA-256 verification, matching dependency-lock and
Python 3.12 import checks. Overwritten files are retained in a rollback archive.
The restarted capture-only daemon reports the deployed scheduler hash. A final
read-only check at 16:24 ET confirmed the same daemon (PID 18320) running.
Manual rehearsals do not fill the historical missed September 22–27 windows.

**Read:** `scripts/capture_scheduler.py` now classifies dated receipts explicitly
and exposes dated and active manual failures in brief status. Manual runs preserve
dated history; only a same-day real manual success can satisfy a prerequisite.
Dry runs cannot replace that receipt. Blank failures preserve prior error detail,
and later completed captures update health. Dated and manual totals can overlap.
**Measured:** replaying the saved secondary state with its capture role shows all
five player `FAIL(137)` receipts; none is counted as a dated completion. The
primary status shows 177 completed dated runs, four failed dated runs, two active
manual failures, one acknowledged miss, and 49 inactive jobs. The four dated
failures are `lineups_tue`, `lineups_sat`, `splash_board_tue`, and
`refresh_last_call_sat_1215` from September 22–26; they remain outstanding.
The local daemon was restarted only after its identity and idle state verified;
PID 28948 reports SHA-256 `5d6933d38c5b37e2c248db830042cd8c17e18d4e9832b865c640a82b372e147a`,
193 enabled jobs, and an unchanged schedule digest. This status repair is not yet
deployed to the secondary. Independent review, 57 focused scheduler tests, the
full 1,645-test suite, Ruff format/lint, and mypy passed. Evidence:
`.tmp/backlog-execution-20260928/scheduler-capture-role-replay.txt`,
`primary-status-fixed.log`, `primary-restart-result.json`, and `status-final-*.log`.

**Measured:** seven previously untried aliases passed on the secondary:
`public_betting_tue`, `public_betting_thu`, the four `odds_private_*` aliases, and
`inactives_sun_early`. Five `player_snapshot_*_0915` jobs exited 137; the host's
cgroup reported 35 OOM kills. No season ranges were shortened to conceal that
failure. **Read:** the repair in `src/nfl_ats/players.py` loads one season at a
time, writes and releases each dataset sequentially, and compacts historical
first-seen keys incrementally. It bypasses nflreadpy caching for the synchronous
fetch and restores the prior cache mode in `finally`; existing caller cache
entries remain untouched. Each snapshot is staged outside the visible timestamp
namespace and renamed only after all datasets and its manifest succeed. **Measured:** fixed-input output values, schemas,
ordering, postseason inclusion, availability bases, and manifest bindings match
the prior path; the first-seen probe fell from 245.9 to 183.1 MiB peak working
set. These local probes do not establish full-history memory use on the server.
Deployment and the real host rerun remain pending explicit approval after
automatic approval review rejected sending the new source to that external host.

**Measured:** encrypted backup status reports four snapshots and 3,203,785,390
stored bytes against the 100 GiB quota. Restoring
`registry/experiments/margin-backtest/20260927T134152Z.json` returned 1,650 bytes
with an identical SHA-256 hash. This verifies that file, not all backup coverage.
**Read:** `scripts/offsite_backup.py` now rejects any missing requested source,
all nonzero backup exits including restic's incomplete-backup exit 3, and failed
or missing repository-size results. An incomplete backup cannot reach retention
pruning or report OK. **Measured:** isolated failure-path probes passed.

**Measured:** the final sync dry run retained five differing observations across
13 capture roots. Local and existing remote feature-table hashes match. The
primary's earlier 15:30 run pulled four gap captures and kept five conflicts;
that run preceded the final removal of all remote snapshot deletion. The final
live scheduler sync is pending explicit approval after automatic approval review
rejected its possible feature-table upload. Local integrity probes and independent
review cover staged transfer, Windows archive paths, empty captures, conflicting
content, and verified feature-table replacement.

**Measured:** `loginctl enable-linger friend` returned `Access denied`. The
fallback daemon remains running. Reboot persistence still requires the server
administrator to enable lingering; no privileged workaround was attempted.
The deployment staging and rollback files are under
`/data/nfl_py3/.deploy/backlog-execution-20260928/`. Local command receipts are
under `.tmp/backlog-execution-20260928/`; raw data and credentials are untracked.

## Original bring-up order (historical)

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

## Second capture host: current repository behavior

**Read, 2026-09-26:** the scheduler selects the platform's uv executable and
runs only capture jobs under `NFL_ATS_SCHEDULER_ROLE=capture`
(`scripts/capture_scheduler.py:23`, `:1598`). The portability changes described
in the original design are implemented. The September 28 verification above
records current measured health; the September 10 account below is historical.

The primary scheduler invokes `scripts/sync_captures.py` every three hours.
Its reconciliation behavior is (**read**, `scripts/sync_captures.py`):

- An identical snapshot directory name is compared using a manifest of relative
  file names and SHA-256 content hashes. Both exact matches and `CONFLICT`
  differences are retained remotely under the collision policy.
- A local snapshot within that source's dedupe window is a `DUP` only when its
  complete manifest exactly matches the remote snapshot. `DUP` and `CONFLICT`
  snapshots are both retained remotely and are not merged into the local snapshot.
- A remote snapshot that fills a local gap is extracted into a staging directory.
  Its content must match stable remote manifests before and after transfer before
  it is installed locally and marked with `capture_host`.
- No snapshot is deleted automatically because the capture producers do not yet
  publish an immutable completion proof. Verified gap fills also remain remotely.
  A changed, empty, unreadable, malformed, or incomplete transfer fails closed.
  Stable file hashes do not establish semantic completeness across all sources.
- `--keep-remote` remains as a compatibility flag; preservation is now the
  default. `--dry` skips copying and feature-table upload, but still queries SSH,
  hashes duplicate candidates, and writes the local reconciliation log.

The default sync retains every remote snapshot: same-name and near-window
observations, content-proven copies, verified gap fills, and every distinct or
unverified observation. Cleanup requires a future immutable completion proof.
Restic backup and remote capture reconciliation remain separate operations.

The feature table is current only when its SHA-256 content hash matches the
remote file (**read**, `scripts/sync_captures.py`). A changed table uploads to a
unique staged path; the local file is rehashed after upload and the staged remote
hash is checked again before atomic replacement. A change or malformed hash
fails the sync job without replacing the current remote table. The table and any
required private environment values must be deployed before all capture jobs can
work. Keep credentials outside tracked repository files.

### Historical deployment record, 2026-09-10

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
  SSH for the twelve capture roots, pulls gap fills through a local staging
  directory, writes a `capture_host` marker, and verifies file names and SHA-256
  hashes against stable remote manifests. All snapshots remain remotely because
  the current producers do not publish an immutable completion proof. Content
  conflicts and invalid or changing snapshots therefore also remain. `--dry`
  lists only (the historical
  empty-server measurement was `OK pulled=0 duplicates=0 roots=12`).
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
`--run-job odds_thu_tnf`: at the time it needed two files that copy lacked.
`config/source_policies.json` is now tracked and arrives with a fresh checkout.
`data/processed/game_features.parquet` supplies the game-id lookup for quotes;
the sync job now pushes the local copy when its SHA-256 hash differs.
Memory during the nflverse capture stayed under 200 MB used.

Original deployment checklist (historical; reconcile it with the completed
steps above before using it for a new deployment):

1. Copy the code:
   `scp -r pyproject.toml uv.lock README.md src scripts registry config deploy backup-server:/data/nfl_py3/`
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
server's fills gaps; every doubly-captured window remains remotely. An exact
file-name and SHA-256 match is logged as `DUP`; a differing observation is
logged as `CONFLICT`. Neither is merged, averaged, or deleted automatically.
