from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import capture_scheduler as scheduler  # noqa: E402

REMOTE_HOST = "backup-server"
REMOTE_REPO = "/data/nfl_py3"
REMOTE_NAME = "backup-server"
LOG_PATH = REPO / "data" / "sync_captures_log.txt"
HOST_MARKER = "capture_host"
FEATURE_TABLE = "data/processed/game_features.parquet"
DEFAULT_WINDOW_MINUTES = 30
STAMP = re.compile(r"^\d{8}T\d{6}Z$")
SSH_TIMEOUT = 600


def capture_roots() -> dict[str, int]:
    roots: dict[str, int] = {}
    for job in scheduler.SCHEDULE:
        if not job.dedupe_dir or not job.name.startswith(scheduler.CAPTURE_JOB_PREFIXES):
            continue
        window = job.dedupe_minutes or DEFAULT_WINDOW_MINUTES
        roots[job.dedupe_dir] = max(roots.get(job.dedupe_dir, 0), window)
    return roots


def log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")
    print(line)


def ssh(command: str, *, timeout: int = SSH_TIMEOUT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE_HOST, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_stamp(name: str) -> datetime:
    return datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def remote_snapshots(roots: dict[str, int]) -> dict[str, list[str]]:
    script = f"cd {REMOTE_REPO} || exit 3; for r in {' '.join(roots)}; do "
    script += '[ -d "$r" ] && find "$r" -mindepth 1 -maxdepth 1 -type d -printf "%p\\n"; done; true'
    result = ssh(script)
    if result.returncode != 0:
        raise SystemExit(f"remote listing failed: {result.stderr.strip()[:300]}")
    found: dict[str, list[str]] = {root: [] for root in roots}
    for line in result.stdout.splitlines():
        path = line.strip().replace("\\", "/")
        if not path:
            continue
        root, _, name = path.rpartition("/")
        if root in found and STAMP.match(name):
            found[root].append(name)
    return found


def local_snapshots(root: str) -> list[str]:
    folder = REPO / root
    if not folder.is_dir():
        return []
    return sorted(
        child.name for child in folder.iterdir() if child.is_dir() and STAMP.match(child.name)
    )


def remote_listing(relative: str) -> dict[str, int]:
    result = ssh(f'cd {REMOTE_REPO} && find "{relative}" -type f -printf "%s %P\\n"')
    if result.returncode != 0:
        raise SystemExit(f"remote listing of {relative} failed: {result.stderr.strip()[:300]}")
    listing: dict[str, int] = {}
    for line in result.stdout.splitlines():
        size, _, name = line.strip().partition(" ")
        if name:
            listing[name.replace("\\", "/")] = int(size)
    return listing


def local_listing(relative: str) -> dict[str, int]:
    folder = REPO / relative
    return {
        path.relative_to(folder).as_posix(): path.stat().st_size
        for path in folder.rglob("*")
        if path.is_file() and path.name != HOST_MARKER
    }


def pull(relative: str) -> None:
    tar = shutil.which("tar")
    if tar is None:
        raise SystemExit("tar is not available on this machine")
    parent = (REPO / relative).parent
    parent.mkdir(parents=True, exist_ok=True)
    root, _, name = relative.rpartition("/")
    source = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_HOST,
            f'cd {REMOTE_REPO}/{root} && tar -cf - "{name}"',
        ],
        stdout=subprocess.PIPE,
    )
    sink = subprocess.run(
        [tar, "-xf", "-", "-C", str(parent)], stdin=source.stdout, timeout=SSH_TIMEOUT
    )
    source.wait(timeout=SSH_TIMEOUT)
    if source.returncode != 0 or sink.returncode != 0:
        shutil.rmtree(REPO / relative, ignore_errors=True)
        raise SystemExit(
            f"pull of {relative} failed (ssh {source.returncode}, tar {sink.returncode})"
        )
    (REPO / relative / HOST_MARKER).write_text(REMOTE_NAME + "\n", encoding="utf-8")


def delete_remote(relative: str) -> None:
    result = ssh(f'cd {REMOTE_REPO} && rm -rf "{relative}"')
    if result.returncode != 0:
        log(f"WARN could not delete {relative} on {REMOTE_NAME}: {result.stderr.strip()[:200]}")


def diff_summary(local_relative: str, remote_relative: str) -> str:
    mine = local_listing(local_relative)
    theirs = remote_listing(remote_relative)
    shared = set(mine) & set(theirs)
    same = sum(1 for name in shared if mine[name] == theirs[name])
    return (
        f"files_local={len(mine)} files_remote={len(theirs)} shared={len(shared)} "
        f"same_size={same} size_differs={len(shared) - same}"
    )


def push_feature_table() -> str:
    source = REPO / FEATURE_TABLE
    if not source.is_file():
        return "feature_table=missing_locally"
    size = source.stat().st_size
    probe = ssh(f'stat -c "%s" {REMOTE_REPO}/{FEATURE_TABLE} 2>/dev/null || echo 0')
    if probe.returncode == 0 and probe.stdout.strip() == str(size):
        return "feature_table=current"
    remote_dir = f"{REMOTE_REPO}/{Path(FEATURE_TABLE).parent.as_posix()}"
    with source.open("rb") as handle:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                REMOTE_HOST,
                f"mkdir -p {remote_dir} && cat > {REMOTE_REPO}/{FEATURE_TABLE}.tmp "
                f"&& mv {REMOTE_REPO}/{FEATURE_TABLE}.tmp {REMOTE_REPO}/{FEATURE_TABLE}",
            ],
            stdin=handle,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT,
        )
    if result.returncode != 0:
        return f"feature_table=push_failed({result.stderr.strip()[:120]})"
    return f"feature_table=pushed({size} bytes)"


def nearest_local(stamps: list[str], target: datetime, window_minutes: int) -> str | None:
    best: tuple[float, str] | None = None
    for name in stamps:
        gap = abs((parse_stamp(name) - target).total_seconds()) / 60.0
        if gap <= window_minutes and (best is None or gap < best[0]):
            best = (gap, name)
    return best[1] if best else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile capture snapshots with the friend's server: pull the ones that fill a "
            "gap here, log a file-level diff for windows both hosts captured, delete the "
            "server copy once it is accounted for."
        )
    )
    parser.add_argument(
        "--dry", action="store_true", help="List what would happen; change nothing."
    )
    parser.add_argument(
        "--keep-remote", action="store_true", help="Never delete a snapshot on the server."
    )
    args = parser.parse_args(argv)
    if scheduler.SCHEDULER_ROLE == "capture":
        print("sync_captures runs only on the primary role")
        return 0

    roots = capture_roots()
    remote = remote_snapshots(roots)
    pulled = duplicates = skipped = 0
    for root, window in sorted(roots.items()):
        mine = local_snapshots(root)
        for name in sorted(remote.get(root, [])):
            relative = f"{root}/{name}"
            if name in mine:
                skipped += 1
                if not args.dry and not args.keep_remote:
                    delete_remote(relative)
                continue
            twin = nearest_local(mine, parse_stamp(name), window)
            if twin is not None:
                duplicates += 1
                summary = "dry" if args.dry else diff_summary(f"{root}/{twin}", relative)
                log(f"DUP {root} local={twin} remote={name} {summary}")
                if not args.dry and not args.keep_remote:
                    delete_remote(relative)
                continue
            if args.dry:
                log(f"WOULD-PULL {relative}")
                pulled += 1
                continue
            pull(relative)
            expected = remote_listing(relative)
            actual = local_listing(relative)
            if expected != actual:
                log(
                    f"FAIL {relative}: pulled listing differs from the server "
                    f"({len(actual)} vs {len(expected)} files)"
                )
                shutil.rmtree(REPO / relative, ignore_errors=True)
                return 1
            pulled += 1
            mine.append(name)
            log(f"PULLED {relative} files={len(actual)}")
            if not args.keep_remote:
                delete_remote(relative)
    table = "feature_table=dry" if args.dry else push_feature_table()
    log(
        f"OK pulled={pulled} duplicates={duplicates} already_present={skipped} "
        f"roots={len(roots)} {table}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
