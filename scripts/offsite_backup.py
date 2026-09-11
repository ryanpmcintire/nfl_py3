from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES: tuple[str, ...] = ("data", "artifacts", "registry")
REPOSITORY = "sftp:backup-server:/data/restic"
PASSWORD_FILE = Path.home() / ".config" / "restic" / "password"
LOG_PATH = REPO / "data" / "offsite_backup_log.txt"
EXCLUDES: tuple[str, ...] = ("scheduler_heartbeat.json", "scheduler_state.json.tmp")
RETENTION: tuple[str, ...] = ("--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "6")
QUOTA_BYTES = 100 * (1 << 30)
QUOTA_WARN_FRACTION = 0.8
WINGET_PACKAGES = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
DESCRIPTION = (
    "Encrypted off-site restic backup of data/, artifacts/ and registry/ "
    "to the friend's server over WireGuard."
)


def restic_binary() -> str:
    found = shutil.which("restic")
    if found:
        return found
    for candidate in sorted(WINGET_PACKAGES.glob("restic.restic_*/restic*.exe")):
        return str(candidate)
    raise SystemExit("restic is not installed: winget install --id restic.restic --scope user")


def restic_env() -> dict[str, str]:
    if not PASSWORD_FILE.exists():
        raise SystemExit(f"missing restic password file: {PASSWORD_FILE}")
    env = dict(os.environ)
    env["RESTIC_REPOSITORY"] = REPOSITORY
    env["RESTIC_PASSWORD_FILE"] = str(PASSWORD_FILE)
    return env


def run(
    binary: str, args: list[str], env: dict[str, str], *, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")
    print(line)


def failure(step: str, result: subprocess.CompletedProcess[str]) -> SystemExit:
    return SystemExit(f"restic {step} failed ({result.returncode}): {result.stderr.strip()[:400]}")


def repository_exists(binary: str, env: dict[str, str]) -> bool:
    probe = run(binary, ["cat", "config"], env, timeout=120)
    if probe.returncode == 0:
        return True
    text = (probe.stderr or "").lower()
    if "is there a repository at the following location" in text or "no such file" in text:
        return False
    raise SystemExit(f"cannot reach repository: {probe.stderr.strip()[:400]}")


def snapshot_count(binary: str, env: dict[str, str]) -> int:
    result = run(binary, ["snapshots", "--json"], env, timeout=300)
    if result.returncode != 0:
        raise failure("snapshots", result)
    return len(json.loads(result.stdout or "[]"))


def repository_bytes(binary: str, env: dict[str, str]) -> int:
    result = run(binary, ["stats", "--mode", "raw-data", "--json"], env, timeout=600)
    if result.returncode != 0:
        return -1
    return int(json.loads(result.stdout).get("total_size", -1))


def backup(binary: str, env: dict[str, str], sources: list[str], *, timeout: int) -> str:
    args = ["backup", "--tag", "scheduled", "--exclude-caches"]
    for pattern in EXCLUDES:
        args += ["--exclude", pattern]
    args += [str(REPO / source) for source in sources if (REPO / source).exists()]
    result = run(binary, args, env, timeout=timeout)
    if result.returncode not in (0, 3):
        raise failure("backup", result)
    summary = [line for line in (result.stdout or "").splitlines() if line.startswith("snapshot ")]
    return summary[-1] if summary else f"backup exit {result.returncode}"


def forget_and_prune(binary: str, env: dict[str, str]) -> None:
    result = run(binary, ["forget", *RETENTION, "--prune"], env, timeout=1800)
    if result.returncode != 0:
        raise failure("forget --prune", result)


def check(binary: str, env: dict[str, str]) -> None:
    result = run(binary, ["check"], env, timeout=1800)
    if result.returncode != 0:
        raise failure("check", result)


def restore_test(binary: str, env: dict[str, str], relative: str) -> int:
    source = REPO / relative
    if not source.is_file():
        raise SystemExit(f"not a file: {source}")
    target = Path(tempfile.mkdtemp(prefix="restic_restore_"))
    include = "/" + source.as_posix().replace(":", "")
    args = ["restore", "latest", "--target", str(target), "--include", include]
    result = run(binary, args, env, timeout=600)
    if result.returncode != 0:
        raise failure("restore", result)
    restored = next(target.rglob(source.name), None)
    if restored is None:
        raise SystemExit(f"restore produced no file under {target}")
    same = (
        hashlib.sha256(restored.read_bytes()).digest()
        == hashlib.sha256(source.read_bytes()).digest()
    )
    shutil.rmtree(target, ignore_errors=True)
    log(f"RESTORE-TEST {relative} bytes={source.stat().st_size} hash_match={same}")
    return 0 if same else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report snapshots and repository size only; write nothing.",
    )
    parser.add_argument("--no-prune", action="store_true", help="Skip forget --prune.")
    parser.add_argument("--no-check", action="store_true", help="Skip restic check.")
    parser.add_argument(
        "--timeout", type=int, default=1500, help="Seconds allowed for the backup step."
    )
    parser.add_argument(
        "--restore-test",
        metavar="RELATIVE_PATH",
        help="Restore one file from the latest snapshot into a temp dir and compare hashes.",
    )
    args = parser.parse_args(argv)

    binary = restic_binary()
    env = restic_env()
    if args.restore_test:
        return restore_test(binary, env, args.restore_test)
    if args.status:
        if not repository_exists(binary, env):
            print(f"no repository yet at {REPOSITORY}")
            return 1
        size = repository_bytes(binary, env)
        count = snapshot_count(binary, env)
        print(f"snapshots={count} repository_bytes={size} quota_bytes={QUOTA_BYTES}")
        return 0

    if not repository_exists(binary, env):
        init = run(binary, ["init"], env, timeout=300)
        if init.returncode != 0:
            raise failure("init", init)
        log(f"INIT {REPOSITORY}")

    summary = backup(binary, env, args.sources, timeout=args.timeout)
    if not args.no_prune:
        forget_and_prune(binary, env)
    if not args.no_check:
        check(binary, env)
    size = repository_bytes(binary, env)
    count = snapshot_count(binary, env)
    warning = " QUOTA-WARN" if size > QUOTA_BYTES * QUOTA_WARN_FRACTION else ""
    log(f"OK {summary} snapshots={count} repository_bytes={size}{warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
