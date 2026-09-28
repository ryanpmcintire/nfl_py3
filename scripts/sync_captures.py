from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

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
WINDOWS_INVALID = frozenset('<>:"\\|?*')
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)


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
    root_args = " ".join(shlex.quote(root) for root in roots)
    script = f"cd {shlex.quote(REMOTE_REPO)} || exit 3; for r in {root_args}; do "
    script += (
        '[ ! -e "$r" ] && continue; [ -d "$r" ] || exit 4; '
        'find "$r" -mindepth 1 -maxdepth 1 -type d -printf "%p\\n" || exit 5; done'
    )
    result = ssh(script)
    if result.returncode != 0:
        raise SystemExit(f"remote listing failed: {result.stderr.strip()[:300]}")
    found: dict[str, list[str]] = {root: [] for root in roots}
    for line in result.stdout.splitlines():
        path = line.strip().replace("\\", "/")
        if not path:
            continue
        root, _, name = path.rpartition("/")
        if root not in found:
            raise SystemExit(f"remote listing returned an unexpected path: {path}")
        if STAMP.match(name):
            found[root].append(name)
    for root, names in found.items():
        if len(names) != len(set(names)):
            raise SystemExit(f"remote listing returned duplicate snapshots for {root}")
    return found


def local_snapshots(root: str) -> list[str]:
    folder = REPO / root
    if not folder.is_dir():
        return []
    return sorted(
        child.name for child in folder.iterdir() if child.is_dir() and STAMP.match(child.name)
    )


def checked_relative(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SystemExit(f"unsafe capture path: {relative}")
    return path


def checked_snapshot_member(raw_name: str, relative: str) -> PurePosixPath:
    name = PurePosixPath(raw_name)
    invalid_character = any(
        character in WINDOWS_INVALID or ord(character) < 32 for character in raw_name
    )
    invalid_part = any(
        part in {"", ".", ".."}
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED
        for part in name.parts
    )
    if name.is_absolute() or invalid_character or invalid_part or name.as_posix() != raw_name:
        raise SystemExit(f"remote manifest of {relative} contains an unsafe path")
    return name


def parse_remote_manifest(output: str, relative: str) -> dict[str, str]:
    if output and not output.endswith("\0"):
        raise SystemExit(f"remote manifest of {relative} is incomplete")
    manifest: dict[str, str] = {}
    windows_names: set[str] = set()
    records = output.split("\0")[:-1] if output else []
    for record in records:
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", record, flags=re.DOTALL)
        if match is None:
            raise SystemExit(f"remote manifest of {relative} is malformed")
        digest, raw_name = match.groups()
        name = checked_snapshot_member(raw_name, relative)
        if len(name.parts) == 1 and name.name.casefold() == HOST_MARKER.casefold():
            raise SystemExit(f"remote manifest of {relative} contains an unsafe path")
        normalized = name.as_posix()
        windows_name = normalized.casefold()
        if normalized in manifest or windows_name in windows_names:
            raise SystemExit(f"remote manifest of {relative} contains duplicate paths")
        manifest[normalized] = digest
        windows_names.add(windows_name)
    return manifest


def remote_manifest(relative: str) -> dict[str, str]:
    checked_relative(relative)
    folder = shlex.quote(f"{REMOTE_REPO}/{relative}")
    marker = shlex.quote(f"./{HOST_MARKER}")
    command = (
        f"cd {folder} || exit 3; files=$(mktemp) || exit 4; "
        'hashes=$(mktemp) || { rm -f "$files"; exit 4; }; '
        'trap \'rm -f "$files" "$hashes"\' EXIT HUP INT TERM; '
        f'find . -type f ! -path {marker} -print0 > "$files" || exit 5; '
        'sort -z "$files" -o "$files" || exit 6; '
        'xargs -0 -r sha256sum -z -- < "$files" > "$hashes" || exit 7; '
        'cat "$hashes"'
    )
    result = ssh(command)
    if result.returncode != 0:
        raise SystemExit(f"remote manifest of {relative} failed: {result.stderr.strip()[:300]}")
    manifest = parse_remote_manifest(result.stdout, relative)
    if not manifest:
        raise SystemExit(f"remote manifest of {relative} is empty")
    return manifest


def folder_manifest(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        raise SystemExit(f"capture snapshot is not a directory: {folder}")
    entries = sorted(folder.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise SystemExit(f"capture snapshot contains a symlink: {folder}")
    root_markers = [
        path
        for path in entries
        if path.is_file()
        and len(path.relative_to(folder).parts) == 1
        and path.name.casefold() == HOST_MARKER.casefold()
    ]
    if any(path.name != HOST_MARKER for path in root_markers):
        raise SystemExit(f"capture snapshot has a reserved root file name: {folder}")
    files = [path for path in entries if path.is_file() and path not in root_markers]
    manifest: dict[str, str] = {}
    for path in files:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SystemExit(f"capture snapshot changed while hashing: {folder}")
        manifest[path.relative_to(folder).as_posix()] = digest.hexdigest()
    current = sorted(
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and path not in root_markers
    )
    if current != sorted(manifest):
        raise SystemExit(f"capture snapshot changed while hashing: {folder}")
    return manifest


def local_manifest(relative: str) -> dict[str, str]:
    checked_relative(relative)
    return folder_manifest(REPO / relative)


def pull(relative: str, expected: dict[str, str]) -> dict[str, str]:
    path = checked_relative(relative)
    if not expected:
        raise SystemExit(f"refusing to pull an empty capture snapshot: {relative}")
    parent = (REPO / relative).parent
    parent.mkdir(parents=True, exist_ok=True)
    target = REPO / relative
    if target.exists():
        raise SystemExit(f"refusing to pull over an existing capture snapshot: {relative}")
    root = PurePosixPath(*path.parts[:-1]).as_posix()
    name = path.name
    with tempfile.TemporaryDirectory(prefix=".sync-captures-", dir=parent) as temporary:
        staging = Path(temporary)
        archive_path = staging / "snapshot.tar"
        extract_root = staging / "extract"
        extract_root.mkdir()
        with archive_path.open("wb") as output:
            transfer = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    REMOTE_HOST,
                    f"cd {shlex.quote(f'{REMOTE_REPO}/{root}')} && "
                    f"tar --exclude={shlex.quote(f'{name}/{HOST_MARKER}')} "
                    f"-cf - -- {shlex.quote(name)}",
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=SSH_TIMEOUT,
            )
        if transfer.returncode != 0:
            raise SystemExit(
                f"pull of {relative} failed: {transfer.stderr.decode(errors='replace')[:300]}"
            )
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            if not members:
                raise SystemExit(f"pull of {relative} produced an empty archive")
            for member in members:
                if member.name == name:
                    pass
                elif member.name.startswith(f"{name}/"):
                    checked_snapshot_member(member.name[len(name) + 1 :], relative)
                else:
                    raise SystemExit(f"pull of {relative} produced an unsafe archive path")
                if not member.isdir() and not member.isfile():
                    raise SystemExit(f"pull of {relative} produced an unsafe archive member")
            archive.extractall(extract_root, members=members, filter="data")
        staged = extract_root / name
        if sorted(child.name for child in extract_root.iterdir()) != [name] or not staged.is_dir():
            raise SystemExit(f"pull of {relative} produced an unexpected archive layout")
        actual = folder_manifest(staged)
        observed = remote_manifest(relative)
        if expected != observed:
            raise SystemExit(f"remote snapshot changed while pulling: {relative}")
        if expected != actual:
            raise SystemExit(f"pulled content differs from the remote manifest: {relative}")
        (staged / HOST_MARKER).write_text(REMOTE_NAME + "\n", encoding="utf-8")
        try:
            staged.rename(target)
        except OSError as exc:
            raise SystemExit(f"could not install pulled snapshot {relative}: {exc}") from exc
    return actual


def diff_summary(mine: dict[str, str], theirs: dict[str, str]) -> str:
    shared = set(mine) & set(theirs)
    same = sum(1 for name in shared if mine[name] == theirs[name])
    return (
        f"files_local={len(mine)} files_remote={len(theirs)} shared={len(shared)} "
        f"same_content={same} content_differs={len(shared) - same} "
        f"local_only={len(set(mine) - set(theirs))} remote_only={len(set(theirs) - set(mine))}"
    )


def stable_file_digest(path: Path) -> str | None:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None
    return digest.hexdigest()


def parse_sha256(output: str) -> str | None:
    value, _, _ = output.strip().partition(" ")
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def discard_remote_file(path: str) -> None:
    ssh(f"rm -f -- {shlex.quote(path)}")


def push_feature_table() -> str:
    source = REPO / FEATURE_TABLE
    if not source.is_file():
        return "feature_table=missing_locally"
    expected = stable_file_digest(source)
    if expected is None:
        return "feature_table=push_failed(local_changed)"
    remote_path = f"{REMOTE_REPO}/{FEATURE_TABLE}"
    quoted_remote = shlex.quote(remote_path)
    probe = ssh(
        f"if [ ! -e {quoted_remote} ]; then echo missing; "
        f"elif [ -L {quoted_remote} ] || [ ! -f {quoted_remote} ]; then exit 4; "
        f"else sha256sum -- {quoted_remote}; fi"
    )
    if probe.returncode != 0:
        return f"feature_table=push_failed(remote_probe:{probe.stderr.strip()[:80]})"
    remote_digest = None if probe.stdout.strip() == "missing" else parse_sha256(probe.stdout)
    if probe.stdout.strip() != "missing" and remote_digest is None:
        return "feature_table=push_failed(malformed_remote_hash)"
    if remote_digest == expected:
        confirmed = stable_file_digest(source)
        if confirmed != expected:
            return "feature_table=push_failed(local_changed)"
        return "feature_table=current"
    remote_dir = f"{REMOTE_REPO}/{Path(FEATURE_TABLE).parent.as_posix()}"
    staged = f"{remote_path}.sync-{uuid.uuid4().hex}"
    quoted_staged = shlex.quote(staged)
    with source.open("rb") as handle:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                REMOTE_HOST,
                f"umask 077; mkdir -p {shlex.quote(remote_dir)} && "
                f"cat > {quoted_staged} && sha256sum -- {quoted_staged}",
            ],
            stdin=handle,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT,
        )
    if result.returncode != 0:
        discard_remote_file(staged)
        return f"feature_table=push_failed({result.stderr.strip()[:120]})"
    if parse_sha256(result.stdout) != expected:
        discard_remote_file(staged)
        return "feature_table=push_failed(staged_hash_mismatch)"
    confirmed = stable_file_digest(source)
    if confirmed != expected:
        discard_remote_file(staged)
        return "feature_table=push_failed(local_changed)"
    finalize = ssh(
        f"actual=$(sha256sum -- {quoted_staged}) || exit 5; "
        "actual=${actual%% *}; "
        f'[ "$actual" = {shlex.quote(expected)} ] || exit 6; '
        f"mv -fT -- {quoted_staged} {quoted_remote}"
    )
    if finalize.returncode != 0:
        discard_remote_file(staged)
        return f"feature_table=push_failed(finalize:{finalize.stderr.strip()[:80]})"
    return f"feature_table=pushed(sha256={expected[:12]})"


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
            "gap here, compare duplicate windows by SHA-256 manifest, and preserve every "
            "server snapshot."
        )
    )
    parser.add_argument(
        "--dry", action="store_true", help="List what would happen; change nothing."
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Compatibility flag; server snapshots are always preserved.",
    )
    args = parser.parse_args(argv)
    if scheduler.SCHEDULER_ROLE == "capture":
        print("sync_captures runs only on the primary role")
        return 0

    roots = capture_roots()
    remote = remote_snapshots(roots)
    pulled = duplicates = skipped = conflicts = 0
    for root, window in sorted(roots.items()):
        mine = local_snapshots(root)
        for name in sorted(remote.get(root, [])):
            relative = f"{root}/{name}"
            if name in mine:
                local_files = local_manifest(relative)
                remote_files = remote_manifest(relative)
                summary = diff_summary(local_files, remote_files)
                if remote_files and local_files == remote_files:
                    skipped += 1
                    log(f"MATCH {relative} {summary} remote=kept(collision-policy)")
                else:
                    conflicts += 1
                    log(f"CONFLICT {relative} exact-name {summary} remote=kept")
                continue
            twin = nearest_local(mine, parse_stamp(name), window)
            if twin is not None:
                local_files = local_manifest(f"{root}/{twin}")
                remote_files = remote_manifest(relative)
                summary = diff_summary(local_files, remote_files)
                if remote_files and local_files == remote_files:
                    duplicates += 1
                    log(
                        f"DUP {root} local={twin} remote={name} {summary} "
                        "remote=kept(collision-policy)"
                    )
                else:
                    conflicts += 1
                    log(f"CONFLICT {root} local={twin} remote={name} {summary} remote=kept")
                continue
            if args.dry:
                log(f"WOULD-PULL {relative}")
                pulled += 1
                continue
            expected = remote_manifest(relative)
            actual = pull(relative, expected)
            pulled += 1
            mine.append(name)
            log(f"PULLED {relative} files={len(actual)} remote=kept(no-immutability-proof)")
    table = "feature_table=dry" if args.dry else push_feature_table()
    failed = table == "feature_table=missing_locally" or table.startswith(
        "feature_table=push_failed("
    )
    log(
        f"{'FAIL' if failed else 'OK'} pulled={pulled} duplicates={duplicates} "
        f"already_present={skipped} "
        f"conflicts={conflicts} roots={len(roots)} {table}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
