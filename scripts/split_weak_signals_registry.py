from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_DIRNAME,
    WEAK_SIGNAL_REGISTRY_FILENAME,
    read_registry_payload,
    read_registry_shards,
    registry_from_payload,
    registry_to_payload,
    write_registry_shards,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split registry/weak_signals.json into registry/weak_signals/ shards."
    )
    parser.add_argument("--registry-dir", type=Path, default=Path("registry"))
    args = parser.parse_args()
    legacy = args.registry_dir / WEAK_SIGNAL_REGISTRY_FILENAME
    shards = args.registry_dir / WEAK_SIGNAL_REGISTRY_DIRNAME
    if shards.exists():
        print(f"refusing: {shards} already exists", file=sys.stderr)
        return 1
    if not legacy.is_file():
        print(f"refusing: {legacy} does not exist", file=sys.stderr)
        return 1
    payload = read_registry_payload(legacy)
    if payload is None:
        print("refusing: legacy registry unreadable", file=sys.stderr)
        return 1
    write_registry_shards(payload, shards)
    rebuilt = read_registry_shards(shards)
    legacy_payload = registry_to_payload(registry_from_payload(payload))
    shard_payload = registry_to_payload(registry_from_payload(rebuilt))
    raw_roundtrip = rebuilt == payload
    if legacy_payload != shard_payload or not raw_roundtrip:
        print("round trip mismatch; shard directory left in place for inspection", file=sys.stderr)
        return 2
    files = sorted(shards.iterdir())
    print(f"shards={len(files)} signals={len(payload['signals'])} legacy_kept={legacy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
