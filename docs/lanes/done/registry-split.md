# registry-split

## Goal
Replace the 14.7 MB, 222k-line registry/weak_signals.json with per-family shards in registry/weak_signals/.

## State
Done 2026-10-20. registry/weak_signals/: _registry.json (version, notes) plus one <family>.json per family and _no_family.json; 392 files, 7,941 signals; largest shard 2.1 MB. weak_signals.py reads/writes shards (only changed shards rewritten; duplicate names, stray files or both layouts present raise; the legacy path falls back to the directory). merged_registry_file() hands model_ledger a merged temp file because model_ledger.py is pinned by the independent-validation source guard (name contains "model"); its bytes are unchanged. Live tests (test_weak_signals live ledger, test_registry_explorer live views, test_model_ledger live artifacts) read the shards and no longer skip. pytest 1645 passed, mypy clean.

## Tried
Worker edited model_ledger.py; reverted to the enrolled bytes (sha bb6c38fc) and routed board_site_content.py and public_board.py through merged_registry_file instead.

## Next
None. After Week 18, model_ledger.py may read shards directly.

## Open
None.
