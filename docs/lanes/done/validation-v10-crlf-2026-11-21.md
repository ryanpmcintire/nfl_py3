# validation-v10-crlf-2026-11-21

## Goal
Restore the v10 independent-validation capture, failing since 2026-11-18 12:29 ET with "source changed".

## State (2026-11-21 09:50 ET, measured)
- Cause: src/nfl_ats/pick_probability.py (pinned) was rewritten 2026-11-18 07:29 with LF line endings during the hou-best-pick-check revert; v10 enrolled the CRLF bytes (sha 4d48a405...). Content identical apart from CR (`git diff --ignore-cr-at-eol` empty); LF form hashes acbe391b.
- Fix: rewrote the file with CRLF endings; hash back to 4d48a405, `independent-validation status` source_unchanged true; `independent-validation capture` exit 0, recorded 12 (the unplayed Week 11 games). Git sees no change (autocrlf).
- Lost captures, not back-filled: Wed 11-18 12:10, Thu 11-19 12:10 and 19:55, Fri 11-20 12:10. IND at HOU (Thu) keeps its enrollment-time observation.

## Tried
- Nothing else; no source content changed.

## Next
- Post-Week-18 pinned bundle: hash LF-normalised bytes in independent_validation._sources so line-ending churn cannot stop a cohort (changes every stored hash, so only at a re-enrolment).
- Session start: grep data/scheduler_log.txt for `independent_validation.*FAIL`.

## Open
- None.
