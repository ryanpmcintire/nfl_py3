# Public forecast archive foundation

Status: **tamper-evident canonical archive implemented 2026-09-02; SKY-06
remains open because publisher authentication is not implemented**.

This work is production-isolated. It did not alter `publish-predictions`, the
published card, any prospective/CLV ledger, the scheduler, model selection, or
the registry. It created no live archive and stores no secret in the
repository.

## What the record preserves

`src/nfl_ats/public_forecast_archive.py` defines one canonical JSON object per
publication batch. Each line includes:

- publication identity and UTC publication timestamp;
- decision label, decision timestamp, and latest declared input-observation
  timestamp;
- model ID, feature profile, probability method, and SHA-256 hashes of the
  model configuration, feature table, and source prediction artifact;
- per-game identity, season/week, teams, kickoff, spread, predicted margin,
  home win/cover probabilities, and the cover/push/loss probability split;
- the previous record's content hash; and
- the current record's deterministic content hash.

Forecast rows are sorted by season, week, kickoff, and game ID. UTC timestamps
are normalized to six fractional digits with a `Z` suffix. JSON is UTF-8 with
sorted keys, no insignificant whitespace, and no NaN/Infinity values. The
content hash is SHA-256 of those canonical bytes before the `content_sha256`
field is added.

These fields preserve the probability and exact quoted line needed for later
calibration. Outcomes are deliberately not appended to an old forecast row;
a future calibration report should join immutable forecasts to results by
`game_id`, leaving the pregame record unchanged.

## Point-in-time contract

Every record enforces:

```text
inputs_observed_through_utc <= decision_at_utc <= published_at_utc < kickoff_utc
```

The final inequality is checked for every game. Publication timestamps must
increase along the file and publication IDs cannot repeat. Probabilities must
be finite and lie in `[0, 1]`; the home-cover-excluding-push, push, and loss
probabilities must sum to one within `1e-9`. Game IDs cannot repeat within a
publication.

The provenance timestamps are declarations pinned into the content hash; the
feature/prediction artifact hashes identify the exact local inputs that must be
audited if the declaration is challenged. The archive does not infer a cutoff
from file modification time.

## Append and verification

`append_public_forecast_record` verifies the complete existing archive before
adding anything, takes an exclusive adjacent lock, rejects duplicate or
backdated publications, opens the archive in binary append mode, flushes the
new canonical line, and never rewrites existing bytes. A crash or manual edit
that leaves a partial line causes later verification and appends to fail
closed; there is no automatic repair that could hide lost history.

The standalone integrity command does not touch the project CLI:

```powershell
.\.tools\uv.exe run --no-sync python -m nfl_ats.public_forecast_archive verify `
  path\to\public_forecasts.jsonl
```

If a head hash was retained or published independently, pin it during
verification:

```powershell
.\.tools\uv.exe run --no-sync python -m nfl_ats.public_forecast_archive verify `
  path\to\public_forecasts.jsonl `
  --expected-head-sha256 <64-character-head-hash>
```

The verifier rejects noncanonical encoding, schema drift, altered content,
broken previous-record links, repeated publication IDs, non-increasing
publication times, invalid decision/kickoff chronology, and a mismatch with
the optional known head.

## Security boundary: a hash is not a signature

The SHA-256 chain provides integrity relative to a separately remembered head:
changing an old record changes its content hash and every later link. It does
not authenticate who published the archive. An attacker able to replace the
whole file can recompute every unkeyed hash and present a different head.

For that reason, this document and the verifier call these values **content
hashes**, never cryptographic signatures. SKY-06's “signed” definition of done
requires an owner decision outside the repository:

1. choose an owner-held signing key and custody/rotation/revocation policy, or
   an external signing/transparency provider;
2. define which canonical bytes are signed and how public keys/checkpoints are
   distributed independently of the archive;
3. add signature verification without committing private key material; and
4. deliberately wire the signed append into publishing, then accrue real
   pre-kickoff records long enough to report calibration over time.

Until that choice is made, the archive foundation is useful and testable but
SKY-06 remains open.

## Focused regression coverage

`tests/test_public_forecast_archive.py` verifies:

- input order does not change a record's deterministic content hash;
- a second append preserves the first record's exact byte prefix and links to
  its hash;
- a known external head can be checked;
- post-decision input provenance and at/after-kickoff publication fail closed;
- edited probabilities, noncanonical encoding, and forged/broken links fail;
- duplicate/backdated appends fail; and
- the standalone verification command reports valid and invalid archives with
  process-friendly return codes.
