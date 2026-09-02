"""Machine-readable acquisition, retention, quota, and publication policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "config" / "source_policies.json"
VALID_RISKS = frozenset({"green", "yellow", "red"})
VALID_RAW_RETENTION = frozenset({"private_local_only", "private_local_existing_only"})


class SourcePolicyError(RuntimeError):
    """An external-source operation violates the reviewed policy registry."""


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    risk: str
    terms_url: str
    terms_reviewed_on: date
    acquisition_allowed: bool
    raw_retention: str
    raw_redistribution: str
    derived_publication: str
    conditions: tuple[str, ...]
    quota: dict[str, int]


def load_source_policies(path: Path = DEFAULT_REGISTRY) -> dict[str, SourcePolicy]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), dict):
        raise SourcePolicyError(f"Invalid source-policy registry schema: {path}")
    policies: dict[str, SourcePolicy] = {}
    for source_id, raw in payload["sources"].items():
        try:
            reviewed = date.fromisoformat(str(raw["terms_reviewed_on"]))
            conditions = tuple(str(value) for value in raw["conditions"])
            quota = {str(key): int(value) for key, value in raw.get("quota", {}).items()}
            policy = SourcePolicy(
                source_id=str(source_id),
                risk=str(raw["risk"]),
                terms_url=str(raw["terms_url"]),
                terms_reviewed_on=reviewed,
                acquisition_allowed=raw["acquisition_allowed"],
                raw_retention=str(raw["raw_retention"]),
                raw_redistribution=str(raw["raw_redistribution"]),
                derived_publication=str(raw["derived_publication"]),
                conditions=conditions,
                quota=quota,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SourcePolicyError(f"Invalid policy for source {source_id}: {error}") from error
        if policy.risk not in VALID_RISKS:
            raise SourcePolicyError(f"Invalid risk for source {source_id}: {policy.risk}")
        if type(policy.acquisition_allowed) is not bool:  # bool, not truthy strings/integers
            raise SourcePolicyError(f"acquisition_allowed must be boolean for {source_id}")
        if policy.raw_retention not in VALID_RAW_RETENTION:
            raise SourcePolicyError(f"Invalid raw retention for source {source_id}")
        if not policy.terms_url.startswith("https://") or not policy.conditions:
            raise SourcePolicyError(f"Terms URL and conditions are required for {source_id}")
        policies[policy.source_id] = policy
    return policies


def require_acquisition(source_id: str, path: Path = DEFAULT_REGISTRY) -> SourcePolicy:
    policies = load_source_policies(path)
    if source_id not in policies:
        raise SourcePolicyError(f"External source is not registered: {source_id}")
    policy = policies[source_id]
    if not policy.acquisition_allowed:
        raise SourcePolicyError(
            f"New acquisition is disabled by source policy for {source_id}; "
            f"review {policy.terms_url} and update the tracked registry first"
        )
    return policy


def validate_review_currency(
    *, path: Path = DEFAULT_REGISTRY, as_of: date | None = None, max_age_days: int = 366
) -> None:
    """Fail when reviewed terms have exceeded the declared annual audit cadence."""

    today = as_of or date.today()
    stale = sorted(
        policy.source_id
        for policy in load_source_policies(path).values()
        if policy.terms_reviewed_on + timedelta(days=max_age_days) < today
    )
    if stale:
        raise SourcePolicyError("Source terms review is stale: " + ", ".join(stale))


def require_private_raw_destination(source_id: str, destination: Path) -> None:
    policy = require_acquisition(source_id)
    resolved = destination.resolve()
    private_roots = tuple(
        (REPO_ROOT / relative).resolve()
        for relative in ("data/raw", "data/market", "data/cfb", "data/players")
    )
    inside_repository = resolved == REPO_ROOT or REPO_ROOT in resolved.parents
    approved_repository_root = any(
        resolved == root or root in resolved.parents for root in private_roots
    )
    if (
        policy.raw_retention.startswith("private_local")
        and inside_repository
        and not approved_repository_root
    ):
        raise SourcePolicyError(
            f"Raw {source_id} data must stay under a gitignored private data root: {destination}"
        )


def require_raw_redistribution(source_id: str) -> None:
    policy = load_source_policies().get(source_id)
    if policy is None or policy.raw_redistribution == "prohibited":
        raise SourcePolicyError(f"Raw redistribution is prohibited for {source_id}")
