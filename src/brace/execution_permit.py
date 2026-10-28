"""One-shot execution permits bound to canonical tool actions."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


def _canonical_payload(
    operation: str,
    resource: str,
    arguments: Mapping[str, Any],
) -> bytes:
    """Encode an action deterministically for binding and audit."""
    document = {
        "schema": "brace.action.v1",
        "operation": operation,
        "resource": resource,
        "arguments": arguments,
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def action_fingerprint(
    operation: str,
    resource: str,
    arguments: Mapping[str, Any],
) -> str:
    """Return the versioned SHA-256 fingerprint of an exact action."""
    return "sha256:" + hashlib.sha256(
        _canonical_payload(operation, resource, arguments)
    ).hexdigest()


class PermitState(str, Enum):
    READY = "ready"
    SPENT = "spent"
    REVOKED = "revoked"


@dataclass(frozen=True)
class PermitCheck:
    accepted: bool
    problems: tuple[str, ...] = ()


@dataclass
class ExecutionPermit:
    """Short-lived authority for one exact action in one task session."""

    permit_id: str
    action_fingerprint: str
    session_id: str
    policy_version: str
    authorization_context: dict[str, Any]
    issued_at: float
    valid_until: float
    state: PermitState = field(default=PermitState.READY)

    def check(
        self,
        expected_fingerprint: str,
        session_id: str,
        *,
        now: float | None = None,
    ) -> PermitCheck:
        observed_at = time.time() if now is None else now
        problems: list[str] = []
        if self.state is PermitState.SPENT:
            problems.append("permit_already_spent")
        elif self.state is PermitState.REVOKED:
            problems.append("permit_revoked")
        if observed_at > self.valid_until:
            problems.append("permit_expired")
        if not secrets.compare_digest(self.action_fingerprint, expected_fingerprint):
            problems.append("action_fingerprint_mismatch")
        if not secrets.compare_digest(self.session_id, session_id):
            problems.append("session_mismatch")
        return PermitCheck(not problems, tuple(problems))


class PermitLedger:
    """In-memory permit registry with atomic validation and consumption."""

    def __init__(self) -> None:
        self._permits: dict[str, ExecutionPermit] = {}
        self._events: list[dict[str, Any]] = []

    def issue(
        self,
        *,
        operation: str,
        resource: str,
        arguments: Mapping[str, Any],
        session_id: str,
        policy_version: str,
        authorization_context: Mapping[str, Any],
        ttl_seconds: float,
        now: float | None = None,
    ) -> ExecutionPermit:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued_at = time.time() if now is None else now
        permit = ExecutionPermit(
            permit_id=secrets.token_urlsafe(24),
            action_fingerprint=action_fingerprint(operation, resource, arguments),
            session_id=session_id,
            policy_version=policy_version,
            authorization_context=dict(authorization_context),
            issued_at=issued_at,
            valid_until=issued_at + ttl_seconds,
        )
        self._permits[permit.permit_id] = permit
        self._record("issued", permit)
        return permit

    def validate(
        self,
        permit: ExecutionPermit,
        *,
        operation: str,
        resource: str,
        arguments: Mapping[str, Any],
        session_id: str,
        now: float | None = None,
    ) -> PermitCheck:
        registered = self._permits.get(permit.permit_id)
        if registered is not permit:
            return PermitCheck(False, ("permit_not_registered",))
        expected = action_fingerprint(operation, resource, arguments)
        return permit.check(expected, session_id, now=now)

    def consume(
        self,
        permit: ExecutionPermit,
        *,
        operation: str,
        resource: str,
        arguments: Mapping[str, Any],
        session_id: str,
        now: float | None = None,
    ) -> PermitCheck:
        check = self.validate(
            permit,
            operation=operation,
            resource=resource,
            arguments=arguments,
            session_id=session_id,
            now=now,
        )
        if not check.accepted:
            self._record("consume_denied", permit, check.problems)
            return check
        permit.state = PermitState.SPENT
        self._record("consumed", permit)
        return PermitCheck(True)

    def revoke(self, permit_id: str) -> bool:
        permit = self._permits.get(permit_id)
        if permit is None or permit.state is not PermitState.READY:
            return False
        permit.state = PermitState.REVOKED
        self._record("revoked", permit)
        return True

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def _record(
        self,
        event: str,
        permit: ExecutionPermit,
        problems: tuple[str, ...] = (),
    ) -> None:
        self._events.append({
            "event": event,
            "permit_id": permit.permit_id,
            "action_fingerprint": permit.action_fingerprint,
            "timestamp": time.time(),
            "problems": list(problems),
        })
