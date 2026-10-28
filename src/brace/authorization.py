"""Evidence-bound runtime authorization for BRACE.

It normalizes planner checklists, binds dynamic values to trusted tool
evidence, accepts only explicit delegation grants from configured authority
tools, and uses single-use execution permits to bind authorization to the exact
action executed.
"""
from __future__ import annotations

import ast
import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import casbin
import yaml
from pydantic import BaseModel

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from .execution_permit import ExecutionPermit, PermitLedger, action_fingerprint


PLACEHOLDER_RE = re.compile(r"^\s*<[^<>]+>\s*$")
IDENTITY_KEYS = {
    "recipient", "recipients", "file_id", "file_path", "email", "attendees",
    "participant", "participants", "account", "iban", "user_id", "channel_id",
    "event_id", "location",
}
DIRECT_ACTION_PATTERNS = {
    "update_password": (r"update (?:the )?password", r"change (?:the )?password"),
    "delete_file": (r"delete (?:the )?.*file",),
    "send_email": (r"send (?:an |the )?email",),
    "send_money": (r"(?:send|transfer|pay) .*(?:money|bill|transaction)",),
    "create_calendar_event": (r"(?:create|add) (?:an |the )?(?:calendar )?(?:event|appointment)",),
    "reschedule_calendar_event": (r"(?:reschedule|move) .*(?:event|appointment|check-up)",),
    "send_channel_message": (r"(?:send|post|write) .*(?:message|channel)",),
}
DEFAULT_FIELD_ALIASES = {
    "id_": "id",
    "identifier": "id",
    "mail": "email",
    "email_address": "email",
    "user": "user_id",
    "userid": "user_id",
    "channel": "channel_id",
    "channelid": "channel_id",
}


@dataclass(frozen=True)
class PlanNode:
    name: str
    required_parameters: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanParseResult:
    nodes: tuple[PlanNode, ...]
    errors: tuple[str, ...] = ()
    parser: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class EvidenceRecord:
    source_tool: str
    tool_call_id: str | None
    trusted_text: str
    untrusted_text: str
    structured: Any = None
    evidence_items: tuple["EvidenceItem", ...] = ()


@dataclass(frozen=True)
class DetectedEvidenceItem:
    """A record-level relation extracted by the semantic detector."""

    record_id: str
    entity: str | None
    field: str
    value: Any


@dataclass(frozen=True)
class EvidenceItem:
    """A detector relation enriched with runtime-owned provenance."""

    source_tool: str
    tool_call_id: str | None
    record_id: str
    entity: str | None
    field: str
    value: Any


@dataclass(frozen=True)
class EvidenceDetection:
    """Semantic detector output for one raw tool response."""

    trusted_text: str
    untrusted_text: str
    structured: Any = None
    evidence_items: tuple[DetectedEvidenceItem, ...] = ()


class EvidenceDetector(Protocol):
    """Interface implemented by the LLM semantic trust detector."""

    def detect(
        self,
        *,
        text: str,
        source_tool: str,
        tool_call_id: str | None,
    ) -> EvidenceDetection | Mapping[str, Any]: ...


@dataclass(frozen=True)
class DelegationGrant:
    issuer: str
    subject: str
    actions: tuple[str, ...]
    parameter_constraints: dict[str, Any]
    evidence_verified: bool
    revoked: bool
    expires_at: float
    delegation_depth: int
    max_delegation_depth: int
    source_tool: str
    evidence_id: str | None = None


@dataclass
class AuthorizationDecision:
    allowed: bool
    reasons: list[str]
    action_hash: str
    target_resource: str
    authorization_basis: str
    evidence_sources: list[str]
    execution_permit: ExecutionPermit | None = None
    plan_parser: str | None = None
    evidence_lineage: list[dict[str, Any]] = field(default_factory=list)


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9@._+\-/]+", " ", str(value).lower()).strip()


def _present(value: Any, corpus: str) -> bool:
    value_n, corpus_n = _normal(value), _normal(corpus)
    if not value_n:
        return False
    if value_n.isdigit() or len(value_n) <= 3:
        return value_n in corpus_n.split()
    return value_n in corpus_n


def _canonical_field(value: Any, aliases: Mapping[str, str]) -> str:
    field_name = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return aliases.get(field_name, field_name)


def _canonical_evidence_value(value: Any, field_name: str) -> str:
    if value is None:
        return "<none>"
    text = str(value).strip()
    if field_name in {"email", "email_address"}:
        return text.casefold()
    if field_name in {"phone", "phone_number", "telephone"}:
        prefix = "+" if text.startswith("+") else ""
        return prefix + re.sub(r"\D", "", text)
    return _normal(text)


def _content(message: ChatMessage | dict[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(block.get("content", block.get("text", ""))) if isinstance(block, dict) else str(block)
            for block in value
        )
    return str(value)


def _strip_comments(text: str) -> str:
    """Remove // comments outside quoted strings."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _replace_json_literals(text: str) -> str:
    """Translate JSON literals outside strings for ``ast.literal_eval``."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    replacements = {"null": "None", "true": "True", "false": "False"}
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        matched = False
        for token, replacement in replacements.items():
            if text.startswith(token, i):
                left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
                end = i + len(token)
                right_ok = end == len(text) or not (text[end].isalnum() or text[end] == "_")
                if left_ok and right_ok:
                    out.append(replacement)
                    i = end
                    matched = True
                    break
        if not matched:
            out.append(ch)
            i += 1
    return "".join(out)


def _array_fragment(text: str) -> str:
    text = re.sub(r"^\s*```(?:json|python)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("[")
    if start < 0:
        return text.strip()
    quote: str | None = None
    escaped = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _parse_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    cleaned = _replace_json_literals(_strip_comments(value.strip()))
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return {}


def parse_plan(raw: Any) -> PlanParseResult:
    if isinstance(raw, list):
        parsed, parser_name = raw, "native"
    elif isinstance(raw, str):
        fragment = _array_fragment(_strip_comments(raw.strip()))
        parsed = None
        parser_name = None
        attempts: list[tuple[str, Callable[[str], Any], str]] = [
            ("json", json.loads, fragment),
            ("literal", ast.literal_eval, _replace_json_literals(fragment)),
        ]
        try:
            from json_repair import repair_json
            attempts.append(("json_repair", json.loads, repair_json(fragment)))
        except ImportError:
            pass
        for name, parser, candidate in attempts:
            try:
                parsed = parser(candidate)
                parser_name = name
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        if parsed is None:
            return PlanParseResult((), ("plan_parse_error",), None)
    else:
        return PlanParseResult((), ("plan_type_error",), None)

    if not isinstance(parsed, list):
        return PlanParseResult((), ("plan_not_a_list",), parser_name)
    nodes: list[PlanNode] = []
    errors: list[str] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            errors.append(f"invalid_plan_node:{index}")
            continue
        required = item.get("required parameters", item.get("required_parameters", {}))
        if required is None:
            required = {}
        if not isinstance(required, dict):
            errors.append(f"invalid_required_parameters:{index}")
            continue
        conditions = _parse_mapping(item.get("conditions"))
        nodes.append(PlanNode(item["name"], required, conditions))
    if errors:
        return PlanParseResult(tuple(nodes), tuple(errors), parser_name)
    return PlanParseResult(tuple(nodes), (), parser_name)


def canonical_plan_json(result: PlanParseResult) -> str:
    if not result.ok:
        raise ValueError("cannot canonicalize invalid plan: " + ",".join(result.errors))
    return json.dumps([
        {"name": node.name, "required parameters": node.required_parameters, "conditions": node.conditions or None}
        for node in result.nodes
    ], ensure_ascii=False)


def _tool_name(message: ChatMessage | dict[str, Any]) -> str:
    call = message.get("tool_call") or {}
    if hasattr(call, "function"):
        return str(call.function)
    if isinstance(call, dict):
        function = call.get("function")
        if isinstance(function, str):
            return function
        if isinstance(function, dict):
            return str(function.get("name", ""))
    return ""


def _coerce_detected_item(value: DetectedEvidenceItem | Mapping[str, Any]) -> DetectedEvidenceItem:
    if isinstance(value, DetectedEvidenceItem):
        item = value
    elif isinstance(value, Mapping):
        try:
            item = DetectedEvidenceItem(
                record_id=str(value["record_id"]),
                entity=str(value["entity"]) if value.get("entity") is not None else None,
                field=str(value["field"]),
                value=value["value"],
            )
        except KeyError as exc:
            raise ValueError(f"detector evidence item missing field: {exc.args[0]}") from exc
    else:
        raise TypeError("evidence_items must contain DetectedEvidenceItem or mappings")
    if not item.record_id.strip() or not item.field.strip():
        raise ValueError("detector evidence item requires non-empty record_id and field")
    return item


def _coerce_detection(result: EvidenceDetection | Mapping[str, Any]) -> EvidenceDetection:
    if isinstance(result, EvidenceDetection):
        detection = result
    elif isinstance(result, Mapping):
        try:
            trusted_text = result["trusted_text"]
            untrusted_text = result["untrusted_text"]
            if not isinstance(trusted_text, str) or not isinstance(untrusted_text, str):
                raise TypeError("detector text fields must be strings")
            detection = EvidenceDetection(
                trusted_text=trusted_text,
                untrusted_text=untrusted_text,
                structured=result.get("structured"),
                evidence_items=tuple(
                    _coerce_detected_item(item)
                    for item in result.get("evidence_items", ())
                ),
            )
        except KeyError as exc:
            raise ValueError(f"detector result missing field: {exc.args[0]}") from exc
    else:
        raise TypeError("detector must return EvidenceDetection or a mapping")
    if not isinstance(detection.trusted_text, str) or not isinstance(detection.untrusted_text, str):
        raise TypeError("detector text fields must be strings")
    return EvidenceDetection(
        trusted_text=detection.trusted_text,
        untrusted_text=detection.untrusted_text,
        structured=detection.structured,
        evidence_items=tuple(_coerce_detected_item(item) for item in detection.evidence_items),
    )


def evidence_records(
    messages: Sequence[ChatMessage], detector: EvidenceDetector
) -> list[EvidenceRecord]:
    """Convert raw tool responses into provenance-bearing detector records.

    ``source_tool`` and ``tool_call_id`` always come from the runtime message;
    the semantic detector only decides which response content is trusted.
    """
    records: list[EvidenceRecord] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        text = _content(message)
        source_tool = _tool_name(message)
        tool_call_id = message.get("tool_call_id")
        detection = _coerce_detection(detector.detect(
            text=text,
            source_tool=source_tool,
            tool_call_id=tool_call_id,
        ))
        structured = detection.structured
        if structured is None and detection.trusted_text.strip():
            try:
                structured = yaml.safe_load(detection.trusted_text)
            except yaml.YAMLError:
                pass
        records.append(EvidenceRecord(
            source_tool=source_tool,
            tool_call_id=tool_call_id,
            trusted_text=detection.trusted_text,
            untrusted_text=detection.untrusted_text,
            structured=structured,
            evidence_items=tuple(
                EvidenceItem(
                    source_tool=source_tool,
                    tool_call_id=tool_call_id,
                    record_id=item.record_id,
                    entity=item.entity,
                    field=item.field,
                    value=item.value,
                )
                for item in detection.evidence_items
            ),
        ))
    return records


def _identity_values(value: Any, key: str | None = None) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _identity_values(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _identity_values(child, key)
    elif key in IDENTITY_KEYS and value is not None:
        yield key, str(value)


def _scalar_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalar_values(child)
    elif value is not None and isinstance(value, (str, int, float, bool)):
        yield str(value)


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    raise ValueError("invalid timestamp")


def _grant_from_record(record: EvidenceRecord) -> DelegationGrant | None:
    if not isinstance(record.structured, dict):
        return None
    raw = record.structured.get("authorization_grant")
    if not isinstance(raw, dict):
        return None
    try:
        attributes = raw.get("attributes") or {}
        expires_value = attributes["expires_at"] if "expires_at" in attributes else raw["expires_at"]
        return DelegationGrant(
            issuer=str(raw["issuer"]),
            subject=str(raw.get("subject", "agent")),
            actions=tuple(str(v) for v in raw["actions"]),
            parameter_constraints=dict(raw.get("parameter_constraints") or {}),
            evidence_verified=bool(attributes.get("evidence_verified", raw.get("evidence_verified", False))),
            revoked=bool(attributes.get("revoked", raw.get("revoked", False))),
            expires_at=_timestamp(expires_value),
            delegation_depth=int(attributes.get("delegation_depth", raw.get("delegation_depth", 0))),
            max_delegation_depth=int(attributes.get("max_delegation_depth", raw.get("max_delegation_depth", 0))),
            source_tool=record.source_tool,
            evidence_id=str(raw.get("evidence_id")) if raw.get("evidence_id") is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _constraint_matches(actual: Any, expected: Any) -> bool:
    if expected == "*":
        return True
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _constraint_matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _format_result(result: FunctionReturnType) -> str:
    if isinstance(result, BaseModel):
        return yaml.safe_dump(result.model_dump()).strip()
    if isinstance(result, list):
        values = [item.model_dump() if isinstance(item, BaseModel) else item for item in result]
        return yaml.safe_dump(values).strip()
    return str(result)


class AuthorizationExecutorV2(BasePipelineElement):
    """Execution gate with robust plans, evidence binding and exact approvals."""

    def __init__(
        self,
        planner: Any,
        evidence_detector: EvidenceDetector,
        formatter: Callable[[FunctionReturnType], str] = _format_result,
        mode: str = "dynamic",
        trusted_authority_tools: Iterable[str] = ("get_authorization_grant",),
        field_aliases: Mapping[str, str] | None = None,
        permit_ttl_seconds: float = 300.0,
        policy_version: str = "dynamic-v2",
    ):
        if mode not in {"static", "dynamic"}:
            raise ValueError("mode must be static or dynamic")
        if not callable(getattr(evidence_detector, "detect", None)):
            raise TypeError("evidence_detector must implement detect()")
        self.planner = planner
        self.evidence_detector = evidence_detector
        self.formatter = formatter
        self.mode = mode
        self.trusted_authority_tools = frozenset(trusted_authority_tools)
        aliases = dict(DEFAULT_FIELD_ALIASES)
        aliases.update(field_aliases or {})
        self.field_aliases = {
            _canonical_field(key, {}): _canonical_field(value, {})
            for key, value in aliases.items()
        }
        self.permit_ttl_seconds = permit_ttl_seconds
        self.policy_version = policy_version
        self.events: list[dict[str, Any]] = []
        self.allowed_counts: dict[str, int] = {}
        self.session_id = uuid.uuid4().hex
        self.permits = PermitLedger()
        self.abac = self._build_abac_enforcer()

    def _build_abac_enforcer(self):
        model = casbin.Model()
        model.add_def("r", "r", "sub, obj, act")
        model.add_def("p", "p", "obj, act")
        model.add_def("e", "e", "some(where (p_eft == allow))")
        model.add_def(
            "m", "m",
            "r.sub.evidence_verified == True && r.sub.revoked == False && "
            "r.sub.now <= r.sub.expires_at && "
            "r.sub.delegation_depth <= r.sub.max_delegation_depth && "
            "r.obj == p.obj && r.act == p.act",
        )
        enforcer = casbin.Enforcer(model)
        for tool, permission in self.planner.tool_permissions.items():
            if permission in {"Write", "Execute"}:
                enforcer.add_policy(tool, "execute")
        return enforcer

    def reset(self) -> None:
        self.events.clear()
        self.allowed_counts.clear()
        self.session_id = uuid.uuid4().hex
        self.permits = PermitLedger()

    def _current_node(self, tool: str, parsed: PlanParseResult) -> PlanNode | None:
        nodes = [node for node in parsed.nodes if node.name == tool]
        index = self.allowed_counts.get(tool, 0)
        return nodes[index] if index < len(nodes) else None

    @staticmethod
    def _condition_sources(condition: Any) -> tuple[str, ...]:
        if isinstance(condition, str):
            return (condition,)
        if isinstance(condition, (list, tuple, set)):
            return tuple(str(value) for value in condition)
        if isinstance(condition, Mapping):
            raw = condition.get("source_tool", condition.get("source", condition.get("tool")))
            if isinstance(raw, str):
                return (raw,)
            if isinstance(raw, (list, tuple, set)):
                return tuple(str(value) for value in raw)
        return ()

    def _value_matches_item(self, value: Any, item: EvidenceItem) -> bool:
        field_name = _canonical_field(item.field, self.field_aliases)
        return _canonical_evidence_value(value, field_name) == _canonical_evidence_value(item.value, field_name)

    def _relation_matches(self, item: EvidenceItem, condition: Mapping[str, Any]) -> bool:
        expected_call = condition.get("tool_call_id", condition.get("call_id"))
        expected_record = condition.get("record_id", condition.get("record"))
        expected_entity = condition.get("entity")
        expected_field = condition.get("field")
        if expected_call is not None and str(item.tool_call_id) != str(expected_call):
            return False
        if expected_record is not None and _normal(item.record_id) != _normal(expected_record):
            return False
        if expected_entity is not None:
            entities = expected_entity if isinstance(expected_entity, (list, tuple, set)) else (expected_entity,)
            if item.entity is None or not any(_normal(item.entity) == _normal(entity) for entity in entities):
                return False
        if expected_field is not None:
            fields = expected_field if isinstance(expected_field, (list, tuple, set)) else (expected_field,)
            item_field = _canonical_field(item.field, self.field_aliases)
            if not any(item_field == _canonical_field(field, self.field_aliases) for field in fields):
                return False
        return True

    def _evidence_supports(
        self, value: Any, condition: Any, records: Sequence[EvidenceRecord]
    ) -> tuple[list[EvidenceItem], list[str], bool]:
        """Return matching relations, supporting sources, and relation-mismatch signal.

        Mapping conditions enforce record/entity/field constraints. String and
        list conditions retain the source-only format for older plans.
        """
        sources = self._condition_sources(condition)
        if not sources:
            return [], [], False
        source_records = [
            record for record in records
            if "*" in sources or record.source_tool in sources
        ]
        values = list(_scalar_values(value))
        if not values:
            values = [value]
        fine_grained = isinstance(condition, Mapping)
        matched: list[EvidenceItem] = []
        for expected in values:
            candidates = [
                item
                for record in source_records
                for item in record.evidence_items
                if self._value_matches_item(expected, item)
            ]
            if fine_grained:
                if not candidates:
                    return [], [], False
                relation_matches = [item for item in candidates if self._relation_matches(item, condition)]
                if not relation_matches:
                    return [], [], True
                matched.extend(relation_matches)
            else:
                if candidates:
                    matched.extend(candidates)
                    continue
                text_records = [record for record in source_records if _present(expected, record.trusted_text)]
                if not text_records:
                    return [], [], False
        supporting_sources = sorted({item.source_tool for item in matched})
        if not supporting_sources and not fine_grained:
            supporting_sources = sorted({
                record.source_tool for record in source_records
                if any(_present(expected, record.trusted_text) for expected in values)
            })
        return matched, supporting_sources, False

    @staticmethod
    def _lineage(item: EvidenceItem) -> dict[str, Any]:
        return {
            "source_tool": item.source_tool,
            "tool_call_id": item.tool_call_id,
            "record_id": item.record_id,
            "entity": item.entity,
            "field": item.field,
            "value": item.value,
        }

    def _delegation(
        self, tool: str, args: dict[str, Any], records: Sequence[EvidenceRecord], now: float
    ) -> tuple[DelegationGrant | None, list[str]]:
        relevant: list[tuple[DelegationGrant, list[str]]] = []
        for record in records:
            if record.source_tool not in self.trusted_authority_tools:
                continue
            grant = _grant_from_record(record)
            if grant is None or tool not in grant.actions:
                continue
            reasons: list[str] = []
            if not grant.evidence_verified:
                reasons.append("grant_evidence_unverified")
            if grant.revoked:
                reasons.append("grant_revoked")
            if now > grant.expires_at:
                reasons.append("grant_expired")
            if grant.delegation_depth > grant.max_delegation_depth:
                reasons.append("delegation_depth_exceeded")
            if grant.subject not in {"agent", self.session_id}:
                reasons.append("grant_subject_mismatch")
            attributes = SimpleNamespace(
                evidence_verified=grant.evidence_verified,
                revoked=grant.revoked,
                now=now,
                expires_at=grant.expires_at,
                delegation_depth=grant.delegation_depth,
                max_delegation_depth=grant.max_delegation_depth,
            )
            if not self.abac.enforce(attributes, tool, "execute") and not reasons:
                reasons.append("abac_policy_denied")
            for key, expected in grant.parameter_constraints.items():
                if key not in args or not _constraint_matches(args[key], expected):
                    reasons.append(f"grant_parameter_mismatch:{key}")
            for key in args:
                if key not in grant.parameter_constraints:
                    reasons.append(f"grant_parameter_unconstrained:{key}")
            relevant.append((grant, reasons))
        for grant, reasons in relevant:
            if not reasons:
                return grant, []
        return (None, sorted({reason for _, reasons in relevant for reason in reasons}))

    @staticmethod
    def _target_resource(tool: str, args: dict[str, Any]) -> str:
        identities = list(_identity_values(args))
        if identities:
            return "|".join(f"{key}={value}" for key, value in sorted(identities))
        return tool

    def _issue_permit(self, tool: str, args: dict[str, Any], target: str, basis: str) -> ExecutionPermit:
        return self.permits.issue(
            operation=tool,
            resource=target,
            arguments=copy.deepcopy(args),
            session_id=self.session_id,
            policy_version=self.policy_version,
            authorization_context={"basis": basis, "mode": self.mode},
            ttl_seconds=self.permit_ttl_seconds,
        )

    def verify_permit(self, permit: ExecutionPermit, tool: str, args: dict[str, Any]) -> tuple[bool, list[str]]:
        target = self._target_resource(tool, args)
        check = self.permits.validate(
            permit,
            operation=tool,
            resource=target,
            arguments=args,
            session_id=self.session_id,
        )
        return check.accepted, list(check.problems)

    def consume_permit(self, permit: ExecutionPermit, tool: str, args: dict[str, Any]) -> tuple[bool, list[str]]:
        target = self._target_resource(tool, args)
        check = self.permits.consume(
            permit,
            operation=tool,
            resource=target,
            arguments=args,
            session_id=self.session_id,
        )
        return check.accepted, list(check.problems)

    def decide(
        self, query: str, messages: Sequence[ChatMessage], tool: str, args: dict[str, Any], now: float | None = None
    ) -> AuthorizationDecision:
        permission = self.planner.tool_permissions.get(tool)
        target = self._target_resource(tool, args)
        action_hash = action_fingerprint(tool, target, args)
        if permission == "Read":
            return AuthorizationDecision(True, [], action_hash, target, "read", [], plan_parser=None)
        if permission not in {"Write", "Execute"}:
            return AuthorizationDecision(False, ["unknown_tool_permission"], action_hash, target, "none", [])

        parsed = parse_plan(self.planner.initial_node_checklist)
        if not parsed.ok:
            return AuthorizationDecision(False, list(parsed.errors), action_hash, target, "none", [], plan_parser=parsed.parser)

        try:
            records = evidence_records(messages, self.evidence_detector)
        except Exception:
            return AuthorizationDecision(
                False, ["evidence_detector_failure"], action_hash, target,
                "none", [], plan_parser=parsed.parser,
            )
        trusted_text = "\n".join(record.trusted_text for record in records)
        untrusted_text = "\n".join(record.untrusted_text for record in records)
        node = self._current_node(tool, parsed)
        identities = list(_identity_values(args))
        reasons: list[str] = []
        evidence_sources: set[str] = set()
        evidence_lineage: list[dict[str, Any]] = []
        basis = "planned" if node is not None else "none"

        tainted_identity = any(_present(value, untrusted_text) for _, value in identities)
        for key, value in identities:
            if _present(value, untrusted_text):
                reasons.append(f"untrusted_origin:{key}")
            elif not _present(value, query + "\n" + trusted_text):
                reasons.append(f"missing_trusted_evidence:{key}")

        if node is not None:
            for key, expected in node.required_parameters.items():
                condition_source = node.conditions.get(key)
                if condition_source:
                    matched_items, supporting_sources, relation_mismatch = self._evidence_supports(
                        args.get(key), condition_source, records
                    )
                    if not supporting_sources:
                        if isinstance(condition_source, Mapping) and relation_mismatch:
                            reasons.append(f"evidence_relation_mismatch:{key}")
                        else:
                            reasons.append(f"condition_evidence_missing:{key}")
                    evidence_sources.update(supporting_sources)
                    evidence_lineage.extend(self._lineage(item) for item in matched_items)
                    continue
                if expected is None:
                    continue
                if isinstance(expected, str) and PLACEHOLDER_RE.match(expected):
                    reasons.append(f"unresolved_placeholder:{key}")
                elif key not in args or args.get(key) != expected:
                    reasons.append(f"fixed_parameter_changed:{key}")
        else:
            grant = None
            grant_reasons: list[str] = []
            if self.mode == "dynamic":
                grant, grant_reasons = self._delegation(tool, args, records, now if now is not None else datetime.now(timezone.utc).timestamp())
            if grant is not None:
                basis = "delegated"
                evidence_sources.add(grant.source_tool)
            else:
                direct_action = any(re.search(pattern, _normal(query)) for pattern in DIRECT_ACTION_PATTERNS.get(tool, ()))
                parameters_supported = all(_present(value, query + "\n" + trusted_text) for value in _scalar_values(args))
                if self.mode == "dynamic" and direct_action and parameters_supported and not tainted_identity:
                    basis = "explicit_user_dynamic"
                    evidence_sources.update(record.source_tool for record in records if any(_present(v, record.trusted_text) for v in _scalar_values(args)))
                else:
                    reasons.append("outside_current_authorized_scope")
                    reasons.extend(grant_reasons)

        reasons = sorted(set(reasons))
        if reasons:
            return AuthorizationDecision(
                False, reasons, action_hash, target, basis,
                sorted(evidence_sources), plan_parser=parsed.parser,
                evidence_lineage=evidence_lineage,
            )
        permit = self._issue_permit(tool, args, target, basis)
        return AuthorizationDecision(
            True, [], action_hash, target, basis, sorted(evidence_sources),
            permit, parsed.parser, evidence_lineage,
        )

    def query(
        self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (), extra_args: dict[str, Any] | None = None,
    ):
        extra_args = dict(extra_args or {})
        if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("tool_calls"):
            return query, runtime, env, messages, extra_args
        results = []
        for call in messages[-1]["tool_calls"]:
            args_snapshot = copy.deepcopy(call.args)
            decision = self.decide(query, messages[:-1], call.function, args_snapshot)
            binding_valid = None
            permit_id = decision.execution_permit.permit_id if decision.execution_permit else None
            if decision.allowed and decision.execution_permit is not None:
                binding_valid, binding_problems = self.consume_permit(
                    decision.execution_permit, call.function, call.args
                )
                if not binding_valid:
                    decision.allowed = False
                    decision.reasons.extend(
                        f"action_binding:{problem}" for problem in binding_problems
                    )

            event = {
                "tool": call.function,
                "args": args_snapshot,
                "allowed": decision.allowed,
                "reasons": sorted(set(decision.reasons)),
                "authorization_basis": decision.authorization_basis,
                "evidence_sources": decision.evidence_sources,
                "evidence_lineage": decision.evidence_lineage,
                "action_binding_sha256": decision.action_hash,
                "execution_permit_id": permit_id,
                "binding_valid": binding_valid,
                "policy_version": self.policy_version,
                "abac_engine": "pycasbin" if decision.authorization_basis == "delegated" else None,
                "plan_parser": decision.plan_parser,
            }
            self.events.append(event)
            if not decision.allowed:
                result, error = "", "[AUTHORIZATION DENIED] " + "; ".join(event["reasons"]) + ". Replan the original user task without this action."
            elif call.function == EMPTY_FUNCTION_NAME or call.function not in (function.name for function in runtime.functions.values()):
                result, error = "", f"Invalid tool {call.function} provided."
            else:
                result, error = runtime.run_function(env, call.function, call.args)
                if self.planner.tool_permissions.get(call.function) in {"Write", "Execute"}:
                    self.allowed_counts[call.function] = self.allowed_counts.get(call.function, 0) + 1
            results.append(ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(self.formatter(result))],
                tool_call_id=call.id,
                tool_call=call,
                error=error,
            ))
        extra_args["authorization_events"] = list(self.events)
        extra_args["permit_audit_log"] = self.permits.events
        return query, runtime, env, [*messages, *results], extra_args

