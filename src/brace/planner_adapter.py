"""Compile and validate planner-independent BRACE checklists."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .authorization import PLACEHOLDER_RE, PlanNode, canonical_plan_json, parse_plan


@dataclass(frozen=True)
class ChecklistResult:
    accepted: bool
    reason: str = ""


class PlannerChecklistAdapter:
    """Immutable compiled view of a planner-produced parameter checklist."""

    def __init__(self, raw_checklist: Any):
        parsed = parse_plan(raw_checklist)
        self.errors = parsed.errors
        self._nodes = parsed.nodes
        self.canonical = canonical_plan_json(parsed) if parsed.ok else None

    @property
    def valid(self) -> bool:
        return not self.errors

    def validate(
        self,
        proposed_calls: Iterable[Mapping[str, Any]],
        *,
        completed_steps: int = 0,
    ) -> ChecklistResult:
        if not self.valid:
            return ChecklistResult(
                False,
                "Parameter checklist cannot be parsed: " + ", ".join(self.errors),
            )
        remaining = self._nodes[max(0, completed_steps):]
        for offset, proposal in enumerate(proposed_calls):
            if offset >= len(remaining):
                # The authorization gate decides whether a dynamic extension is valid.
                return ChecklistResult(True)
            outcome = self._validate_one(remaining[offset], proposal)
            if not outcome.accepted:
                return outcome
        return ChecklistResult(True)

    @staticmethod
    def _validate_one(node: PlanNode, proposal: Mapping[str, Any]) -> ChecklistResult:
        function = proposal.get("function")
        if not isinstance(function, Mapping) or function.get("name") != node.name:
            return ChecklistResult(False, "The function name does not align with checklist.")
        arguments = PlannerChecklistAdapter._arguments(function.get("arguments"))
        if arguments is None:
            return ChecklistResult(False, "Function arguments are not valid JSON.")
        for parameter, fixed_value in node.required_parameters.items():
            if PlannerChecklistAdapter._is_dynamic(node, parameter, fixed_value):
                continue
            if parameter not in arguments:
                return ChecklistResult(
                    False,
                    f"The checklist parameter '{parameter}' is missing.",
                )
            if arguments[parameter] != fixed_value:
                return ChecklistResult(
                    False,
                    f"The function parameter '{parameter}' does not equal the checklist value.",
                )
        return ChecklistResult(True)

    @staticmethod
    def _arguments(raw: Any) -> dict[str, Any] | None:
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, Mapping) else None

    @staticmethod
    def _is_dynamic(node: PlanNode, parameter: str, value: Any) -> bool:
        return (
            value is None
            or parameter in node.conditions
            or (isinstance(value, str) and PLACEHOLDER_RE.match(value) is not None)
        )
