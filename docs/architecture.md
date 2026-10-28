# BRACE implementation overview

BRACE sits between an agent's proposed tool call and the tool runtime.

1. The planner checklist is parsed into canonical tool/parameter requirements.
2. The injected LLM semantic detector partitions each raw tool response into
   trusted and untrusted content and extracts record--entity--field--value
   relations; the runtime attaches the originating tool name and call ID.
3. Each sensitive argument is jointly matched against the required source,
   call, record, entity, field, and value relation, or against a constrained
   delegation or explicit-request authorization path.
4. The policy layer evaluates the proposed operation and resource.
5. An allowed action receives a short-lived, single-use execution permit bound to the exact canonical tool name and arguments.
6. The runtime atomically verifies and consumes that permit before execution.

`authorization.py` defines the detector interface, record-level evidence
types, field-alias canonicalization, relational matching, and audit lineage.
`execution_permit.py` implements exact-action execution permits. `planner_adapter.py`
normalizes checklist output and validates proposed calls without depending on
a particular planner implementation.

The concrete semantic trust detector used in the reported experiment is not yet present in this local release tree; it must be added from the separate experiment machine before claiming full paper reproduction. The authorization core no longer reads benchmark attack markers.

